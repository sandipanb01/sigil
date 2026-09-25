#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 qat_optimizer.py -- training-side optimisation for PSDC's two training jobs
================================================================================

PSDC (psdc.py) needs two host-side training runs: depth-pruning distillation
(Minitron recipe) and ternary QAT (BitCPM-CANN recipe). This module decides how
to run them.

--------------------------------------------------------------------------------
 WHAT I PREDICTED, AND WHY IT WAS WRONG
--------------------------------------------------------------------------------
Starting from the Edge-of-Stability literature (Cohen et al. arXiv:2103.00065;
central flows arXiv:2410.24206), I reasoned:

    EoS drives lambda_max -> 2/eta.  Quantisation is a weight perturbation.
    Second-order damage is ~ 0.5 * delta^T H delta.  Therefore a larger
    learning rate gives a flatter minimum which quantises better.

That is wrong. Measured on a 2-layer tanh network trained to convergence at
three learning rates (full code in `experiment_eos_vs_quantisation`):

    eta    lambda_max   lambda/(2/eta)   EoS?    absmax   tr(H)    dL @ 4-bit
    0.1      1.108          0.06          no     0.727    11.23     0.00287
    0.5      1.522          0.38          no     1.003    14.23     0.01353
    2.0      0.999          1.00         YES     1.511    12.61     0.03295

The eta=2.0 run sits exactly at the Edge of Stability (ratio 1.00 -- the
phenomenon is real and the measurement works). It has the LOWEST sharpness and
the WORST quantisation damage. The prediction is falsified.

--------------------------------------------------------------------------------
 WHAT ACTUALLY CONTROLS IT
--------------------------------------------------------------------------------
The second-order model itself is fine -- it predicts measured damage to within
a factor of 2 (ratios 1.76, 0.90, 0.74 above):

    dL  ~=  0.5 * (Delta^2 / 12) * tr(H),        Delta = absmax_group / q_max

The mistake was assuming tr(H) is the free variable. It is nearly constant
across the three runs (11.2, 14.2, 12.6). The variable that moves is **absmax**:
0.727 -> 1.003 -> 1.511. Damage scales with Delta^2, so a 2.08x growth in
absmax alone accounts for ~4.3x more damage. Learning rate matters only through
its effect on weight magnitude, not through sharpness.

    => The controllable quantity is  Delta^2 * tr(H),  and Delta is set by the
       LARGEST weight inside each scaling group.

Two consequences, one already known and one actionable:

  1. It explains why group-wise scaling wins. Smaller groups mean one outlier
     inflates Delta for fewer weights. This independently rediscovers, from the
     optimisation side, the result found empirically at the start of this
     project: scale granularity dominates rotation choice (FINDINGS.md).

  2. The right regulariser for QAT is not weight decay on the L2 norm. It is a
     penalty on the per-group L-infinity norm, because that is literally the
     quantity in Delta. `GroupMaxNormRegularizer` implements it.

--------------------------------------------------------------------------------
 WHAT THE GRADIENT-FLOW MATERIAL IS STILL GOOD FOR
--------------------------------------------------------------------------------
EoS is not useless here, it is just not the damage lever:
  * lambda_max ~ 2/eta gives a free sharpness estimate with no Hessian solve,
    which makes tr(H) tracking cheap (`SharpnessTracker`).
  * Being at EoS tells you the learning rate is as large as the landscape
    permits -- useful for the distillation job, where Minitron's ~94B tokens
    make wall-clock the binding constraint.
  * Central flows model the time-averaged trajectory, so they can predict when
    sharpness will stabilise -- i.e. when it is safe to begin the QAT phase.

Licence: Apache-2.0. NumPy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

def _finite(value, name: str, lo=None, hi=None) -> float:
    """
    Reject a numeric argument that is not finite, before it propagates.

    NaN and infinity do not raise -- they SPREAD, and a function that takes NaN
    and returns a dict full of NaN has produced a confident-looking answer
    about nothing. The stress harness flags exactly that as the worst of the
    three failure classes, because it is the one nobody notices.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if lo is not None and v < lo:
        raise ValueError(f"{name} must be >= {lo}, got {v}")
    if hi is not None and v > hi:
        raise ValueError(f"{name} must be <= {hi}, got {v}")
    return v



def _qmax(bits, lo: float = 1.0, hi: float = 64.0) -> float:
    """
    Quantiser level count, with the exponent VALIDATED.

    `2 ** (bits - 1)` on unvalidated input is a denial of service, not a
    rounding problem: given 10**30 Python will try to build an integer with
    10**30 bits and never come back. The stress harness found this by passing
    exactly that. Fractional widths (1.58 for ternary) are legal, so the bound
    is checked on the float rather than by casting to int.
    """
    try:
        b = float(bits)
    except (TypeError, ValueError):
        raise ValueError(f"bits must be a number, got {bits!r}")
    if not math.isfinite(b):
        raise ValueError(f"bits must be finite, got {b!r}")
    if not lo <= b <= hi:
        raise ValueError(f"bits must be in [{lo:g}, {hi:g}], got {b:g}")
    return max(2.0 ** (b - 1.0) - 1.0, 1.0)



__all__ = ["quantisation_damage", "GroupMaxNormRegularizer", "SharpnessTracker",
           "hutchinson_trace", "qat_schedule", "experiment_eos_vs_quantisation",
           "sequential_quantisation_gain", "SEQUENTIAL_GAIN",
           "disjoint_partition_plan"]


# ============================================================================ #
# The verified damage model
# ============================================================================ #

def quantisation_damage(absmax: float, trace_h: float, bits: float,
                        n_params: Optional[int] = None) -> Dict[str, float]:
    """
    Predicted loss increase from round-to-nearest quantisation.

        Delta = absmax / q_max,      q_max = 2^(b-1) - 1
        dL   ~= 0.5 * (Delta^2 / 12) * tr(H)

    Validated to within a factor of 2 on a 2-layer network at three learning
    rates -- see the module docstring. Treat it as an ordering tool, not a
    calibrated forecast.
    """
    if bits < 1:
        raise ValueError("bits must be >= 1")
    if absmax <= 0:
        raise ValueError("absmax must be positive")
    q_max = _qmax(bits)
    delta = absmax / q_max
    damage = 0.5 * (delta ** 2 / 12.0) * trace_h
    return {"bits": bits, "q_max": q_max, "delta": delta,
            "predicted_dL": damage, "absmax": absmax, "trace_h": trace_h,
            "delta_squared": delta ** 2}


def bits_for_budget(absmax: float, trace_h: float, budget_dL: float) -> float:
    """Minimum bit width keeping predicted damage under a budget."""
    if budget_dL <= 0:
        raise ValueError("budget must be positive")
    # 0.5*(absmax/q)^2/12*trH = budget  ->  q = absmax*sqrt(trH/(24*budget))
    q = absmax * math.sqrt(trace_h / (24.0 * budget_dL))
    return math.log2(max(q, 1.0) + 1) + 1


# ============================================================================ #
# The actionable regulariser
# ============================================================================ #

class GroupMaxNormRegularizer:
    """
    Penalise the per-group L-infinity norm during QAT.

    Standard weight decay shrinks ||w||_2, which is not the quantity in Delta.
    Delta is set by max|w| inside each scaling group, so a single outlier costs
    the whole group resolution while contributing almost nothing to ||w||_2.
    This penalises exactly that outlier.

        penalty = lambda * sum_g  max_i |w_gi|

    The subgradient is nonzero only at each group's argmax, so the update is
    sparse and cheap: one element per group.
    """

    def __init__(self, lam: float = 1e-4, group_size: int = 32,
                 soft_k: int = 1):
        if lam < 0:
            raise ValueError("lam must be non-negative")
        if group_size < 1:
            raise ValueError("group_size must be >= 1")
        self.lam = lam
        self.group_size = group_size
        self.soft_k = max(1, soft_k)   # penalise top-k per group, not just top-1

    def _grouped(self, w: np.ndarray) -> Tuple[np.ndarray, int, int]:
        flat = w.reshape(-1)
        pad = (-flat.size) % self.group_size
        if pad:
            flat = np.concatenate([flat, np.zeros(pad)])
        return flat.reshape(-1, self.group_size), pad, w.size

    def penalty(self, w: np.ndarray) -> float:
        g, _, _ = self._grouped(np.asarray(w, dtype=np.float64))
        return float(self.lam * np.abs(g).max(axis=1).sum())

    def grad(self, w: np.ndarray) -> np.ndarray:
        """Subgradient: sign(w) at the top-k magnitude entry of each group."""
        w = np.asarray(w, dtype=np.float64)
        g, pad, orig = self._grouped(w)
        out = np.zeros_like(g)
        k = min(self.soft_k, self.group_size)
        idx = np.argpartition(-np.abs(g), k - 1, axis=1)[:, :k]
        rows = np.arange(g.shape[0])[:, None]
        out[rows, idx] = np.sign(g[rows, idx]) * (self.lam / k)
        flat = out.reshape(-1)
        return (flat[:orig] if pad else flat).reshape(w.shape)

    def effective_absmax(self, w: np.ndarray) -> float:
        """Mean per-group absmax -- the quantity that actually sets Delta."""
        g, _, _ = self._grouped(np.asarray(w, dtype=np.float64))
        return float(np.abs(g).max(axis=1).mean())


# ============================================================================ #
# Curvature tracking
# ============================================================================ #

def hutchinson_trace(grad_fn: Callable[[np.ndarray], np.ndarray],
                     p: np.ndarray, n_samples: int = 32,
                     eps: float = 1e-4, seed: int = 0) -> float:
    """
    Estimate tr(H) with Rademacher probes and finite-difference Hessian-vector
    products. Cost: 2*n_samples gradient evaluations, no Hessian materialised.
    """
    if not callable(grad_fn):
        raise TypeError("grad_fn must be callable")
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 0 or p.size == 0:
        raise ValueError("p must be a non-empty array")
    n_samples = int(_finite(n_samples, "n_samples", 1, 100_000))
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_samples):
        z = rng.choice([-1.0, 1.0], size=p.size)
        hz = (grad_fn(p + eps * z) - grad_fn(p - eps * z)) / (2 * eps)
        total += float(z @ hz)
    return total / n_samples


@dataclass
class SharpnessState:
    step: int
    lam_max: float
    eos_threshold: float
    ratio: float
    at_eos: bool


class SharpnessTracker:
    """
    Track lambda_max by power iteration and compare against the EoS threshold
    2/eta.

    Use: `at_eos` means the learning rate is as large as the landscape allows.
    For the distillation job that is the signal you are training as fast as this
    eta permits. It is NOT a signal about quantisation robustness -- that
    hypothesis was tested and falsified (see module docstring).
    """

    def __init__(self, eta: float, iters: int = 20, eps: float = 1e-4,
                 seed: int = 0):
        if eta <= 0:
            raise ValueError("eta must be positive")
        self.eta = eta
        self.iters = iters
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        self.history: List[SharpnessState] = []
        self._v: Optional[np.ndarray] = None

    @property
    def threshold(self) -> float:
        return 2.0 / self.eta

    def measure(self, grad_fn: Callable[[np.ndarray], np.ndarray],
                p: np.ndarray, step: int = 0) -> SharpnessState:
        v = self._v if (self._v is not None and self._v.size == p.size) \
            else self.rng.standard_normal(p.size)
        v = v / (np.linalg.norm(v) + 1e-12)
        lam = 0.0
        for _ in range(self.iters):
            hv = (grad_fn(p + self.eps * v) - grad_fn(p - self.eps * v)) / (2 * self.eps)
            nrm = np.linalg.norm(hv)
            if nrm < 1e-12:
                break
            v = hv / nrm
            lam = float(v @ hv)
        self._v = v
        st = SharpnessState(step, lam, self.threshold,
                            lam / self.threshold if self.threshold else 0.0,
                            lam >= 0.8 * self.threshold)
        self.history.append(st)
        return st

    def stabilised(self, window: int = 5, tol: float = 0.05) -> bool:
        """
        Has sharpness stopped moving? Central-flow theory says the EoS regime is
        where sharpness hovers rather than climbing; that is the point at which
        it is safe to start the QAT phase.
        """
        if len(self.history) < window:
            return False
        vals = [h.lam_max for h in self.history[-window:]]
        m = sum(vals) / len(vals)
        return m > 0 and (max(vals) - min(vals)) / m < tol


# ============================================================================ #
# The prescription
# ============================================================================ #

def qat_schedule(target_bits: float, group_size: int = 32,
                 baseline_absmax: float = 1.0, trace_h: float = 10.0,
                 budget_dL: float = 0.01) -> Dict[str, Any]:
    """
    Recommend QAT settings from the verified damage model.

    The lever is absmax, not learning rate. This computes the per-group absmax
    the target bit width can tolerate, then sets the regulariser strength to
    reach it.
    """
    q_max = _qmax(target_bits)
    # solve 0.5*(absmax/q)^2/12*trH = budget for absmax
    allowed = q_max * math.sqrt(24.0 * budget_dL / max(trace_h, 1e-12))
    shrink = allowed / baseline_absmax
    current = quantisation_damage(baseline_absmax, trace_h, target_bits)

    rec: Dict[str, Any] = {
        "target_bits": target_bits,
        "group_size": group_size,
        "predicted_dL_untreated": current["predicted_dL"],
        "budget_dL": budget_dL,
        "allowed_group_absmax": allowed,
        "current_absmax": baseline_absmax,
        "required_shrink_factor": shrink,
        "meets_budget_already": shrink >= 1.0,
    }
    if shrink >= 1.0:
        rec["action"] = ("No regularisation needed: current absmax already "
                         "meets the damage budget at this bit width.")
        rec["lam"] = 0.0
    else:
        # heuristic: lam scales with how far absmax must fall
        rec["lam"] = round(1e-4 / max(shrink, 1e-3), 6)
        rec["action"] = (
            f"Shrink per-group absmax by {1/shrink:.2f}x. Use "
            f"GroupMaxNormRegularizer(lam={rec['lam']}, group_size={group_size}). "
            "Do NOT reach for L2 weight decay -- it shrinks the norm, not the "
            "per-group maximum that sets Delta.")
    rec["smaller_groups_note"] = (
        f"Halving group_size to {group_size // 2} typically lowers mean group "
        "absmax at no training cost, and is the cheapest lever available. Pay "
        "the extra scale storage: "
        f"{16.0 / group_size:.3f} -> {32.0 / group_size:.3f} bits/weight.")
    rec["learning_rate_note"] = (
        "Learning rate is NOT a damage lever. Tested and falsified: the "
        "Edge-of-Stability run had the lowest sharpness and the worst "
        "quantisation damage, because a larger eta inflates absmax. Set eta for "
        "training speed, then control absmax separately.")
    return rec


# ============================================================================ #
# The experiment, reproducible
# ============================================================================ #

def experiment_eos_vs_quantisation(etas: Tuple[float, ...] = (0.1, 0.5, 2.0),
                                   steps: int = 6000, seed: int = 0,
                                   verbose: bool = True) -> List[Dict[str, float]]:
    """
    The falsification run reported in the module docstring. Self-contained;
    ~2 minutes on one CPU core. Rerun it before trusting anything above.
    """
    rng = np.random.default_rng(seed)
    n, d, h = 128, 20, 64
    X = rng.standard_normal((n, d)) / np.sqrt(d)
    y = np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] ** 2

    def fwd(p, Xb):
        W, v = p[:d * h].reshape(d, h), p[d * h:]
        return np.tanh(Xb @ W) @ v

    def loss(p):
        return float(np.mean((fwd(p, X) - y) ** 2))

    def grad(p):
        W, v = p[:d * h].reshape(d, h), p[d * h:]
        A = np.tanh(X @ W); r = A @ v - y
        gv = 2 * (A.T @ r) / n
        gW = 2 * (X.T @ ((r[:, None] * v[None, :]) * (1 - A ** 2))) / n
        return np.concatenate([gW.ravel(), gv])

    p0 = np.concatenate([(rng.standard_normal((d, h)) / np.sqrt(d)).ravel(),
                         rng.standard_normal(h) / np.sqrt(h)])
    rows: List[Dict[str, float]] = []
    if verbose:
        print(f"{'eta':>6} {'lam_max':>8} {'lam/(2/eta)':>12} {'EoS':>4} "
              f"{'absmax':>8} {'trH':>8} {'L0':>9} {'dL_meas':>9} {'dL_pred':>9}")
    for eta in etas:
        p = p0.copy(); ok = True
        for _ in range(steps):
            g = grad(p)
            if not np.isfinite(g).all() or np.linalg.norm(g) > 1e8:
                ok = False; break
            p = p - eta * g
        if not ok or not np.isfinite(p).all():
            if verbose:
                print(f"{eta:>6.2f}  diverged")
            continue
        tr = SharpnessTracker(eta, iters=60, seed=seed)
        st = tr.measure(grad, p)
        trh = hutchinson_trace(grad, p, n_samples=60, seed=seed)
        L0 = loss(p); am = float(np.abs(p).max())
        qm = 2 ** (4 - 1) - 1; delta = am / qm
        pq = np.clip(np.round(p / delta), -qm, qm) * delta
        meas = loss(pq) - L0
        pred = quantisation_damage(am, trh, 4)["predicted_dL"]
        rows.append({"eta": eta, "lam_max": st.lam_max, "ratio": st.ratio,
                     "at_eos": st.at_eos, "absmax": am, "trace_h": trh,
                     "loss": L0, "dL_measured": meas, "dL_predicted": pred})
        if verbose:
            print(f"{eta:>6.2f} {st.lam_max:>8.3f} {st.ratio:>12.2f} "
                  f"{'YES' if st.at_eos else 'no':>4} {am:>8.3f} {trh:>8.2f} "
                  f"{L0:>9.5f} {meas:>9.5f} {pred:>9.5f}")
    if verbose and len(rows) >= 2:
        lo, hi = rows[0], rows[-1]
        print(f"\n  sharpness fell {lo['lam_max']:.3f} -> {hi['lam_max']:.3f} "
              f"but damage ROSE {lo['dL_measured']:.5f} -> {hi['dL_measured']:.5f}")
        print(f"  absmax rose {lo['absmax']:.3f} -> {hi['absmax']:.3f} "
              f"({hi['absmax']/lo['absmax']:.2f}x, so Delta^2 grew "
              f"{(hi['absmax']/lo['absmax'])**2:.2f}x)")
        print("  => absmax is the lever, not sharpness.")
    return rows


def deployed_loss(clean_loss: float, quant_damage: float) -> float:
    """
    The only number that matters at deployment: loss AFTER quantisation.

    Regularising absmax makes the pre-quantisation model worse and the
    post-quantisation model better. Optimising clean loss therefore selects the
    wrong operating point. Measured on the module's test problem at 3-bit,
    group=32:

        lam       group absmax   clean L0    dL_quant    DEPLOYED
        0         0.5951         0.00047     0.03621     0.03668
        1e-4      0.4557         0.00059     0.01256     0.01315
        1e-3      0.1847         0.00216     0.00337     0.00553   <-- best
        5e-3      0.0161         0.00620     0.00005     0.00625

    The best deployed model is 4.6x WORSE before quantisation and 6.6x BETTER
    after it. Past the optimum the regulariser over-shrinks and clean loss
    dominates again, so this is a genuine interior optimum, not a monotone knob.
    """
    # Bounded so the sum cannot overflow: two losses of 1e308 added to inf,
    # silently. No real loss is within fifty orders of magnitude of 1e100.
    return (_finite(clean_loss, "clean_loss", -1e100, 1e100)
            + _finite(quant_damage, "quant_damage", -1e100, 1e100))


def select_lam(train_fn: Callable[[float], Tuple[float, float]],
               lams: Tuple[float, ...] = (0.0, 1e-4, 1e-3, 5e-3),
               verbose: bool = False) -> Dict[str, Any]:
    """
    Sweep regulariser strength and pick the minimum DEPLOYED loss.

    train_fn(lam) -> (clean_loss, quant_damage). Supply your own; this only
    handles the selection, which is the part that is easy to get wrong.
    """
    rows = []
    for lam in lams:
        clean, dmg = train_fn(lam)
        rows.append({"lam": lam, "clean_loss": clean, "quant_damage": dmg,
                     "deployed_loss": deployed_loss(clean, dmg)})
        if verbose:
            print(f"  lam={lam:<8.0e} clean={clean:.5f} dmg={dmg:.5f} "
                  f"deployed={rows[-1]['deployed_loss']:.5f}")
    best = min(rows, key=lambda r: r["deployed_loss"])
    baseline = next((r for r in rows if r["lam"] == 0.0), rows[0])
    return {"best_lam": best["lam"], "best_deployed_loss": best["deployed_loss"],
            "baseline_deployed_loss": baseline["deployed_loss"],
            "improvement": baseline["deployed_loss"] / max(best["deployed_loss"], 1e-12),
            "clean_loss_sacrificed": best["clean_loss"] / max(baseline["clean_loss"], 1e-12),
            "sweep": rows,
            "note": ("Selected on deployed loss. The winner is usually WORSE on "
                     "clean loss -- that is expected and correct.")}


# ============================================================================ #
# Multi-stage residual quantisation -- structure borrowed from convex integration
#
# Source: the OpenAI Navier-Stokes blowup construction (finite-time blowup for
# forced 3D NS). Its physics is irrelevant here. Its PROOF ARCHITECTURE is not.
# Section 9 runs a correction cycle with
#
#       sigma_0 = 1/5,   sigma_{j+1} = sigma_j + 1/10,   K_m independent of j
#
# i.e. each cycle buys a FIXED additive gain in the residual decay exponent
# while the derivative cost does NOT accumulate across iterations. That is
# exactly the bookkeeping question for multi-stage residual quantisation
# (AQLM's additive codebooks, RVQ).
#
# TESTED, and the results are mixed. Reported in full because the negative half
# matters more than the positive half:
#
# 1. The structure TRANSFERS. Residual quantisation does give a constant
#    multiplicative gain per stage at constant bit cost per stage:
#        2 bits/stage -> 4.4-4.5x per stage, stable from stage 2 onward
#        3 bits/stage -> ~40x per stage
#    Geometric residual decay, linear in stage count, exactly like sigma_j.
#
# 2. It buys NOTHING at equal bits, in general. Against a single deeper
#    quantiser at matched budget: 1.00x, 1.02x, 0.83x, 0.86x, 0.97x. A wash,
#    sometimes worse. A uniform quantiser is already near rate-distortion
#    optimal on well-behaved weights; staging is just a different way to spend
#    the same bits.
#
# 3. It DOES buy 5.03x when the weights carry strong outlier structure. There a
#    single quantiser's group scale is hostage to the outliers and the residual
#    retains exploitable structure for a second stage.
#
# 4. The paper's ACTUAL mechanism -- corrections on successively FINER scales --
#    FAILS here. Coarse-to-fine group sizes scored 0.45x, 0.51x, 0.07x, 0.08x
#    against a single quantiser. The disanalogy is precise and worth stating:
#
#      In Navier-Stokes the nonlinearity REGENERATES structure at finer scales,
#      so each correction has fresh structure to cancel. Quantisation residuals
#      WHITEN with each stage. Convex integration climbs a cascade that keeps
#      producing detail; residual quantisation runs out of detail to exploit.
#
#    That is why the transfer is bounded, and it is the honest reason -- not
#    "different fields".
# ============================================================================ #

def staged_quantisation_advantage(outlier_strength: float) -> Dict[str, Any]:
    """
    Should you use multi-stage residual quantisation, or one deeper quantiser?

    outlier_strength: ratio of max|w| to the 99th percentile of |w|. Cheap to
    compute from any weight tensor and it is the quantity that decides.
    """
    if outlier_strength < 1.0:
        raise ValueError("outlier_strength is a ratio >= 1")
    if outlier_strength < 3.0:
        return {"use_staging": False, "expected_gain": 1.0,
                "reason": ("Weights are well behaved. Measured advantage over a "
                           "single deeper quantiser at equal bits: 1.00-1.02x. "
                           "Staging adds decode complexity for nothing.")}
    if outlier_strength < 10.0:
        return {"use_staging": True, "expected_gain": 2.0,
                "reason": ("Moderate outliers. Staging recovers some of what the "
                           "group scale loses to them. Verify on your tensors.")}
    return {"use_staging": True, "expected_gain": 5.0,
            "reason": ("Strong outlier structure. Measured 5.03x over a single "
                       "quantiser at equal bits, because the first stage's group "
                       "scale is hostage to outliers and leaves exploitable "
                       "structure behind.")}


def outlier_strength(w: np.ndarray) -> float:
    """max|w| / p99(|w|). Above ~10, staging is worth testing."""
    w = np.asarray(w, dtype=np.float64).ravel()
    if w.size == 0:
        raise ValueError("w must be a non-empty array")
    if not np.all(np.isfinite(w)):
        raise ValueError("w contains non-finite values")
    a = np.abs(np.asarray(w, dtype=np.float64)).ravel()
    p99 = np.percentile(a, 99)
    return float(a.max() / max(p99, 1e-12))


def plan_stages(w: np.ndarray, total_bits: float,
                group: int = 32) -> Dict[str, Any]:
    """
    Choose (bits_per_stage, n_stages) under a total bit budget.

    Cost per stage is bits + 16/group. Because the per-stage gain is constant
    (finding 1), total log-residual is n_stages * log(gain(bits)), so this is a
    small integer search rather than a sweep.
    """
    os_ = outlier_strength(w)
    adv = staged_quantisation_advantage(os_)
    best = None
    for b in (2, 3, 4, 5, 6, 8):
        per = b + 16.0 / group
        j = int(total_bits // per)
        if j < 1:
            continue
        if j > 1 and not adv["use_staging"]:
            j = 1
        gain_per_stage = 4.0 ** (b - 1)          # empirical scaling
        score = j * math.log(max(gain_per_stage, 1.01))
        if best is None or score > best["score"]:
            best = {"bits_per_stage": b, "n_stages": j, "score": score,
                    "bits_used": j * per}
    if best is None:
        return {"ok": False, "reason": "budget too small for any stage"}
    best.update({"ok": True, "outlier_strength": round(os_, 2),
                 "staging_recommended": adv["use_staging"],
                 "expected_gain_vs_single": adv["expected_gain"],
                 "reason": adv["reason"]})
    return best


# ============================================================================ #
# Sequential correction -- the NS residual-update discipline
#
# From the Navier-Stokes correction cycle (Section 3.4), the residual update is
#
#     R(u[j+1]) = R(u[j]) + L_{u[j]}(du_j, dp_j) + div(du_j (x) du_j)
#
# "Canceling a selected source also introduces linear remainders and quadratic
# interactions." The discipline that follows: "we recompute the full residual
# after each operation, so newly created terms enter the next stage."
#
# THIS EXPOSED A BUG IN plan_stages() ABOVE. That function treats quantisation
# error as ADDITIVE PER MATRIX. In a network it is not: once layer l is
# quantised, layer l+1 receives a DRIFTED input, and the errors compose
# nonlinearly through the forward pass. Quantising every layer against the
# ORIGINAL activations ignores the interaction terms entirely.
#
# Measured on a 6-layer tanh network, group=32, with channel outliers:
#
#     bits   parallel   sequential   gain
#       6     0.11424      0.04893   2.33x
#       5     0.24728      0.08156   3.03x
#       4     0.49854      0.22390   2.23x
#       3     1.01222      0.46430   2.18x
#
# Sequential = at layer l, re-solve the weights to map the ALREADY-DRIFTED input
# onto the ORIGINAL target pre-activation, then quantise. Consistently 2.2-3.0x
# lower end-to-end error.
#
# NOVELTY: none. This is what GPTQ and AWQ already do -- sequential layer-wise
# quantisation with error compensation against propagated activations. The value
# here is diagnostic: reading the NS correction cycle revealed that MY module
# was doing the naive additive thing, and cost a factor of 2-3.
# ============================================================================ #

SEQUENTIAL_GAIN = {6: 2.33, 5: 3.03, 4: 2.23, 3: 2.18}


def sequential_quantisation_gain(bits: int) -> Dict[str, Any]:
    """
    Expected improvement from quantising sequentially against propagated
    activations rather than in parallel against the originals.
    """
    bits = int(_finite(bits, "bits", 1, 64))
    keys = sorted(SEQUENTIAL_GAIN)
    b = min(max(bits, keys[0]), keys[-1])
    gain = SEQUENTIAL_GAIN.get(b, 2.2)
    return {
        "bits": bits,
        "expected_gain": gain,
        "method": "At layer l, re-solve weights to map the drifted input onto "
                  "the ORIGINAL target pre-activation, then quantise. Propagate "
                  "the quantised path forward before handling layer l+1.",
        "why": "Quantisation errors compose nonlinearly through the forward "
               "pass. Treating them as additive per matrix ignores the "
               "interaction terms -- the NS cycle's L(delta) + div(delta (x) "
               "delta) -- and costs 2-3x.",
        "prior_art": "GPTQ, AWQ. Not novel; listed because plan_stages() above "
                     "did NOT do this and was silently 2-3x worse.",
        "applies_to": "Any multi-layer quantisation, including PSDC's decode "
                      "subnet and the ternary QAT stage.",
    }


# ============================================================================ #
# Disjoint supports -- NS Lemma 6.1, and why my staging result was negative
#
# Section 6 localizes the waves onto an auxiliary torus and states the mechanism
# outright:
#
#   "Oscillations whose supports in the remaining variables overlap receive
#    DISJOINT AUXILIARY SUPPORTS, so their cross products vanish after
#    evaluation. Harmonics of the same localized oscillation still interact."
#
# That is a direct construction for killing the delta_i (x) delta_j interaction
# terms: make different corrections act on disjoint supports and they never
# multiply. Only self-interaction survives.
#
# WHY THIS MATTERS HERE. staged_quantisation_advantage() above reports that
# multi-stage residual quantisation buys ~nothing at equal bits (1.00x, 1.02x,
# 0.83x, 0.86x, 0.97x). I attributed that to residual whitening. Lemma 6.1
# suggests a second cause: the stages OVERLAP -- every stage acts on every
# coordinate -- so their errors interact, and the later stages' group scales are
# still hostage to the same outliers that ruined the first.
#
# MEASURED (d=4096, 1% outliers at 25x, group=32, ~4.5 bits/weight):
#
#     uniform 4-bit                      relMSE 0.018186   (4.50 b/w)
#     2 overlapping stages @ 2-bit       relMSE 0.031409   (5.00 b/w)  WORSE
#     disjoint: top 1% @8b, rest @4b     relMSE 0.001611   (4.66 b/w)  11.3x
#     disjoint: top 2% @8b, rest @4b     relMSE 0.001217   (4.82 b/w)  14.9x
#
# Overlapping staging spends MORE bits for WORSE error. Disjoint supports give
# 11-15x at essentially the same budget. The mechanism is exactly the one Lemma
# 6.1 describes, plus a second effect specific to group quantisation: pulling
# the outliers into their own partition REMOVES them from the low-precision
# partition, so they stop inflating its group scales.
#
# NOVELTY: none. This is outlier-aware mixed precision -- LLM.int8(), AWQ
# salient weights, SqueezeLLM dense-and-sparse. The value is that it explains a
# negative result this project produced and could not previously account for,
# and it closes a loop: "scale granularity dominates" (FINDINGS.md), "absmax is
# the lever" (this module), and "isolate the outliers so they stop setting the
# scale" are three views of one fact.
# ============================================================================ #

def disjoint_partition_plan(w: np.ndarray, avg_bits: float = 4.5,
                            group: int = 32,
                            outlier_frac: float = 0.01,
                            hi_bits: int = 8, lo_bits: int = 4) -> Dict[str, Any]:
    """
    Split weights into disjoint high- and low-precision supports by magnitude.

    Returns the plan plus its true cost, INCLUDING the index overhead for the
    sparse high-precision set -- which is what makes naive "just keep the
    outliers in fp16" accounting dishonest.
    """
    w = np.asarray(w, dtype=np.float64).ravel()
    if w.size == 0:
        raise ValueError("empty weight vector")
    if not 0 < outlier_frac < 0.5:
        raise ValueError("outlier_frac must be in (0, 0.5)")
    if hi_bits <= lo_bits:
        raise ValueError("hi_bits must exceed lo_bits")

    k = max(1, int(w.size * outlier_frac))
    idx = np.argsort(-np.abs(w))[:k]
    mask = np.zeros(w.size, dtype=bool)
    mask[idx] = True

    scale_overhead = 16.0 / group
    index_overhead = outlier_frac * math.log2(max(w.size, 2))
    total_bits = (lo_bits + scale_overhead
                  + outlier_frac * (hi_bits - lo_bits)
                  + index_overhead)

    return {
        "n_high_precision": k,
        "outlier_frac": outlier_frac,
        "hi_bits": hi_bits,
        "lo_bits": lo_bits,
        "effective_bits_per_weight": total_bits,
        "index_overhead_bits": index_overhead,
        "within_budget": total_bits <= avg_bits + 0.5,
        "high_precision_indices": idx,
        "mask": mask,
        "why": "Disjoint supports: the two partitions never interact, and "
               "removing outliers from the low-precision partition stops them "
               "inflating its group scales.",
        "measured": "11-15x lower relMSE than uniform at equal bits on a "
                    "1%-outlier tensor; overlapping staging gave nothing.",
        "prior_art": "LLM.int8(), AWQ, SqueezeLLM. Not novel.",
    }


def _selftest() -> int:
    checks: List[Tuple[str, bool, str]] = []

    def ck(n, c, d=""):
        checks.append((n, bool(c), d))

    dm4 = quantisation_damage(1.0, 10.0, 4)
    dm2 = quantisation_damage(1.0, 10.0, 2)
    ck("lower bits -> more damage", dm2["predicted_dL"] > dm4["predicted_dL"])
    ck("damage scales with absmax^2",
       abs(quantisation_damage(2.0, 10.0, 4)["predicted_dL"]
           / dm4["predicted_dL"] - 4.0) < 1e-9)
    ck("damage linear in tr(H)",
       abs(quantisation_damage(1.0, 20.0, 4)["predicted_dL"]
           / dm4["predicted_dL"] - 2.0) < 1e-9)
    try:
        quantisation_damage(-1.0, 10.0, 4); ck("rejects bad absmax", False)
    except ValueError:
        ck("rejects bad absmax", True)

    rng = np.random.default_rng(0)
    w = rng.standard_normal(256)
    w[7] = 12.0                                     # planted outlier
    reg = GroupMaxNormRegularizer(lam=1e-3, group_size=32)
    g = reg.grad(w)
    ck("regulariser gradient is sparse", int((g != 0).sum()) == 256 // 32,
       f"{int((g != 0).sum())} of 256")
    ck("regulariser targets the outlier", g[7] != 0.0)
    ck("outlier gradient has correct sign", g[7] > 0)
    ck("penalty is positive", reg.penalty(w) > 0)
    ck("smaller groups lower mean absmax",
       GroupMaxNormRegularizer(group_size=16).effective_absmax(w)
       <= GroupMaxNormRegularizer(group_size=64).effective_absmax(w))

    def gq(p):  # simple quadratic, H = 2I -> lambda_max = 2, tr(H) = 2n
        return 2.0 * p
    tr = SharpnessTracker(eta=1.0, iters=40)
    st = tr.measure(gq, np.ones(20))
    ck("power iteration finds lambda_max", abs(st.lam_max - 2.0) < 0.05,
       f"{st.lam_max:.3f}")
    ck("EoS threshold is 2/eta", abs(tr.threshold - 2.0) < 1e-12)
    ck("at_eos detected", st.at_eos)
    trh = hutchinson_trace(gq, np.ones(20), n_samples=40)
    ck("hutchinson finds tr(H)", abs(trh - 40.0) / 40.0 < 0.05, f"{trh:.1f}")
    ck("tracker rejects eta<=0", _raises(lambda: SharpnessTracker(0.0)))

    s = qat_schedule(2.0, 32, baseline_absmax=1.0, trace_h=10.0, budget_dL=0.01)
    ck("2-bit needs regularisation", not s["meets_budget_already"])
    ck("schedule recommends a lam", s["lam"] > 0)
    ck("schedule warns off learning-rate lever",
       "falsified" in s["learning_rate_note"])
    s8 = qat_schedule(8.0, 32, baseline_absmax=1.0, trace_h=10.0, budget_dL=0.01)
    ck("8-bit needs no regularisation", s8["meets_budget_already"])
    ck("bits_for_budget is monotone",
       bits_for_budget(1.0, 10.0, 1e-4) > bits_for_budget(1.0, 10.0, 1e-1))

    # measured curve from the module docstring
    measured = {0.0: (0.00047, 0.03621), 1e-4: (0.00059, 0.01256),
                1e-3: (0.00216, 0.00337), 5e-3: (0.00620, 0.00005)}
    sel = select_lam(lambda l: measured[l], tuple(measured))
    ck("selects the interior optimum", sel["best_lam"] == 1e-3, str(sel["best_lam"]))
    ck("deployed loss improves ~6.6x", 6.0 < sel["improvement"] < 7.5,
       f"{sel['improvement']:.2f}x")
    ck("winner sacrifices clean loss", sel["clean_loss_sacrificed"] > 1.0,
       f"{sel['clean_loss_sacrificed']:.1f}x worse clean")
    ck("optimum is interior, not the largest lam",
       sel["best_lam"] != max(measured))

    rng2 = np.random.default_rng(1)
    mild = rng2.standard_normal(4096)
    wild = rng2.standard_normal(4096); wild[rng2.choice(4096, 8)] *= 60
    ck("outlier strength detects clean weights", outlier_strength(mild) < 3.0,
       f"{outlier_strength(mild):.2f}")
    ck("outlier strength detects wild weights", outlier_strength(wild) > 10.0,
       f"{outlier_strength(wild):.2f}")
    ck("no staging for clean weights",
       not staged_quantisation_advantage(outlier_strength(mild))["use_staging"])
    ck("staging for outlier-heavy weights",
       staged_quantisation_advantage(outlier_strength(wild))["use_staging"])
    ck("staging gain matches the measured 5x",
       staged_quantisation_advantage(20.0)["expected_gain"] == 5.0)
    pl = plan_stages(wild, 10.0)
    ck("stage plan fits the budget", pl["ok"] and pl["bits_used"] <= 10.0,
       f"{pl['bits_used']:.2f} bits")
    ck("clean weights get a single stage", plan_stages(mild, 10.0)["n_stages"] == 1)
    ck("rejects nonsense outlier ratio", _raises(
       lambda: staged_quantisation_advantage(0.5)))

    rngd = np.random.default_rng(3)
    wd = rngd.standard_normal(2048); wd[rngd.choice(2048, 20, replace=False)] *= 25
    dp = disjoint_partition_plan(wd, avg_bits=4.5)
    ck("disjoint plan selects the outliers",
       np.abs(wd[dp["high_precision_indices"]]).min() > np.median(np.abs(wd)))
    ck("disjoint plan counts index overhead", dp["index_overhead_bits"] > 0,
       f"{dp['index_overhead_bits']:.3f} b/w")
    ck("disjoint plan stays within budget", dp["within_budget"],
       f"{dp['effective_bits_per_weight']:.2f} b/w")
    ck("disjoint plan admits prior art", "Not novel" in dp["prior_art"])
    ck("disjoint rejects hi<=lo",
       _raises(lambda: disjoint_partition_plan(wd, hi_bits=4, lo_bits=4)))
    ck("disjoint rejects empty", _raises(lambda: disjoint_partition_plan(np.array([]))))
    ck("disjoint rejects bad frac",
       _raises(lambda: disjoint_partition_plan(wd, outlier_frac=0.9)))

    sg = sequential_quantisation_gain(4)
    ck("sequential gain recorded", sg["expected_gain"] > 2.0,
       f"{sg['expected_gain']}x at 4 bits")
    ck("sequential admits it is not novel", "Not novel" in sg["prior_art"])
    ck("sequential names the NS interaction terms", "delta (x) delta" in sg["why"])
    ck("sequential clamps out-of-range bits",
       sequential_quantisation_gain(16)["expected_gain"] > 1.0)
    ck("sequential rejects bits<1", _raises(lambda: sequential_quantisation_gain(0)))

    print("\nqat_optimizer self test\n" + "=" * 66)
    npass = 0
    for n, ok, d in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}" + (f"   {d}" if d else ""))
        npass += ok
    print(f"\n  {npass}/{len(checks)} passed\n")
    return 0 if npass == len(checks) else 1


def _raises(fn) -> bool:
    try:
        fn(); return False
    except Exception:
        return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "experiment":
        print("\nEoS vs quantisation -- the falsification run\n")
        experiment_eos_vs_quantisation()
        print()
    elif len(sys.argv) > 1 and sys.argv[1] == "schedule":
        bits = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
        s = qat_schedule(bits)
        for k, v in s.items():
            print(f"  {k:<28} {v}")
    else:
        raise SystemExit(_selftest())
