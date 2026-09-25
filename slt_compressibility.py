#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 slt_compressibility.py -- singular learning theory applied to on-device
                           quantisation, including where it did NOT work
================================================================================

Sources read for this module:
  * Urdshals, Lau, Hoogland, van Wingerden, Murfet, "Compressibility Measures
    Complexity: Minimum Description Length Meets Singular Learning Theory"
    (arXiv:2510.12077, Timaeus + UK AISI, Oct 2025)
  * Watanabe (2009) via Theorem 2 therein; Lau et al., "The Local Learning
    Coefficient: A Singularity-Aware Complexity Measure" (AISTATS 2025)
  * timaeus.co/research (Timaeus has since merged into Resolution)

--------------------------------------------------------------------------------
 WHY THIS MATTERS FOR THIS PROJECT
--------------------------------------------------------------------------------
qat_optimizer.py models quantisation damage as

        dL  ~=  0.5 * (Delta^2 / 12) * tr(H)

and it predicts measured damage to within a factor of 2. But SMDL says that
model is built on the wrong invariant:

    "In contrast to the classical treatment of MDL, where geometric invariants
     like the curvature determined by the Hessian appear in the description
     length, the important geometric feature in the singular case is
     DEGENERACY."

Neural networks are singular: the parameter-to-distribution map is not
one-to-one and the Fisher information matrix is degenerate. Classical MDL
assumes neither. So the Hessian is a leading-order quantity only for REGULAR
models, and networks are not regular.

The replacement invariant comes from Watanabe's theorem (resolution of
singularities, Hironaka 1964):

    Vol({ w : L(w) - L0 <= eps })  ~  c * eps^lambda * (-log eps)^(m-1)

lambda is the real log canonical threshold (RLCT), also called the learning
coefficient; m is its multiplicity. For a REGULAR model the sublevel sets are
ellipsoidal with volume ~ eps^(d/2), giving lambda = d/2. Deviation of lambda
from d/2 is exactly the degeneracy the Hessian cannot see.

SMDL then proves the two-part-code redundancy is

    R_n = lambda * log n - (m-1) * log log n + O_p(1)

and reports empirically on Pythia (to 6.9B) that for QUANTIZATION specifically
there is a close, in places LINEAR, relationship between estimated LLC and
compressibility measured in bits.

--------------------------------------------------------------------------------
 WHAT I TRIED, AND WHAT FAILED
--------------------------------------------------------------------------------
I implemented the volume-scaling estimator directly from Watanabe's theorem:
sample uniformly in a ball around the trained parameter, count the fraction with
loss below each threshold, fit the log-log slope to recover lambda.

It does not resolve. Measured on 2-layer tanh networks:

  target    width    D     L0        lambda_hat   bits to 10% tolerance
  simple      16    144   0.00002       2.02              8
  simple      32    288   0.00002       2.33             10
  simple      64    576   0.00002       2.24              9
  complex     16    144   0.00201       2.37              7
  complex     32    288   0.00169       2.34              7
  complex     64    576   0.00154       2.21              7

lambda_hat sits at ~2.2 for every configuration -- both targets, every width.
It carries no signal about compressibility here.

**This is a negative result about MY ESTIMATOR, not about SMDL.** Timaeus
publish several papers specifically on LLC estimation methodology ("From Global
to Local: A Scalable Benchmark for Local Posterior Sampling"; "Guide for
Sampling Hyperparameter Selection") and state plainly that "we lack theoretical
knowledge of the true LLC for large transformer models". Their results use
SGLD-based local posterior sampling, not naive ball sampling. Naive volume
estimation in high dimension fails because almost all the mass of a ball sits
near its surface, so the sample never explores the degenerate directions that
carry the signal.

**Correct tool: github.com/timaeus-research/devinterp.** Do not use the
estimator below for anything load-bearing; it is here to document the failure
and to be replaced.

--------------------------------------------------------------------------------
 THE ONE THING THAT DOES SURVIVE, AND IT IS USEFUL
--------------------------------------------------------------------------------
The absmax result from qat_optimizer.py and the SLT picture are not rivals. They
are different factors of the same product:

    damage  =  (how far quantisation moves you)  x  (what the landscape charges
                                                     you for moving that far)
                    ^ Delta, set by absmax             ^ degeneracy, i.e. lambda

qat_optimizer.py controls the first factor and can do so today, with a measured
6.6x improvement in deployed loss. The second factor is real, is the
theoretically correct leading-order term, and I could not measure it. Both
statements belong in the record.

Licence: Apache-2.0. NumPy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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



__all__ = ["regular_lambda", "volume_scaling_lambda", "degeneracy_ratio",
           "bits_to_tolerance", "singular_mdl_redundancy",
           "damage_decomposition", "LLC_ESTIMATION_WARNING"]


LLC_ESTIMATION_WARNING = (
    "The volume-scaling estimator in this module returned ~2.2 for every model "
    "tested -- it carries no signal. Naive ball sampling fails in high dimension "
    "because nearly all the mass sits near the surface and never explores the "
    "degenerate directions. Use github.com/timaeus-research/devinterp (SGLD "
    "local posterior sampling) for any real LLC estimate."
)


def regular_lambda(d: int) -> float:
    """
    Learning coefficient of a REGULAR model: lambda = d/2, because sublevel sets
    are ellipsoidal with volume ~ eps^(d/2). Networks are not regular, so this
    is the upper reference against which degeneracy is measured, never the
    answer.
    """
    d = _finite(d, "d", 0.0, 1e12)
    if d < 1:
        raise ValueError("d must be >= 1")
    return d / 2.0


def volume_scaling_lambda(loss_fn: Callable[[np.ndarray], float],
                          p: np.ndarray,
                          n_samples: int = 3000,
                          radius_frac: float = 0.05,
                          seed: int = 0,
                          n_thresholds: int = 12) -> Dict[str, Any]:
    """
    Estimate lambda from Vol(eps) ~ eps^lambda by ball sampling.

    KNOWN TO FAIL -- see LLC_ESTIMATION_WARNING and the module docstring. The
    return dict always carries `reliable: False` so no caller can use this
    result without seeing that.
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(p, dtype=np.float64)
    L0 = loss_fn(p)
    R = radius_frac * math.sqrt(p.size) * float(np.std(p))
    if R <= 0:
        return {"lambda_hat": float("nan"), "reliable": False,
                "reason": "degenerate parameter scale",
                "warning": LLC_ESTIMATION_WARNING}
    z = rng.standard_normal((n_samples, p.size))
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    rad = R * rng.random(n_samples) ** (1.0 / p.size)
    dl = np.array([loss_fn(p + rad[i] * z[i]) - L0 for i in range(n_samples)])
    dl = dl[dl > 0]
    if dl.size < 50:
        return {"lambda_hat": float("nan"), "reliable": False,
                "reason": "too few positive loss increases",
                "warning": LLC_ESTIMATION_WARNING}
    eps = np.exp(np.linspace(math.log(np.percentile(dl, 5)),
                             math.log(np.percentile(dl, 60)), n_thresholds))
    frac = np.array([(dl <= e).mean() for e in eps])
    ok = frac > 0
    if ok.sum() < 3:
        return {"lambda_hat": float("nan"), "reliable": False,
                "reason": "insufficient distinct thresholds",
                "warning": LLC_ESTIMATION_WARNING}
    slope = float(np.polyfit(np.log(eps[ok]), np.log(frac[ok]), 1)[0])
    return {"lambda_hat": slope, "reliable": False,
            "n_effective": int(dl.size), "radius": R,
            "reason": "estimator does not resolve; documented failure",
            "warning": LLC_ESTIMATION_WARNING}


def degeneracy_ratio(lambda_hat: float, d: int) -> Dict[str, Any]:
    """
    lambda / (d/2). 1.0 means regular; below 1 means degenerate; the lower the
    value, the more redundant the parameterisation and (per SMDL) the more
    compressible the model.
    """
    # |lambda| cannot exceed d/2 for a regular model, and d <= 1e12, so 1e12
    # is generous; an unbounded lambda of 1e308 returned ratio=inf silently.
    lambda_hat = _finite(lambda_hat, "lambda_hat", -1e12, 1e12)
    d = int(_finite(d, "d", 1, 10 ** 12))
    reg = regular_lambda(d)
    ratio = lambda_hat / reg
    if ratio > 0.9:
        verdict = "near-regular: little redundancy, expect poor compressibility"
    elif ratio > 0.3:
        verdict = "moderately degenerate"
    else:
        verdict = "highly degenerate: substantial redundancy, expect good compressibility"
    return {"lambda_hat": lambda_hat, "regular_lambda": reg,
            "ratio": ratio, "verdict": verdict,
            "caveat": "Only meaningful with a RELIABLE lambda estimate. "
                      + LLC_ESTIMATION_WARNING}


def bits_to_tolerance(loss_fn: Callable[[np.ndarray], float],
                      p: np.ndarray, tol_frac: float = 0.10,
                      group: int = 32, max_bits: int = 16) -> Dict[str, Any]:
    """
    SMDL's operational compressibility: the fewest bits per weight keeping loss
    within (1 + tol_frac) * L0. This is measurable without any LLC estimate,
    which is why it is the quantity to report.
    """
    if not 0 < tol_frac < 10:
        raise ValueError("tol_frac must be a positive fraction")
    p = np.asarray(p, dtype=np.float64)
    L0 = loss_fn(p)
    budget = L0 * (1.0 + tol_frac)
    n_full = (p.size // group) * group
    for b in range(2, max_bits + 1):
        qmax = _qmax(b)
        f = p[:n_full].reshape(-1, group)
        s = np.abs(f).max(axis=1, keepdims=True) / qmax
        s[s == 0] = 1e-12
        pq = p.copy()
        pq[:n_full] = (np.clip(np.round(f / s), -qmax, qmax) * s).ravel()
        if loss_fn(pq) <= budget:
            return {"bits": b, "baseline_loss": L0, "tolerance": budget,
                    "achieved_loss": loss_fn(pq), "group": group,
                    "effective_bits": b + 16.0 / group}
    return {"bits": None, "baseline_loss": L0, "tolerance": budget,
            "group": group,
            "reason": f"no bit width <= {max_bits} met the tolerance"}


def singular_mdl_redundancy(lambda_: float, n: int, multiplicity: int = 1) -> float:
    """
    SMDL Theorem 1:  R_n = lambda*log n - (m-1)*log log n + O_p(1).

    The two-part-code redundancy, i.e. the bits needed to specify the model
    itself. Note this GROWS with lambda: a more complex (less degenerate) model
    costs more to describe and is therefore less compressible.
    """
    n = _finite(n, "n", 1.0, 1e30)
    lambda_ = _finite(lambda_, "lambda_", -1e12, 1e12)
    if n < 3:
        raise ValueError("n must be >= 3 for log log n")
    if multiplicity < 1:
        raise ValueError("multiplicity must be >= 1")
    return lambda_ * math.log(n) - (multiplicity - 1) * math.log(math.log(n))


def damage_decomposition(absmax: float, bits: float, trace_h: float,
                         lambda_hat: Optional[float] = None,
                         d: Optional[int] = None) -> Dict[str, Any]:
    """
    Separate the two factors of quantisation damage.

        damage = (displacement) x (landscape price of displacement)
                  ^ Delta, set by absmax   ^ degeneracy (lambda), NOT tr(H)

    The Hessian term is retained because it is measurable and predicts within a
    factor of 2 in practice; it is flagged as theoretically the wrong
    leading-order invariant for a singular model, per SMDL.
    """
    q_max = _qmax(bits)
    delta = absmax / q_max
    hessian_estimate = 0.5 * (delta ** 2 / 12.0) * trace_h
    out: Dict[str, Any] = {
        "displacement_factor": {"delta": delta, "delta_squared": delta ** 2,
                                "controlled_by": "absmax within each scaling group",
                                "actionable": True,
                                "measured_effect": "6.6x better deployed loss via "
                                                   "group L-inf regularisation"},
        "landscape_factor": {"hessian_proxy": trace_h,
                             "theoretically_correct": "degeneracy (RLCT lambda)",
                             "caveat": "SMDL: for singular models the Hessian is "
                                       "NOT the leading-order term; degeneracy is."},
        "predicted_damage_hessian_model": hessian_estimate,
        "model_accuracy": "within a factor of 2 on the tested problem",
    }
    if lambda_hat is not None and d is not None:
        out["landscape_factor"]["degeneracy"] = degeneracy_ratio(lambda_hat, d)
    return out


def _selftest() -> int:
    checks: List[Tuple[str, bool, str]] = []

    def ck(n, c, dd=""):
        checks.append((n, bool(c), dd))

    ck("regular lambda is d/2", regular_lambda(100) == 50.0)
    ck("regular lambda rejects d<1", _raises(lambda: regular_lambda(0)))

    dr = degeneracy_ratio(2.2, 576)
    ck("wide net reads as highly degenerate", dr["ratio"] < 0.01,
       f"ratio {dr['ratio']:.4f}")
    ck("degeneracy verdict mentions compressibility",
       "compressib" in dr["verdict"])
    ck("degeneracy carries the estimator caveat", "devinterp" in dr["caveat"])
    ck("regular model reads as near-regular",
       degeneracy_ratio(50.0, 100)["ratio"] == 1.0)

    r1 = singular_mdl_redundancy(2.0, 1000)
    r2 = singular_mdl_redundancy(4.0, 1000)
    ck("redundancy grows with lambda", r2 > r1, f"{r1:.2f} -> {r2:.2f}")
    ck("redundancy grows with n",
       singular_mdl_redundancy(2.0, 10000) > singular_mdl_redundancy(2.0, 1000))
    ck("multiplicity reduces redundancy",
       singular_mdl_redundancy(2.0, 1000, 3) < singular_mdl_redundancy(2.0, 1000, 1))
    ck("redundancy rejects tiny n", _raises(lambda: singular_mdl_redundancy(2.0, 2)))
    ck("redundancy rejects m<1", _raises(lambda: singular_mdl_redundancy(2.0, 1000, 0)))

    rng = np.random.default_rng(0)
    w = rng.standard_normal(512) * 0.3
    target = w.copy()

    def quad_loss(p):
        return float(np.mean((p - target) ** 2)) + 1e-6

    est = volume_scaling_lambda(quad_loss, w, n_samples=400, seed=0)
    ck("estimator always reports unreliable", est["reliable"] is False)
    ck("estimator ships its warning", "devinterp" in est["warning"])

    btt = bits_to_tolerance(quad_loss, w, tol_frac=0.10)
    ck("bits-to-tolerance returns a bit width", btt["bits"] is not None,
       f"{btt['bits']} bits")
    ck("effective bits include scale overhead",
       btt["effective_bits"] > btt["bits"])
    loose = bits_to_tolerance(quad_loss, w, tol_frac=1.0)
    tight = bits_to_tolerance(quad_loss, w, tol_frac=0.001)
    ck("looser tolerance needs no more bits", loose["bits"] <= tight["bits"],
       f"{loose['bits']} <= {tight['bits']}")
    ck("bits_to_tolerance rejects bad tolerance",
       _raises(lambda: bits_to_tolerance(quad_loss, w, tol_frac=-1)))

    dd = damage_decomposition(absmax=1.0, bits=4, trace_h=10.0,
                              lambda_hat=2.2, d=576)
    ck("decomposition separates two factors",
       "displacement_factor" in dd and "landscape_factor" in dd)
    ck("displacement factor is flagged actionable",
       dd["displacement_factor"]["actionable"])
    ck("landscape factor names degeneracy as correct",
       "RLCT" in dd["landscape_factor"]["theoretically_correct"])
    ck("decomposition warns Hessian is not leading order",
       "NOT the leading-order term" in dd["landscape_factor"]["caveat"])
    ck("damage scales with absmax squared",
       abs(damage_decomposition(2.0, 4, 10.0)["predicted_damage_hessian_model"]
           / dd["predicted_damage_hessian_model"] - 4.0) < 1e-9)

    print("\nslt_compressibility self test\n" + "=" * 68)
    npass = 0
    for n, ok, dd_ in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}" + (f"   {dd_}" if dd_ else ""))
        npass += ok
    print(f"\n  {npass}/{len(checks)} passed")
    print(f"\n  {LLC_ESTIMATION_WARNING}\n")
    return 0 if npass == len(checks) else 1


def _raises(fn) -> bool:
    try:
        fn(); return False
    except Exception:
        return True


if __name__ == "__main__":
    raise SystemExit(_selftest())
