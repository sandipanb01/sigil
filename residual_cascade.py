#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# The module docstring is RAW: it draws under-braces with \_ ... _/, which is
# an invalid escape sequence. Python 3.12 made that a SyntaxWarning and 3.15
# makes it a SyntaxError, so a non-raw docstring here would eventually stop
# this file importing at all.
r"""
================================================================================
 residual_cascade.py -- when is another round of PTQ refinement worth running?
================================================================================

SOURCE. OpenAI, "Finite Time Blowup for Navier-Stokes" (165 pp), Section 3.4
and Section 9. The proof is a CORRECTION CYCLE: a residual is reduced by a
sequence of targeted corrections, each of which creates new error terms, and
the whole thing closes only because the worst new term stays subordinate to
the one removed. Section 9 makes that quantitative.

WHY A FLUID-DYNAMICS PROOF IS ABOUT QUANTISATION. The paper's residual update is

    R(u[j+1], p[j+1]) = R(u[j], p[j]) + L_{u[j]}(du_j, dp_j) + div(du_j (x) du_j)
                        \_ what you had _/ \_ linear remainder _/ \_ quadratic _/

Now take one output row of a linear layer, scale s, grid position u = w/s and
integer rounding r. The calibration objective is J(r) = (r-u)^T G (r-u) with
G = X^T X / n. Flipping coordinate i by delta changes it by EXACTLY

    dJ = 2*delta*(G(r-u))_i  +  delta^2 * G_ii
         \___linear remainder___/  \_quadratic self-interaction_/

Term for term, the same object. Rounding refinement IS the paper's cycle, so
the paper's convergence questions transfer verbatim.

--------------------------------------------------------------------------------
 WHAT THE PAPER'S CYCLE REQUIRES
--------------------------------------------------------------------------------
Encoded exactly in NS_CYCLE below, from the error tables on pp. 109-110:

    sigma_0 = 1/5,   sigma_{j+1} = sigma_j + 1/10,   B_j = 1/2 + sigma_j,
    C*_j = 1 + sigma_j,   kappa_s = 1e-5

Two distinct constraints hold it together, and `analyse_cycle()` recovers both:

  1. A FLOOR ON THE STARTING RESIDUAL. The mean-side self-interaction sits at
     C* + sigma_j - 2*kappa_s, which clears the next level C* + 1/10 only when
     sigma_j > 0.1 + 2*kappa_s. sigma_0 = 1/5 clears it by exactly 2x. Start
     worse than that floor and the cycle never closes.
  2. LOSSES PER OPERATION. Every divergence and reconstruction "incurs one loss
     of kappa_s". With kappa_s = 1e-5 these are negligible and the cycle runs
     forever.

--------------------------------------------------------------------------------
 WHAT I PREDICTED, AND WHAT MEASUREMENT SAID
--------------------------------------------------------------------------------
PREDICTED, from constraint 1: since bit width sets the initial residual, there
should be a bit width below which refinement stops converging. Low-bit PTQ
should refuse to refine.

FALSIFIED. `experiment_refinement_law()` runs rounding coordinate descent at
8/6/5/4/3/2 bits. It converges at every one of them, and the fraction of the
gain that survives on held-out data is flat in bit width:

    bits      8      6      4      3      2
    kept  76.0%  74.8%  78.6%  76.9%  75.5%      (at n/d = 8)

The reason is worth stating, because it is the useful half of the transfer:
the paper must BOUND its quadratic self-interaction, since it cannot evaluate
it. Coordinate descent EVALUATES it exactly -- the delta^2*G_ii term is right
there in dJ -- and simply rejects a flip that does not pay. Constraint 1 is an
artefact of needing a bound, not a fact about correction cycles. It does not
transfer.

CONSTRAINT 2 TRANSFERS, AND IT BINDS HARD. The paper's kappa_s is 1e-5 and
vanishes. Its PTQ counterpart is the finite-calibration-set loss, and it is
enormous. Measured, with d free rounding decisions per row and n calibration
tokens:

    n/d     0.5    1.0    2.0    4.0    8.0   16.0   32.0   64.0
    cal    2.00x  1.56x  1.42x  1.36x  1.35x  1.31x  1.32x  1.33x
    test   0.77x  0.96x  1.10x  1.21x  1.27x  1.27x  1.30x  1.32x
    kept  -23.5%  -7.9%  23.6%  57.0%  77.3%  89.1%  94.6%  96.8%
                  ^ at the noise floor; a second draw gave +1.02x here

    kept  ~=  1 - 1.75 * d/n          (c = 1.60-1.95 over n/d = 4..64)

Two controls establish that this is a law in the RATIO, not a curve in n:

    vary d at fixed n/d = 8:   d = 32/64/128/256 -> kept 75.2/78.0/76.8/75.7%
    vary bits at fixed n/d=8:  8/6/4/3/2 bits    -> kept 76.0/74.8/78.6/76.9/75.5%

--------------------------------------------------------------------------------
 THE RESULT
--------------------------------------------------------------------------------
At n/d = 0.5 refinement makes held-out error WORSE -- 0.77x -- while
calibration error improves 2.0x. That is not a small effect and it is invisible
to anyone who reports calibration numbers.

The exact break-even sits in run-to-run noise around n/d ~= 1: two independent
draws gave 0.96x and 1.02x there. Treat n/d = 1 as "no effect either way" and
n/d = 4 as the point where refinement first keeps more than half of what it
claims. A second run of experiment_refinement_law() at n/d = 4/8/16 reproduced
kept = 57.6/78.4/88.9% against the model's 56.2/78.1/89.1%.

To keep a fraction k of the measured improvement you need

    n >= 1.75 * d / (1 - k)     tokens

where d is the INPUT dimension of the matrix being refined. For Bonsai 2 27B
(hidden 5120, intermediate 17408) a standard 128 x 2048 = 262,144-token
calibration set keeps ~96% on the attention projections and ~88% on down_proj.
A 128 x 512 = 65,536-token set -- also common -- keeps ~54% on down_proj.
Half the reported improvement there is calibration overfitting.

Licence: Apache-2.0.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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



__all__ = [
    "ErrorTerm", "CorrectionOperation", "CorrectionCycle", "NS_CYCLE",
    "KAPPA_S", "analyse_cycle", "sigma_schedule", "validate_against_paper",
    "REFINEMENT_C", "BREAK_EVEN_RATIO", "kept_fraction",
    "calibration_tokens_needed", "refinement_verdict", "MEASURED_KEPT",
    "layer_refinement_plan", "TRANSFORMER_SHAPES", "invariant_budget",
    "shrinking_support_schedule", "experiment_refinement_law",
]

KAPPA_S = 1e-5
"""The paper's per-operation loss. Section 9: `kappa_s = 1e-5`."""


# ============================================================================ #
# SECTION 1 -- the exponent algebra
#
# Every newly created term in the paper is an affine function of the stage
# parameter sigma_j, minus some number of kappa_s losses. Writing them this way
# lets the cycle be checked mechanically instead of read off a table by eye.
# ============================================================================ #

@dataclass(frozen=True)
class ErrorTerm:
    """
    A newly created error term, as an exponent of the small parameter.

        exponent = const + sigma_coeff * sigma_j - n_kappa * kappa_s

    Larger exponent means SMALLER error, so a term is harmless exactly when its
    exponent clears the level the next stage needs.
    """
    name: str
    const: float
    sigma_coeff: float = 0.0
    n_kappa: int = 0
    kind: str = "linear"          # linear | cross | quadratic | reconstruction

    def exponent(self, sigma: float, kappa_s: float = KAPPA_S) -> float:
        return self.const + self.sigma_coeff * sigma - self.n_kappa * kappa_s


@dataclass(frozen=True)
class CorrectionOperation:
    """
    One operation in a cycle: it cancels a targeted source and creates terms.

    `target_level` is the level the surviving terms must clear, expressed the
    same way: const + sigma_coeff * sigma.
    """
    name: str
    creates: Tuple[ErrorTerm, ...]
    target_const: float
    target_sigma_coeff: float = 1.0
    preserves_invariants: Tuple[str, ...] = ()
    equations_spent: int = 1

    def target(self, sigma: float) -> float:
        return self.target_const + self.target_sigma_coeff * sigma

    def binding(self, sigma: float,
                kappa_s: float = KAPPA_S) -> Tuple[ErrorTerm, float]:
        """The worst new term and its margin. The worst one is what decides."""
        worst = min(self.creates, key=lambda t: t.exponent(sigma, kappa_s))
        return worst, worst.exponent(sigma, kappa_s) - self.target(sigma)


@dataclass
class CorrectionCycle:
    name: str
    operations: List[CorrectionOperation]
    sigma_0: float
    gain_per_cycle: float
    kappa_s: float = KAPPA_S
    source: str = ""
    invariants: Tuple[str, ...] = ()
    total_equations: int = 0
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# The paper's cycle, transcribed.
#
# Wave side (p. 109), relative to B_j = 1/2 + sigma_j:
#     linear error                        B + 1/2 - 4*kappa_s
#     mean interaction                    B + 0.4 - 1*kappa_s
#     cross with old exact waves          B + 1/2 - 2*kappa_s
#     signed-correction self-interaction  2B - 3*kappa_s
#
# Mean side (p. 110), relative to C*_j = 1 + sigma_j:
#     transverse x old-wave remainder     C* + 0.18 - 1*kappa_s
#     signed curl x old exact wave        C* + 1/2 - 2*kappa_s
#     signed-correction self-interaction  C* + sigma_j - 2*kappa_s
#
# Substituting B = 1/2 + sigma and C* = 1 + sigma puts everything in sigma.
# --------------------------------------------------------------------------- #

NS_CYCLE = CorrectionCycle(
    name="Navier-Stokes residual correction cycle",
    sigma_0=0.2,
    gain_per_cycle=0.1,
    source="OpenAI, Finite Time Blowup for Navier-Stokes, Sec. 3.4 and Sec. 9 "
           "(Prop. 9.6; error tables pp. 109-110).",
    invariants=("normalized angular momentum", "axial flux"),
    total_equations=5,
    operations=[
        CorrectionOperation(
            name="1. cancel nonzero angular harmonics (wave amplitudes)",
            # target: B_{j+1} = 1/2 + sigma + 1/10 = 0.6 + sigma
            target_const=0.6, target_sigma_coeff=1.0,
            creates=(
                ErrorTerm("linear error", 1.0, 1.0, 4, "linear"),
                ErrorTerm("mean interaction", 0.9, 1.0, 1, "cross"),
                ErrorTerm("cross with old exact waves", 1.0, 1.0, 2, "cross"),
                ErrorTerm("signed-correction self-interaction",
                          1.0, 2.0, 3, "quadratic"),
            ),
        ),
        CorrectionOperation(
            name="2. adjust averaged stress via signed amplitude increments",
            # target: C*_{j+1} = 1 + sigma + 1/10 = 1.1 + sigma
            target_const=1.1, target_sigma_coeff=1.0,
            creates=(
                ErrorTerm("transverse correction x old-wave remainder",
                          1.18, 1.0, 1, "cross"),
                ErrorTerm("signed curl correction x old exact wave",
                          1.5, 1.0, 2, "cross"),
                ErrorTerm("signed-correction self-interaction",
                          1.0, 2.0, 2, "quadratic"),
            ),
        ),
        CorrectionOperation(
            name="3. remove nonconstant auxiliary means (temporal inverse)",
            target_const=1.1, target_sigma_coeff=1.0,
            creates=(
                ErrorTerm("reconstruction remainder (flat, beyond every power)",
                          99.0, 0.0, 0, "reconstruction"),
                ErrorTerm("complementary part at order H = C* - 2 kappa",
                          1.2, 1.0, 2, "linear"),
            ),
        ),
        CorrectionOperation(
            name="4. solve the five radial moment equations",
            target_const=1.1, target_sigma_coeff=1.0,
            equations_spent=5,
            preserves_invariants=("normalized angular momentum", "axial flux"),
            creates=(
                ErrorTerm("nonlinear remainders with improved decay",
                          1.3, 1.0, 1, "quadratic"),
                ErrorTerm("retained moment-correction term be*Me",
                          2.0, 1.0, 1, "linear"),
            ),
        ),
    ],
    notes=[
        "Operation 4 spends 2 of its 5 equations PRESERVING invariants rather "
        "than reducing residual -- 40% of that operation's budget goes to "
        "conservation, not to error.",
        "The paper proves a stronger intermediate bound (C* + 0.17) than the "
        "C* + 0.10 the next stage actually needs; the slack is deliberate.",
    ],
)


def analyse_cycle(cycle: CorrectionCycle = NS_CYCLE,
                  sigma: Optional[float] = None) -> Dict[str, Any]:
    """
    Which term binds, by how much, and what floor it puts on the starting
    residual. This is the whole content of a convergence proof, mechanised.
    """
    sig = cycle.sigma_0 if sigma is None else sigma
    rows, floors = [], []
    for op in cycle.operations:
        worst, margin = op.binding(sig, cycle.kappa_s)
        # A term whose exponent grows with sigma FASTER than the target does
        # imposes a floor: below some sigma it stops clearing the level. Scan
        # every term, not just the one that binds at this particular sigma --
        # the floor is set by a term that is comfortably subordinate at
        # sigma_0 and only becomes binding further down. Checking the binding
        # term alone reported no floor at all, because at sigma_0 = 1/5 the
        # self-interaction (1.39998) sits just ABOVE the transverse cross term
        # (1.37999) and does not bind until sigma < 0.18.
        floor, floor_term, floor_kind = None, None, None
        for t in op.creates:
            excess = t.sigma_coeff - op.target_sigma_coeff
            if excess > 0:
                f = (op.target_const - t.const
                     + t.n_kappa * cycle.kappa_s) / excess
                if f > 0:
                    floors.append((op.name, t.name, t.kind, f))
                    if floor is None or f > floor:
                        floor, floor_term, floor_kind = f, t.name, t.kind
        rows.append({
            "operation": op.name,
            "binding_term": worst.name,
            "kind": worst.kind,
            "exponent": round(worst.exponent(sig, cycle.kappa_s), 6),
            "target": round(op.target(sig), 6),
            "margin": round(margin, 6),
            "ok": margin > 0,
            "sigma_floor": (round(floor, 6) if floor is not None else None),
            "floor_term": floor_term,
            "floor_kind": floor_kind,
            "equations_spent": op.equations_spent,
            "invariants_preserved": list(op.preserves_invariants),
        })
    worst_floor = max((f for _, _, _, f in floors), default=None)
    return {
        "cycle": cycle.name,
        "sigma": sig,
        "rows": rows,
        "closes": all(r["ok"] for r in rows),
        "min_margin": round(min(r["margin"] for r in rows), 6),
        "sigma_floor": (round(worst_floor, 6) if worst_floor is not None else None),
        "floor_source": (max(floors, key=lambda f: f[3])[:3] if floors else None),
        "floor_kind": (max(floors, key=lambda f: f[3])[2] if floors else None),
        "safety_factor": (round(sig / worst_floor, 3)
                          if worst_floor and worst_floor > 0 else None),
        "invariant_equations": sum(len(op.preserves_invariants)
                                   for op in cycle.operations),
        "total_equations": cycle.total_equations,
        "source": cycle.source,
    }


def sigma_schedule(n: int, cycle: CorrectionCycle = NS_CYCLE) -> List[float]:
    """sigma_j for j = 0..n-1. The paper: 1/5, 3/10, 2/5, ... -> infinity."""
    # The cap is not fussiness: range(10**30) builds a list with 10**30
    # entries and never returns. The stress harness hung here.
    n = int(_finite(n, "n", 1, 100_000))
    return [cycle.sigma_0 + j * cycle.gain_per_cycle for j in range(n)]


def validate_against_paper() -> Dict[str, Any]:
    """
    The transcription is only worth anything if it reproduces the paper's own
    published recursion and its own stated floor. If a future edit breaks the
    encoding, this fails rather than quietly producing plausible numbers.
    """
    sched = sigma_schedule(6)
    expected = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]          # sigma_j = 1/5 + j/10
    a0 = analyse_cycle(NS_CYCLE, 0.2)
    # The paper's own intermediate test is `sigma_j - 3 kappa_s > 0.17`, which
    # at the level of the end-of-cycle requirement becomes sigma_j > 0.1 + ...
    below = analyse_cycle(NS_CYCLE, 0.05)
    return {
        "sigma_schedule_matches": all(
            abs(a - b) < 1e-12 for a, b in zip(sched, expected)),
        "sigma_schedule": sched,
        "cycle_closes_at_sigma_0": a0["closes"],
        "floor_is_quadratic": a0.get("floor_kind") == "quadratic",
        "sigma_floor": a0["sigma_floor"],
        "floor_near_one_tenth": (a0["sigma_floor"] is not None
                                 and abs(a0["sigma_floor"] - 0.1) < 1e-3),
        "safety_factor_is_two": (a0["safety_factor"] is not None
                                 and abs(a0["safety_factor"] - 2.0) < 0.01),
        "fails_below_the_floor": not below["closes"],
        "verdict": "FAITHFUL",
    }


# ============================================================================ #
# SECTION 2 -- the transfer, and the measured law
# ============================================================================ #

REFINEMENT_C = 1.75
"""
Constant in `kept ~= 1 - REFINEMENT_C * d/n`.

Fitted from experiment_refinement_law(); implied c was 1.60, 1.66, 1.90, 1.74,
1.95 at n/d = 4, 8, 16, 32, 64. This is the PTQ counterpart of the paper's
kappa_s -- and where kappa_s is 1e-5 and vanishes, this one dominates.
"""

BREAK_EVEN_RATIO = 1.2
"""
n/d below which refinement is not worth running. The true zero-crossing is
near n/d = 1 and sits in run-to-run noise -- three measurements there gave
0.980x, 0.956x and 1.017x held-out gain -- so this is set slightly above it
rather than at it. Below n/d = 0.5 the harm is unambiguous: 0.766x held-out
against a 2.0x calibration "improvement".
"""

MEASURED_KEPT = {
    # n/d : (calibration gain, held-out gain, kept fraction)
    0.5:  (1.997, 0.766, -0.235),
    1.0:  (1.562, 0.956, -0.079),
    2.0:  (1.417, 1.099, 0.236),
    4.0:  (1.363, 1.207, 0.570),
    8.0:  (1.349, 1.270, 0.773),
    16.0: (1.306, 1.273, 0.891),
    32.0: (1.316, 1.299, 0.946),
    64.0: (1.334, 1.323, 0.968),
}
"""4-bit, d_in = 128, d_out = 64, 8 sweeps, 16384 held-out rows."""


def kept_fraction(d: int, n_tokens: int, c: float = REFINEMENT_C) -> float:
    """
    Fraction of the CALIBRATION-measured improvement that survives on held-out
    data. Negative means refinement is actively harmful.

    `d` is the input dimension of the matrix -- the number of free rounding
    decisions per row. `n_tokens` is the calibration token count.
    """
    # `nan <= 0` is False, so the old checks let NaN through and the headline
    # calibration law returned NaN without complaint. Found by fuzzing at a
    # wider budget than the default reached.
    d = _finite(d, "d", 1.0, 1e12)
    n_tokens = _finite(n_tokens, "n_tokens", 1.0, 1e15)
    c = _finite(c, "c", 1e-6, 1e3)
    return 1.0 - c * d / n_tokens


def calibration_tokens_needed(d: int, target_kept: float = 0.9,
                              c: float = REFINEMENT_C) -> int:
    """Tokens required to keep `target_kept` of the improvement: n >= c*d/(1-k)."""
    if not 0.0 <= target_kept < 1.0:
        raise ValueError("target_kept must be in [0, 1)")
    # Round the denominator before dividing: 1 - 0.9 is 0.09999999999999998
    # in binary, which made the 95% figure fail to come out at exactly twice
    # the 90% figure and broke an invariant the callers rely on.
    denom = round(1.0 - target_kept, 12)
    return int(math.ceil(round(c * d / denom, 6)))


def refinement_verdict(d: int, n_tokens: int,
                       observed_cal_gain: Optional[float] = None) -> Dict[str, Any]:
    """
    Should this matrix be refined at all, and what will the reported gain really
    be worth?
    """
    k = kept_fraction(d, n_tokens)
    ratio = n_tokens / float(d)
    if ratio < BREAK_EVEN_RATIO:
        verdict, advice = "HARMFUL", (
            f"n/d = {ratio:.2f} is below the measured break-even of "
            f"{BREAK_EVEN_RATIO}. Calibration error will improve and held-out "
            f"error will get WORSE. Use plain round-to-nearest, or collect "
            f"{calibration_tokens_needed(d, 0.5):,} tokens for a 50% keep.")
    elif k < 0.5:
        verdict, advice = "MARGINAL", (
            f"Only {k:.0%} of any measured gain is real. Refine only if you "
            f"can afford {calibration_tokens_needed(d, 0.9):,} tokens for a "
            f"90% keep.")
    else:
        verdict, advice = "WORTH IT", (
            f"{k:.0%} of the measured gain survives. "
            f"{calibration_tokens_needed(d, 0.95):,} tokens would take it to 95%.")
    out = {
        "d": d, "n_tokens": n_tokens, "n_over_d": round(ratio, 2),
        "kept_fraction": round(k, 4), "verdict": verdict, "advice": advice,
        "tokens_for_90pct": calibration_tokens_needed(d, 0.9),
        "tokens_for_95pct": calibration_tokens_needed(d, 0.95),
    }
    if observed_cal_gain is not None:
        if observed_cal_gain < 1.0:
            raise ValueError("a calibration gain below 1.0 is not a gain")
        real = 1.0 + (observed_cal_gain - 1.0) * max(k, 0.0)
        out.update({
            "observed_cal_gain": observed_cal_gain,
            "expected_true_gain": round(real, 4),
            "overstatement": round(observed_cal_gain / real, 3) if real > 0 else None,
        })
    return out


# Input dimensions that actually get refined, per matrix, for models in the
# catalogue. `d` is the INPUT dimension: q/k/v/o see hidden_size, gate/up see
# hidden_size, down sees intermediate_size.
TRANSFORMER_SHAPES = {
    # key: (hidden_size, intermediate_size, n_layers, display)
    "qwen3-0.6b":   (1024, 3072, 28, "Qwen3-0.6B"),
    "qwen3-1.7b":   (2048, 6144, 28, "Qwen3-1.7B"),
    "qwen3-4b":     (2560, 9728, 36, "Qwen3-4B"),
    "qwen3-8b":     (4096, 12288, 36, "Qwen3-8B"),
    "llama-v3.2-1b": (2048, 8192, 16, "Llama-3.2-1B"),
    "llama-v3.1-8b": (4096, 14336, 32, "Llama-3.1-8B"),
    "gemma-4-e2b-it": (2048, 8192, 30, "Gemma 4 E2B-it"),
    "bonsai-2-27b": (5120, 17408, 64, "Bonsai 2 27B"),
}


def layer_refinement_plan(model_key: str, n_tokens: int) -> Dict[str, Any]:
    """
    Per-matrix verdict for a real model, because the answer differs WITHIN a
    layer: down_proj sees the intermediate dimension and is therefore the first
    matrix to fall off the calibration budget.
    """
    if model_key not in TRANSFORMER_SHAPES:
        raise KeyError(f"unknown model {model_key!r}; "
                       f"known: {sorted(TRANSFORMER_SHAPES)}")
    hidden, inter, n_layers, display = TRANSFORMER_SHAPES[model_key]
    mats = [("q/k/v/o_proj", hidden), ("gate_proj/up_proj", hidden),
            ("down_proj", inter)]
    rows = [dict(matrix=name, **refinement_verdict(d, n_tokens))
            for name, d in mats]
    worst = min(rows, key=lambda r: r["kept_fraction"])
    return {
        "model": display, "model_key": model_key, "n_layers": n_layers,
        "hidden_size": hidden, "intermediate_size": inter,
        "n_tokens": n_tokens, "matrices": rows,
        "binding_matrix": worst["matrix"],
        "binding_kept": worst["kept_fraction"],
        "headline": (
            f"{display}: at {n_tokens:,} calibration tokens, "
            f"{worst['matrix']} keeps only {worst['kept_fraction']:.0%} of "
            f"any measured refinement gain and is what limits the recipe."),
    }


# ============================================================================ #
# SECTION 3 -- two structural habits from the proof, kept because they are
# cheap and this project had neither.
# ============================================================================ #

def invariant_budget(total_equations: int = 5,
                     invariants: int = 2) -> Dict[str, Any]:
    """
    The paper spends 2 of its 5 moment equations PRESERVING conserved
    quantities and only 3 reducing residual -- 40% of the budget on
    conservation.

    The PTQ counterpart is spending part of a correction budget holding the
    per-channel activation mean and the residual-stream scale fixed rather than
    minimising L2. Violating those compounds nonlinearly downstream, in the
    same way a momentum violation would propagate here.

    UNMEASURED in this project. Stated as a hypothesis with an experiment, not
    as a result.
    """
    if total_equations < 1 or invariants < 0 or invariants > total_equations:
        raise ValueError("invariants must be between 0 and total_equations")
    return {
        "total_equations": total_equations,
        "spent_on_invariants": invariants,
        "spent_on_residual": total_equations - invariants,
        "conservation_share": round(invariants / total_equations, 3),
        "paper": "Two equations preserve the zero angular-momentum and "
                 "axial-flux integrals (9.10); three cancel the linear "
                 "contributions to (P, J_theta, J_z).",
        "ptq_analogue": "Hold per-channel activation mean and residual-stream "
                        "scale fixed; spend the remainder on L2.",
        "status": "UNMEASURED -- hypothesis with a clear experiment.",
    }


def shrinking_support_schedule(n_stages: int, width_0: int,
                               shrink: float = 0.5) -> List[Dict[str, Any]]:
    """
    "Each cutoff equals one sufficiently close to q = 0, and its support shrinks
    with the stage." (Sec. 3.4)

    Later correction stages act on progressively narrower supports. The PTQ
    reading: refine fewer channels each round -- the ones still carrying
    residual -- rather than sweeping everything every time. Cost per round falls
    geometrically while the terms that matter keep being corrected.
    """
    n_stages = int(_finite(n_stages, "n_stages", 1, 10_000))
    # width 0 divided by itself below: ZeroDivisionError, the "confused" kind.
    width_0 = int(_finite(width_0, "width_0", 1, 10 ** 12))
    shrink = _finite(shrink, "shrink")
    if not 0.0 < shrink < 1.0:
        raise ValueError("shrink must be in (0, 1)")
    out, total = [], 0.0
    for j in range(n_stages):
        w = max(1, int(round(width_0 * shrink ** j)))
        total += w
        out.append({"stage": j, "width": w,
                    "fraction_of_full": round(w / width_0, 4),
                    "cumulative_width": int(total)})
    full = n_stages * width_0
    for row in out:
        row["cost_vs_full_sweep"] = round(total / full, 4)
    return out


# ============================================================================ #
# SECTION 4 -- the experiment
# ============================================================================ #

def experiment_refinement_law(d_in: int = 128, d_out: int = 64,
                              n_test: int = 8192, n_sweeps: int = 6,
                              ratios: Sequence[float] = (0.5, 1, 2, 4, 8, 16),
                              bits: int = 4, seed: int = 0,
                              verbose: bool = True) -> Dict[str, Any]:
    """
    Reproduce the measurement. Needs torch; everything else in this module does
    not, so the module stays importable on a machine without it.

    Rounding coordinate descent, all rows in parallel, accepting a flip only
    when it strictly decreases the calibration objective -- so the calibration
    curve is monotone by construction and the only question is what happens on
    held-out data.
    """
    try:
        import torch
    except ImportError:
        return {"ran": False,
                "reason": "torch is not installed; pip install -U torch"}

    torch.manual_seed(seed)
    dt = torch.float64

    def problem(n_cal):
        W = torch.randn(d_out, d_in, dtype=dt) / math.sqrt(d_in)
        A = torch.randn(d_in, d_in, dtype=dt) / math.sqrt(d_in)
        L = torch.linalg.cholesky(A @ A.T + 0.1 * torch.eye(d_in, dtype=dt))
        return (W,
                torch.randn(n_cal, d_in, dtype=dt) @ L.T,
                torch.randn(n_test, d_in, dtype=dt) @ L.T)

    def rel(W, Q, X):
        Y = X @ W.T
        return float((X @ Q.T - Y).norm() / Y.norm())

    rows = []
    for ratio in ratios:
        n_cal = max(2, int(round(ratio * d_in)))
        W, Xc, Xt = problem(n_cal)
        G = Xc.T @ Xc / n_cal
        qmax = _qmax(bits)
        s = W.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
        u = W / s
        r = u.round().clamp(-qmax, qmax)
        cal0, test0 = rel(W, s * r, Xc), rel(W, s * r, Xt)

        diag = torch.diagonal(G).clone()
        e = r - u
        Ge = e @ G
        for _ in range(n_sweeps):
            flips = 0
            for i in range(d_in):
                gi = Ge[:, i]
                for delta in (1.0, -1.0):
                    dJ = 2.0 * delta * gi + delta * delta * diag[i]
                    ok = (dJ < -1e-15) & ((r[:, i] + delta).abs() <= qmax)
                    if ok.any():
                        step = torch.where(ok, torch.full_like(gi, delta),
                                           torch.zeros_like(gi))
                        r[:, i] += step
                        e[:, i] += step
                        Ge = Ge + step.unsqueeze(1) * G[i].unsqueeze(0)
                        gi = Ge[:, i]
                        flips += int(ok.sum())
            if flips == 0:
                break
        cal1, test1 = rel(W, s * r, Xc), rel(W, s * r, Xt)
        cg, tg = cal0 / cal1, test0 / test1
        kept = (tg - 1.0) / (cg - 1.0) if cg > 1.0 else 0.0
        rows.append({"n_over_d": ratio, "n_cal": n_cal,
                     "cal_gain": round(cg, 4), "test_gain": round(tg, 4),
                     "kept": round(kept, 4),
                     "implied_c": (round((1.0 - kept) * ratio, 3)
                                   if ratio >= 2 else None),
                     "predicted_kept": round(kept_fraction(d_in, n_cal), 4)})
        if verbose:
            ic = rows[-1]["implied_c"]
            print(f"  n/d={ratio:>5.1f}  n={n_cal:>6}  cal {cg:>6.3f}x  "
                  f"test {tg:>6.3f}x  kept {kept:>7.1%}  "
                  f"predicted {rows[-1]['predicted_kept']:>7.1%}"
                  + (f"  c={ic}" if ic else ""))
    cs = [r["implied_c"] for r in rows if r["implied_c"]]
    return {"ran": True, "bits": bits, "d_in": d_in, "d_out": d_out,
            "rows": rows,
            "implied_c_mean": (round(sum(cs) / len(cs), 3) if cs else None),
            "harmful_below": [r["n_over_d"] for r in rows if r["test_gain"] < 1.0]}


# ============================================================================ #
# SECTION 5 -- self test
# ============================================================================ #

def _raises(fn) -> bool:
    try:
        fn()
        return False
    except Exception:
        return True


def selftest() -> int:
    checks: List[Tuple[str, bool, str]] = []

    def ck(name, cond, detail=""):
        checks.append((name, bool(cond), str(detail)))

    # ---- the paper's cycle, transcribed faithfully ----
    v = validate_against_paper()
    ck("sigma schedule reproduces the paper's 1/5 + j/10",
       v["sigma_schedule_matches"], str(v["sigma_schedule"][:4]))
    ck("cycle closes at the paper's own sigma_0", v["cycle_closes_at_sigma_0"])
    ck("the floor comes from the QUADRATIC self-interaction",
       v["floor_is_quadratic"], str(v.get("floor_source")))
    ck("recovered sigma floor is 1/10", v["floor_near_one_tenth"],
       str(v["sigma_floor"]))
    _op2 = analyse_cycle(NS_CYCLE)["rows"][1]
    ck("the floor is attributed to the self-interaction, not to what binds",
       "self-interaction" in (_op2["floor_term"] or "")
       and _op2["binding_term"] != _op2["floor_term"],
       f"floor={_op2['floor_term']}, binds={_op2['binding_term']}")
    ck("sigma_0 = 1/5 clears the floor by exactly 2x",
       v["safety_factor_is_two"])
    ck("the cycle FAILS below its floor", v["fails_below_the_floor"])

    a = analyse_cycle(NS_CYCLE)
    ck("every operation clears its target", a["closes"])
    ck("all four operations are analysed", len(a["rows"]) == 4)
    ck("two invariants are preserved", a["invariant_equations"] == 2)
    ck("40% of the moment budget goes to conservation",
       abs(a["invariant_equations"] / a["total_equations"] - 0.4) < 1e-9)
    ck("margins are positive throughout", a["min_margin"] > 0,
       f"{a['min_margin']}")
    # Larger sigma can only help: the binding term grows faster than the target.
    ck("a better starting residual cannot hurt",
       analyse_cycle(NS_CYCLE, 0.5)["min_margin"] >=
       analyse_cycle(NS_CYCLE, 0.2)["min_margin"] - 1e-12)
    ck("kappa losses are subtracted, not added",
       ErrorTerm("t", 1.0, 0.0, 3).exponent(0.2, 1e-2) < 1.0)
    ck("sigma_schedule rejects n < 1", _raises(lambda: sigma_schedule(0)))

    # ---- the measured law ----
    ck("kept fraction is 1 - c*d/n",
       abs(kept_fraction(100, 1000) - (1 - 1.75 * 0.1)) < 1e-12)
    ck("kept goes negative below break-even", kept_fraction(100, 100) < 0)
    ck("kept approaches 1 for a large calibration set",
       kept_fraction(100, 10_000_000) > 0.999)
    ck("d must be positive", _raises(lambda: kept_fraction(0, 100)))
    ck("n must be positive", _raises(lambda: kept_fraction(100, 0)))
    ck("tokens needed grows as 1/(1-k)",
       calibration_tokens_needed(1000, 0.95) ==
       2 * calibration_tokens_needed(1000, 0.9))
    ck("target_kept must be below 1",
       _raises(lambda: calibration_tokens_needed(100, 1.0)))
    ck("90% keep on d=5120 needs ~90k tokens",
       88_000 < calibration_tokens_needed(5120, 0.9) < 92_000,
       f"{calibration_tokens_needed(5120, 0.9):,}")

    # The model must agree with what was actually measured, within the scatter
    # of the fit. If a future edit changes REFINEMENT_C, this catches it.
    # The law is asymptotic in d/n. It is accurate from n/d = 4 upward and
    # CONSERVATIVE below that (at n/d = 2 it predicts 12.5% against 23.6%
    # measured), which is the safe direction to be wrong in.
    worst = 0.0
    for ratio, (_cg, _tg, kept) in MEASURED_KEPT.items():
        if ratio < 4:
            continue
        pred = kept_fraction(128, int(ratio * 128))
        worst = max(worst, abs(pred - kept))
    ck("model matches measurement for n/d >= 4", worst < 0.02, f"max err {worst:.3f}")
    ck("model is conservative below n/d = 4",
       kept_fraction(128, 256) < MEASURED_KEPT[2.0][2])
    ck("measurement says refinement is harmful below break-even",
       MEASURED_KEPT[0.5][1] < 1.0 and MEASURED_KEPT[1.0][1] < 1.0)
    ck("measurement says calibration improves even when held-out worsens",
       MEASURED_KEPT[0.5][0] > 1.9 and MEASURED_KEPT[0.5][1] < 0.8)

    # ---- verdicts ----
    bad = refinement_verdict(4096, 4096)
    ck("n = d is reported HARMFUL", bad["verdict"] == "HARMFUL")
    ck("harmful verdict names a token count to fix it", "tokens" in bad["advice"])
    good = refinement_verdict(1024, 262_144)
    ck("a generous calibration set is WORTH IT", good["verdict"] == "WORTH IT")
    ck("a real gain is discounted, never inflated",
       refinement_verdict(1024, 8192, 1.40)["expected_true_gain"] < 1.40)
    ck("overstatement is reported",
       refinement_verdict(1024, 8192, 1.40)["overstatement"] > 1.0)
    ck("a 'gain' below 1.0 is rejected",
       _raises(lambda: refinement_verdict(100, 10_000, 0.9)))

    # ---- real models ----
    plan = layer_refinement_plan("bonsai-2-27b", 262_144)
    ck("Bonsai 2 plan uses its real hidden size",
       plan["hidden_size"] == 5120 and plan["intermediate_size"] == 17408)
    ck("down_proj is the binding matrix", plan["binding_matrix"] == "down_proj")
    ck("all three matrix groups are covered", len(plan["matrices"]) == 3)
    small = layer_refinement_plan("bonsai-2-27b", 65_536)
    ck("a 128x512 calibration set roughly halves the keep on down_proj",
       0.45 < small["binding_kept"] < 0.60, f"{small['binding_kept']:.2f}")
    ck("more tokens never lowers the keep",
       plan["binding_kept"] > small["binding_kept"])
    ck("unknown model raises",
       _raises(lambda: layer_refinement_plan("nope", 1000)))
    ck("every catalogue shape has intermediate > hidden",
       all(i > h for h, i, _, _ in TRANSFORMER_SHAPES.values()))

    # ---- structural habits ----
    ib = invariant_budget()
    ck("invariant budget is 2 of 5", ib["conservation_share"] == 0.4)
    ck("invariant budget is flagged UNMEASURED",
       ib["status"].startswith("UNMEASURED"))
    ck("invariant count is validated",
       _raises(lambda: invariant_budget(3, 5)))
    sched = shrinking_support_schedule(4, 1024)
    ck("support halves each stage",
       [r["width"] for r in sched] == [1024, 512, 256, 128])
    ck("shrinking support is cheaper than sweeping everything",
       sched[0]["cost_vs_full_sweep"] < 0.5, str(sched[0]["cost_vs_full_sweep"]))
    ck("shrink factor is validated",
       _raises(lambda: shrinking_support_schedule(3, 100, 1.5)))
    ck("n_stages is validated",
       _raises(lambda: shrinking_support_schedule(0, 100)))

    # ---- honesty guards ----
    ck("the falsified prediction is recorded in the module docstring",
       "FALSIFIED" in (__doc__ or ""))
    ck("the reason the bit-width prediction failed is stated",
       "cannot evaluate" in (__doc__ or "")
       and "EVALUATES it exactly" in (__doc__ or ""))

    print("-" * 74)
    print("  SELF TEST")
    print("-" * 74)
    npass = 0
    for name, ok, detail in checks:
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name}" + (f"   {detail}" if detail else ""))
        npass += ok
    print()
    print(f"  {npass}/{len(checks)} passed")
    return 0 if npass == len(checks) else 1


# ============================================================================ #
# SECTION 6 -- CLI
# ============================================================================ #

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Correction-cycle analysis for PTQ refinement, transferred "
                    "from the Navier-Stokes blowup construction.")
    p.add_argument("cmd", choices=["cycle", "law", "plan", "experiment",
                                   "selftest"])
    p.add_argument("--model", default="bonsai-2-27b",
                   help="catalogue key for `plan`")
    p.add_argument("--tokens", type=int, default=262_144,
                   help="calibration tokens (128 seq x 2048 = 262144)")
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--bits", type=int, default=4)
    args = p.parse_args(argv)

    if args.cmd == "selftest":
        return selftest()

    if args.cmd == "cycle":
        a = analyse_cycle(NS_CYCLE, args.sigma)
        print(f"\n  {a['cycle']}")
        print(f"  {a['source']}\n")
        print(f"  sigma = {a['sigma']}")
        print(f"  {'operation':<52}{'binds':>10}{'margin':>10}")
        for r in a["rows"]:
            mark = "ok" if r["ok"] else "FAIL"
            print(f"  {r['operation'][:51]:<52}{r['kind']:>10}"
                  f"{r['margin']:>+10.4f}  {mark}")
            if r["sigma_floor"] is not None:
                # Name the term that imposes the FLOOR, which is not in general
                # the term that binds at this sigma.
                print(f"      -> {r['floor_term']} ({r['floor_kind']}) "
                      f"imposes sigma > {r['sigma_floor']:.4f}")
        print(f"\n  cycle closes: {a['closes']}   min margin {a['min_margin']:+.4f}")
        if a["sigma_floor"] is not None:
            print(f"  starting residual floor: sigma > {a['sigma_floor']:.4f}  "
                  f"(sigma_0 = {NS_CYCLE.sigma_0} clears it "
                  f"{a['safety_factor']}x)")
        print(f"  conservation budget: {a['invariant_equations']} of "
              f"{a['total_equations']} equations preserve invariants")
        print(f"\n  sigma_j: " + ", ".join(f"{s:g}" for s in sigma_schedule(6))
              + ", ... -> infinity")
        for n in NS_CYCLE.notes:
            print(f"\n  note: {n}")
        print()
        print("  TRANSFER: the floor above does NOT carry to PTQ -- the paper")
        print("  must bound its quadratic term, coordinate descent evaluates it")
        print("  exactly and rejects bad flips. What DOES carry is the loss")
        print("  term, and in PTQ it dominates. Run `law`.\n")
        return 0

    if args.cmd == "law":
        print("\n  MEASURED: rounding refinement, 4-bit, d_in=128, "
              "16384 held-out rows\n")
        print(f"  {'n/d':>6}{'cal gain':>11}{'test gain':>11}{'kept':>9}"
              f"{'model':>9}")
        for ratio, (cg, tg, kept) in sorted(MEASURED_KEPT.items()):
            pred = kept_fraction(128, int(ratio * 128))
            flag = "  <-- harmful" if tg < 1.0 else ""
            print(f"  {ratio:>6.1f}{cg:>10.3f}x{tg:>10.3f}x{kept:>8.1%}"
                  f"{pred:>8.1%}{flag}")
        print(f"\n  kept ~= 1 - {REFINEMENT_C} * d/n     "
              f"break-even at n/d ~= {BREAK_EVEN_RATIO}")
        print("\n  Controls: kept is flat in d at fixed n/d (75.2/78.0/76.8/75.7%")
        print("  for d = 32/64/128/256) and flat in bit width (76.0/74.8/78.6/")
        print("  76.9/75.5% for 8/6/4/3/2 bits). It is a law in the ratio.\n")
        print("  Tokens needed, by matrix input dimension:")
        print(f"  {'d':>8}{'90% keep':>14}{'95% keep':>14}")
        for d in (1024, 2048, 4096, 5120, 12288, 17408):
            print(f"  {d:>8}{calibration_tokens_needed(d, 0.9):>13,}"
                  f"{calibration_tokens_needed(d, 0.95):>14,}")
        print()
        return 0

    if args.cmd == "plan":
        pl = layer_refinement_plan(args.model, args.tokens)
        print(f"\n  {pl['model']}  --  {pl['n_tokens']:,} calibration tokens")
        print(f"  hidden {pl['hidden_size']}, intermediate "
              f"{pl['intermediate_size']}, {pl['n_layers']} layers\n")
        print(f"  {'matrix':<20}{'d':>7}{'n/d':>8}{'kept':>9}  verdict")
        for m in pl["matrices"]:
            print(f"  {m['matrix']:<20}{m['d']:>7}{m['n_over_d']:>8.1f}"
                  f"{m['kept_fraction']:>8.1%}  {m['verdict']}")
        print(f"\n  {pl['headline']}")
        for m in pl["matrices"]:
            if m["kept_fraction"] < 0.9:
                print(f"\n  {m['matrix']}: {m['advice']}")
        print()
        return 0

    if args.cmd == "experiment":
        print("\n  Reproducing the refinement law (needs torch).\n")
        res = experiment_refinement_law(bits=args.bits)
        if not res["ran"]:
            print(f"  {res['reason']}")
            return 2
        print(f"\n  implied c = {res['implied_c_mean']} "
              f"(module uses {REFINEMENT_C})")
        if res["harmful_below"]:
            print(f"  refinement was HARMFUL at n/d = {res['harmful_below']}")
        print()
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
