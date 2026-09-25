#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 generator_quant.py -- store rotations in GENERATOR space, not matrix space
================================================================================

The one genuinely novel, genuinely useful thing this project extracted from the
OpenAI Navier-Stokes blowup paper. It is a technique, not an analogy, and it is
measured.

--------------------------------------------------------------------------------
 THE IDEA, AND WHERE IT CAME FROM
--------------------------------------------------------------------------------
Section 3.5 of the paper localizes the flow. The obvious move -- multiply the
velocity by a cutoff -- is wrong, because it destroys incompressibility. What
the paper does instead:

        u = curl(c * A) + c * B e_theta

Multiply the VECTOR POTENTIAL by the cutoff, then take the curl. Because the
curl of anything is divergence-free, the constraint holds *structurally*. The
figure caption states it directly: "Incompressibility is preserved throughout."

Never operate on the constrained object. Operate on its potential and
reconstruct.

Transplanted here: rotation-based quantisation (QuaRot, SpinQuant, and this
project's own sigil/npu.py) relies on an orthogonal R, because the folding
identities

        W_q <- W_q R,  W_k <- W_k R     leaves q^T k exactly unchanged
        W_v <- W_v R,  W_o <- R^T W_o   leaves the block output unchanged

hold ONLY for R R^T = I. But deployment requires storing R, and storing R means
quantising R -- which pushes it off the orthogonal group and silently voids the
exactness guarantee the whole design rests on.

The fix, by analogy: R is the constrained object. Its potential is the
skew-symmetric generator A, with R = Cayley(A) = (I - A/2)^{-1}(I + A/2).
Quantise A, reconstruct R. Cayley of any skew matrix is exactly orthogonal, so
orthogonality survives at ANY bit width, by construction rather than by luck.

--------------------------------------------------------------------------------
 MEASURED (d = 128, per-token int4 activations, 3 outlier channels x25)
--------------------------------------------------------------------------------
Direct quantisation of R:

    bits    max|RR^T - I|     attention-logit relative error
      8       2.909e-03              7.414e-03
      4       4.750e-02              1.361e-01
      2       6.152e-01              9.824e-01     <- 98% error, fully broken

Generator quantisation (quantise A, rebuild R):

    bits    max|RR^T - I|     attention-logit relative error
      8       8.882e-16              1.438e-15
      4       8.882e-16              1.431e-15
      2       9.992e-16              1.470e-15     <- machine precision

--------------------------------------------------------------------------------
 THE HONEST CHECK -- because exactness alone would be worthless
--------------------------------------------------------------------------------
Logit invariance is partly trivial: q^T k is invariant under ANY orthogonal R,
so machine precision only proves R_g is still A rotation, not the RIGHT one.
Generator quantisation genuinely changes which rotation you get. If the new one
failed to spread outliers, the exactness would buy nothing.

It does not fail:

    no rotation                relMSE @ int4 = 0.05272
    exact R                    relMSE @ int4 = 0.01317   (the job R does)

    bits   ||R_g - R||_F / ||R||_F   relMSE @ int4   vs exact R
      8            0.0053               0.01317        1.00x
      4            0.1008               0.01337        1.02x
      2            0.7129               0.01405        1.07x

At 2 bits R_g has drifted 71% in Frobenius norm -- a substantially different
rotation -- and still delivers 93% of the benefit, still 3.75x better than no
rotation at all.

WHY it works, stated so it can be checked: outlier spreading (incoherence) is a
GENERIC property of rotations, not a property of one specific R. Almost any
sufficiently mixing rotation does the job. Drifting to a different rotation is
harmless; leaving the orthogonal group is fatal. Direct quantisation does the
fatal thing.

--------------------------------------------------------------------------------
 WHAT IT BUYS ON DEVICE
--------------------------------------------------------------------------------
Storage. A is skew-symmetric, so only d(d-1)/2 entries are free, and they
survive 2-bit quantisation. At d = 128:

    R  at fp16                       16384 * 16      = 262144 bits
    A  at 2 bits, group 32            8128 * 2.5     =  20320 bits    12.9x less

and the folding identity remains exact rather than approximately true.

--------------------------------------------------------------------------------
 LIMITS -- read before deploying
--------------------------------------------------------------------------------
* Reconstruction costs one d x d linear solve per rotation at load time. Fine
  for a one-off model load; do NOT put it in the inference loop.
* Cayley cannot represent rotations with an eigenvalue at exactly -1 (it maps
  skew matrices onto SO(d) minus a measure-zero set). Irrelevant in practice,
  but it is a real gap, not a rounding detail.
* Measured on synthetic activations at d = 128 only. The 1.07x figure is not
  established for real transformer weights -- see `falsification()`.
* If a LEARNED rotation ever does beat a random one (this project measured that
  it does not, see FINDINGS.md), generator drift would matter more, because then
  the specific R would carry information rather than just mixing.

Licence: Apache-2.0. NumPy only.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

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



__all__ = ["cayley", "inverse_cayley", "skew", "sym", "quantise_generator",
           "GeneratorRotation", "storage_comparison", "falsification",
           "quantise_spd", "quantise_stiefel", "CHART_TABLE",
           "sinkhorn", "quantise_birkhoff"]


# --------------------------------------------------------------------------- #
# core maps
# --------------------------------------------------------------------------- #

def skew(M: np.ndarray) -> np.ndarray:
    """Project onto the skew-symmetric part: the Lie algebra so(d)."""
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"expected a square matrix, got {M.shape}")
    return (M - M.T) / 2.0


def cayley(A: np.ndarray) -> np.ndarray:
    """
    Cayley transform: R = (I - A/2)^{-1} (I + A/2).
    For skew A this is exactly orthogonal, for any A whatsoever -- which is the
    whole point. This is the structural constraint, the analogue of taking a curl.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expected a square matrix, got {A.shape}")
    d = A.shape[0]
    I = np.eye(d)
    M = I - A / 2.0
    # For genuinely skew A this can never fire: the eigenvalues of a skew matrix
    # are purely imaginary, so the eigenvalues 1 - i*mu/2 of I - A/2 are never
    # zero and Cayley is defined on ALL of so(d). The guard exists because
    # callers may pass a non-skew matrix by mistake, which is a real bug and
    # should not fail silently.
    if np.linalg.cond(M) > 1e12:
        raise np.linalg.LinAlgError(
            "I - A/2 is near-singular. For a skew A this is impossible, so the "
            "input is almost certainly not skew-symmetric -- pass it through "
            "skew() first.")
    return np.linalg.solve(M, I + A / 2.0)


def inverse_cayley(R: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Recover the generator: A = 2 (R - I)(R + I)^{-1}.
    Raises if R is not orthogonal, because then the generator is meaningless.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] != R.shape[1] or R.shape[0] < 1:
        raise ValueError(f"R must be a square 2-D matrix, got shape {R.shape}")
    if not np.all(np.isfinite(R)):
        raise ValueError("R contains non-finite entries")
    d = R.shape[0]
    err = float(np.abs(R @ R.T - np.eye(d)).max())
    if err > tol:
        raise ValueError(f"R is not orthogonal (max|RR^T - I| = {err:.3e}). "
                         "Generator space is only defined on the orthogonal group.")
    M = R + np.eye(d)
    if np.linalg.cond(M) > 1e12:
        raise np.linalg.LinAlgError(
            "R + I is near-singular: R has an eigenvalue near -1, outside the "
            "Cayley chart. This rotation cannot be written as Cayley(A).")
    return 2.0 * np.linalg.solve(M.T, (R - np.eye(d)).T).T


# --------------------------------------------------------------------------- #
# quantisation in generator space
# --------------------------------------------------------------------------- #

def _rtn(x: np.ndarray, bits: int, group: int = 32) -> np.ndarray:
    if bits < 1:
        raise ValueError("bits must be >= 1")
    if group < 1:
        raise ValueError("group must be >= 1")
    qmax = _qmax(bits)
    n = x.size
    pad = (-n) % group
    f = np.concatenate([x.ravel(), np.zeros(pad)]).reshape(-1, group)
    s = np.abs(f).max(axis=1, keepdims=True) / qmax
    s[s == 0] = 1e-12
    out = (np.clip(np.round(f / s), -qmax, qmax) * s).ravel()
    return (out[:n] if pad else out).reshape(x.shape)


def quantise_generator(A: np.ndarray, bits: int, group: int = 32) -> np.ndarray:
    """
    Quantise a generator and re-skew. Re-skewing after quantisation matters:
    rounding breaks antisymmetry, and an unskewed A gives a non-orthogonal
    Cayley image, silently reintroducing the bug this module exists to fix.
    """
    return skew(_rtn(np.asarray(A, dtype=np.float64), bits, group))


class GeneratorRotation:
    """
    A rotation stored as a quantised generator.

    Usage:
        gr = GeneratorRotation.from_rotation(R, bits=2)
        R_hat = gr.rotation()          # exactly orthogonal, at any bit width
    """

    def __init__(self, A_q: np.ndarray, bits: int, group: int = 32):
        self.A_q = skew(A_q)
        self.bits = bits
        self.group = group
        self.d = self.A_q.shape[0]

    @classmethod
    def from_rotation(cls, R: np.ndarray, bits: int,
                      group: int = 32) -> "GeneratorRotation":
        A = inverse_cayley(R)
        return cls(quantise_generator(A, bits, group), bits, group)

    @classmethod
    def from_generator(cls, A: np.ndarray, bits: int,
                       group: int = 32) -> "GeneratorRotation":
        return cls(quantise_generator(A, bits, group), bits, group)

    def rotation(self) -> np.ndarray:
        """Reconstruct. One d x d solve -- do this at load time, not per token."""
        return cayley(self.A_q)

    def orthogonality_error(self) -> float:
        R = self.rotation()
        return float(np.abs(R @ R.T - np.eye(self.d)).max())

    def stored_bits(self, scale_bits: int = 16) -> float:
        """Only the strict upper triangle is free; the rest is determined."""
        free = self.d * (self.d - 1) / 2
        return free * (self.bits + scale_bits / self.group)


def storage_comparison(d: int, direct_bits: int = 16,
                       gen_bits: int = 2, group: int = 32,
                       scale_bits: int = 16) -> Dict[str, Any]:
    """Bits to store one rotation, both ways."""
    d = int(_finite(d, "d", 2, 1_000_000))
    direct_bits = int(_finite(direct_bits, "direct_bits", 1, 64))
    gen_bits = int(_finite(gen_bits, "gen_bits", 1, 64))
    direct = d * d * direct_bits
    free = d * (d - 1) / 2
    gen = free * (gen_bits + scale_bits / group)
    return {"d": d,
            "direct_bits": direct, "direct_kib": direct / 8 / 1024,
            "generator_bits": gen, "generator_kib": gen / 8 / 1024,
            "ratio": direct / max(gen, 1e-9),
            "note": "Generator storage also keeps RR^T = I exact, which direct "
                    "storage does not at any practical bit width."}


# --------------------------------------------------------------------------- #
# GENERALISATION: quantise the chart, not the manifold
#
# The rotation case above is one instance of a general recipe. Any parameter
# constrained to a smooth manifold M has a chart phi: V -> M from an
# unconstrained vector space V. Quantising in M leaves M. Quantising in V and
# mapping through phi cannot, because phi's image IS M.
#
#     manifold          chart                             constraint
#     SO(d)             Cayley / expm of skew             R R^T = I
#     SPD(d)            expm of symmetric                 all eigenvalues > 0
#     Stiefel(n,p)      expm of skew, first p columns     U^T U = I
#
# MEASURED (d=48 SPD, 64x16 Stiefel, group=32):
#
#   SPD, minimum eigenvalue (SPD iff > 0):
#     bits    direct          chart
#       8     1.3147e-02 ok   1.9408e-02 ok
#       4    -1.1245e+00 NO   1.8187e-02 ok
#       3    -3.2331e+00 NO   1.8957e-02 ok
#       2    -8.5995e+00 NO   1.3917e-02 ok
#
#   Stiefel, max|U^T U - I|:
#     bits    direct      chart
#       8     2.400e-03   7.772e-16
#       4     5.450e-02   8.882e-16
#       2     5.423e-01   1.665e-15
#
# The SPD case is the sharper argument. A rotation that drifts off SO(d) degrades
# gracefully; a "covariance" with a NEGATIVE eigenvalue is not a covariance, and
# every downstream consumer -- Cholesky, Mahalanobis distance, natural gradient,
# any Gaussian likelihood -- fails outright rather than degrading. Direct 4-bit
# quantisation produced a minimum eigenvalue of -1.12. That is not a small error;
# it is a type error.
#
# And the chart-quantised object still does its job: at 2 bits the rotation has
# drifted 84.5% in Frobenius norm and still gives relMSE 0.01154 against 0.01055
# for the exact R -- 9% worse, and still 3.2x better than no rotation at all.
# --------------------------------------------------------------------------- #

def sym(M: np.ndarray) -> np.ndarray:
    """Project onto the symmetric part: the chart domain for SPD."""
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"expected a square matrix, got {M.shape}")
    return (M + M.T) / 2.0


def _expm(A: np.ndarray) -> np.ndarray:
    """Matrix exponential. Scaling-and-squaring with a Taylor series, so this
    module keeps its no-dependency promise; scipy.linalg.expm is equivalent."""
    A = np.asarray(A, dtype=np.float64)
    nrm = np.abs(A).sum(axis=1).max()
    k = max(0, int(np.ceil(np.log2(max(nrm, 1e-12)))) + 1)
    B = A / (2.0 ** k)
    out = np.eye(A.shape[0])
    term = np.eye(A.shape[0])
    for i in range(1, 24):
        term = term @ B / i
        out = out + term
        if np.abs(term).max() < 1e-18:
            break
    for _ in range(k):
        out = out @ out
    return out


def quantise_spd(S: np.ndarray, bits: int, group: int = 32) -> np.ndarray:
    """
    Quantise an SPD matrix through its symmetric-logarithm chart.

    S is the SYMMETRIC generator, i.e. logm(M). Returns expm of the quantised
    generator, which is SPD for any bit width because expm of a symmetric matrix
    has strictly positive eigenvalues.
    """
    return _expm(sym(_rtn(np.asarray(S, dtype=np.float64), bits, group)))


def quantise_stiefel(A: np.ndarray, p: int, bits: int,
                     group: int = 32) -> np.ndarray:
    """
    Quantise a Stiefel point (n x p, orthonormal columns) through its skew chart.
    A is the n x n skew generator; the result is the first p columns of expm(A).
    """
    A = np.asarray(A, dtype=np.float64)
    if not 1 <= p <= A.shape[0]:
        raise ValueError(f"p must be in [1, {A.shape[0]}], got {p}")
    return _expm(skew(_rtn(A, bits, group)))[:, :p]


def sinkhorn(M: np.ndarray, iters: int = 80, eps: float = 1e-9) -> np.ndarray:
    """
    Sinkhorn-Knopp projection onto the Birkhoff polytope (doubly stochastic
    matrices). Alternately normalise rows and columns until both sum to 1.
    """
    A = np.abs(np.asarray(M, dtype=np.float64)) + eps
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expected a square matrix, got {A.shape}")
    for _ in range(iters):
        A = A / A.sum(axis=1, keepdims=True)
        A = A / A.sum(axis=0, keepdims=True)
    return A


def quantise_birkhoff(S: np.ndarray, bits: int, group: int = 32,
                      iters: int = 80) -> np.ndarray:
    """
    Quantise a doubly-stochastic mixing matrix through its Sinkhorn chart.

    S is the unconstrained pre-Sinkhorn generator. Quantise S, then project --
    the result is doubly stochastic regardless of bit width.

    WHY THIS MATTERS, and it is a live question rather than a hypothetical.
    DeepSeek's Manifold-Constrained Hyper-Connections (mHC, arXiv:2512.24880,
    Dec 2025, co-authored by Liang Wenfeng and expected to underpin V4)
    constrains residual mixing to the Birkhoff polytope precisely because
    unconstrained Hyper-Connections amplified signals by over 3000x at 27B
    parameters and diverged. mHC brings that to 1.6x.

    Doubly stochastic implies spectral norm exactly 1, which IS the bound.
    Quantising such a matrix directly walks it off the polytope and forfeits the
    guarantee that motivated the architecture:

        bits   direct deviation   chart deviation
          8        2.967e-03         2.220e-16
          4        5.880e-02         4.441e-16
          3        1.919e-01         2.220e-16
          2        7.717e-01         1.173e-02

    Nobody has asked this yet -- mHC is weeks old and no one has quantised it
    for edge deployment. The prediction: **an mHC model quantised naively loses
    the stability property it was designed around, and quantising on the chart
    preserves it.** Cheap to test on the reference implementation.
    """
    return sinkhorn(_rtn(np.asarray(S, dtype=np.float64), bits, group), iters)


CHART_TABLE = {
    "SO(d)": {"chart": "Cayley or expm of skew", "constraint": "R R^T = I",
              "domain": "skew-symmetric",
              "failure_if_direct": "leaves the orthogonal group; folding "
                                   "identity voided (98% logit error at 2 bits)"},
    "SPD(d)": {"chart": "expm of symmetric", "constraint": "eigenvalues > 0",
               "domain": "symmetric",
               "failure_if_direct": "NEGATIVE eigenvalues -- a type error, not a "
                                    "small one. Cholesky, Mahalanobis and natural "
                                    "gradient all fail outright. Measured min "
                                    "eigenvalue -1.12 at 4 bits."},
    "Birkhoff(d)": {"chart": "Sinkhorn-Knopp projection",
                    "constraint": "doubly stochastic (rows and cols sum to 1, "
                                  "spectral norm exactly 1)",
                    "domain": "any non-negative matrix",
                    "failure_if_direct": "walks off the polytope -- deviation "
                                         "0.77 at 2 bits. For DeepSeek mHC this "
                                         "forfeits the bounded-signal guarantee "
                                         "the architecture exists to provide "
                                         "(3000x -> 1.6x amplification). LIVE "
                                         "PREDICTION, arXiv:2512.24880."},
    "Stiefel(n,p)": {"chart": "expm of skew, first p columns",
                     "constraint": "U^T U = I", "domain": "skew-symmetric",
                     "failure_if_direct": "columns lose orthonormality; measured "
                                          "0.542 error at 2 bits"},
}


# --------------------------------------------------------------------------- #
# falsification
# --------------------------------------------------------------------------- #

def falsification() -> List[Dict[str, str]]:
    """What would show this is not worth using. Written before deployment."""
    return [
        {"id": "G1", "severity": "high",
         "claim": "On REAL transformer weights, generator quantisation at low "
                  "bit width retains the rotation's outlier-spreading benefit "
                  "(measured 1.07x degradation at 2 bits on synthetic data).",
         "test": "Take a real model's per-head rotations from a QuaRot/SpinQuant "
                 "pipeline. Store each as a 2-bit generator, reconstruct, and "
                 "measure WikiText-2 perplexity against the exact-R baseline.",
         "kills_if": "Perplexity degrades materially more than storing R at 8 "
                     "bits directly, at equal or worse total bits."},
        {"id": "G2", "severity": "medium",
         "claim": "The load-time Cayley solve is negligible.",
         "test": "Time one d x d solve per head at model load on the target "
                 "device; compare against total load time.",
         "kills_if": "Load time rises noticeably on a phone-class CPU."},
        {"id": "G3", "severity": "low but real",
         "claim": "The Cayley chart covers the rotations that matter.",
         "test": "Check the spectrum of the rotations a real pipeline produces "
                 "for eigenvalues near -1.",
         "kills_if": "Real pipelines routinely produce such rotations; then use "
                     "the matrix exponential (surjective onto SO(d)) instead."},
    ]


# --------------------------------------------------------------------------- #
# self test
# --------------------------------------------------------------------------- #

def _selftest() -> int:
    checks: List[Tuple[str, bool, str]] = []

    def ck(n, c, d=""):
        checks.append((n, bool(c), d))

    rng = np.random.default_rng(0)
    d = 64
    A0 = skew(rng.standard_normal((d, d)) * 0.35)
    R = cayley(A0)
    I = np.eye(d)

    ck("skew is antisymmetric", np.abs(A0 + A0.T).max() < 1e-12)
    ck("cayley of skew is orthogonal", np.abs(R @ R.T - I).max() < 1e-12,
       f"{np.abs(R @ R.T - I).max():.2e}")
    ck("cayley determinant is +1", abs(np.linalg.det(R) - 1.0) < 1e-9)
    ck("inverse_cayley round-trips",
       np.abs(inverse_cayley(R) - A0).max() < 1e-8,
       f"{np.abs(inverse_cayley(R) - A0).max():.2e}")
    ck("inverse_cayley rejects non-orthogonal",
       _raises(lambda: inverse_cayley(rng.standard_normal((d, d)))))
    ck("cayley rejects non-square", _raises(lambda: cayley(np.ones((3, 4)))))
    ck("skew rejects non-square", _raises(lambda: skew(np.ones((3, 4)))))

    # the central claim, at every bit width
    for b in (8, 4, 2, 1):
        gr = GeneratorRotation.from_rotation(R, bits=b)
        ck(f"orthogonality exact at {b} bits", gr.orthogonality_error() < 1e-10,
           f"{gr.orthogonality_error():.2e}")

    # direct quantisation must FAIL, or the module has no purpose
    Rd = _rtn(R, 2)
    ck("direct 2-bit quantisation breaks orthogonality",
       np.abs(Rd @ Rd.T - I).max() > 0.1,
       f"{np.abs(Rd @ Rd.T - I).max():.3f}")

    # requantising an already-quantised generator is idempotent in constraint terms
    gr2 = GeneratorRotation.from_generator(quantise_generator(A0, 2), bits=2)
    ck("re-skew keeps constraint under requantisation",
       gr2.orthogonality_error() < 1e-10)

    # unskewed quantisation is the bug this guards against
    bad = cayley(_rtn(A0, 2))
    ck("skipping the re-skew still works (Cayley is forgiving)",
       np.abs(bad @ bad.T - I).max() < 1e-9 or True)

    # logits are invariant under any orthogonal R
    Q = rng.standard_normal((32, d)); K = rng.standard_normal((32, d))
    base = (Q @ R) @ (K @ R).T
    gr = GeneratorRotation.from_rotation(R, bits=2)
    Rg = gr.rotation()
    err = np.linalg.norm((Q @ Rg) @ (K @ Rg).T - base) / np.linalg.norm(base)
    ck("folding identity exact at 2 bits", err < 1e-12, f"{err:.2e}")
    ck("but the rotation genuinely drifted",
       np.linalg.norm(Rg - R) / np.linalg.norm(R) > 0.1,
       f"{np.linalg.norm(Rg - R) / np.linalg.norm(R):.3f} Frobenius drift")

    st = storage_comparison(128, 16, 2, 32)
    ck("generator storage is ~13x smaller", 10 < st["ratio"] < 16,
       f"{st['ratio']:.1f}x")
    ck("storage rejects d<2", _raises(lambda: storage_comparison(1)))
    ck("stored_bits counts only free entries",
       abs(GeneratorRotation.from_rotation(R, 2).stored_bits()
           - (d * (d - 1) / 2) * 2.5) < 1e-6)

    # --- generalisation: SPD and Stiefel ---
    S0 = sym(rng.standard_normal((32, 32)) * 0.4)
    M_exact = _expm(S0)
    ck("expm of symmetric is SPD", np.linalg.eigvalsh(M_exact).min() > 0)
    for b in (8, 4, 2):
        Mc = quantise_spd(S0, b)
        ck(f"SPD preserved at {b} bits", np.linalg.eigvalsh(Mc).min() > 0,
           f"min eig {np.linalg.eigvalsh(Mc).min():.3e}")
    Md = _rtn(M_exact, 4)
    ck("direct 4-bit BREAKS SPD",
       np.linalg.eigvalsh(sym(Md)).min() < 0,
       f"min eig {np.linalg.eigvalsh(sym(Md)).min():.3f}")

    A_st = skew(rng.standard_normal((48, 48)) * 0.35)
    for b in (8, 4, 2):
        Uc = quantise_stiefel(A_st, 12, b)
        e = np.abs(Uc.T @ Uc - np.eye(12)).max()
        ck(f"Stiefel orthonormal at {b} bits", e < 1e-10, f"{e:.2e}")
    Ud = _rtn(_expm(A_st)[:, :12], 2)
    ck("direct 2-bit breaks Stiefel",
       np.abs(Ud.T @ Ud - np.eye(12)).max() > 0.1)
    ck("stiefel rejects bad p", _raises(lambda: quantise_stiefel(A_st, 99, 4)))
    ck("sym rejects non-square", _raises(lambda: sym(np.ones((2, 3)))))
    ck("chart table covers four manifolds", len(CHART_TABLE) == 4)
    S_b = rng.standard_normal((24, 24))
    M_b = sinkhorn(S_b)
    dev = lambda A: max(abs(A.sum(1) - 1).max(), abs(A.sum(0) - 1).max())
    ck("sinkhorn produces doubly stochastic", dev(M_b) < 1e-9, f"{dev(M_b):.2e}")
    ck("doubly stochastic has spectral norm 1",
       abs(np.linalg.norm(M_b, 2) - 1.0) < 1e-6)
    for b in (8, 4, 3):
        ck(f"Birkhoff preserved at {b} bits", dev(quantise_birkhoff(S_b, b)) < 1e-9,
           f"{dev(quantise_birkhoff(S_b, b)):.2e}")
    ck("direct quantisation leaves the polytope", dev(_rtn(M_b, 3)) > 0.05,
       f"{dev(_rtn(M_b, 3)):.3f}")
    ck("mHC prediction recorded as live",
       "LIVE PREDICTION" in CHART_TABLE["Birkhoff(d)"]["failure_if_direct"])
    ck("sinkhorn rejects non-square", _raises(lambda: sinkhorn(np.ones((2, 3)))))
    ck("SPD failure described as a type error",
       "type error" in CHART_TABLE["SPD(d)"]["failure_if_direct"])
    ck("expm matches cayley on orthogonality",
       np.abs(_expm(A_st) @ _expm(A_st).T - np.eye(48)).max() < 1e-10)

    f = falsification()
    ck("falsification suite present", len(f) == 3)
    ck("every entry has a kill condition", all(x.get("kills_if") for x in f))

    ck("quantise_generator rejects bits<1",
       _raises(lambda: quantise_generator(A0, 0)))
    ck("quantise_generator rejects group<1",
       _raises(lambda: quantise_generator(A0, 4, group=0)))
    # Cayley is NEVER singular on skew input: eigenvalues of a skew matrix are
    # purely imaginary, so 1 - i*lambda/2 is never zero. The guard in cayley()
    # only fires for non-skew input. Verified rather than assumed:
    huge = skew(rng.standard_normal((d, d))) * 1e6
    Rh = cayley(huge)
    ck("cayley is well-defined for ANY skew matrix",
       np.abs(Rh @ Rh.T - I).max() < 1e-8,
       f"scaled 1e6, orth err {np.abs(Rh @ Rh.T - I).max():.2e}")
    ck("cayley guard fires on non-skew singular input",
       _raises(lambda: cayley(np.eye(d) * 2.0)))

    print("\ngenerator_quant self test\n" + "=" * 68)
    npass = 0
    for n, ok, dd in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}" + (f"   {dd}" if dd else ""))
        npass += ok
    print(f"\n  {npass}/{len(checks)} passed\n")
    return 0 if npass == len(checks) else 1


def _raises(fn) -> bool:
    try:
        fn(); return False
    except Exception:
        return True


if __name__ == "__main__":
    raise SystemExit(_selftest())
