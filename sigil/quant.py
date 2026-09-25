"""
sigil.quant
===========
Quantisers and the fidelity metrics SIGIL is scored against.

Two families, because the optimal input distribution differs between them and
that difference is the whole reason SIGIL parameterises its target:

  IntRTN   symmetric uniform integer quantiser, per-channel or per-token scale.
           At a fixed clipping range the MSE-optimal input marginal is UNIFORM.
           This is what llama.cpp Q4_0/Q8_0 and the Hexagon INT4/INT8 dataflow do.

  NFCodebook  normal-float codebook: levels placed at equiprobable quantiles of
           N(0,1), then rescaled per block. The MSE-optimal input marginal is
           GAUSSIAN. This is the NF4 / lattice / companded family.

So: quantiser choice determines which sigil.gof target you should optimise.
Getting this pairing wrong is the most common way rotation methods leave
performance on the table, and it is why SIGIL exposes the target as a knob
rather than hard-coding Gaussianity.

Metrics
-------
rel_mse             plain reconstruction error.
inner_product_error error in q . k  -- the quantity attention actually consumes.
                    This is the right KV-cache metric (cf. TurboQuant); a method
                    can win on rel_mse and still distort attention scores.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

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



__all__ = ["IntRTN", "NFCodebook", "quantize_kv", "rel_mse",
           "inner_product_error", "effective_bits"]


# --------------------------------------------------------------------------- #
# quantisers
# --------------------------------------------------------------------------- #

class IntRTN:
    """
    Symmetric uniform integer quantiser, round-to-nearest.

    axis=0 -> one scale per column (per-channel; correct for Keys, which carry
              channel-structured outliers)
    axis=1 -> one scale per row (per-token; correct for Values)

    clip: fraction of the absmax used as the clipping range. 1.0 is plain
    absmax. Values below 1.0 trade clipping error for resolution and are
    searched by `search_clip`.
    """

    def __init__(self, bits: int, axis: int = 0, clip: float = 1.0,
                 group: int | None = None):
        if not 1 <= bits <= 16:
            raise ValueError("bits must be in [1, 16]")
        self.bits = bits
        self.axis = axis
        self.clip = float(clip)
        self.group = group
        self.qmax = _qmax(bits)

    def _scales(self, X: np.ndarray) -> np.ndarray:
        amax = np.abs(X).max(axis=self.axis, keepdims=True)
        return np.maximum(amax * self.clip, 1e-12) / self.qmax

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.group:
            return self._grouped(X)
        s = self._scales(X)
        q = np.clip(np.round(X / s), -self.qmax, self.qmax)
        return q * s

    def _grouped(self, X: np.ndarray) -> np.ndarray:
        """Block-wise scales along the feature axis, as in GGUF Q4_0/Q2_K."""
        n, d = X.shape
        g = self.group
        pad = (-d) % g
        Xp = np.pad(X, ((0, 0), (0, pad))) if pad else X
        B = Xp.reshape(n, -1, g)
        amax = np.abs(B).max(axis=2, keepdims=True)
        s = np.maximum(amax * self.clip, 1e-12) / self.qmax
        Q = np.clip(np.round(B / s), -self.qmax, self.qmax) * s
        out = Q.reshape(n, -1)
        return out[:, :d] if pad else out

    def search_clip(self, X: np.ndarray, grid=None) -> "IntRTN":
        """Pick the clipping ratio minimising MSE on X. Cheap 1-D search."""
        grid = grid if grid is not None else np.linspace(0.5, 1.0, 11)
        best, best_e = self.clip, np.inf
        for c in grid:
            self.clip = float(c)
            e = float(np.mean((self(X) - X) ** 2))
            if e < best_e:
                best, best_e = float(c), e
        self.clip = best
        return self


class NFCodebook:
    """
    Normal-float codebook. Levels are the equiprobable quantiles of N(0,1),
    symmetric and including zero, then scaled per block by absmax. NF4 is the
    bits=4 case.
    """

    def __init__(self, bits: int, group: int = 64):
        self.bits = bits
        self.group = group
        _qmax(bits, 1, 24)          # validate the exponent
        k = 2 ** int(bits)
        # equiprobable quantiles, offset to avoid infinite tails
        p = (np.arange(k) + 0.5) / k
        lv = norm.ppf(p)
        lv = lv / np.abs(lv).max()
        # snap the nearest level to exact zero, which matters for sparsity
        lv[np.argmin(np.abs(lv))] = 0.0
        self.levels = np.sort(lv)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        g = self.group
        pad = (-d) % g
        Xp = np.pad(X, ((0, 0), (0, pad))) if pad else X
        B = Xp.reshape(n, -1, g)
        s = np.maximum(np.abs(B).max(axis=2, keepdims=True), 1e-12)
        Bn = B / s
        idx = np.abs(Bn[..., None] - self.levels[None, None, None, :]).argmin(axis=-1)
        out = (self.levels[idx] * s).reshape(n, -1)
        return out[:, :d] if pad else out


def quantize_kv(K: np.ndarray, V: np.ndarray, bits: int,
                quantiser: str = "int", group: int | None = None,
                search: bool = True):
    """
    KIVI-convention KV quantisation: per-channel for Keys, per-token for Values.
    Returns (K_hat, V_hat).
    """
    if quantiser == "nf":
        q = NFCodebook(bits, group=group or 64)
        return q(K), q(V)
    qk = IntRTN(bits, axis=0, group=group)
    qv = IntRTN(bits, axis=1, group=group)
    if search:
        qk.search_clip(K)
        qv.search_clip(V)
    return qk(K), qv(V)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def rel_mse(X: np.ndarray, Xh: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    return float(np.sum((Xh - X) ** 2) / (np.sum(X ** 2) + 1e-12))


def inner_product_error(Q: np.ndarray, K: np.ndarray, Kh: np.ndarray) -> float:
    """
    Relative error in the attention logits q . k induced by quantising K.
    Rotations are orthogonal, so this is measured in the original basis and is
    directly comparable across methods.
    """
    A = Q @ K.T
    Ah = Q @ Kh.T
    return float(np.sqrt(np.sum((Ah - A) ** 2) / (np.sum(A ** 2) + 1e-12)))


def effective_bits(bits: int, group: int | None, d: int,
                   scale_bits: int = 16) -> float:
    """Bits per weight including the stored per-group scale."""
    if not group:
        return bits + scale_bits / d
    return bits + scale_bits / group
