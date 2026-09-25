"""
sigil.gof
=========
Sliced goodness-of-fit statistics with analytic gradients.

These are the objective functions SIGIL optimises rotations against. Each one
answers: "how far is the marginal of this data, along a given set of directions,
from the distribution the quantiser wants to see?"

Three targets are implemented:

  EppsPulley  -> target N(0,1).  Characteristic-function L2 distance, Gaussian
                 weighting. This is the LeJEPA/SIGReg test (Balestriero & LeCun,
                 arXiv:2511.08544). Correct target for lattice / normal-float /
                 companded codebooks, and for the isotropy-based OOD score.

  UniformCvM  -> target U(-1,1). Cramer-von Mises statistic. Correct target for a
                 symmetric fixed-range uniform integer quantiser (this is the
                 regime DartQuant, arXiv:2511.04063, argues for).

  Kurtosis    -> excess kurtosis, |k - 3|. A single-moment proxy for Gaussianity.
                 Included only as the KurTail (arXiv:2503.01483) baseline.

Slicing directions
------------------
`slices="axes"`   : the coordinate axes. This is what a per-channel or per-token
                    scalar quantiser actually acts on, so it is the default.
`slices="random"` : random unit directions. This is the LeJEPA formulation; it
                    controls the *joint* law, which is what a vector quantiser
                    and the chi-square routing score need.

All statistics operate on Z = X @ R with a single rotation-invariant global
scale s = sqrt(mean(X**2)), so isotropy (equal per-coordinate variance) is part
of what is being optimised rather than being normalised away per column.

Every statistic returns (value, dL/dZ) so the Cayley optimiser in
sigil.rotation needs no autodiff framework.
"""

from __future__ import annotations

import numpy as np

__all__ = ["EppsPulley", "UniformCvM", "Kurtosis", "global_scale", "make_slices"]


def global_scale(X: np.ndarray) -> float:
    """Rotation-invariant RMS scale. Orthogonal R leaves ||X||_F unchanged."""
    return float(np.sqrt(np.mean(np.asarray(X, dtype=np.float64) ** 2)) + 1e-12)


def make_slices(d: int, n_slices: int, mode: str, rng: np.random.Generator):
    """
    Return a (d, m) matrix of slicing directions, or None for the identity
    (axis) case, which is handled without a matmul.
    """
    if mode == "axes":
        return None
    if mode == "random":
        U = rng.standard_normal((d, n_slices))
        U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-12
        return U
    raise ValueError(f"unknown slice mode {mode!r}")


class _SlicedStat:
    """Shared plumbing: project Z onto slices, evaluate a 1-D statistic, chain back."""

    def __init__(self, n_slices: int = 0, slices: str = "axes", seed: int = 0):
        self.n_slices = n_slices
        self.slices = slices
        self.rng = np.random.default_rng(seed)
        self._U = None

    def _directions(self, d: int):
        if self.slices == "axes":
            return None
        if self._U is None or self._U.shape[0] != d:
            self._U = make_slices(d, self.n_slices, self.slices, self.rng)
        return self._U

    def __call__(self, Z: np.ndarray, need_grad: bool = True):
        Z = np.asarray(Z, dtype=np.float64)
        U = self._directions(Z.shape[1])
        P = Z if U is None else Z @ U          # (n, m)
        val, dP = self._stat_1d(P, need_grad)
        if not need_grad:
            return val, None
        dZ = dP if U is None else dP @ U.T
        return val, dZ

    def _stat_1d(self, P, need_grad):           # pragma: no cover - interface
        raise NotImplementedError


class EppsPulley(_SlicedStat):
    """
    Discretised Epps-Pulley characteristic-function test against N(0,1).

        T = sum_k w_k * [ (C_k - exp(-t_k^2/2))^2 + S_k^2 ]
        C_k = mean_i cos(t_k * p_i),   S_k = mean_i sin(t_k * p_i)

    with w_k a Gaussian window over the frequency grid. Consistent against every
    fixed alternative (it compares the whole characteristic function), unlike a
    single moment such as kurtosis.

    `num_points` mirrors the grid size used by the reference LeJEPA
    implementation (galilai-group/lejepa uses EppsPulley(num_points=17)).
    """

    def __init__(self, num_points: int = 17, t_max: float = 3.0, bandwidth: float = 1.5,
                 n_slices: int = 0, slices: str = "axes", seed: int = 0):
        super().__init__(n_slices=n_slices, slices=slices, seed=seed)
        self.t = np.linspace(t_max / num_points, t_max, num_points)
        self.w = np.exp(-(self.t ** 2) / (2.0 * bandwidth ** 2))
        self.w = self.w / self.w.sum()
        self.ref = np.exp(-(self.t ** 2) / 2.0)

    def _stat_1d(self, P, need_grad):
        n, m = P.shape
        t = self.t[None, None, :]                       # (1,1,K)
        A = P[:, :, None] * t                           # (n,m,K)
        cosA, sinA = np.cos(A), np.sin(A)
        C = cosA.mean(axis=0)                           # (m,K)
        S = sinA.mean(axis=0)
        dC = C - self.ref[None, :]
        per_slice = ((dC ** 2 + S ** 2) * self.w[None, :]).sum(axis=1)   # (m,)
        val = float(per_slice.mean())
        if not need_grad:
            return val, None
        # d/dP_ij of mean over slices of sum_k w_k[(C-ref)^2 + S^2]
        coefC = (2.0 * dC * self.w[None, :]) / (n * m)   # (m,K)
        coefS = (2.0 * S * self.w[None, :]) / (n * m)
        dP = (coefC[None, :, :] * (-t) * sinA + coefS[None, :, :] * t * cosA).sum(axis=2)
        return val, dP

    def per_slice(self, Z: np.ndarray) -> np.ndarray:
        """Per-direction statistic, used for bit allocation and diagnostics."""
        Z = np.asarray(Z, dtype=np.float64)
        U = self._directions(Z.shape[1])
        P = Z if U is None else Z @ U
        t = self.t[None, None, :]
        A = P[:, :, None] * t
        C = np.cos(A).mean(axis=0)
        S = np.sin(A).mean(axis=0)
        return (((C - self.ref[None, :]) ** 2 + S ** 2) * self.w[None, :]).sum(axis=1)


class UniformCvM(_SlicedStat):
    """
    Smooth Cramer-von Mises statistic against U(-a, a), a = sqrt(3) so the target
    has unit variance and is directly comparable to the N(0,1) target.

    The empirical CDF is smoothed with a logistic kernel of width `h` so the
    statistic is differentiable:

        F_hat(x) = mean_i sigmoid((x - p_i)/h)
        T = mean over a grid of (F_hat(x) - F_target(x))^2
    """

    def __init__(self, num_points: int = 33, h: float = 0.08,
                 n_slices: int = 0, slices: str = "axes", seed: int = 0):
        super().__init__(n_slices=n_slices, slices=slices, seed=seed)
        self.a = np.sqrt(3.0)
        self.x = np.linspace(-self.a * 1.15, self.a * 1.15, num_points)
        self.h = h
        self.F = np.clip((self.x + self.a) / (2 * self.a), 0.0, 1.0)

    def _stat_1d(self, P, need_grad):
        n, m = P.shape
        x = self.x[None, None, :]
        z = (x - P[:, :, None]) / self.h
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        Fh = sig.mean(axis=0)                            # (m,K)
        d = Fh - self.F[None, :]
        val = float((d ** 2).mean(axis=1).mean())
        if not need_grad:
            return val, None
        K = self.x.size
        coef = (2.0 * d) / (K * n * m)                   # (m,K)
        dsig = sig * (1.0 - sig) * (-1.0 / self.h)       # d sigmoid / d P
        dP = (coef[None, :, :] * dsig).sum(axis=2)
        return val, dP


class Kurtosis(_SlicedStat):
    """
    KurTail-style baseline: mean over slices of (excess kurtosis)^2.
    Uses detached mean/variance, matching the layer-wise KurTail formulation.
    """

    def _stat_1d(self, P, need_grad):
        n, m = P.shape
        mu = P.mean(axis=0, keepdims=True)
        c = P - mu
        var = (c ** 2).mean(axis=0, keepdims=True) + 1e-12
        k = (c ** 4).mean(axis=0) / (var[0] ** 2)
        e = k - 3.0
        val = float((e ** 2).mean())
        if not need_grad:
            return val, None
        # gradient through the 4th moment only (variance treated as detached,
        # as in the reference layer-wise implementation)
        dP = (2.0 * e)[None, :] * (4.0 * c ** 3) / (n * var * m)
        return val, dP
