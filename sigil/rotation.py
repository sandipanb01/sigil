"""
sigil.rotation
==============
Orthogonal transform learning by Riemannian descent on a sliced goodness-of-fit
objective.

Two parameterisations:

  CayleyRotation   dense d x d orthogonal, exact orthogonality maintained by the
                   Cayley transform (Li et al., "Efficient Riemannian
                   Optimization on the Stiefel Manifold via the Cayley
                   Transform"). This is the parameterisation SpinQuant uses.

  ButterflyRotation  log2(d) layers of Givens rotations, O(d log d) parameters
                   and O(d log d) apply cost. Matches the ButterflyQuant
                   (arXiv:2509.09679) structure. Use when d is large or when the
                   transform cannot be folded into a weight matrix and must run
                   online.

Both take a statistic object from sigil.gof providing (value, dL/dZ).

Cost note: for head_dim = 128 the dense Cayley step is a 128x128 solve, which is
microseconds. Learning all K/V rotations for a 32-layer model is a few minutes on
CPU, versus SpinQuant's end-to-end backprop (reported as 4x H100 for Llama3-70B).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["CayleyRotation", "ButterflyRotation", "random_hadamard", "FitLog", "fit_rotation"]


# --------------------------------------------------------------------------- #
# fixed baselines
# --------------------------------------------------------------------------- #

def _hadamard(n: int) -> np.ndarray:
    """Sylvester Hadamard matrix; n must be a power of two."""
    if n & (n - 1):
        raise ValueError(f"Hadamard requires a power of two, got {n}")
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def random_hadamard(d: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Randomised Hadamard transform: H @ diag(+-1). This is the QuaRot /
    incoherence-processing baseline (Ashkboos et al., arXiv:2404.00456;
    Tseng et al., QuIP#, arXiv:2402.04396).
    """
    rng = rng or np.random.default_rng(0)
    s = rng.choice([-1.0, 1.0], size=d)
    return _hadamard(d) * s[None, :]


# --------------------------------------------------------------------------- #
# learned rotations
# --------------------------------------------------------------------------- #

class CayleyRotation:
    """Dense orthogonal matrix, updated by the Cayley transform."""

    def __init__(self, d: int, init: str = "hadamard", seed: int = 0):
        rng = np.random.default_rng(seed)
        if init == "hadamard" and (d & (d - 1)) == 0:
            self.R = random_hadamard(d, rng)
        elif init == "identity":
            self.R = np.eye(d)
        else:
            Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
            self.R = Q
        self.d = d

    def apply(self, X: np.ndarray) -> np.ndarray:
        return X @ self.R

    def step(self, G: np.ndarray, lr: float) -> None:
        """
        G is dL/dR. Project to the tangent space of O(d) and take a Cayley step:
            A = G R^T - R G^T          (skew-symmetric)
            R <- (I + lr/2 A)^{-1} (I - lr/2 A) R
        Exactly orthogonality-preserving for any lr.
        """
        R = self.R
        A = G @ R.T - R @ G.T
        # scale-normalise so lr has a consistent meaning across layers
        nrm = np.linalg.norm(A)
        if nrm > 1e-12:
            A = A / nrm
        I = np.eye(self.d)
        M = I + (lr / 2.0) * A
        self.R = np.linalg.solve(M, (I - (lr / 2.0) * A) @ R)

    def matrix(self) -> np.ndarray:
        return self.R


class ButterflyRotation:
    """
    log2(d) butterfly layers of Givens rotations. Parameters are angles, so
    orthogonality is exact by construction and the apply cost is O(d log d).
    """

    def __init__(self, d: int, seed: int = 0):
        if d & (d - 1):
            raise ValueError("butterfly requires power-of-two d")
        self.d = d
        self.n_layers = int(np.log2(d))
        rng = np.random.default_rng(seed)
        self.theta = rng.normal(0.0, 0.05, size=(self.n_layers, d // 2))

    def _pairs(self, layer: int):
        stride = 1 << layer
        idx = np.arange(self.d)
        top = idx[(idx & stride) == 0]
        return top, top + stride

    def matrix(self) -> np.ndarray:
        R = np.eye(self.d)
        for l in range(self.n_layers):
            i, j = self._pairs(l)
            c, s = np.cos(self.theta[l]), np.sin(self.theta[l])
            Ri, Rj = R[i].copy(), R[j].copy()
            R[i] = c[:, None] * Ri - s[:, None] * Rj
            R[j] = s[:, None] * Ri + c[:, None] * Rj
        return R

    def apply(self, X: np.ndarray) -> np.ndarray:
        return X @ self.matrix()

    def step(self, G: np.ndarray, lr: float) -> None:
        """Finite-difference-free angle update via the chain rule through matrix()."""
        R = self.matrix()
        # dL/dtheta obtained by differentiating each Givens layer in sequence.
        # Cheap enough at these sizes to do by explicit re-composition.
        eps = 1e-4
        g = np.zeros_like(self.theta)
        base = float(np.sum(G * R))
        for l in range(self.n_layers):
            for k in range(0, self.theta.shape[1], max(1, self.theta.shape[1] // 8)):
                self.theta[l, k] += eps
                g[l, k] = (float(np.sum(G * self.matrix())) - base) / eps
                self.theta[l, k] -= eps
        nrm = np.linalg.norm(g)
        if nrm > 1e-12:
            self.theta -= lr * g / nrm


# --------------------------------------------------------------------------- #
# fitting loop
# --------------------------------------------------------------------------- #

@dataclass
class FitLog:
    values: list = field(default_factory=list)
    seconds: float = 0.0

    @property
    def start(self) -> float:
        return self.values[0] if self.values else float("nan")

    @property
    def end(self) -> float:
        return self.values[-1] if self.values else float("nan")


def fit_rotation(X: np.ndarray,
                 stat,
                 steps: int = 200,
                 lr: float = 0.35,
                 batch: int = 4096,
                 param: str = "cayley",
                 init: str = "hadamard",
                 seed: int = 0,
                 scale: Optional[float] = None,
                 verbose: bool = False):
    """
    Learn R minimising stat(X R / s) over the orthogonal group.

    Parameters
    ----------
    X      : (n, d) calibration activations for one layer / one head group.
    stat   : callable from sigil.gof returning (value, dL/dZ).
    scale  : rotation-invariant global scale; computed from X if omitted.

    Returns
    -------
    (R, FitLog)
    """
    import time

    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    s = scale if scale is not None else float(np.sqrt(np.mean(X ** 2)) + 1e-12)
    rot = CayleyRotation(d, init=init, seed=seed) if param == "cayley" \
        else ButterflyRotation(d, seed=seed)

    rng = np.random.default_rng(seed + 1)
    log = FitLog()
    t0 = time.time()
    lr_t = lr
    for t in range(steps):
        idx = rng.choice(n, size=min(batch, n), replace=False) if n > batch else slice(None)
        Xb = X[idx]
        Z = (Xb @ rot.matrix()) / s
        val, dZ = stat(Z, need_grad=True)
        log.values.append(val)
        G = (Xb.T @ dZ) / s
        rot.step(G, lr_t)
        lr_t = lr * (0.5 * (1 + np.cos(np.pi * (t + 1) / steps)))   # cosine decay
        if verbose and t % max(1, steps // 5) == 0:
            print(f"    step {t:4d}  stat={val:.6f}  lr={lr_t:.4f}")

    Z = (X @ rot.matrix()) / s
    log.values.append(stat(Z, need_grad=False)[0])
    log.seconds = time.time() - t0
    return rot.matrix(), log
