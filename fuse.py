"""
fuse.py — P/S region covariance matrix fusion library
------------------------------------------------------
All functions accept two (n, d, d) or (d, d) SPD matrix arrays
and return a fused array of the same shape.

Registry:
    FUSION_METHODS : dict[str, callable]  —  name → function

Usage:
    from fuse import FUSION_METHODS
    fused = FUSION_METHODS["arith_mean"](P_batch, S_batch)   # (n, 8, 8)

Adding a new method:
    1. Define a function f(P, S) -> np.ndarray (same shape as P).
    2. Add it to FUSION_METHODS at the bottom of this file.
"""

import numpy as np
from scipy.linalg import expm, logm


# =============================================================================
# Fusion methods
# =============================================================================

def arith_mean(P: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Arithmetic (Euclidean) mean: (P + S) / 2.

    Preserves SPD: convex combination of two PD matrices is PD.
    Fastest option; ignores the curved geometry of the SPD manifold.
    """
    return (P + S) * 0.5


def log_euclidean_mean(P: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Log-Euclidean mean: expm((logm(P) + logm(S)) / 2).

    Geodesic midpoint under the log-Euclidean metric.
    Respects the manifold structure of SPD matrices (never produces
    near-singular results). Slower due to per-trial matrix log/exp.
    """
    def _logm(X: np.ndarray) -> np.ndarray:
        return logm(X).real

    def _expm(X: np.ndarray) -> np.ndarray:
        return expm(X).real

    if P.ndim == 2:
        return _expm((_logm(P) + _logm(S)) * 0.5)

    return np.stack([
        _expm((_logm(p) + _logm(s)) * 0.5)
        for p, s in zip(P, S)
    ])


def matrix_product(P: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Congruence product: P @ S @ P.

    Maps S into the coordinate system defined by P.
    Result is SPD whenever P and S are SPD
    (proof: for any x, x^T (PSP) x = (Px)^T S (Px) > 0).
    Note: eigenvalues scale as λ_P² · λ_S, so the result may need
    SPD projection if P has very large eigenvalues.
    """
    if P.ndim == 2:
        return P @ S @ P
    return (P @ S) @ P   # batched: (n,d,d) @ (n,d,d) @ (n,d,d)


def p_only(P: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Baseline: use P-region covariance only (ignore S)."""
    return P.copy()


def s_only(P: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Baseline: use S-region covariance only (ignore P)."""
    return S.copy()


# =============================================================================
# Registry  —  add new methods here
# =============================================================================

FUSION_METHODS: dict[str, callable] = {
    "arith_mean":     arith_mean,
    "log_euclidean":  log_euclidean_mean,
    "matrix_product": matrix_product,
    "p_only":         p_only,
    "s_only":         s_only,
}
