"""
fuse.py — P/S intra-brain covariance fusion library
----------------------------------------------------
Hyperscanning context: each G## session has two simultaneous subjects.
The 16×16 joint covariance has blocks:
    [ P_intra (8×8) | inter ]
    [ inter^T       | S_intra ]
P = Primary's intra-brain 8×8 block (G##_EC_p.npy)
S = Secondary's intra-brain 8×8 block (G##_EC_s.npy)
inter = off-diagonal cross-brain block (G##_EC_inter.npy)  ← optional
Note: intra >> inter in magnitude (within-brain EEG correlates much more than cross-brain).

All functions have the signature f(P, S, inter=None) and return an (n, d, d) or (d, d)
array of the same leading shape as P.

Registry:
    FUSION_METHODS : dict[str, callable]  —  name → function

Usage:
    from fuse import FUSION_METHODS
    fused = FUSION_METHODS["arith_mean"](P_batch, S_batch)       # (n, 8, 8)
    fused = FUSION_METHODS["inter_gram"](P_batch, S_batch, inter) # requires inter.npy

Adding a new method:
    1. Define a function f(P, S, inter=None) -> np.ndarray (same shape as P).
    2. Add it to FUSION_METHODS at the bottom of this file.
"""

import warnings

import numpy as np
from scipy.linalg import expm, logm


# =============================================================================
# Fusion methods
# =============================================================================

def arith_mean(P: np.ndarray, S: np.ndarray, inter=None) -> np.ndarray:
    """Arithmetic (Euclidean) mean: (P + S) / 2.

    Preserves SPD: convex combination of two PD matrices is PD.
    Fastest option; ignores the curved geometry of the SPD manifold.
    """
    return (P + S) * 0.5


def log_euclidean_mean(P: np.ndarray, S: np.ndarray, inter=None) -> np.ndarray:
    """Log-Euclidean mean: expm((logm(P) + logm(S)) / 2).

    Geodesic midpoint under the log-Euclidean metric.
    Respects the manifold structure of SPD matrices (never produces
    near-singular results). Slower due to per-trial matrix log/exp.
    """
    def _logm(X: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning,
                                    message="logm result may be inaccurate")
            return logm(X).real

    def _expm(X: np.ndarray) -> np.ndarray:
        return expm(X).real

    if P.ndim == 2:
        return _expm((_logm(P) + _logm(S)) * 0.5)

    return np.stack([
        _expm((_logm(p) + _logm(s)) * 0.5)
        for p, s in zip(P, S)
    ])


def matrix_product(P: np.ndarray, S: np.ndarray, inter=None) -> np.ndarray:
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


def p_only(P: np.ndarray, S: np.ndarray, inter=None) -> np.ndarray:
    """Baseline: use Primary intra-brain covariance only (ignore S and inter)."""
    return P.copy()


def s_only(P: np.ndarray, S: np.ndarray, inter=None) -> np.ndarray:
    """Baseline: use Secondary intra-brain covariance only (ignore P and inter)."""
    return S.copy()


def inter_gram(P: np.ndarray, S: np.ndarray, inter: np.ndarray) -> np.ndarray:
    """Inter-brain Gram matrix: inter @ inter.T.

    The raw inter block (8×8) is a cross-covariance between Primary and
    Secondary channels — not symmetric, not SPD.  The Gram matrix inter @ inter.T
    is always PSD (made SPD by ensure_spd in the evaluation loop) and captures
    inter-brain coupling strength: larger eigenvalues mean more synchrony.

    Requires G##_*_inter.npy files alongside _p.npy / _s.npy.
    In the hyperscanning data the inter block is the bottom-left 8×8 of the
    16×16 joint covariance (Primary rows × Secondary cols).
    """
    if inter is None:
        raise ValueError(
            "inter_gram requires the inter-brain block. "
            "Save the bottom-left 8×8 of the 16×16 joint covariance "
            "as G##_<cond>_inter.npy alongside the _p.npy / _s.npy files."
        )
    if inter.ndim == 2:
        return inter @ inter.T
    # batched: (n, 8, 8) @ (n, 8, 8)^T
    return inter @ inter.transpose(0, 2, 1)


# =============================================================================
# Registry  —  add new methods here
# =============================================================================

FUSION_METHODS: dict[str, callable] = {
    "arith_mean":     arith_mean,
    "log_euclidean":  log_euclidean_mean,
    "matrix_product": matrix_product,
    "s_only":         s_only,
    "inter_gram":     inter_gram,   # inter-brain coupling baseline
    "p_only":         p_only,       # included for completeness; scientifically weak
}
