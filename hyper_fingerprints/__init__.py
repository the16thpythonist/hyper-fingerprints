"""Hyper Fingerprints: hyperdimensional molecular fingerprints."""

import numpy as np

from hyper_fingerprints.encoder import Encoder
from hyper_fingerprints.features import DEFAULT_ATOM_TYPES, DEFAULT_FEATURES


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between two sets of vectors.

    Parameters
    ----------
    a : np.ndarray
        Shape ``(N, D)`` or ``(D,)``.
    b : np.ndarray
        Shape ``(M, D)`` or ``(D,)``.

    Returns
    -------
    np.ndarray
        Shape ``(N, M)``. Entry ``[i, j]`` is the cosine similarity
        between ``a[i]`` and ``b[j]``.
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


__all__ = ["Encoder", "DEFAULT_ATOM_TYPES", "DEFAULT_FEATURES", "cosine_similarity"]
