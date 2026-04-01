"""
Feature encoder: maps discrete feature tuples to hypervectors via codebooks.
"""

from __future__ import annotations

import numpy as np

from hyper_fingerprints.utils import TupleIndexer


class FeatureEncoder:
    """Encodes multi-dimensional feature tuples into HRR hypervectors."""

    def __init__(
        self,
        dim: int,
        num_categories: int,
        indexer: TupleIndexer,
        *,
        seed: int | None = None,
        codebook: np.ndarray | None = None,
    ) -> None:
        self.dim = dim
        self.num_categories = num_categories
        self.indexer = indexer

        if codebook is not None:
            self.codebook = codebook
        else:
            rng = np.random.default_rng(seed)
            self.codebook = self._generate_codebook(rng)

    def _generate_codebook(self, rng: np.random.Generator) -> np.ndarray:
        cb = rng.standard_normal((self.num_categories, self.dim))
        norms = np.linalg.norm(cb, axis=-1, keepdims=True)
        cb = cb / norms
        return cb

    def encode_indices(self, data: np.ndarray) -> np.ndarray:
        """Map feature tuples ``[N, F]`` to flat codebook indices ``[N]``."""
        data = data.astype(np.int64)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        tup = list(map(tuple, data.tolist()))
        return np.array(self.indexer.get_idxs(tup), dtype=np.int64)

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode feature tuples ``[N, F]`` into hypervectors ``[N, D]``."""
        idxs = self.encode_indices(data)
        return self.codebook[idxs]
