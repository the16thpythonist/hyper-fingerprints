"""
Feature encoder: maps discrete feature tuples to hypervectors via codebooks.
"""

from __future__ import annotations

import torch
import torchhd

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
        dtype: str = "float64",
        device: torch.device | str = "cpu",
    ) -> None:
        self.dim = dim
        self.num_categories = num_categories
        self.indexer = indexer
        self.dtype = dtype
        self.device = torch.device(device)

        if seed is not None:
            torch.manual_seed(seed)
        self.codebook = self._generate_codebook()

    def _generate_codebook(self) -> torch.Tensor:
        dt = torch.float64 if self.dtype == "float64" else torch.float32
        cb = torchhd.random(
            self.num_categories, self.dim, vsa="HRR", device="cpu", dtype=dt
        )
        return cb.to(self.device)

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Encode feature tuples ``[N, F]`` into hypervectors ``[N, D]``."""
        data = data.long()
        if data.dim() == 1:
            data = data.unsqueeze(-1)
        tup = list(map(tuple, data.tolist()))
        idxs = self.indexer.get_idxs(tup)
        idxs_tens = torch.tensor(idxs, dtype=torch.long, device=self.device)
        return self.codebook[idxs_tens]
