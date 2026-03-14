"""
Low-level utilities: TupleIndexer, scatter_hd, HRR algebra helpers.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union

import numpy as np


# ───────────────────── Graph data structures ─────────────────────


@dataclass
class GraphData:
    """Minimal graph representation for a single molecule."""
    x: np.ndarray          # [N, 5]
    edge_index: np.ndarray  # [2, E]


@dataclass
class GraphBatch:
    """Batched graph representation for multiple molecules."""
    x: np.ndarray          # [total_N, 5]
    edge_index: np.ndarray  # [2, total_E]
    batch: np.ndarray      # [total_N]


def batch_from_data_list(data_list: list[GraphData]) -> GraphBatch:
    """Concatenate a list of GraphData into a single GraphBatch."""
    xs = []
    edge_indices = []
    batch_indices = []
    node_offset = 0

    for i, data in enumerate(data_list):
        num_nodes = data.x.shape[0]
        xs.append(data.x)

        if data.edge_index.shape[1] > 0:
            edge_indices.append(data.edge_index + node_offset)
        else:
            edge_indices.append(data.edge_index)

        batch_indices.append(np.full(num_nodes, i, dtype=np.int64))
        node_offset += num_nodes

    return GraphBatch(
        x=np.concatenate(xs, axis=0) if xs else np.empty((0, 5), dtype=np.float64),
        edge_index=np.concatenate(edge_indices, axis=1) if edge_indices else np.empty((2, 0), dtype=np.int64),
        batch=np.concatenate(batch_indices) if batch_indices else np.empty(0, dtype=np.int64),
    )


# ───────────────────── HRR algebra ─────────────────────


def hrr_identity(n: int, d: int) -> np.ndarray:
    """HRR identity vectors: ``[n, d]`` with ``[:,0] = 1``, rest zero."""
    out = np.zeros((n, d), dtype=np.float64)
    out[:, 0] = 1.0
    return out


def hrr_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """HRR binding via circular convolution (element-wise FFT multiply)."""
    return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))


def hrr_multibundle(x: np.ndarray) -> np.ndarray:
    """HRR bundling: sum along the second-to-last axis."""
    return np.sum(x, axis=-2)


# ───────────────────── Scatter ─────────────────────


def scatter_hd(
    src: np.ndarray,
    index: np.ndarray,
    *,
    op: str,
    dim_size: int | None = None,
) -> np.ndarray:
    """Scatter-reduce hypervectors along dim=0.

    Parameters
    ----------
    src : np.ndarray
        Hypervector batch ``[N, D]``.
    index : np.ndarray
        Bucket indices ``[N]``.
    op : str
        ``"bundle"`` (element-wise sum) or ``"bind"`` (element-wise product).
    dim_size : int, optional
        Number of output buckets.
    """
    d = src.shape[-1]

    if index.size == 0:
        if dim_size is None:
            dim_size = 1
        return hrr_identity(dim_size, d)

    if dim_size is None:
        dim_size = int(index.max()) + 1

    if op == "bundle":
        out = np.zeros((dim_size, d), dtype=np.float64)
        np.add.at(out, index, src)
    else:  # bind
        out = hrr_identity(dim_size, d)
        np.multiply.at(out, index, src)

    return out


# ───────────────────── TupleIndexer ─────────────────────


class TupleIndexer:
    """Bijection between feature tuples and flat indices."""

    def __init__(self, sizes: Sequence[int]) -> None:
        sizes = [s for s in sizes if s]
        self.sizes = sizes
        self.idx_to_tuple: list[tuple[int, ...]] = (
            list(itertools.product(*(range(N) for N in sizes))) if sizes else []
        )
        self.tuple_to_idx: dict[tuple[int, ...], int] = (
            {t: idx for idx, t in enumerate(self.idx_to_tuple)} if sizes else {}
        )

    def get_tuple(self, idx: int) -> tuple[int, ...]:
        return self.idx_to_tuple[idx]

    def get_tuples(self, idxs: list[int]) -> list[tuple[int, ...]]:
        return [self.idx_to_tuple[idx] for idx in idxs]

    def get_idx(self, tup: Union[tuple[int, ...], int]) -> int | None:
        if isinstance(tup, int):
            return self.tuple_to_idx.get((tup,))
        return self.tuple_to_idx.get(tup)

    def get_idxs(self, tuples: list[Union[tuple[int, ...], int]]) -> list[int]:
        return [self.get_idx(tup) for tup in tuples]

    def size(self) -> int:
        return len(self.idx_to_tuple)
