"""
Low-level utilities: TupleIndexer, scatter_hd, cartesian_bind_tensor.

These are adapted from graph_hdc and will eventually be replaced with
a pure-NumPy implementation once the PyTorch dependency is removed.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Union

import torch
import torchhd
from torch import Tensor
from torchhd import HRRTensor


def scatter_hd(
    src: Tensor,
    index: Tensor,
    *,
    op: str,
    dim_size: int | None = None,
) -> Tensor:
    """Scatter-reduce hypervectors along dim=0.

    Parameters
    ----------
    src : Tensor
        Hypervector batch ``[N, D]``.
    index : Tensor
        Bucket indices ``[N]``.
    op : str
        ``"bundle"`` (element-wise sum) or ``"bind"`` (element-wise product).
    dim_size : int, optional
        Number of output buckets.
    """
    from torch_geometric.utils import scatter

    index = index.to(src.device, dtype=torch.long, non_blocking=True)

    if index.numel() == 0:
        if dim_size is None:
            dim_size = 1
        return torchhd.identity(
            num_vectors=dim_size,
            dimensions=src.shape[-1],
            vsa="HRR",
            device=src.device,
        )

    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    if isinstance(src, HRRTensor) and op == "bundle":
        reduce = "sum"
    elif op == "bundle":
        reduce = "sum"
    else:
        reduce = "mul"

    idx_dim = int(index.max().item()) + 1
    result = scatter(src, index, dim=0, dim_size=idx_dim, reduce=reduce)

    if (num_identity := dim_size - idx_dim) > 0:
        identities = torchhd.identity(
            num_vectors=num_identity,
            dimensions=src.shape[-1],
            vsa="HRR",
            device=src.device,
        )
        result = torch.cat([result, identities])

    return result


def cartesian_bind_tensor(tensors: list[Tensor]) -> Tensor:
    """Cartesian product of hypervector sets, bound together.

    Parameters
    ----------
    tensors : list[Tensor]
        Each ``[Ni, D]``.

    Returns
    -------
    Tensor
        ``[N1*N2*..., D]``
    """
    tensors = [t for t in tensors if t is not None]
    if not tensors:
        raise ValueError("Need at least one set")

    if len(tensors) == 1:
        t = tensors[0]
        return t.unsqueeze(-1) if t.dim() == 1 else t

    sizes = [t.shape[0] for t in tensors]
    idx_grids = torch.cartesian_prod(
        *[torch.arange(n, device=tensors[0].device) for n in sizes]
    )

    hv_list = []
    for k, t in enumerate(tensors):
        idxs = idx_grids[:, k]
        hv = t[idxs] if t.dim() != 1 else t[idxs].unsqueeze(-1)
        hv_list.append(hv)

    stacked = torch.stack(hv_list, dim=1)
    return torchhd.multibind(stacked)


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
