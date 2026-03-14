"""
Encoder: hyperdimensional molecular fingerprints via HRR message passing.
"""

from __future__ import annotations

import math
from typing import Union

import torch
import torchhd
from rdkit import Chem
from torch import Tensor
from torch_geometric.data import Batch, Data

from hyper_fingerprints.codebook import FeatureEncoder
from hyper_fingerprints.features import (
    DEFAULT_ATOM_TYPES,
    atom_type_map,
    feature_bins,
    mol_to_data,
)
from hyper_fingerprints.utils import (
    TupleIndexer,
    cartesian_bind_tensor,
    scatter_hd,
)


class Encoder:
    """Hyperdimensional molecular fingerprint encoder.

    Encodes molecules into fixed-dimensional hypervectors using HRR
    (Holographic Reduced Representations) with message passing.

    Parameters
    ----------
    dimension : int
        Hypervector dimensionality (default 256).
    depth : int
        Number of message-passing layers (default 3).
    atom_types : list[str], optional
        Supported atom symbols. Defaults to
        ``["Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]``.
    seed : int, optional
        Random seed for reproducible codebook generation.
    normalize : bool
        L2-normalize hypervectors after each message-passing layer
        (default False).

    Examples
    --------
    >>> enc = Encoder(dimension=256, depth=3)
    >>> emb = enc.encode("CCO")            # single SMILES
    >>> emb.shape
    torch.Size([1, 256])
    >>> embs = enc.encode(["CCO", "c1ccccc1"])  # batch
    >>> embs.shape
    torch.Size([2, 256])
    """

    def __init__(
        self,
        dimension: int = 256,
        depth: int = 3,
        atom_types: list[str] | None = None,
        seed: int | None = None,
        normalize: bool = False,
    ) -> None:
        self.dimension = dimension
        self.depth = depth
        self.atom_types = atom_types if atom_types is not None else list(DEFAULT_ATOM_TYPES)
        self.seed = seed
        self.normalize = normalize

        self._atom_to_idx = atom_type_map(self.atom_types)
        self._feature_bins = feature_bins(self.atom_types)

        # Build codebook
        if seed is not None:
            torch.manual_seed(seed)
        num_categories = math.prod(self._feature_bins)
        self._indexer = TupleIndexer(self._feature_bins)
        self._encoder = FeatureEncoder(
            dim=dimension,
            num_categories=num_categories,
            indexer=self._indexer,
            dtype="float64",
        )
        self._codebook = self._encoder.codebook

    def __repr__(self) -> str:
        return (
            f"Encoder(dimension={self.dimension}, depth={self.depth}, "
            f"atom_types={self.atom_types}, seed={self.seed})"
        )

    @property
    def feature_bins(self) -> list[int]:
        """Feature bin sizes: ``[num_atom_types, 6, 3, 4, 2]``."""
        return list(self._feature_bins)

    # ───────────────────── Public API ─────────────────────

    def encode(
        self,
        molecules: Union[str, Chem.Mol, Data, list[str], list[Chem.Mol], list[Data]],
    ) -> Tensor:
        """Encode molecules into order-N hypervector fingerprints.

        The order-N embedding bundles node information across all
        message-passing layers (0 through ``depth``).

        Parameters
        ----------
        molecules
            A single molecule or a list of molecules.  Each molecule
            can be a SMILES string, an RDKit ``Mol``, or a PyG ``Data``
            object (with ``x`` and ``edge_index``).

        Returns
        -------
        Tensor
            ``[batch_size, dimension]``
        """
        batch = self._prepare_batch(molecules)
        return self._encode_batch(batch)["graph_embedding"]

    def encode_joint(
        self,
        molecules: Union[str, Chem.Mol, Data, list[str], list[Chem.Mol], list[Data]],
    ) -> Tensor:
        """Encode molecules into joint order-0 + order-N fingerprints.

        Returns the concatenation of the order-0 embedding (node features
        only, no structural context) and the order-N embedding (full
        message-passing).

        Parameters
        ----------
        molecules
            Same as :meth:`encode`.

        Returns
        -------
        Tensor
            ``[batch_size, 2 * dimension]``
        """
        batch = self._prepare_batch(molecules)
        result = self._encode_batch(batch)
        return torch.cat([result["node_terms"], result["graph_embedding"]], dim=-1)

    # ───────────────────── Internal ─────────────────────

    def _prepare_batch(
        self,
        molecules: Union[str, Chem.Mol, Data, list],
    ) -> Batch:
        """Convert flexible input into a PyG Batch."""
        # Normalize to list
        if isinstance(molecules, (str, Chem.Mol, Data)):
            molecules = [molecules]

        data_list: list[Data] = []
        for mol_input in molecules:
            if isinstance(mol_input, str):
                mol = Chem.MolFromSmiles(mol_input)
                if mol is None:
                    raise ValueError(f"Invalid SMILES: {mol_input!r}")
                data_list.append(mol_to_data(mol, self._atom_to_idx))
            elif isinstance(mol_input, Chem.Mol):
                data_list.append(mol_to_data(mol_input, self._atom_to_idx))
            elif isinstance(mol_input, Data):
                data_list.append(mol_input)
            else:
                raise TypeError(
                    f"Expected str, Chem.Mol, or Data, got {type(mol_input).__name__}"
                )

        return Batch.from_data_list(data_list)

    def _encode_node_features(self, x: Tensor) -> Tensor:
        """Map node feature matrix ``[N, 5]`` to hypervectors ``[N, D]``."""
        return self._encoder.encode(x)

    def _encode_batch(self, batch: Batch) -> dict[str, Tensor]:
        """Run the full HRR message-passing pipeline on a Batch.

        Returns
        -------
        dict
            ``graph_embedding`` : ``[B, D]`` — order-N embedding
            ``node_terms``      : ``[B, D]`` — order-0 embedding
        """
        node_hv = self._encode_node_features(batch.x)

        edge_index = batch.edge_index
        srcs, dsts = edge_index
        node_dim = batch.x.size(0)

        # Stack for message-passing layers: [depth+1, N, D]
        node_hv_stack = node_hv.new_zeros(self.depth + 1, node_dim, self.dimension)
        node_hv_stack[0] = node_hv

        for layer in range(self.depth):
            messages = node_hv_stack[layer][dsts]
            aggregated = scatter_hd(messages, srcs, dim_size=node_dim, op="bundle")
            hr = torchhd.bind(node_hv_stack[layer].clone(), aggregated)

            if self.normalize:
                norm = hr.norm(dim=-1, keepdim=True)
                node_hv_stack[layer + 1] = hr / (norm + 1e-8)
            else:
                node_hv_stack[layer + 1] = hr

        # Graph readout
        # node_hv_stack: [depth+1, N, D] -> transpose to [N, depth+1, D]
        all_layers = node_hv_stack.transpose(0, 1)
        node_hv_bundled = torchhd.multibundle(all_layers)

        # Order-0: bundle of initial node HVs per graph
        node_terms = scatter_hd(src=node_hv, index=batch.batch, op="bundle")
        # Order-N: bundle of all-layer node HVs per graph
        graph_embedding = scatter_hd(src=node_hv_bundled, index=batch.batch, op="bundle")

        return {
            "graph_embedding": graph_embedding,
            "node_terms": node_terms,
        }
