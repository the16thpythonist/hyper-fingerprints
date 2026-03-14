"""
Encoder: hyperdimensional molecular fingerprints via HRR message passing.
"""

from __future__ import annotations

import json
import math
import os
from typing import Union

import numpy as np
from rdkit import Chem

from hyper_fingerprints.codebook import FeatureEncoder
from hyper_fingerprints.features import (
    DEFAULT_ATOM_TYPES,
    atom_type_map,
    feature_bins,
    mol_to_data,
)
from hyper_fingerprints.utils import (
    GraphBatch,
    GraphData,
    TupleIndexer,
    batch_from_data_list,
    hrr_bind,
    hrr_multibundle,
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
    codebook : np.ndarray, optional
        Pre-generated codebook array. If provided, ``seed`` is ignored
        for codebook generation.

    Examples
    --------
    >>> enc = Encoder(dimension=256, depth=3)
    >>> emb = enc.encode("CCO")            # single SMILES
    >>> emb.shape
    (1, 256)
    >>> embs = enc.encode(["CCO", "c1ccccc1"])  # batch
    >>> embs.shape
    (2, 256)
    """

    def __init__(
        self,
        dimension: int = 256,
        depth: int = 3,
        atom_types: list[str] | None = None,
        seed: int | None = None,
        normalize: bool = False,
        codebook: np.ndarray | None = None,
    ) -> None:
        self.dimension = dimension
        self.depth = depth
        self.atom_types = atom_types if atom_types is not None else list(DEFAULT_ATOM_TYPES)
        self.seed = seed
        self.normalize = normalize

        self._atom_to_idx = atom_type_map(self.atom_types)
        self._feature_bins = feature_bins(self.atom_types)

        # Build codebook
        num_categories = math.prod(self._feature_bins)
        self._indexer = TupleIndexer(self._feature_bins)
        self._encoder = FeatureEncoder(
            dim=dimension,
            num_categories=num_categories,
            indexer=self._indexer,
            seed=seed,
            codebook=codebook,
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

    # ───────────────────── Save / Load ─────────────────────

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Save the encoder to an ``.npz`` file.

        Parameters
        ----------
        path
            Destination file path. A ``.npz`` extension is appended
            automatically by NumPy if not already present.
        """
        config = json.dumps({
            "dimension": self.dimension,
            "depth": self.depth,
            "atom_types": self.atom_types,
            "normalize": self.normalize,
            "seed": self.seed,
        })
        np.savez(path, config=np.array(config), codebook=self._codebook)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> Encoder:
        """Load an encoder from an ``.npz`` file.

        Parameters
        ----------
        path
            Path to the ``.npz`` file previously created by :meth:`save`.

        Returns
        -------
        Encoder
            A fully reconstructed encoder with the saved codebook.
        """
        data = np.load(path, allow_pickle=False)
        config = json.loads(str(data["config"]))
        return cls(
            dimension=config["dimension"],
            depth=config["depth"],
            atom_types=config["atom_types"],
            normalize=config["normalize"],
            seed=config["seed"],
            codebook=data["codebook"],
        )

    # ───────────────────── Public API ─────────────────────

    def encode(
        self,
        molecules: Union[str, Chem.Mol, list[str], list[Chem.Mol]],
    ) -> np.ndarray:
        """Encode molecules into order-N hypervector fingerprints.

        The order-N embedding bundles node information across all
        message-passing layers (0 through ``depth``).

        Parameters
        ----------
        molecules
            A single molecule or a list of molecules.  Each molecule
            can be a SMILES string or an RDKit ``Mol``.

        Returns
        -------
        np.ndarray
            ``[batch_size, dimension]``
        """
        batch = self._prepare_batch(molecules)
        return self._encode_batch(batch)["graph_embedding"]

    def encode_joint(
        self,
        molecules: Union[str, Chem.Mol, list[str], list[Chem.Mol]],
    ) -> np.ndarray:
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
        np.ndarray
            ``[batch_size, 2 * dimension]``
        """
        batch = self._prepare_batch(molecules)
        result = self._encode_batch(batch)
        return np.concatenate([result["node_terms"], result["graph_embedding"]], axis=-1)

    # ───────────────────── Internal ─────────────────────

    def _prepare_batch(
        self,
        molecules: Union[str, Chem.Mol, list],
    ) -> GraphBatch:
        """Convert flexible input into a GraphBatch."""
        # Normalize to list
        if isinstance(molecules, (str, Chem.Mol)):
            molecules = [molecules]
        elif not isinstance(molecules, list):
            raise TypeError(
                f"Expected str or Chem.Mol, got {type(molecules).__name__}"
            )

        data_list: list[GraphData] = []
        for mol_input in molecules:
            if isinstance(mol_input, str):
                mol = Chem.MolFromSmiles(mol_input)
                if mol is None:
                    raise ValueError(f"Invalid SMILES: {mol_input!r}")
                data_list.append(mol_to_data(mol, self._atom_to_idx))
            elif isinstance(mol_input, Chem.Mol):
                data_list.append(mol_to_data(mol_input, self._atom_to_idx))
            else:
                raise TypeError(
                    f"Expected str or Chem.Mol, got {type(mol_input).__name__}"
                )

        return batch_from_data_list(data_list)

    def _encode_node_features(self, x: np.ndarray) -> np.ndarray:
        """Map node feature matrix ``[N, 5]`` to hypervectors ``[N, D]``."""
        return self._encoder.encode(x)

    def _encode_batch(self, batch: GraphBatch) -> dict[str, np.ndarray]:
        """Run the full HRR message-passing pipeline on a GraphBatch.

        Returns
        -------
        dict
            ``graph_embedding`` : ``[B, D]`` -- order-N embedding
            ``node_terms``      : ``[B, D]`` -- order-0 embedding
        """
        node_hv = self._encode_node_features(batch.x)

        edge_index = batch.edge_index
        srcs, dsts = edge_index[0], edge_index[1]
        node_dim = batch.x.shape[0]

        # Stack for message-passing layers: [depth+1, N, D]
        node_hv_stack = np.zeros((self.depth + 1, node_dim, self.dimension), dtype=np.float64)
        node_hv_stack[0] = node_hv

        for layer in range(self.depth):
            messages = node_hv_stack[layer][dsts]
            aggregated = scatter_hd(messages, srcs, dim_size=node_dim, op="bundle")
            hr = hrr_bind(node_hv_stack[layer], aggregated)

            if self.normalize:
                norm = np.linalg.norm(hr, axis=-1, keepdims=True)
                node_hv_stack[layer + 1] = hr / (norm + 1e-8)
            else:
                node_hv_stack[layer + 1] = hr

        # Graph readout
        # node_hv_stack: [depth+1, N, D] -> transpose to [N, depth+1, D]
        all_layers = node_hv_stack.transpose(1, 0, 2)
        node_hv_bundled = hrr_multibundle(all_layers)

        # Order-0: bundle of initial node HVs per graph
        node_terms = scatter_hd(src=node_hv, index=batch.batch, op="bundle")
        # Order-N: bundle of all-layer node HVs per graph
        graph_embedding = scatter_hd(src=node_hv_bundled, index=batch.batch, op="bundle")

        return {
            "graph_embedding": graph_embedding,
            "node_terms": node_terms,
        }
