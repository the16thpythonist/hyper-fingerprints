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

from hyper_fingerprints.codebook import CompositeEncoder, FeatureEncoder
from hyper_fingerprints.features import (
    DEFAULT_ATOM_TYPES,
    DEFAULT_FEATURES,
    atom_type_map,
    feature_bins,
    mol_to_data,
    resolve_features,
    resolve_graph_features,
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

try:
    from hyper_fingerprints._core import encode_batch_rs, prepare_batch_rs
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


# Atom features the Rust SMILES parser (``prepare_batch_rs``) can extract itself.
RUST_FEATURES = frozenset({"element", "degree", "charge", "hydrogens", "aromatic"})


def _needs_rdkit_canonicalization(smiles: str) -> bool:
    """Whether a SMILES string may contain aromatic rings written in Kekulé form.

    The Rust parser does not perceive aromaticity, so ``C1=CC=CC=C1`` would be
    featurized differently from ``c1ccccc1``. A Kekulé aromatic ring always
    has at least two double bonds between the ring-opening and ring-closing
    digits, so any ring span with >= 2 ``=`` (or an explicit aromatic ``:``
    bond) is routed through RDKit first. False positives (e.g. quinones) only
    cost speed; they yield identical fingerprints either way.
    """
    open_rings: dict[str, int] = {}
    i, n = 0, len(smiles)
    while i < n:
        c = smiles[i]
        if c == "[":
            close = smiles.find("]", i)
            i = n if close < 0 else close + 1
            continue
        if c == ":":
            return True
        if c.isdigit() or c == "%":
            if c == "%":
                label, width = smiles[i : i + 3], 3
            else:
                label, width = c, 1
            start = open_rings.pop(label, None)
            if start is None:
                open_rings[label] = i
            elif smiles.count("=", start, i) >= 2:
                return True
            i += width
            continue
        i += 1
    return False


def _canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization with ``torch.nn.functional.normalize`` semantics."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


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
    features : list[str] | dict[str, dict], optional
        Atom features to extract. A **list** of names selects the legacy
        product-codebook encoder (one random vector per feature tuple).
        Available: ``"element"``, ``"degree"``, ``"charge"``,
        ``"hydrogens"``, ``"implicit_hydrogens"``, ``"aromatic"``. Default
        (None) uses the first five. A **dict** ``{name: cfg}`` selects the
        composite encoder in which every feature has its own codebook and the
        per-feature hypervectors are bound together; ``cfg["kind"]`` is
        ``"categorical"`` or ``"continuous"`` (FHRR fractional power encoding,
        requires ``cfg["bandwidth"]``). See :mod:`hyper_fingerprints.presets`.
    graph_features : dict[str, dict], optional
        Graph-level features (``"graph_size"``, ``"graph_diameter"``,
        ``"graph_total_charge"``) with the same config format. Composite
        mode only.
    graph_mode : str
        How graph-level feature hypervectors enter the fingerprint:
        ``"bundle"`` (normalized sum, graph_hdc convention), ``"bind"``
        (circular convolution) or ``"concat"`` (appended block, output has
        ``2 * dimension`` columns). Default ``"bundle"``.
    bidirectional : bool
        If True (default) messages flow both ways along every bond. If False
        each bond is a single directed edge (graph_hdc default).
    unknown_atoms : str
        ``"error"`` (default) raises for atoms outside ``atom_types``;
        ``"slot"`` maps them to a shared extra codebook row. Composite mode
        only.
    node_codebooks, graph_codebooks : dict[str, np.ndarray], optional
        Pre-generated per-feature base arrays for composite mode (used by
        :meth:`load` and :meth:`from_legacy_codebook`).
    backend : str
        Encoding backend: ``"auto"`` (use Rust if available, else NumPy),
        ``"rust"`` (require Rust extension), or ``"numpy"`` (force pure
        NumPy). Default ``"auto"``.

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
        features: Union[list[str], dict[str, dict], None] = None,
        backend: str = "auto",
        graph_features: dict[str, dict] | None = None,
        graph_mode: str = "bundle",
        bidirectional: bool = True,
        unknown_atoms: str = "error",
        node_codebooks: dict[str, np.ndarray] | None = None,
        graph_codebooks: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.dimension = dimension
        self.depth = depth
        self.atom_types = atom_types if atom_types is not None else list(DEFAULT_ATOM_TYPES)
        self.seed = seed
        self.normalize = normalize
        self.features = features if features is not None else list(DEFAULT_FEATURES)
        self.backend = backend
        self.graph_features = graph_features
        self.graph_mode = graph_mode
        self.bidirectional = bidirectional
        self.unknown_atoms = unknown_atoms

        self.mode = "composite" if isinstance(self.features, dict) else "legacy"

        if graph_mode not in ("bundle", "bind", "concat"):
            raise ValueError(
                f"graph_mode must be 'bundle', 'bind' or 'concat', got {graph_mode!r}"
            )
        if unknown_atoms not in ("error", "slot"):
            raise ValueError(
                f"unknown_atoms must be 'error' or 'slot', got {unknown_atoms!r}"
            )
        if self.mode == "legacy" and (graph_features or unknown_atoms != "error"):
            raise ValueError(
                "graph_features and unknown_atoms='slot' require the composite "
                "encoder (pass `features` as a dict)"
            )

        self._atom_to_idx = atom_type_map(self.atom_types)
        self._feature_defs = resolve_features(self.features, self.atom_types)
        self._feature_bins = feature_bins(self.atom_types, self.features)
        self._graph_feature_defs = resolve_graph_features(graph_features)

        if self.mode == "legacy":
            # Build product codebook
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
            self._node_encoder = None
            self._graph_encoder = None
        else:
            if codebook is not None:
                raise ValueError("`codebook` is only valid for the legacy (list) feature mode")
            self._codebook = None
            node_cfgs = {}
            for fd, (name, cfg) in zip(self._feature_defs, self.features.items()):
                cfg = dict(cfg)
                if cfg.get("kind", "categorical") == "categorical":
                    cfg["kind"] = "categorical"
                    cfg["num_categories"] = fd.bins + (
                        1 if name == "element" and unknown_atoms == "slot" else 0
                    )
                node_cfgs[name] = cfg
            self._node_encoder = CompositeEncoder(
                dimension, node_cfgs, seed=seed, codebooks=node_codebooks
            )
            self._graph_encoder = None
            if self._graph_feature_defs:
                for name, cfg in graph_features.items():
                    if cfg.get("kind") != "continuous":
                        raise ValueError(
                            f"graph feature {name!r} must be 'continuous' with a bandwidth"
                        )
                self._graph_encoder = CompositeEncoder(
                    dimension, dict(graph_features), seed=seed, codebooks=graph_codebooks
                )

    def __repr__(self) -> str:
        return (
            f"Encoder(dimension={self.dimension}, depth={self.depth}, "
            f"atom_types={self.atom_types}, seed={self.seed}, mode={self.mode!r})"
        )

    @property
    def feature_bins(self) -> list[int]:
        """Feature bin sizes derived from the active feature set."""
        return list(self._feature_bins)

    # ───────────────────── Save / Load ─────────────────────

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Save the encoder to an ``.npz`` file.

        Legacy encoders write ``config`` + ``codebook`` (unchanged format).
        Composite encoders write ``config`` + one ``node/<name>`` /
        ``graph/<name>`` array per feature.

        Parameters
        ----------
        path
            Destination file path. A ``.npz`` extension is appended
            automatically by NumPy if not already present.
        """
        config = {
            "dimension": self.dimension,
            "depth": self.depth,
            "atom_types": self.atom_types,
            "normalize": self.normalize,
            "seed": self.seed,
            "features": self.features,
        }
        if self.mode == "legacy":
            np.savez(path, config=np.array(json.dumps(config)), codebook=self._codebook)
            return

        config.update({
            "format_version": 2,
            "mode": "composite",
            "graph_features": self.graph_features,
            "graph_mode": self.graph_mode,
            "bidirectional": self.bidirectional,
            "unknown_atoms": self.unknown_atoms,
        })
        arrays = {f"node/{k}": v for k, v in self._node_encoder.arrays().items()}
        if self._graph_encoder is not None:
            arrays.update({f"graph/{k}": v for k, v in self._graph_encoder.arrays().items()})
        np.savez(path, config=np.array(json.dumps(config)), **arrays)

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
            A fully reconstructed encoder with the saved codebook(s).
        """
        data = np.load(path, allow_pickle=False)
        config = json.loads(str(data["config"]))
        if config.get("mode") != "composite":
            return cls(
                dimension=config["dimension"],
                depth=config["depth"],
                atom_types=config["atom_types"],
                normalize=config["normalize"],
                seed=config["seed"],
                codebook=data["codebook"],
                features=config.get("features"),
            )
        return cls._from_composite_npz(data, config)

    @classmethod
    def _from_composite_npz(cls, data, config: dict, **overrides) -> Encoder:
        node_cbs = {k[len("node/"):]: data[k] for k in data.files if k.startswith("node/")}
        graph_cbs = {k[len("graph/"):]: data[k] for k in data.files if k.startswith("graph/")}

        # Every declared feature must have its array; never silently regenerate.
        graph_features = overrides.get("graph_features", config.get("graph_features"))
        missing = [f"node/{n}" for n in config["features"] if n not in node_cbs]
        missing += [f"graph/{n}" for n in (graph_features or {}) if n not in graph_cbs]
        if missing:
            raise ValueError(f"codebook file is missing arrays for: {missing}")

        kwargs = dict(
            dimension=config["dimension"],
            depth=config["depth"],
            atom_types=config["atom_types"],
            normalize=config["normalize"],
            seed=config.get("seed"),
            features=config["features"],
            graph_features=config.get("graph_features") or None,
            graph_mode=config.get("graph_mode", "bundle"),
            bidirectional=config.get("bidirectional", True),
            unknown_atoms=config.get("unknown_atoms", "error"),
            node_codebooks=node_cbs,
            graph_codebooks=graph_cbs or None,
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    @classmethod
    def from_legacy_codebook(
        cls,
        path: Union[str, os.PathLike],
        *,
        depth: int | None = None,
        bidirectional: bool | None = None,
        graph_features: Union[dict, bool, None] = None,
    ) -> Encoder:
        """Build an encoder from a codebook exported by ``graph_hdc``.

        The file is produced by ``HyperNet.export_legacy_codebook`` and
        carries the exact base vectors / spectra of the graph_hdc encoders
        together with the message-passing configuration, so that both
        libraries compute identical fingerprints.

        Parameters
        ----------
        path
            Path to the exported ``.npz``.
        depth, bidirectional
            Optional overrides of the exported values.
        graph_features
            ``None`` keeps the exported graph features; ``False`` drops
            them; a dict replaces the config (arrays are matched by name).
        """
        data = np.load(path, allow_pickle=False)
        config = json.loads(str(data["config"]))
        if config.get("pooling", "sum") != "sum":
            raise ValueError(f"Unsupported pooling {config['pooling']!r}; only 'sum'")
        if config.get("normalize_all", False):
            raise ValueError("normalize_all=True is not supported")
        overrides = {}
        if depth is not None:
            overrides["depth"] = depth
        if bidirectional is not None:
            overrides["bidirectional"] = bidirectional
        if graph_features is False:
            overrides["graph_features"] = None
            overrides["graph_codebooks"] = None
        elif isinstance(graph_features, dict):
            overrides["graph_features"] = graph_features
        return cls._from_composite_npz(data, config, **overrides)

    # ───────────────────── Public API ─────────────────────

    def encode(
        self,
        molecules: Union[str, Chem.Mol, list[str], list[Chem.Mol]],
    ) -> np.ndarray:
        """Encode molecules into order-N hypervector fingerprints.

        The order-N embedding bundles node information across all
        message-passing layers (0 through ``depth``).

        Note: in legacy (list-features) mode, single-atom molecules encode
        differently alone than inside a batch containing bonded molecules
        (an all-edgeless batch aggregates to the HRR identity, a mixed batch
        to zero). This historical behaviour is kept for output stability;
        composite mode is batch-invariant.

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
        result = self._encode_full(molecules)
        return result["graph_embedding"]

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
        result = self._encode_full(molecules)
        return np.concatenate([result["node_terms"], result["graph_embedding"]], axis=-1)

    def _encode_full(
        self,
        molecules: Union[str, Chem.Mol, list[str], list[Chem.Mol]],
    ) -> dict[str, np.ndarray]:
        """Route to the fastest available pipeline."""
        # Normalize to list
        if isinstance(molecules, (str, Chem.Mol)):
            molecules = [molecules]
        elif not isinstance(molecules, list):
            raise TypeError(
                f"Expected str or Chem.Mol, got {type(molecules).__name__}"
            )

        if len(molecules) == 0:
            width = self.dimension
            if self.mode == "composite" and self._graph_encoder is not None and self.graph_mode == "concat":
                width = 2 * self.dimension
            return {
                "graph_embedding": np.zeros((0, width), dtype=np.float64),
                "node_terms": np.zeros((0, self.dimension), dtype=np.float64),
            }

        if self.mode == "composite":
            if self.backend == "rust":
                raise RuntimeError(
                    "backend='rust' is not yet supported for the composite "
                    "(dict features) encoder; use backend='numpy' or 'auto'"
                )
            batch = self._prepare_batch_from_list(molecules)
            return self._encode_batch_numpy(batch)

        use_rust = (self.backend == "rust") or (self.backend == "auto" and _HAS_RUST)
        # The Rust SMILES parser only knows the built-in feature set; other
        # legacy feature lists take the RDKit preparation path (and can still
        # use Rust for the message passing).
        rust_parse_ok = all(name in RUST_FEATURES for name in self.features)
        all_smiles = use_rust and rust_parse_ok and all(isinstance(m, str) for m in molecules)

        if all_smiles:
            if not _HAS_RUST:
                raise RuntimeError(
                    "backend='rust' requested but Rust extension is not installed"
                )
            # Kekulé-form aromatic rings must be aromatized by RDKit first.
            molecules = [
                _canonical_smiles(smi) if _needs_rdkit_canonicalization(smi) else smi
                for smi in molecules
            ]
            return self._encode_smiles_rust(molecules)

        # Fall back to Python preparation + optional Rust encoding
        batch = self._prepare_batch_from_list(molecules)
        return self._encode_batch(batch)

    # ───────────────────── Internal ─────────────────────

    def _prepare_batch_from_list(
        self,
        molecules: list,
    ) -> GraphBatch:
        """Convert a list of molecules into a GraphBatch (Python/RDKit path)."""
        data_list: list[GraphData] = []
        kwargs = dict(
            graph_feature_defs=self._graph_feature_defs,
            bidirectional=self.bidirectional,
            unknown_atoms=self.unknown_atoms,
        )
        for mol_input in molecules:
            if isinstance(mol_input, str):
                mol = Chem.MolFromSmiles(mol_input)
                if mol is None:
                    raise ValueError(f"Invalid SMILES: {mol_input!r}")
                data_list.append(mol_to_data(mol, self._atom_to_idx, self._feature_defs, **kwargs))
            elif isinstance(mol_input, Chem.Mol):
                data_list.append(mol_to_data(mol_input, self._atom_to_idx, self._feature_defs, **kwargs))
            else:
                raise TypeError(
                    f"Expected str or Chem.Mol, got {type(mol_input).__name__}"
                )

        return batch_from_data_list(data_list)

    def _encode_smiles_rust(
        self,
        smiles_list: list[str],
    ) -> dict[str, np.ndarray]:
        """Full Rust pipeline: SMILES parsing + feature extraction + encoding."""
        feature_indices, edge_index, batch_indices, num_graphs = prepare_batch_rs(
            smiles_list,
            self._atom_to_idx,
            self.features,
            self._feature_bins,
        )

        graph_embedding, node_terms = encode_batch_rs(
            self._codebook,
            np.asarray(feature_indices),
            np.asarray(edge_index),
            np.asarray(batch_indices),
            num_graphs,
            self.depth,
            self.normalize,
        )

        return {
            "graph_embedding": np.asarray(graph_embedding),
            "node_terms": np.asarray(node_terms),
        }

    def _encode_node_features(self, x: np.ndarray) -> np.ndarray:
        """Map node feature matrix ``[N, F]`` to hypervectors ``[N, D]``."""
        if self.mode == "composite":
            return self._node_encoder.encode(x)
        return self._encoder.encode(x)

    def _encode_batch(self, batch: GraphBatch) -> dict[str, np.ndarray]:
        """Run the full HRR message-passing pipeline on a GraphBatch.

        Returns
        -------
        dict
            ``graph_embedding`` : ``[B, D]`` -- order-N embedding
            ``node_terms``      : ``[B, D]`` -- order-0 embedding
        """
        use_rust = (self.backend == "rust") or (self.backend == "auto" and _HAS_RUST)
        if use_rust:
            if not _HAS_RUST:
                raise RuntimeError(
                    "backend='rust' requested but Rust extension is not installed"
                )
            return self._encode_batch_rust(batch)
        return self._encode_batch_numpy(batch)

    def _encode_batch_rust(self, batch: GraphBatch) -> dict[str, np.ndarray]:
        """Rust-accelerated message-passing pipeline."""
        feature_indices = self._encoder.encode_indices(batch.x)
        num_graphs = int(batch.batch.max()) + 1 if batch.batch.size > 0 else 1

        graph_embedding, node_terms = encode_batch_rs(
            self._codebook,
            feature_indices,
            batch.edge_index,
            batch.batch,
            num_graphs,
            self.depth,
            self.normalize,
        )

        return {
            "graph_embedding": np.asarray(graph_embedding),
            "node_terms": np.asarray(node_terms),
        }

    def _encode_batch_numpy(self, batch: GraphBatch) -> dict[str, np.ndarray]:
        """Pure-NumPy message-passing pipeline (fallback)."""
        node_hv = self._encode_node_features(batch.x)

        edge_index = batch.edge_index
        srcs, dsts = edge_index[0], edge_index[1]
        node_dim = batch.x.shape[0]

        # Stack for message-passing layers: [depth+1, N, D]
        node_hv_stack = np.zeros((self.depth + 1, node_dim, self.dimension), dtype=np.float64)
        node_hv_stack[0] = node_hv

        for layer in range(self.depth):
            messages = node_hv_stack[layer][dsts]
            # Composite mode: a node without incoming messages aggregates to
            # zero regardless of the rest of the batch (graph_hdc semantics).
            # Legacy mode keeps the identity fallback for an all-edgeless batch.
            aggregated = scatter_hd(
                messages, srcs, dim_size=node_dim, op="bundle",
                empty="zeros" if self.mode == "composite" else "identity",
            )
            hr = hrr_bind(node_hv_stack[layer], aggregated)

            if self.normalize and self.mode == "composite":
                # torch.nn.functional.normalize semantics (graph_hdc parity)
                node_hv_stack[layer + 1] = _l2_normalize(hr)
            elif self.normalize:
                # legacy formula, shared with the Rust backend
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

        # Graph-level properties (composite mode only)
        if self._graph_encoder is not None and batch.g is not None:
            graph_hv = self._encode_graph_features(batch.g)
            if self.graph_mode == "bundle":
                graph_embedding = _l2_normalize(
                    _l2_normalize(graph_embedding) + _l2_normalize(graph_hv)
                )
            elif self.graph_mode == "bind":
                graph_embedding = hrr_bind(graph_embedding, graph_hv)
            else:  # concat
                graph_embedding = np.concatenate([graph_embedding, graph_hv], axis=-1)

        return {
            "graph_embedding": graph_embedding,
            "node_terms": node_terms,
        }

    def _encode_graph_features(self, g: np.ndarray) -> np.ndarray:
        """Sum of per-property FHRR hypervectors ``[B, D]`` (graph_hdc convention)."""
        g = np.asarray(g, dtype=np.float64)
        out = np.zeros((g.shape[0], self.dimension), dtype=np.float64)
        for col, cb in enumerate(self._graph_encoder.codebooks.values()):
            out += cb.encode(g[:, col])
        return out
