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

from hyper_fingerprints.codebook import CompositeEncoder, FeatureEncoder, is_unit_modulus
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
    from hyper_fingerprints._core import (
        encode_batch_hv_rs,
        encode_batch_rs,
        prepare_batch_rs,
    )
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
            # Explicit hydrogens ([H], [2H]) are removed by RDKit but kept by
            # the Rust parser; let RDKit normalize them as well.
            body = smiles[i + 1 : close].lstrip("0123456789")
            if body[:1] == "H" and not body[1:2].islower():
                return True
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
    return Chem.MolToSmiles(_mol_from_smiles(smiles))


def _mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return mol


def _check_mol(mol: Chem.Mol) -> Chem.Mol:
    if mol.GetNumAtoms() == 0:
        raise ValueError("Molecule has no atoms")
    if mol.NeedsUpdatePropertyCache():
        raise ValueError("Molecule is not sanitized; pass a sanitized RDKit Mol")
    return mol


_FEATURE_CFG_KEYS = {"kind", "bandwidth", "size", "seed_offset", "clip"}


def _validate_feature_cfg(name: str, cfg, *, allow_categorical: bool) -> None:
    if not isinstance(cfg, dict):
        raise ValueError(f"config for feature {name!r} must be a dict, got {type(cfg).__name__}")
    unknown = set(cfg) - _FEATURE_CFG_KEYS
    if unknown:
        raise ValueError(
            f"unknown keys {sorted(unknown)} in config for feature {name!r}; "
            f"allowed: {sorted(_FEATURE_CFG_KEYS)}"
        )
    kind = cfg.get("kind", "categorical")
    if kind == "continuous":
        bw = cfg.get("bandwidth")
        if not isinstance(bw, (int, float)) or isinstance(bw, bool) or not bw > 0:
            raise ValueError(f"feature {name!r}: continuous features need a bandwidth > 0, got {bw!r}")
    elif kind == "categorical":
        if not allow_categorical:
            raise ValueError(f"graph feature {name!r} must be 'continuous' with a bandwidth")
        if "clip" in cfg and (not isinstance(cfg["clip"], int) or cfg["clip"] < 0):
            raise ValueError(f"feature {name!r}: clip must be a non-negative int, got {cfg['clip']!r}")
    else:
        raise ValueError(
            f"feature {name!r}: kind must be 'categorical' or 'continuous', got {kind!r}"
        )
    if "seed_offset" in cfg and not isinstance(cfg["seed_offset"], int):
        raise ValueError(f"feature {name!r}: seed_offset must be an int")


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
        (default False). Each layer multiplies magnitudes, so leaving this
        off reaches ~1e2 by ``depth=3`` (~1e4 to 1e7 with
        ``unit_modulus=False``) and overflows float64 at ``depth`` ~10 (~9)
        -- where the NumPy backend returns inf/NaN and the Rust backend
        raises ``ValueError``. Prefer ``normalize=True`` beyond ``depth=4``.
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
        requires ``cfg["bandwidth"]``). Keep ``bandwidth`` on the order of the
        feature's value range: the value is encoded as
        ``spectrum ** (value / bandwidth)``, so a much smaller bandwidth makes
        neighbouring values nearly orthogonal. With ``unit_modulus=False`` the
        random spectral magnitudes are raised to that power as well, which
        drives the embedding magnitude to numerically useless sizes
        (bandwidth 0.05 on a feature reaching 3 gives ~1e63). See
        :mod:`hyper_fingerprints.presets`.
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
    unit_modulus : bool
        If True (default) every sampled codebook vector -- legacy product
        codebook rows, categorical feature vectors and the base spectra of
        continuous features -- has Fourier coefficients of magnitude one
        (FHRR): the random draw keeps only its phases. Binding then preserves
        norms and repeated binding during message passing does not
        concentrate the fingerprint on a few Fourier components. False
        reproduces the Gaussian codebooks of versions before 0.5.0 bit for
        bit (same seed, same draw). Codebooks that are passed in
        (``codebook``, ``node_codebooks``, ``graph_codebooks``, :meth:`load`)
        are used as given; when every codebook is passed in, the attribute
        ``unit_modulus`` is set from the arrays themselves, so it always
        describes the vectors in use.
    backend : str
        Encoding backend: ``"auto"`` (use Rust if available, else NumPy),
        ``"rust"`` (require Rust extension), or ``"numpy"`` (force pure
        NumPy). Default ``"auto"``. Both modes run their message passing in
        Rust; composite mode additionally builds its node hypervectors in
        NumPy and is always parsed by RDKit (the Rust SMILES parser handles
        the legacy feature set only).

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
        unit_modulus: bool = True,
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
        self.unit_modulus = unit_modulus

        if isinstance(self.features, dict):
            self.mode = "composite"
        elif isinstance(self.features, (list, tuple)):
            self.features = list(self.features)
            self.mode = "legacy"
        else:
            raise ValueError(
                f"features must be a list of names or a dict of configs, "
                f"got {type(self.features).__name__}"
            )
        if not self.graph_features:
            self.graph_features = None

        self._validate_config(codebook)

        self._atom_to_idx = atom_type_map(self.atom_types)
        self._feature_defs = resolve_features(self.features, self.atom_types)
        self._feature_bins = feature_bins(self.atom_types, self.features)
        self._graph_feature_defs = resolve_graph_features(self.graph_features)

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
                unit_modulus=unit_modulus,
            )
            self._codebook = self._encoder.codebook
            self._node_encoder = None
            self._graph_encoder = None
        else:
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
                dimension, node_cfgs, seed=seed, codebooks=node_codebooks,
                unit_modulus=unit_modulus,
            )
            self._graph_encoder = None
            if self._graph_feature_defs:
                self._graph_encoder = CompositeEncoder(
                    dimension, dict(self.graph_features), seed=seed, codebooks=graph_codebooks,
                    unit_modulus=unit_modulus,
                )

        # When nothing was sampled, the flag describes the passed-in arrays.
        given = self._given_codebook_arrays(codebook, node_codebooks, graph_codebooks)
        if given is not None:
            self.unit_modulus = all(is_unit_modulus(arr) for arr in given)

    def _given_codebook_arrays(self, codebook, node_codebooks, graph_codebooks):
        """All codebook arrays if every one of them was passed in, else None."""
        if self.mode == "legacy":
            return None if codebook is None else [self._codebook]
        node_cbs = node_codebooks or {}
        graph_cbs = graph_codebooks or {}
        if any(name not in node_cbs for name in self.features):
            return None
        if any(name not in graph_cbs for name in (self.graph_features or {})):
            return None
        arrays = list(self._node_encoder.arrays().values())
        if self._graph_encoder is not None:
            arrays += list(self._graph_encoder.arrays().values())
        return arrays

    # ───────────────────── Validation ─────────────────────

    def _validate_config(self, codebook) -> None:
        """Reject invalid configurations here, so that ``encode`` never has to."""
        def _is_int(x):
            return isinstance(x, (int, np.integer)) and not isinstance(x, bool)

        def _is_bool(x):
            return isinstance(x, (bool, np.bool_))

        if not _is_int(self.dimension) or self.dimension < 1:
            raise ValueError(f"dimension must be a positive integer, got {self.dimension!r}")
        if not _is_int(self.depth) or self.depth < 0:
            raise ValueError(f"depth must be a non-negative integer, got {self.depth!r}")
        if self.seed is not None and (not _is_int(self.seed) or self.seed < 0):
            raise ValueError(f"seed must be None or a non-negative integer, got {self.seed!r}")
        for name in ("normalize", "bidirectional", "unit_modulus"):
            if not _is_bool(getattr(self, name)):
                raise ValueError(f"{name} must be a bool, got {getattr(self, name)!r}")
        if self.backend not in ("auto", "rust", "numpy"):
            raise ValueError(
                f"backend must be 'auto', 'rust' or 'numpy', got {self.backend!r}"
            )
        if self.backend == "rust" and not _HAS_RUST:
            raise ValueError("backend='rust' requested but the Rust extension is not installed")
        if self.graph_mode not in ("bundle", "bind", "concat"):
            raise ValueError(
                f"graph_mode must be 'bundle', 'bind' or 'concat', got {self.graph_mode!r}"
            )
        if self.unknown_atoms not in ("error", "slot"):
            raise ValueError(
                f"unknown_atoms must be 'error' or 'slot', got {self.unknown_atoms!r}"
            )

        # atom vocabulary
        if not isinstance(self.atom_types, (list, tuple)) or len(self.atom_types) == 0:
            raise ValueError("atom_types must be a non-empty list of element symbols")
        if len(set(self.atom_types)) != len(self.atom_types):
            raise ValueError(f"atom_types contains duplicates: {self.atom_types}")
        table = Chem.GetPeriodicTable()
        for sym in self.atom_types:
            try:
                table.GetAtomicNumber(sym)
            except Exception:
                raise ValueError(f"atom_types contains unknown element symbol {sym!r}") from None

        # features
        if len(self.features) == 0:
            raise ValueError("features must contain at least one feature")
        if self.mode == "legacy":
            if len(set(self.features)) != len(self.features):
                raise ValueError(f"features contains duplicates: {self.features}")
            if codebook is not None:
                expected = (math.prod(feature_bins(self.atom_types, self.features)), self.dimension)
                shape = tuple(np.shape(codebook))
                if shape != expected:
                    raise ValueError(f"codebook has shape {shape}, expected {expected}")
            if self.graph_features or self.unknown_atoms != "error":
                raise ValueError(
                    "graph_features and unknown_atoms='slot' require the composite "
                    "encoder (pass `features` as a dict)"
                )
            return

        if codebook is not None:
            raise ValueError("`codebook` is only valid for the legacy (list) feature mode")
        for name, cfg in self.features.items():
            _validate_feature_cfg(name, cfg, allow_categorical=True)
        if self.features.get("element", {}).get("kind") == "continuous":
            raise ValueError("feature 'element' cannot be continuous (its value is a vocabulary index)")
        for name, cfg in (self.graph_features or {}).items():
            _validate_feature_cfg(name, cfg, allow_categorical=False)

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
            "unit_modulus": self.unit_modulus,
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
    def load(cls, path: Union[str, os.PathLike], *, backend: str = "auto") -> Encoder:
        """Load an encoder from an ``.npz`` file.

        The backend is a runtime preference and is not stored in the file.

        Parameters
        ----------
        path
            Path to the ``.npz`` file previously created by :meth:`save`.
        backend
            Backend for the loaded encoder (``"auto"`` | ``"rust"`` | ``"numpy"``).

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
                backend=backend,
                # files written before 0.5.0 carry Gaussian codebooks (the
                # flag is re-derived from the stored arrays anyway)
                unit_modulus=config.get("unit_modulus", False),
            )
        return cls._from_composite_npz(data, config, backend=backend)

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
            # files written before 0.5.0 carry Gaussian codebooks (re-derived
            # from the stored arrays when all of them are present)
            unit_modulus=config.get("unit_modulus", False),
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
            # The Rust SMILES parser only emits legacy product-codebook indices,
            # so composite input is always prepared with RDKit; the message
            # passing can still run on either backend.
            batch = self._prepare_batch_from_list(molecules)
            return self._encode_batch(batch)

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
                mol = _mol_from_smiles(mol_input)
                data_list.append(mol_to_data(mol, self._atom_to_idx, self._feature_defs, **kwargs))
            elif isinstance(mol_input, Chem.Mol):
                mol = _check_mol(mol_input)
                data_list.append(mol_to_data(mol, self._atom_to_idx, self._feature_defs, **kwargs))
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

    def _encode_node_features(self, x: np.ndarray, *, use_rust: bool = False) -> np.ndarray:
        """Map node feature matrix ``[N, F]`` to hypervectors ``[N, D]``."""
        if self.mode == "composite":
            # Encoding depends only on an atom's own feature row, and those rows
            # repeat heavily across a batch, so each distinct one is computed
            # once. Bit-identical to encoding every row separately.
            return self._node_encoder.encode_unique(x, use_rust=use_rust)
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
        num_graphs = int(batch.batch.max()) + 1 if batch.batch.size > 0 else 1

        if self.mode == "composite":
            node_hv = self._encode_node_features(batch.x, use_rust=True)
            graph_embedding, node_terms = encode_batch_hv_rs(
                np.ascontiguousarray(node_hv, dtype=np.float64),
                batch.edge_index,
                batch.batch,
                num_graphs,
                self.depth,
                "l2" if self.normalize else "none",
                False,  # empty="zeros" scatter semantics
            )
        else:
            feature_indices = self._encoder.encode_indices(batch.x)
            graph_embedding, node_terms = encode_batch_rs(
                self._codebook,
                feature_indices,
                batch.edge_index,
                batch.batch,
                num_graphs,
                self.depth,
                self.normalize,
            )

        graph_embedding = self._apply_graph_features(np.asarray(graph_embedding), batch)

        return {
            "graph_embedding": graph_embedding,
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

        graph_embedding = self._apply_graph_features(graph_embedding, batch)

        return {
            "graph_embedding": graph_embedding,
            "node_terms": node_terms,
        }

    def _apply_graph_features(
        self, graph_embedding: np.ndarray, batch: GraphBatch
    ) -> np.ndarray:
        """Fold graph-level property hypervectors into the readout.

        Backend-independent: both the NumPy and Rust message-passing paths
        call this on their ``[B, D]`` order-N embedding. No-op outside
        composite mode, where ``graph_features`` is unavailable.
        """
        if self._graph_encoder is None or batch.g is None:
            return graph_embedding

        graph_hv = self._encode_graph_features(batch.g)
        if self.graph_mode == "bundle":
            return _l2_normalize(
                _l2_normalize(graph_embedding) + _l2_normalize(graph_hv)
            )
        if self.graph_mode == "bind":
            return hrr_bind(graph_embedding, graph_hv)
        # concat
        return np.concatenate([graph_embedding, graph_hv], axis=-1)

    def _encode_graph_features(self, g: np.ndarray) -> np.ndarray:
        """Sum of per-property FHRR hypervectors ``[B, D]`` (graph_hdc convention)."""
        g = np.asarray(g, dtype=np.float64)
        out = np.zeros((g.shape[0], self.dimension), dtype=np.float64)
        for col, cb in enumerate(self._graph_encoder.codebooks.values()):
            out += cb.encode(g[:, col])
        return out
