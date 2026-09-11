"""
Molecular feature extraction: SMILES / RDKit Mol -> GraphData.

Features are defined via a registry of named extractors. Each extractor
maps an RDKit atom (or, for graph-level features, a molecule) to a value and
declares how many bins it has when used categorically. The set of active
features is configurable per Encoder, either as a list of names (legacy
product-codebook mode) or as a ``{name: cfg}`` dict (composite mode) where
``cfg["kind"]`` is ``"categorical"`` or ``"continuous"``.

Default feature set (backward-compatible):
  ["element", "degree", "charge", "hydrogens", "aromatic"]
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Union

import numpy as np
from rdkit import Chem

from hyper_fingerprints.utils import GraphData

DEFAULT_ATOM_TYPES: list[str] = ["Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]

DEFAULT_FEATURES: list[str] = ["element", "degree", "charge", "hydrogens", "aromatic"]

FeatureSpec = Union[list, dict]


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureDef:
    """Definition of a single atom feature.

    ``bins`` is the number of categories when the feature is used
    categorically. When ``continuous`` is True the extractor returns the raw
    (unclipped) value; otherwise integer count features are clipped to
    ``bins - 1``.
    """

    name: str
    bins: int
    extract: Callable[[Chem.Atom, dict], float]
    continuous: bool = False


@dataclass(frozen=True)
class GraphFeatureDef:
    """Definition of a single graph-level (whole-molecule) feature."""

    name: str
    extract: Callable[[Chem.Mol, dict], float]


def _clip(value: int, fd: FeatureDef) -> float:
    return float(value if fd.continuous else min(value, fd.bins - 1))


def _extract_element(atom: Chem.Atom, ctx: dict) -> float:
    sym = atom.GetSymbol()
    atom_to_idx = ctx["atom_to_idx"]
    if sym not in atom_to_idx:
        if ctx.get("unknown_atoms") == "slot":
            return float(len(atom_to_idx))
        raise ValueError(
            f"Atom '{sym}' not in supported atom types: {list(atom_to_idx)}"
        )
    return float(atom_to_idx[sym])


def _extract_degree(atom: Chem.Atom, ctx: dict) -> float:
    return _clip(atom.GetDegree(), ctx["fd"])


def _extract_charge(atom: Chem.Atom, ctx: dict) -> float:
    charge = atom.GetFormalCharge()
    if ctx["fd"].continuous:
        return float(charge)
    return float(0 if charge == 0 else (1 if charge > 0 else 2))


def _extract_hydrogens(atom: Chem.Atom, ctx: dict) -> float:
    return _clip(atom.GetTotalNumHs(), ctx["fd"])


def _extract_implicit_hydrogens(atom: Chem.Atom, ctx: dict) -> float:
    return _clip(atom.GetNumImplicitHs(), ctx["fd"])


def _extract_aromatic(atom: Chem.Atom, ctx: dict) -> float:
    return float(atom.GetIsAromatic())


def _make_element_def(atom_types: list[str]) -> FeatureDef:
    """Create the element feature def with the correct bin count."""
    return FeatureDef(name="element", bins=len(atom_types), extract=_extract_element)


# Fixed features (bin count doesn't depend on atom_types)
_FIXED_FEATURES: dict[str, FeatureDef] = {
    "degree": FeatureDef(name="degree", bins=6, extract=_extract_degree),
    "charge": FeatureDef(name="charge", bins=3, extract=_extract_charge),
    "hydrogens": FeatureDef(name="hydrogens", bins=4, extract=_extract_hydrogens),
    "implicit_hydrogens": FeatureDef(
        name="implicit_hydrogens", bins=4, extract=_extract_implicit_hydrogens
    ),
    "aromatic": FeatureDef(name="aromatic", bins=2, extract=_extract_aromatic),
}

AVAILABLE_FEATURES: list[str] = [
    "element", "degree", "charge", "hydrogens", "implicit_hydrogens", "aromatic",
]


# ---------------------------------------------------------------------------
# Graph-level features
# ---------------------------------------------------------------------------


def _extract_graph_size(mol: Chem.Mol, ctx: dict) -> float:
    return float(mol.GetNumAtoms())


def _extract_graph_diameter(mol: Chem.Mol, ctx: dict) -> float:
    if mol.GetNumAtoms() <= 1:
        return 0.0
    dist = np.asarray(Chem.GetDistanceMatrix(mol), dtype=np.float64)
    # Disconnected fragments yield very large sentinel distances; ignore them.
    finite = dist[dist < 1e6]
    return float(finite.max()) if finite.size else 0.0


def _extract_graph_total_charge(mol: Chem.Mol, ctx: dict) -> float:
    return float(Chem.GetFormalCharge(mol))


_GRAPH_FEATURES: dict[str, GraphFeatureDef] = {
    "graph_size": GraphFeatureDef(name="graph_size", extract=_extract_graph_size),
    "graph_diameter": GraphFeatureDef(name="graph_diameter", extract=_extract_graph_diameter),
    "graph_total_charge": GraphFeatureDef(
        name="graph_total_charge", extract=_extract_graph_total_charge
    ),
}

AVAILABLE_GRAPH_FEATURES: list[str] = list(_GRAPH_FEATURES)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _lookup(name: str, atom_types: list[str]) -> FeatureDef:
    if name == "element":
        return _make_element_def(atom_types)
    if name in _FIXED_FEATURES:
        return _FIXED_FEATURES[name]
    raise ValueError(f"Unknown feature {name!r}. Available: {AVAILABLE_FEATURES}")


def resolve_features(
    features: FeatureSpec,
    atom_types: list[str],
) -> list[FeatureDef]:
    """Resolve feature names (list) or configs (dict) into FeatureDef objects.

    In dict form, ``cfg["kind"] == "continuous"`` marks the feature as
    continuous and ``cfg.get("clip")`` optionally overrides the categorical
    bin count (``bins = clip + 1``).
    """
    defs: list[FeatureDef] = []
    if isinstance(features, dict):
        for name, cfg in features.items():
            fd = _lookup(name, atom_types)
            kind = cfg.get("kind", "categorical")
            if kind == "continuous":
                fd = replace(fd, continuous=True)
            elif kind == "categorical":
                if "clip" in cfg:
                    fd = replace(fd, bins=int(cfg["clip"]) + 1)
            else:
                raise ValueError(
                    f"Unknown feature kind {kind!r} for {name!r}; "
                    "expected 'categorical' or 'continuous'"
                )
            defs.append(fd)
    else:
        for name in features:
            defs.append(_lookup(name, atom_types))
    return defs


def resolve_graph_features(features: FeatureSpec) -> list[GraphFeatureDef]:
    """Resolve graph-level feature names/configs into GraphFeatureDef objects."""
    names = list(features) if features is not None else []
    defs = []
    for name in names:
        if name not in _GRAPH_FEATURES:
            raise ValueError(
                f"Unknown graph feature {name!r}. Available: {AVAILABLE_GRAPH_FEATURES}"
            )
        defs.append(_GRAPH_FEATURES[name])
    return defs


# ---------------------------------------------------------------------------
# Public helpers (backward-compatible)
# ---------------------------------------------------------------------------


def atom_type_map(atom_types: list[str]) -> dict[str, int]:
    """Build symbol -> index mapping from a list of atom type symbols."""
    return {sym: idx for idx, sym in enumerate(atom_types)}


def feature_bins(
    atom_types: list[str],
    feature_names: FeatureSpec | None = None,
) -> list[int]:
    """Return the feature bin sizes for a given configuration."""
    if feature_names is None:
        feature_names = DEFAULT_FEATURES
    defs = resolve_features(feature_names, atom_types)
    return [d.bins for d in defs]


def mol_to_data(
    mol: Chem.Mol,
    atom_to_idx: dict[str, int],
    feature_defs: list[FeatureDef] | None = None,
    atom_types: list[str] | None = None,
    graph_feature_defs: list[GraphFeatureDef] | None = None,
    *,
    bidirectional: bool = True,
    unknown_atoms: str = "error",
) -> GraphData:
    """Convert an RDKit molecule to a GraphData object.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.
    atom_to_idx : dict[str, int]
        Atom symbol -> feature index mapping.
    feature_defs : list[FeatureDef], optional
        Feature extractors to use. If None, uses the default 5-feature set.
    atom_types : list[str], optional
        Atom type vocabulary. Only needed when feature_defs is None.
    graph_feature_defs : list[GraphFeatureDef], optional
        Graph-level extractors; if given, ``GraphData.g`` is filled.
    bidirectional : bool
        If True (default) every bond yields two directed edges. If False,
        each bond yields a single edge ``min(i, j) -> max(i, j)`` (graph_hdc
        convention).
    unknown_atoms : str
        ``"error"`` (default) raises on symbols outside ``atom_to_idx``;
        ``"slot"`` maps them to the extra index ``len(atom_to_idx)``.

    Returns
    -------
    GraphData
    """
    if feature_defs is None:
        at = atom_types if atom_types is not None else list(atom_to_idx.keys())
        feature_defs = resolve_features(DEFAULT_FEATURES, at)

    ctx = {"atom_to_idx": atom_to_idx, "unknown_atoms": unknown_atoms}

    x = []
    for atom in mol.GetAtoms():
        row = []
        for fd in feature_defs:
            ctx["fd"] = fd
            row.append(fd.extract(atom, ctx))
        x.append(row)

    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if bidirectional:
            src += [i, j]
            dst += [j, i]
        else:
            src.append(min(i, j))
            dst.append(max(i, j))

    if src:
        edge_index = np.array([src, dst], dtype=np.int64)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)

    g = None
    if graph_feature_defs:
        g = np.array([gd.extract(mol, ctx) for gd in graph_feature_defs], dtype=np.float64)

    return GraphData(
        x=np.array(x, dtype=np.float64),
        edge_index=edge_index,
        g=g,
    )
