"""
Molecular feature extraction: SMILES / RDKit Mol -> GraphData.

Features are defined via a registry of named extractors. Each extractor
maps an RDKit atom to a discrete bin index and declares how many bins it
has. The set of active features is configurable per Encoder.

Default feature set (backward-compatible):
  ["element", "degree", "charge", "hydrogens", "aromatic"]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from rdkit import Chem

from hyper_fingerprints.utils import GraphData

DEFAULT_ATOM_TYPES: list[str] = ["Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]

DEFAULT_FEATURES: list[str] = ["element", "degree", "charge", "hydrogens", "aromatic"]


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureDef:
    """Definition of a single discrete atom feature."""

    name: str
    bins: int
    extract: Callable[[Chem.Atom, dict[str, int]], float]


def _extract_element(atom: Chem.Atom, ctx: dict[str, int]) -> float:
    sym = atom.GetSymbol()
    atom_to_idx = ctx["atom_to_idx"]
    if sym not in atom_to_idx:
        raise ValueError(
            f"Atom '{sym}' not in supported atom types: {list(atom_to_idx)}"
        )
    return float(atom_to_idx[sym])


def _extract_degree(atom: Chem.Atom, ctx: dict[str, int]) -> float:
    return float(min(atom.GetDegree(), 5))


def _extract_charge(atom: Chem.Atom, ctx: dict[str, int]) -> float:
    charge = atom.GetFormalCharge()
    return float(0 if charge == 0 else (1 if charge > 0 else 2))


def _extract_hydrogens(atom: Chem.Atom, ctx: dict[str, int]) -> float:
    return float(min(atom.GetTotalNumHs(), 3))


def _extract_aromatic(atom: Chem.Atom, ctx: dict[str, int]) -> float:
    return float(atom.GetIsAromatic())


def _make_element_def(atom_types: list[str]) -> FeatureDef:
    """Create the element feature def with the correct bin count."""
    return FeatureDef(name="element", bins=len(atom_types), extract=_extract_element)


# Fixed features (bin count doesn't depend on atom_types)
_FIXED_FEATURES: dict[str, FeatureDef] = {
    "degree": FeatureDef(name="degree", bins=6, extract=_extract_degree),
    "charge": FeatureDef(name="charge", bins=3, extract=_extract_charge),
    "hydrogens": FeatureDef(name="hydrogens", bins=4, extract=_extract_hydrogens),
    "aromatic": FeatureDef(name="aromatic", bins=2, extract=_extract_aromatic),
}

AVAILABLE_FEATURES: list[str] = ["element", "degree", "charge", "hydrogens", "aromatic"]


def resolve_features(
    feature_names: list[str],
    atom_types: list[str],
) -> list[FeatureDef]:
    """Resolve a list of feature names into FeatureDef objects."""
    element_def = _make_element_def(atom_types)
    defs = []
    for name in feature_names:
        if name == "element":
            defs.append(element_def)
        elif name in _FIXED_FEATURES:
            defs.append(_FIXED_FEATURES[name])
        else:
            raise ValueError(
                f"Unknown feature {name!r}. "
                f"Available: {AVAILABLE_FEATURES}"
            )
    return defs


# ---------------------------------------------------------------------------
# Public helpers (backward-compatible)
# ---------------------------------------------------------------------------


def atom_type_map(atom_types: list[str]) -> dict[str, int]:
    """Build symbol -> index mapping from a list of atom type symbols."""
    return {sym: idx for idx, sym in enumerate(atom_types)}


def feature_bins(
    atom_types: list[str],
    feature_names: list[str] | None = None,
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

    Returns
    -------
    GraphData
    """
    if feature_defs is None:
        at = atom_types if atom_types is not None else list(atom_to_idx.keys())
        feature_defs = resolve_features(DEFAULT_FEATURES, at)

    ctx = {"atom_to_idx": atom_to_idx}

    x = []
    for atom in mol.GetAtoms():
        row = [fd.extract(atom, ctx) for fd in feature_defs]
        x.append(row)

    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]

    if src:
        edge_index = np.array([src, dst], dtype=np.int64)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)

    return GraphData(
        x=np.array(x, dtype=np.float64),
        edge_index=edge_index,
    )
