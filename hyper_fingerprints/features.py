"""
Molecular feature extraction: SMILES / RDKit Mol -> PyG Data.

Fixed 5-feature scheme per atom:
  [atom_type_idx, degree, formal_charge, total_Hs, is_aromatic]

Default atom vocabulary:
  Br, C, Cl, F, I, N, O, P, S  (sorted alphabetically, 9 types)

Feature bins:
  [num_atom_types, 6, 3, 4, 2]
    degree:         0-5  (6 values)
    formal_charge:  0=neutral, 1=positive, 2=negative  (3 values)
    total_Hs:       0-3  (4 values)
    is_aromatic:    0/1  (2 values)
"""

from __future__ import annotations

import torch
from rdkit import Chem
from torch_geometric.data import Data

DEFAULT_ATOM_TYPES: list[str] = ["Br", "C", "Cl", "F", "I", "N", "O", "P", "S"]

# Non-atom-type bin sizes (fixed)
DEGREE_BINS = 6       # 0..5
CHARGE_BINS = 3       # neutral, positive, negative
HYDROGEN_BINS = 4     # 0..3
AROMATIC_BINS = 2     # 0, 1


def atom_type_map(atom_types: list[str]) -> dict[str, int]:
    """Build symbol -> index mapping from a list of atom type symbols."""
    return {sym: idx for idx, sym in enumerate(atom_types)}


def feature_bins(atom_types: list[str]) -> list[int]:
    """Return the feature bin sizes for a given atom type vocabulary."""
    return [len(atom_types), DEGREE_BINS, CHARGE_BINS, HYDROGEN_BINS, AROMATIC_BINS]


def mol_to_data(
    mol: Chem.Mol,
    atom_to_idx: dict[str, int],
) -> Data:
    """Convert an RDKit molecule to a PyG Data object.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule (must have explicit hydrogens info via
        ``GetTotalNumHs``).
    atom_to_idx : dict[str, int]
        Atom symbol -> feature index mapping.

    Returns
    -------
    Data
        PyG Data with ``x`` of shape ``[num_atoms, 5]`` and ``edge_index``.

    Raises
    ------
    ValueError
        If the molecule contains an atom type not in ``atom_to_idx``.
    """
    x = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in atom_to_idx:
            raise ValueError(
                f"Atom '{sym}' not in supported atom types: {list(atom_to_idx)}"
            )
        charge = atom.GetFormalCharge()
        x.append([
            float(atom_to_idx[sym]),
            float(min(atom.GetDegree(), DEGREE_BINS - 1)),
            float(0 if charge == 0 else (1 if charge > 0 else 2)),
            float(min(atom.GetTotalNumHs(), HYDROGEN_BINS - 1)),
            float(atom.GetIsAromatic()),
        ])

    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
    )
