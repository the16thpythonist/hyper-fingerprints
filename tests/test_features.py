"""Tests for molecular feature extraction."""

from __future__ import annotations

import pytest
from rdkit import Chem

from hyper_fingerprints.features import (
    DEFAULT_ATOM_TYPES,
    atom_type_map,
    feature_bins,
    mol_to_data,
)


class TestAtomTypeMap:

    def test_default_has_9_types(self):
        m = atom_type_map(DEFAULT_ATOM_TYPES)
        assert len(m) == 9

    def test_indices_contiguous(self):
        m = atom_type_map(DEFAULT_ATOM_TYPES)
        assert set(m.values()) == set(range(9))

    def test_custom_types(self):
        m = atom_type_map(["C", "N", "O"])
        assert m == {"C": 0, "N": 1, "O": 2}


class TestFeatureBins:

    def test_default_bins(self):
        bins = feature_bins(DEFAULT_ATOM_TYPES)
        assert bins == [9, 6, 3, 4, 2]

    def test_custom_bins(self):
        bins = feature_bins(["C", "N", "O"])
        assert bins == [3, 6, 3, 4, 2]


class TestMolToData:

    def test_ethanol(self):
        mol = Chem.MolFromSmiles("CCO")
        a2i = atom_type_map(DEFAULT_ATOM_TYPES)
        data = mol_to_data(mol, a2i)
        assert data.x.shape == (3, 5)
        assert data.edge_index.shape[0] == 2
        # 2 bonds * 2 directions = 4 edges
        assert data.edge_index.shape[1] == 4

    def test_benzene(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        a2i = atom_type_map(DEFAULT_ATOM_TYPES)
        data = mol_to_data(mol, a2i)
        assert data.x.shape == (6, 5)
        # All atoms should be aromatic
        assert data.x[:, 4].sum() == 6

    def test_unsupported_atom_raises(self):
        mol = Chem.MolFromSmiles("[Fe]")
        a2i = atom_type_map(DEFAULT_ATOM_TYPES)
        with pytest.raises(ValueError, match="Atom 'Fe' not in supported"):
            mol_to_data(mol, a2i)

    def test_feature_ranges(self):
        """All features should be within their expected bin ranges."""
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "c1cc(O)ccc1N"]
        a2i = atom_type_map(DEFAULT_ATOM_TYPES)
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            data = mol_to_data(mol, a2i)
            x = data.x
            assert (x[:, 0] >= 0).all() and (x[:, 0] < 9).all(), f"atom_type out of range for {smi}"
            assert (x[:, 1] >= 0).all() and (x[:, 1] < 6).all(), f"degree out of range for {smi}"
            assert (x[:, 2] >= 0).all() and (x[:, 2] < 3).all(), f"charge out of range for {smi}"
            assert (x[:, 3] >= 0).all() and (x[:, 3] < 4).all(), f"Hs out of range for {smi}"
            assert (x[:, 4] >= 0).all() and (x[:, 4] < 2).all(), f"aromatic out of range for {smi}"

    def test_isolated_atom(self):
        """Single atom with no bonds."""
        mol = Chem.MolFromSmiles("[CH4]")
        a2i = atom_type_map(DEFAULT_ATOM_TYPES)
        data = mol_to_data(mol, a2i)
        assert data.x.shape == (1, 5)
        assert data.edge_index.shape[1] == 0
