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


# ─────────────────────── Composite-mode extraction ───────────────────────

from hyper_fingerprints.features import (  # noqa: E402
    resolve_features,
    resolve_graph_features,
)


class TestContinuousExtraction:

    def _x(self, smiles, features, **kw):
        mol = Chem.MolFromSmiles(smiles)
        a2i = atom_type_map(DEFAULT_ATOM_TYPES)
        defs = resolve_features(features, DEFAULT_ATOM_TYPES)
        return mol_to_data(mol, a2i, defs, **kw)

    def test_continuous_degree_is_unclipped(self):
        # central atom of neopentane-like C(C)(C)(C)C... use a 6-coordinate S
        smi = "FS(F)(F)(F)(F)F"
        cat = self._x(smi, ["degree"]).x[:, 0]
        cont = self._x(smi, {"degree": {"kind": "continuous", "bandwidth": 1.0}}).x[:, 0]
        assert cat.max() == 5
        assert cont.max() == 6

    def test_implicit_vs_total_hydrogens(self):
        smi = "c1cc[nH]c1"
        d = self._x(smi, ["hydrogens", "implicit_hydrogens"])
        n_idx = [a.GetIdx() for a in Chem.MolFromSmiles(smi).GetAtoms() if a.GetSymbol() == "N"][0]
        assert d.x[n_idx, 0] == 1  # total Hs counts the explicit [nH]
        assert d.x[n_idx, 1] == 0  # implicit Hs does not

    def test_charge_bins(self):
        # 0 = neutral, 1 = positive, 2 = negative
        d = self._x("C[N+](C)(C)C.CC(=O)[O-]", ["element", "charge"])
        mol = Chem.MolFromSmiles("C[N+](C)(C)C.CC(=O)[O-]")
        for atom in mol.GetAtoms():
            expected = {0: 0.0, 1: 1.0, -1: 2.0}[atom.GetFormalCharge()]
            assert d.x[atom.GetIdx(), 1] == expected, atom.GetSymbol()
        assert (d.x[:, 1] == 1.0).sum() == 1 and (d.x[:, 1] == 2.0).sum() == 1
        cont = self._x("C[N+](C)(C)C.CC(=O)[O-]", {"charge": {"kind": "continuous", "bandwidth": 1.0}})
        assert set(cont.x[:, 0]) == {0.0, 1.0, -1.0}

    def test_categorical_clip_override(self):
        d = self._x("FS(F)(F)(F)(F)F", {"degree": {"kind": "categorical", "clip": 7}})
        assert d.x[:, 0].max() == 6

    def test_unknown_atom_slot(self):
        with pytest.raises(ValueError):
            self._x("C[Si](C)C", ["element"])
        d = self._x("C[Si](C)C", ["element"], unknown_atoms="slot")
        assert d.x[1, 0] == len(DEFAULT_ATOM_TYPES)

    def test_unidirectional_edges(self):
        d = self._x("CCO", ["element"], bidirectional=False)
        assert d.edge_index.shape == (2, 2)
        assert (d.edge_index[0] < d.edge_index[1]).all()
        # Ring-closure bond is stored by RDKit as (3, 0): must be normalized
        # to src < dst rather than emitted verbatim.
        ring = self._x("C1CCC1", ["element"], bidirectional=False)
        assert ring.edge_index.shape == (2, 4)
        assert (ring.edge_index[0] < ring.edge_index[1]).all()
        assert [3, 0] not in ring.edge_index.T.tolist()
        assert [0, 3] in ring.edge_index.T.tolist()

    def test_graph_features(self):
        defs = resolve_graph_features({"graph_size": {}, "graph_diameter": {}, "graph_total_charge": {}})
        a2i = atom_type_map(DEFAULT_ATOM_TYPES)
        el = resolve_features(["element"], DEFAULT_ATOM_TYPES)
        cases = {
            "C": (1, 0, 0),
            "CCCC": (4, 3, 0),
            "c1ccccc1": (6, 3, 0),
            "C[N+](C)(C)C": (5, 2, 1),
            # disconnected: diameter is the largest *finite* distance, not
            # RDKit's inter-fragment sentinel
            "CC.O": (3, 1, 0),
            "CCC.CC(=O)[O-]": (7, 2, -1),
        }
        for smi, expected in cases.items():
            g = mol_to_data(Chem.MolFromSmiles(smi), a2i, el, graph_feature_defs=defs).g
            assert tuple(g) == expected, smi

    def test_unknown_graph_feature(self):
        with pytest.raises(ValueError):
            resolve_graph_features(["graph_volume"])
