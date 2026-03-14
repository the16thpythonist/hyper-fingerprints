"""Tests for the Encoder class."""

from __future__ import annotations

import pytest
import torch
from rdkit import Chem
from torch_geometric.data import Data

from hyper_fingerprints import Encoder


# ─────────────────────── Helpers ───────────────────────


def _make_ethanol_data() -> Data:
    """CCO as a raw PyG Data object (ZINC-style 5 features)."""
    # C: atom=1, degree=1(idx=1), charge=0, Hs=3, aromatic=0
    # C: atom=1, degree=2(idx=2), charge=0, Hs=2, aromatic=0
    # O: atom=6, degree=1(idx=1), charge=0, Hs=1, aromatic=0
    x = torch.tensor([
        [1, 1, 0, 3, 0],
        [1, 2, 0, 2, 0],
        [6, 1, 0, 1, 0],
    ], dtype=torch.float32)
    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1],
    ], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


# ─────────────────────── Creation ───────────────────────


class TestCreation:

    def test_default_creation(self):
        enc = Encoder()
        assert enc.dimension == 256
        assert enc.depth == 3
        assert len(enc.atom_types) == 9

    def test_custom_dimension_and_depth(self):
        enc = Encoder(dimension=128, depth=5)
        assert enc.dimension == 128
        assert enc.depth == 5

    def test_custom_atom_types(self):
        enc = Encoder(atom_types=["C", "N", "O"])
        assert enc.atom_types == ["C", "N", "O"]
        assert enc.feature_bins[0] == 3

    def test_feature_bins(self):
        enc = Encoder()
        # [9 atom_types, 6 degree, 3 charge, 4 Hs, 2 aromatic]
        assert enc.feature_bins == [9, 6, 3, 4, 2]

    def test_repr(self):
        enc = Encoder(dimension=128, depth=2, seed=42)
        r = repr(enc)
        assert "128" in r
        assert "depth=2" in r


# ─────────────────────── Encoding from SMILES ───────────────────────


class TestEncodeSmiles:

    def test_single_smiles(self):
        enc = Encoder()
        emb = enc.encode("CCO")
        assert emb.shape == (1, 256)

    def test_batch_smiles(self):
        enc = Encoder()
        embs = enc.encode(["CCO", "c1ccccc1", "CC(=O)O"])
        assert embs.shape == (3, 256)

    def test_invalid_smiles_raises(self):
        enc = Encoder()
        with pytest.raises(ValueError, match="Invalid SMILES"):
            enc.encode("not_a_molecule_XYZ!!!")

    def test_unsupported_atom_raises(self):
        enc = Encoder(atom_types=["C", "O"])
        with pytest.raises(ValueError, match="Atom 'N' not in supported"):
            enc.encode("CN")

    def test_single_atom_molecule(self):
        """Methane: single carbon, no bonds."""
        enc = Encoder()
        emb = enc.encode("[CH4]")
        assert emb.shape == (1, 256)


# ─────────────────────── Encoding from Mol ───────────────────────


class TestEncodeMol:

    def test_single_mol(self):
        enc = Encoder()
        mol = Chem.MolFromSmiles("CCO")
        emb = enc.encode(mol)
        assert emb.shape == (1, 256)

    def test_batch_mol(self):
        enc = Encoder()
        mols = [Chem.MolFromSmiles(s) for s in ["CCO", "c1ccccc1"]]
        embs = enc.encode(mols)
        assert embs.shape == (2, 256)

    def test_mol_matches_smiles(self):
        """Encoding a Mol and its SMILES should give the same result."""
        enc = Encoder(seed=42)
        mol = Chem.MolFromSmiles("CCO")
        emb_mol = enc.encode(mol)
        emb_smi = enc.encode("CCO")
        torch.testing.assert_close(emb_mol, emb_smi)


# ─────────────────────── Encoding from Data ───────────────────────


class TestEncodeData:

    def test_single_data(self):
        enc = Encoder()
        data = _make_ethanol_data()
        emb = enc.encode(data)
        assert emb.shape == (1, 256)

    def test_batch_data(self):
        enc = Encoder()
        data_list = [_make_ethanol_data(), _make_ethanol_data()]
        embs = enc.encode(data_list)
        assert embs.shape == (2, 256)

    def test_data_matches_smiles(self):
        """Data created from SMILES should match direct SMILES encoding."""
        enc = Encoder(seed=42)
        emb_smi = enc.encode("CCO")
        emb_data = enc.encode(_make_ethanol_data())
        torch.testing.assert_close(emb_smi, emb_data)


# ─────────────────────── encode_joint ───────────────────────


class TestEncodeJoint:

    def test_joint_shape(self):
        enc = Encoder()
        emb = enc.encode_joint("CCO")
        assert emb.shape == (1, 512)  # 2 * 256

    def test_joint_batch(self):
        enc = Encoder(dimension=128)
        embs = enc.encode_joint(["CCO", "c1ccccc1"])
        assert embs.shape == (2, 256)  # 2 * 128

    def test_joint_contains_both_parts(self):
        """The joint embedding should be [node_terms | graph_embedding]."""
        enc = Encoder(seed=42)
        joint = enc.encode_joint("CCO")
        order0 = joint[:, :enc.dimension]
        orderN = joint[:, enc.dimension:]
        # orderN should equal encode()
        emb = enc.encode("CCO")
        torch.testing.assert_close(orderN, emb)


# ─────────────────────── Determinism ───────────────────────


class TestDeterminism:

    def test_same_seed_same_output(self):
        enc1 = Encoder(seed=42)
        enc2 = Encoder(seed=42)
        emb1 = enc1.encode("CCO")
        emb2 = enc2.encode("CCO")
        torch.testing.assert_close(emb1, emb2)

    def test_different_seed_different_output(self):
        enc1 = Encoder(seed=1)
        enc2 = Encoder(seed=2)
        emb1 = enc1.encode("CCO")
        emb2 = enc2.encode("CCO")
        assert not torch.allclose(emb1, emb2)

    def test_different_molecules_different_output(self):
        enc = Encoder(seed=42)
        emb1 = enc.encode("CCO")
        emb2 = enc.encode("c1ccccc1")
        assert not torch.allclose(emb1, emb2)


# ─────────────────────── Type errors ───────────────────────


class TestTypeErrors:

    def test_bad_type_raises(self):
        enc = Encoder()
        with pytest.raises(TypeError, match="Expected str, Chem.Mol, or Data"):
            enc.encode(42)

    def test_bad_type_in_list_raises(self):
        enc = Encoder()
        with pytest.raises(TypeError):
            enc.encode(["CCO", 42])
