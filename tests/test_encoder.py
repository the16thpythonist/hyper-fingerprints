"""Tests for the Encoder class."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from hyper_fingerprints import Encoder


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
        np.testing.assert_allclose(emb_mol, emb_smi, atol=1e-12, rtol=0)


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
        np.testing.assert_allclose(orderN, emb, atol=1e-12, rtol=0)


# ─────────────────────── Determinism ───────────────────────


class TestDeterminism:

    def test_same_seed_same_output(self):
        enc1 = Encoder(seed=42)
        enc2 = Encoder(seed=42)
        emb1 = enc1.encode("CCO")
        emb2 = enc2.encode("CCO")
        np.testing.assert_allclose(emb1, emb2, atol=1e-12, rtol=0)

    def test_different_seed_different_output(self):
        enc1 = Encoder(seed=1)
        enc2 = Encoder(seed=2)
        emb1 = enc1.encode("CCO")
        emb2 = enc2.encode("CCO")
        assert not np.allclose(emb1, emb2)

    def test_different_molecules_different_output(self):
        enc = Encoder(seed=42)
        emb1 = enc.encode("CCO")
        emb2 = enc.encode("c1ccccc1")
        assert not np.allclose(emb1, emb2)


# ─────────────────────── Type errors ───────────────────────


class TestTypeErrors:

    def test_bad_type_raises(self):
        enc = Encoder()
        with pytest.raises(TypeError, match="Expected str or Chem.Mol"):
            enc.encode(42)

    def test_bad_type_in_list_raises(self):
        enc = Encoder()
        with pytest.raises(TypeError):
            enc.encode(["CCO", 42])


# ─────────────────────── Save / Load ───────────────────────


class TestSaveLoad:

    def test_round_trip(self, tmp_path: Path):
        enc = Encoder(seed=42)
        emb_before = enc.encode(["CCO", "c1ccccc1"])

        path = tmp_path / "encoder.npz"
        enc.save(path)
        loaded = Encoder.load(path)

        assert loaded.dimension == enc.dimension
        assert loaded.depth == enc.depth
        assert loaded.atom_types == enc.atom_types
        assert loaded.normalize == enc.normalize
        assert loaded.seed == enc.seed
        np.testing.assert_array_equal(loaded._codebook, enc._codebook)
        np.testing.assert_allclose(
            loaded.encode(["CCO", "c1ccccc1"]), emb_before, atol=1e-12, rtol=0,
        )

    def test_round_trip_custom_params(self, tmp_path: Path):
        enc = Encoder(
            dimension=128, depth=5, atom_types=["C", "N", "O"],
            normalize=True, seed=99,
        )
        emb_before = enc.encode("CCO")

        path = tmp_path / "encoder.npz"
        enc.save(path)
        loaded = Encoder.load(path)

        assert loaded.dimension == 128
        assert loaded.depth == 5
        assert loaded.atom_types == ["C", "N", "O"]
        assert loaded.normalize is True
        assert loaded.seed == 99
        np.testing.assert_allclose(
            loaded.encode("CCO"), emb_before, atol=1e-12, rtol=0,
        )

    def test_round_trip_no_seed(self, tmp_path: Path):
        enc = Encoder()
        emb_before = enc.encode("CCO")

        path = tmp_path / "encoder.npz"
        enc.save(path)
        loaded = Encoder.load(path)

        assert loaded.seed is None
        np.testing.assert_allclose(
            loaded.encode("CCO"), emb_before, atol=1e-12, rtol=0,
        )
