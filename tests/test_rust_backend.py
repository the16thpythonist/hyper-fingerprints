"""Cross-backend verification and edge-case tests for the Rust extension.

Ensures the Rust and Python backends produce identical results across
a wide range of inputs, including edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from hyper_fingerprints.encoder import Encoder, _HAS_RUST

ATOL = 1e-10
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "regression_data.npz"

pytestmark = pytest.mark.skipif(not _HAS_RUST, reason="Rust extension not built")

BACKENDS = ["numpy", "rust"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fixture():
    raw = np.load(FIXTURE_PATH, allow_pickle=False)
    config = json.loads(bytes(raw["config"]))
    smiles = json.loads(bytes(raw["smiles"]))
    return config, smiles, raw


def _make_encoder(config, codebook, backend):
    return Encoder(
        dimension=config["dimension"],
        depth=config["depth"],
        seed=config["seed"],
        codebook=codebook,
        backend=backend,
    )


# ---------------------------------------------------------------------------
# Cross-backend regression tests (parameterized over both backends)
# ---------------------------------------------------------------------------


class TestCrossBackendRegression:
    """Both backends must match the recorded regression fixtures."""

    @pytest.fixture(scope="class")
    def fixture_data(self):
        if not FIXTURE_PATH.exists():
            pytest.skip("Regression fixtures not found.")
        return _load_fixture()

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_individual_encode(self, fixture_data, backend):
        config, smiles, raw = fixture_data
        enc = _make_encoder(config, raw["codebook"], backend)
        for i, smi in enumerate(smiles):
            expected = raw[f"ind_encode_{i}"]
            result = np.asarray(enc.encode(smi))
            np.testing.assert_allclose(
                result, expected, atol=ATOL, rtol=0,
                err_msg=f"[{backend}] encode() mismatch for {smi!r}",
            )

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_individual_encode_joint(self, fixture_data, backend):
        config, smiles, raw = fixture_data
        enc = _make_encoder(config, raw["codebook"], backend)
        for i, smi in enumerate(smiles):
            expected = raw[f"ind_joint_{i}"]
            result = np.asarray(enc.encode_joint(smi))
            np.testing.assert_allclose(
                result, expected, atol=ATOL, rtol=0,
                err_msg=f"[{backend}] encode_joint() mismatch for {smi!r}",
            )

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_batch_encode(self, fixture_data, backend):
        config, smiles, raw = fixture_data
        enc = _make_encoder(config, raw["codebook"], backend)
        result = np.asarray(enc.encode(smiles))
        np.testing.assert_allclose(result, raw["batch_encode"], atol=ATOL, rtol=0,
                                   err_msg=f"[{backend}] batch encode mismatch")

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_batch_encode_joint(self, fixture_data, backend):
        config, smiles, raw = fixture_data
        enc = _make_encoder(config, raw["codebook"], backend)
        result = np.asarray(enc.encode_joint(smiles))
        np.testing.assert_allclose(result, raw["batch_encode_joint"], atol=ATOL, rtol=0,
                                   err_msg=f"[{backend}] batch encode_joint mismatch")


# ---------------------------------------------------------------------------
# Rust vs Python direct comparison (no fixtures needed)
# ---------------------------------------------------------------------------


class TestRustVsPython:
    """Rust and Python backends must produce identical results."""

    SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CC(=O)Nc1ccc(O)cc1", "c1ccncc1"]

    @pytest.mark.parametrize("dim", [128, 256, 512])
    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_encode_matches(self, dim, depth):
        enc_rs = Encoder(dimension=dim, depth=depth, seed=42, backend="rust")
        enc_py = Encoder(dimension=dim, depth=depth, seed=42, backend="numpy")
        rs = enc_rs.encode(self.SMILES)
        py = enc_py.encode(self.SMILES)
        np.testing.assert_allclose(rs, py, atol=ATOL, rtol=0)

    @pytest.mark.parametrize("dim", [128, 256, 512])
    def test_encode_joint_matches(self, dim):
        enc_rs = Encoder(dimension=dim, depth=2, seed=42, backend="rust")
        enc_py = Encoder(dimension=dim, depth=2, seed=42, backend="numpy")
        rs = enc_rs.encode_joint(self.SMILES)
        py = enc_py.encode_joint(self.SMILES)
        np.testing.assert_allclose(rs, py, atol=ATOL, rtol=0)

    def test_normalize_matches(self):
        enc_rs = Encoder(dimension=256, depth=2, seed=42, normalize=True, backend="rust")
        enc_py = Encoder(dimension=256, depth=2, seed=42, normalize=True, backend="numpy")
        rs = enc_rs.encode(self.SMILES)
        py = enc_py.encode(self.SMILES)
        np.testing.assert_allclose(rs, py, atol=ATOL, rtol=0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases that must work identically across backends."""

    def _assert_backends_match(self, smiles, dim=128, depth=2, **kwargs):
        """Encode with both backends and assert results match."""
        enc_rs = Encoder(dimension=dim, depth=depth, seed=42, backend="rust", **kwargs)
        enc_py = Encoder(dimension=dim, depth=depth, seed=42, backend="numpy", **kwargs)
        rs = enc_rs.encode(smiles)
        py = enc_py.encode(smiles)
        np.testing.assert_allclose(rs, py, atol=ATOL, rtol=0)
        return rs

    def test_single_atom_molecule(self):
        """Methane: 1 atom, 0 edges."""
        result = self._assert_backends_match("C")
        assert result.shape == (1, 128)

    def test_single_atom_in_batch(self):
        """Single-atom molecule mixed with multi-atom molecules."""
        self._assert_backends_match(["C", "CCO", "c1ccccc1"])

    def test_multiple_single_atoms(self):
        """Batch of only single-atom molecules (no edges in entire batch)."""
        self._assert_backends_match(["C", "[NH3]", "O"])

    def test_disconnected_graph(self):
        """Disconnected molecule (dot notation in SMILES)."""
        result = self._assert_backends_match("CC.OO")
        assert result.shape == (1, 128)

    def test_disconnected_in_batch(self):
        """Disconnected molecule in a batch with normal molecules."""
        self._assert_backends_match(["CC.OO", "CCO", "c1ccccc1"])

    def test_two_atom_molecule(self):
        """Simplest molecule with one bond."""
        result = self._assert_backends_match("CC")
        assert result.shape == (1, 128)

    def test_large_molecule(self):
        """Larger molecule with many atoms and rings."""
        smi = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
        self._assert_backends_match(smi, dim=256, depth=3)

    def test_depth_zero(self):
        """Depth 0: only node features, no message passing."""
        self._assert_backends_match(["CCO", "c1ccccc1"], depth=0)

    def test_depth_one(self):
        """Depth 1: single layer of message passing."""
        self._assert_backends_match(["CCO", "c1ccccc1"], depth=1)

    def test_large_batch(self):
        """Larger batch to exercise parallelism."""
        smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CF", "CBr", "CSC"] * 50
        self._assert_backends_match(smiles, dim=256, depth=2)

    def test_all_supported_elements(self):
        """Molecule touching all 9 default atom types."""
        # Br, C, Cl, F, I, N, O, P, S
        smiles = [
            "c1ccc(Br)cc1",       # Br, C
            "ClC(F)I",            # Cl, C, F, I
            "NCS",                # N, C, S
            "OP(O)(O)=O",         # O, P
        ]
        self._assert_backends_match(smiles, dim=256, depth=2)

    def test_aromatic_molecule(self):
        """Aromatic ring with heteroatom."""
        self._assert_backends_match(["c1ccncc1", "c1ccc2[nH]ccc2c1"])

    def test_charged_molecule(self):
        """Molecule with formal charges."""
        self._assert_backends_match("[NH4+].[Cl-]", atom_types=["N", "Cl", "H"])

    def test_determinism_across_runs(self):
        """Multiple calls with same input must give identical results."""
        enc = Encoder(dimension=128, depth=2, seed=42, backend="rust")
        r1 = enc.encode(["CCO", "c1ccccc1"])
        r2 = enc.encode(["CCO", "c1ccccc1"])
        np.testing.assert_array_equal(r1, r2)
