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
from hyper_fingerprints.presets import (
    FEATURES_CONTINUOUS,
    FEATURES_GRAPH_HDC,
    GRAPH_FEATURES_GRAPH_HDC,
)

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


@pytest.fixture(scope="module")
def fixture_data():
    if not FIXTURE_PATH.exists():
        pytest.skip("Regression fixtures not found.")
    return _load_fixture()


class TestCrossBackendRegression:
    """Both backends must match the recorded regression fixtures.

    ``test_regression.py`` runs with ``backend="auto"`` (Rust when built), so
    the ``[numpy]`` leg here is the only fixture check of the NumPy path.
    """

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

    # Includes Kekulé spellings: the Rust parser does not perceive
    # aromaticity, so these must be aromatized before reaching it.
    SMILES = [
        "CCO", "c1ccccc1", "CC(=O)O", "CC(=O)Nc1ccc(O)cc1", "c1ccncc1",
        "C1=CC=CC=C1", "C1=CC=CN=C1", "O1C=CC=C1", "O=C1C=CC(=O)C=C1",
    ]

    def test_kekule_smiles_matches_aromatic_and_mol(self):
        enc = Encoder(dimension=256, depth=2, seed=42, backend="rust")
        pairs = [("C1=CC=CC=C1", "c1ccccc1"), ("C1=CC=CN=C1", "c1ccncc1"), ("O1C=CC=C1", "c1ccoc1")]
        for kekule, aromatic in pairs:
            from_kekule = enc.encode(kekule)
            np.testing.assert_allclose(from_kekule, enc.encode(aromatic), atol=ATOL, rtol=0)
            np.testing.assert_allclose(
                from_kekule, enc.encode(Chem.MolFromSmiles(kekule)), atol=ATOL, rtol=0
            )

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

    @pytest.mark.parametrize("features", [
        ["element"],
        ["element", "degree"],
        ["element", "aromatic", "charge"],
        ["degree", "hydrogens", "aromatic"],
    ])
    def test_reduced_feature_lists(self, features):
        """The Rust parser dispatches extractors by name; every subset must match."""
        smiles = ["CCO", "c1ccccc1", "CC(=O)[O-]", "C[N+](C)(C)C", "c1cc[nH]c1"]
        self._assert_backends_match(smiles, features=features)

    def test_feature_unsupported_by_rust_parser(self):
        """A legacy feature the Rust SMILES parser lacks falls back to RDKit
        preparation (Rust still does the message passing); SMILES and Mol
        input must agree with the pure-NumPy result."""
        features = ["element", "implicit_hydrogens", "aromatic"]
        smiles = ["CCO", "c1cc[nH]c1", "CC(=O)N"]
        rs = self._assert_backends_match(smiles, features=features)
        enc_rs = Encoder(dimension=128, depth=2, seed=42, backend="rust", features=features)
        from_mol = enc_rs.encode([Chem.MolFromSmiles(smi) for smi in smiles])
        np.testing.assert_allclose(from_mol, rs, atol=ATOL, rtol=0)

    def test_determinism_across_runs(self):
        """Multiple calls with same input must give identical results."""
        enc = Encoder(dimension=128, depth=2, seed=42, backend="rust")
        r1 = enc.encode(["CCO", "c1ccccc1"])
        r2 = enc.encode(["CCO", "c1ccccc1"])
        np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Composite (dict-features) encoder across backends
# ---------------------------------------------------------------------------


# Unnormalized composite embeddings grow quickly with depth (magnitudes reach
# ~1e6 at depth 3), so parity is asserted relatively as well as absolutely.
# Observed worst-case relative deviation across this matrix is ~6e-12.
RTOL = 1e-11


class TestCompositeCrossBackend:
    """The composite encoder must give the same result on either backend.

    Composite node hypervectors are built in NumPy from the per-feature
    codebooks; Rust only runs the message-passing loop over them. That loop
    differs from the legacy one in two ways the Rust side has to absorb:
    ``empty="zeros"`` scatter semantics and the ``max(norm, 1e-12)``
    normalization formula.
    """

    SMILES = [
        "CCO", "c1ccccc1", "CC(=O)O", "CC(=O)Nc1ccc(O)cc1",
        "c1ccncc1", "CC(=O)Oc1ccccc1C(=O)O", "CS(C)=O", "[NH4+]",
    ]

    def _assert_match(self, smiles, **kwargs):
        enc_rs = Encoder(backend="rust", **kwargs)
        enc_py = Encoder(backend="numpy", **kwargs)
        rs, py = enc_rs.encode(smiles), enc_py.encode(smiles)
        np.testing.assert_allclose(rs, py, atol=ATOL, rtol=RTOL)
        rs_j, py_j = enc_rs.encode_joint(smiles), enc_py.encode_joint(smiles)
        np.testing.assert_allclose(rs_j, py_j, atol=ATOL, rtol=RTOL)
        return rs

    @pytest.mark.parametrize("features", [FEATURES_CONTINUOUS, FEATURES_GRAPH_HDC])
    @pytest.mark.parametrize("dim", [128, 256])
    @pytest.mark.parametrize("depth", [0, 1, 3])
    def test_encode_matches(self, features, dim, depth):
        self._assert_match(
            self.SMILES, dimension=dim, depth=depth, seed=42, features=features
        )

    @pytest.mark.parametrize("features", [FEATURES_CONTINUOUS, FEATURES_GRAPH_HDC])
    def test_normalize_matches(self, features):
        """Exercises the ``max(norm, 1e-12)`` branch (NormMode::L2Clamp)."""
        self._assert_match(
            self.SMILES, dimension=256, depth=3, seed=42,
            normalize=True, features=features,
        )

    @pytest.mark.parametrize("graph_mode", ["bundle", "bind", "concat"])
    def test_graph_features_match(self, graph_mode):
        rs = self._assert_match(
            self.SMILES, dimension=256, depth=3, seed=42,
            features=FEATURES_GRAPH_HDC,
            graph_features=GRAPH_FEATURES_GRAPH_HDC,
            graph_mode=graph_mode,
        )
        expected = 512 if graph_mode == "concat" else 256
        assert rs.shape == (len(self.SMILES), expected)

    def test_graph_hdc_preset_end_to_end(self):
        """The full graph_hdc-parity configuration, single-directional edges."""
        self._assert_match(
            self.SMILES, dimension=256, depth=3, seed=42, normalize=True,
            features=FEATURES_GRAPH_HDC,
            graph_features=GRAPH_FEATURES_GRAPH_HDC,
            graph_mode="bundle", bidirectional=False,
        )

    def test_bidirectional_false_matches(self):
        self._assert_match(
            self.SMILES, dimension=128, depth=2, seed=7,
            features=FEATURES_CONTINUOUS, bidirectional=False,
        )

    def test_unknown_atoms_slot_matches(self):
        """Out-of-vocabulary elements share the extra codebook row."""
        smiles = ["CCO", "[Se]C", "[Si](C)(C)C", "c1ccccc1"]
        self._assert_match(
            smiles, dimension=128, depth=2, seed=7,
            features=FEATURES_CONTINUOUS, unknown_atoms="slot",
        )

    def test_single_atom_alone_and_in_batch(self):
        """Composite mode is batch-invariant, on both backends."""
        kwargs = dict(dimension=128, depth=2, seed=42, features=FEATURES_CONTINUOUS)
        alone = self._assert_match(["C"], **kwargs)
        batched = self._assert_match(["C", "CCO"], **kwargs)
        np.testing.assert_allclose(alone[0], batched[0], atol=ATOL, rtol=RTOL)

    def test_all_edgeless_batch(self):
        """Every molecule bondless: composite scatters to zeros, so layers
        1..depth vanish and the readout is just the layer-0 bundle."""
        enc = Encoder(dimension=128, depth=3, seed=42, backend="rust",
                      features=FEATURES_CONTINUOUS)
        rs = self._assert_match(["C", "O", "N"], dimension=128, depth=3, seed=42,
                                features=FEATURES_CONTINUOUS)
        # Order-N must equal order-0 here (no message passing contributes).
        joint = enc.encode_joint(["C", "O", "N"])
        np.testing.assert_allclose(joint[:, :128], rs, atol=ATOL, rtol=RTOL)

    def test_disconnected_and_empty_input(self):
        self._assert_match(["CC.OO", "C.C.C"], dimension=128, depth=2, seed=42,
                           features=FEATURES_CONTINUOUS)
        enc = Encoder(dimension=128, depth=2, seed=42, backend="rust",
                      features=FEATURES_CONTINUOUS)
        assert enc.encode([]).shape == (0, 128)

    def test_mol_input_matches_smiles(self):
        mols = [Chem.MolFromSmiles(smi) for smi in self.SMILES]
        kwargs = dict(dimension=128, depth=2, seed=42, features=FEATURES_CONTINUOUS)
        from_smiles = self._assert_match(self.SMILES, **kwargs)
        from_mols = self._assert_match(mols, **kwargs)
        np.testing.assert_allclose(from_smiles, from_mols, atol=ATOL, rtol=RTOL)

    def test_large_batch_parallel(self):
        """Exercises the rayon fan-out across molecules."""
        smiles = self.SMILES * 40
        self._assert_match(smiles, dimension=128, depth=3, seed=42,
                           features=FEATURES_CONTINUOUS)

    def test_determinism_across_runs(self):
        enc = Encoder(dimension=128, depth=3, seed=42, backend="rust",
                      features=FEATURES_CONTINUOUS)
        np.testing.assert_array_equal(enc.encode(self.SMILES), enc.encode(self.SMILES))

    def test_save_load_roundtrip_on_rust(self, tmp_path):
        """A composite encoder reloaded from disk still runs on Rust."""
        enc = Encoder(dimension=128, depth=2, seed=42, features=FEATURES_GRAPH_HDC,
                      graph_features=GRAPH_FEATURES_GRAPH_HDC, backend="numpy")
        path = tmp_path / "composite.npz"
        enc.save(path)
        loaded_rs = Encoder.load(path, backend="rust")
        np.testing.assert_allclose(
            loaded_rs.encode(self.SMILES), enc.encode(self.SMILES),
            atol=ATOL, rtol=RTOL,
        )


# ---------------------------------------------------------------------------
# Composite node encoding (per-feature codebooks + FHRR)
# ---------------------------------------------------------------------------


class TestCompositeNodeEncoding:
    """`CompositeEncoder.encode_rust` must agree with the NumPy reference.

    Node hypervectors are compared directly rather than through a finished
    embedding: message passing and normalization would smear a discrepancy.

    The only intentional divergence is that Rust always evaluates ``z ** e`` as
    ``exp(e * ln z)``, while NumPy switches to repeated multiplication for real
    integer exponents with ``|e| < 100``. On the exponents that actually occur
    (``value / bandwidth``, typically <= ~7) this is worth about 1 ULP.
    """

    # Comfortably above the ~1.3e-15 observed worst case, far below anything
    # that would indicate a real algorithmic difference.
    NODE_ATOL = 1e-12

    SMILES = [
        "CCO", "c1ccccc1", "CC(=O)O", "CC(=O)Nc1ccc(O)cc1", "c1ccncc1",
        "CC(=O)Oc1ccccc1C(=O)O", "CS(C)=O", "[NH4+]", "CC(=O)[O-]",
        "C[N+](C)(C)C", "c1cc[nH]c1", "C", "O", "CC.OO",
    ]

    def _node_hvs(self, **kwargs):
        enc = Encoder(**kwargs)
        x = enc._prepare_batch_from_list(self.SMILES).x
        return enc._node_encoder.encode(x), enc._node_encoder.encode_rust(x)

    # Odd dimensions have no Nyquist bin and even ones do; whether the DC and
    # Nyquist bins come out negative — which is what breaks the Hermitian
    # symmetry of the powered spectrum — depends on the seed. Both axes matter.
    @pytest.mark.parametrize("dim", [64, 127, 128, 255, 256, 257, 512])
    @pytest.mark.parametrize("seed", [0, 1, 42, 7])
    @pytest.mark.parametrize("features", [FEATURES_CONTINUOUS, FEATURES_GRAPH_HDC])
    def test_matches_numpy(self, dim, seed, features):
        ref, got = self._node_hvs(
            dimension=dim, depth=2, seed=seed, features=features
        )
        assert got.shape == ref.shape
        np.testing.assert_allclose(got, ref, atol=self.NODE_ATOL, rtol=0)

    def test_non_hermitian_powered_spectrum_is_handled(self):
        """Guards the subtlest failure mode in the port.

        `generate_hermitian_spectrum` makes the DC bin (and, for even `dim`,
        the Nyquist bin) real — and often negative. A negative real raised to a
        non-integer power leaves the real axis, so `spectrum ** e` is NOT
        Hermitian and `ifft(...).real` discards a real imaginary component. A
        `c2r` transform, which assumes symmetry, would silently produce a
        different vector. This pins a configuration where that actually bites.
        """
        from hyper_fingerprints.codebook import generate_hermitian_spectrum

        spectrum = generate_hermitian_spectrum(256, np.random.default_rng(42))
        assert spectrum[0].imag == 0.0 and spectrum[0].real < 0, "DC bin is not negative real"
        assert spectrum[128].imag == 0.0 and spectrum[128].real < 0, "Nyquist is not negative real"
        powered = spectrum ** 1.5
        residual = np.abs(powered - np.conj(powered[(-np.arange(256)) % 256])).max()
        assert residual > 1.0, "expected a badly non-Hermitian powered spectrum"

        # dim=256/seed=42 is exactly that configuration; degree 3 / bandwidth 2
        # gives the exponent 1.5 used above.
        ref, got = self._node_hvs(
            dimension=256, depth=2, seed=42, features=FEATURES_CONTINUOUS
        )
        np.testing.assert_allclose(got, ref, atol=self.NODE_ATOL, rtol=0)

    @pytest.mark.parametrize("bandwidth", [0.5, 1.0, 2.0, 3.7, 19.0])
    def test_bandwidth_sweep(self, bandwidth):
        """Bandwidth sets the exponent scale, and with it which NumPy
        ``**`` branch is taken. ``bandwidth=1.0`` drives integer exponents
        (the repeated-multiplication branch); fractional ones drive exp/log."""
        features = {
            "element": {"kind": "categorical"},
            "degree": {"kind": "continuous", "bandwidth": bandwidth},
            "hydrogens": {"kind": "continuous", "bandwidth": bandwidth},
        }
        ref, got = self._node_hvs(dimension=256, depth=2, seed=42, features=features)
        np.testing.assert_allclose(got, ref, atol=self.NODE_ATOL, rtol=0)

    def test_large_integer_exponents(self):
        """A pathological bandwidth pushes ``graph_size`` into the large
        integer-exponent regime, where NumPy's two ``**`` branches diverge
        most. Pins the actual deviation instead of assuming it stays tiny."""
        enc = Encoder(
            dimension=128, depth=1, seed=42,
            features={"element": {"kind": "categorical"},
                      "degree": {"kind": "continuous", "bandwidth": 1.0}},
            graph_features={"graph_size": {"kind": "continuous", "bandwidth": 1.0}},
        )
        g = enc._prepare_batch_from_list(self.SMILES).g
        ce = enc._graph_encoder
        ref, got = ce.encode(g), ce.encode_rust(g)
        # Exponents here reach the molecule size (up to 13), so the vectors
        # themselves reach ~2e2 and the absolute deviation grows with them.
        # Relative agreement stays at ~1.7e-15, i.e. still about one ULP.
        assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-13
        np.testing.assert_allclose(got, ref, atol=1e-11, rtol=1e-13)

    def test_all_categorical(self):
        """No continuous features: pure lookups bound together."""
        features = {
            "element": {"kind": "categorical"},
            "charge": {"kind": "categorical"},
            "aromatic": {"kind": "categorical"},
        }
        ref, got = self._node_hvs(dimension=128, depth=2, seed=3, features=features)
        np.testing.assert_allclose(got, ref, atol=self.NODE_ATOL, rtol=0)

    def test_single_feature(self):
        for features in ({"element": {"kind": "categorical"}},
                         {"degree": {"kind": "continuous", "bandwidth": 2.0}}):
            ref, got = self._node_hvs(
                dimension=128, depth=1, seed=5, features=features
            )
            np.testing.assert_allclose(got, ref, atol=self.NODE_ATOL, rtol=0)

    def test_unknown_atoms_slot(self):
        """The extra codebook row must be reachable from the Rust lookup."""
        enc = Encoder(dimension=128, depth=2, seed=7,
                      features=FEATURES_CONTINUOUS, unknown_atoms="slot")
        x = enc._prepare_batch_from_list(["CCO", "[Se]C", "[Si](C)(C)C"]).x
        ce = enc._node_encoder
        np.testing.assert_allclose(
            ce.encode_rust(x), ce.encode(x), atol=self.NODE_ATOL, rtol=0
        )

    def test_empty_and_single_row(self):
        enc = Encoder(dimension=128, depth=2, seed=42, features=FEATURES_CONTINUOUS)
        ce = enc._node_encoder
        x = enc._prepare_batch_from_list(["CCO"]).x
        assert ce.encode_rust(np.zeros((0, x.shape[1]))).shape == (0, 128)
        np.testing.assert_allclose(
            ce.encode_rust(x[:1]), ce.encode(x[:1]), atol=self.NODE_ATOL, rtol=0
        )

    def test_invalid_categorical_index_raises(self):
        """An out-of-range lookup must raise, not read out of bounds."""
        enc = Encoder(dimension=64, depth=1, seed=0, features=FEATURES_CONTINUOUS)
        ce = enc._node_encoder
        x = enc._prepare_batch_from_list(["CCO"]).x.copy()
        x[0, 0] = 999.0
        with pytest.raises(ValueError, match="not a valid index"):
            ce.encode_rust(x)

    def test_determinism(self):
        enc = Encoder(dimension=256, depth=2, seed=42, features=FEATURES_CONTINUOUS)
        x = enc._prepare_batch_from_list(self.SMILES).x
        ce = enc._node_encoder
        np.testing.assert_array_equal(ce.encode_rust(x), ce.encode_rust(x))
