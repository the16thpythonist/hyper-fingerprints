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


# Composite parity is measured as deviation scaled by each ROW's own magnitude.
#
# A global `atol`/`rtol` pair is close to meaningless here: unnormalized
# composite embeddings reach ~1e7 at depth 3 and ~1e15 at depth 4, and the
# magnitude swings with the seed, so any fixed absolute budget is really a
# statement about whichever seed was chosen. The previous `atol=1e-10,
# rtol=1e-11` consumed 99% of its budget at seed 42 / dim 128 / depth 3 and
# would have FAILED at seed 7 (7.6x over) or seed 1 / depth 4 (16.5x over) --
# the suite was green by seed luck rather than by agreement.
#
# Row-scaled deviation is stable instead: measured worst across dims
# {3,8,64,128,256,512} x seeds {0,1,7,42,123} x depth 0-3 x normalize {T,F} is
# 4.1e-13, and 2.5e-12 out to depth 6. ROW_TOL leaves ~400x headroom over that,
# so a real regression trips it and a seed change does not.
ROW_TOL = 1e-9


def row_scaled_deviation(rs: np.ndarray, py: np.ndarray) -> float:
    """Max |rs - py| scaled by the magnitude of the row it occurs in."""
    scale = np.maximum(np.abs(py).max(axis=-1, keepdims=True), 1e-300)
    return float((np.abs(rs - py) / scale).max())


def assert_backends_agree(rs: np.ndarray, py: np.ndarray, tol: float = ROW_TOL) -> None:
    assert rs.shape == py.shape, f"shape {rs.shape} != {py.shape}"
    assert np.all(np.isfinite(py)) == np.all(np.isfinite(rs)), (
        "backends disagree on whether the result is finite"
    )
    if not np.all(np.isfinite(py)):
        return  # both non-finite: nothing meaningful to compare
    dev = row_scaled_deviation(rs, py)
    assert dev <= tol, f"row-scaled deviation {dev:.3e} exceeds {tol:.1e}"


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
        assert_backends_agree(rs, py)
        assert_backends_agree(enc_rs.encode_joint(smiles), enc_py.encode_joint(smiles))
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
        assert_backends_agree(alone[0][None], batched[0][None])

    def test_all_edgeless_batch(self):
        """Every molecule bondless: composite scatters to zeros, so layers
        1..depth vanish and the readout is just the layer-0 bundle."""
        enc = Encoder(dimension=128, depth=3, seed=42, backend="rust",
                      features=FEATURES_CONTINUOUS)
        rs = self._assert_match(["C", "O", "N"], dimension=128, depth=3, seed=42,
                                features=FEATURES_CONTINUOUS)
        # Order-N must equal order-0 here (no message passing contributes).
        joint = enc.encode_joint(["C", "O", "N"])
        assert_backends_agree(joint[:, :128], rs)

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
        assert_backends_agree(from_smiles, from_mols)

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
        assert_backends_agree(loaded_rs.encode(self.SMILES), enc.encode(self.SMILES))


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


# ---------------------------------------------------------------------------
# Broad parameter sweeps
# ---------------------------------------------------------------------------

SWEEP_SMILES = [
    "CCO", "c1ccccc1", "CC(=O)O", "CC(=O)Nc1ccc(O)cc1", "c1ccncc1",
    "CC(=O)Oc1ccccc1C(=O)O", "CS(C)=O", "[NH4+]", "CC(=O)[O-]", "C", "O",
    "CC.OO", "C[N+](C)(C)C", "c1cc[nH]c1", "FC(F)(F)c1ccccc1", "OP(O)(O)=O",
]


class TestCompositeSweep:
    """Cross-backend agreement over a wide parameter grid.

    The earlier suite pinned a single seed (42), dimensions no smaller than 128
    and depth no greater than 3, which left the tolerance resting on seed luck
    and never exercised the regimes where the backends actually part company.
    These sweeps vary every axis that turned out to matter.
    """

    @pytest.mark.parametrize("dim", [3, 8, 64, 128, 256, 512])
    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
    def test_seed_and_dimension_grid(self, dim, seed):
        """Every seed x dimension, across depth and normalize.

        Seed matters because it sets the sign of the spectrum's DC and Nyquist
        bins, which decides whether the non-Hermitian branch of the continuous
        encoder is exercised at all. Dimension matters because odd dimensions
        have no Nyquist bin.
        """
        for depth in (0, 1, 2, 3):
            for normalize in (True, False):
                kw = dict(dimension=dim, depth=depth, seed=seed,
                          normalize=normalize, features=FEATURES_CONTINUOUS)
                py = Encoder(**kw, backend="numpy").encode(SWEEP_SMILES)
                rs = Encoder(**kw, backend="rust").encode(SWEEP_SMILES)
                assert_backends_agree(rs, py)

    @pytest.mark.parametrize("depth", [4, 5, 6])
    @pytest.mark.parametrize("normalize", [True, False])
    def test_deeper_than_the_old_matrix(self, depth, normalize):
        """Depth 4-6 was previously untested; error grows ~3x per layer."""
        for seed in (0, 42, 123):
            kw = dict(dimension=256, depth=depth, seed=seed,
                      normalize=normalize, features=FEATURES_CONTINUOUS)
            py = Encoder(**kw, backend="numpy").encode(SWEEP_SMILES)
            rs = Encoder(**kw, backend="rust").encode(SWEEP_SMILES)
            assert_backends_agree(rs, py)

    @pytest.mark.parametrize("bandwidth", [0.25, 1.0, 2.0, 7.5, 100.0, 1e6])
    def test_bandwidth_range(self, bandwidth):
        """Bandwidth sets the exponent scale (`exponent = value / bandwidth`).

        Bounded below at 0.25 deliberately: feature values here reach 5, so
        smaller bandwidths push the exponent past ~20 and are covered by
        `test_tiny_bandwidth_diverges` instead.
        """
        feats = {"element": {"kind": "categorical"},
                 "degree": {"kind": "continuous", "bandwidth": bandwidth},
                 "charge": {"kind": "continuous", "bandwidth": bandwidth}}
        for seed in (0, 7, 42):
            kw = dict(dimension=128, depth=2, seed=seed, features=feats)
            py = Encoder(**kw, backend="numpy").encode(SWEEP_SMILES)
            rs = Encoder(**kw, backend="rust").encode(SWEEP_SMILES)
            assert_backends_agree(rs, py)

    def test_tiny_bandwidth_diverges(self):
        """Pins the one regime where the backends measurably disagree.

        A bandwidth far below the feature's value range drives the exponent
        large (`degree=3 / 0.05 = 60`), and `spectrum ** 60` sends the
        embedding to ~1e63. Rust evaluates `z ** e` as `exp(e * ln z)` while
        NumPy switches to repeated multiplication for real integer exponents
        with `|e| < 100`; at that magnitude the two differ by ~5.6e-7
        relative, well past `ROW_TOL`.

        This is the documented cost of not replicating NumPy's `npy_cpow`
        internals, not a defect -- and an embedding of 1e63 is numerically
        useless on *either* backend, so the real lesson is to keep bandwidth on
        the order of the feature's value range. The assertion is deliberately
        two-sided: it fails if the divergence grows, and also if it silently
        disappears (which would mean this documentation went stale).
        """
        feats = {"element": {"kind": "categorical"},
                 "degree": {"kind": "continuous", "bandwidth": 0.05},
                 "charge": {"kind": "continuous", "bandwidth": 0.05}}
        kw = dict(dimension=128, depth=2, seed=7, features=feats)
        py = Encoder(**kw, backend="numpy").encode(SWEEP_SMILES)
        rs = Encoder(**kw, backend="rust").encode(SWEEP_SMILES)

        assert np.abs(py).max() > 1e40, "expected the embedding to blow up here"
        dev = row_scaled_deviation(rs, py)
        assert 1e-9 < dev < 1e-4, f"divergence moved to {dev:.2e}; update the docs"

    @pytest.mark.parametrize("n", [1, 2, 3, 17, 64, 200])
    def test_batch_sizes(self, n):
        """Composite mode is batch-invariant, so batch size must not matter."""
        mols = (SWEEP_SMILES * 20)[:n]
        kw = dict(dimension=128, depth=3, seed=1, features=FEATURES_CONTINUOUS)
        py = Encoder(**kw, backend="numpy").encode(mols)
        rs = Encoder(**kw, backend="rust").encode(mols)
        assert_backends_agree(rs, py)
        assert rs.shape == (n, 128)

    @pytest.mark.parametrize("graph_mode", ["bundle", "bind", "concat"])
    @pytest.mark.parametrize("bidirectional", [True, False])
    @pytest.mark.parametrize("unknown_atoms", ["error", "slot"])
    def test_option_combinations(self, graph_mode, bidirectional, unknown_atoms):
        """The cross product of the switches, over several seeds."""
        mols = SWEEP_SMILES + (["[Se]C", "[Si](C)(C)C"] if unknown_atoms == "slot" else [])
        for seed in (0, 42):
            kw = dict(dimension=128, depth=3, seed=seed, normalize=True,
                      features=FEATURES_GRAPH_HDC,
                      graph_features=GRAPH_FEATURES_GRAPH_HDC,
                      graph_mode=graph_mode, bidirectional=bidirectional,
                      unknown_atoms=unknown_atoms)
            py = Encoder(**kw, backend="numpy").encode(mols)
            rs = Encoder(**kw, backend="rust").encode(mols)
            assert_backends_agree(rs, py)

    def test_randomized_configurations(self):
        """Deterministic pseudo-random sweep over the whole option space.

        A sweep of this shape is what surfaced the two crash sites and the
        seed-luck tolerance; keeping one in the suite means the next such bug
        fails a test rather than a user's job. The RNG seed is fixed so a
        failure is reproducible.
        """
        import random

        rng = random.Random(20260915)
        all_feats = ["element", "degree", "charge", "hydrogens",
                     "implicit_hydrogens", "aromatic"]
        checked = 0

        for _ in range(60):
            names = rng.sample(all_feats, rng.randint(1, len(all_feats)))
            feats = {}
            for nm in names:
                if nm != "element" and rng.random() < 0.5:
                    feats[nm] = {"kind": "continuous",
                                 "bandwidth": rng.choice([0.5, 1.0, 2.0, 3.3, 11.0])}
                else:
                    feats[nm] = {"kind": "categorical"}
            gfeats = None
            if rng.random() < 0.5:
                gfeats = {nm: {"kind": "continuous",
                               "bandwidth": rng.choice([1.0, 4.0, 18.5])}
                          for nm in rng.sample(
                              ["graph_size", "graph_diameter", "graph_total_charge"],
                              rng.randint(1, 3))}
            kw = dict(
                dimension=rng.choice([3, 8, 16, 64, 127, 128, 256, 384]),
                depth=rng.choice([0, 1, 2, 3, 4]),
                seed=rng.randrange(10_000),
                normalize=rng.choice([True, False]),
                bidirectional=rng.choice([True, False]),
                graph_mode=rng.choice(["bundle", "bind", "concat"]),
                unknown_atoms=rng.choice(["error", "slot"]),
                features=feats,
                graph_features=gfeats,
            )
            mols = rng.sample(SWEEP_SMILES, rng.choice([1, 2, 5, 12]))

            py_enc = Encoder(**kw, backend="numpy")
            rs_enc = Encoder(**kw, backend="rust")
            for fn in ("encode", "encode_joint"):
                py = getattr(py_enc, fn)(mols)
                # A config that overflows must fail cleanly, not abort the
                # process; PanicException would not be caught by `except
                # Exception` and would take the whole run down.
                try:
                    rs = getattr(rs_enc, fn)(mols)
                except ValueError:
                    assert not np.all(np.isfinite(py)), (
                        f"rust rejected a config NumPy handled finitely: {kw}"
                    )
                    continue
                assert_backends_agree(rs, py)
                checked += 1

        assert checked > 100, f"sweep only compared {checked} results"


class TestNumericalLimits:
    """Overflow must surface as a catchable error, never abort the process.

    A Rust `panic!` reaches Python as `pyo3_runtime.PanicException`, which
    derives from `BaseException` rather than `Exception`. A caller looping over
    a dataset with `try: enc.encode(chunk) except Exception:` would not catch
    it and the whole job would die. Both FFT sites therefore return an error
    that PyO3 converts to `ValueError`.
    """

    def test_tiny_bandwidth_raises_valueerror(self):
        """Node encoding: `spectrum ** e` overflows for a huge exponent."""
        feats = {"element": {"kind": "categorical"},
                 "degree": {"kind": "continuous", "bandwidth": 1e-6}}
        enc = Encoder(dimension=64, depth=2, seed=0, features=feats, backend="rust")
        with pytest.raises(ValueError) as excinfo:
            enc.encode(["CCO"])
        msg = str(excinfo.value)
        assert "bandwidth" in msg, f"message should name the cause: {msg}"
        assert "overflow" in msg.lower()

    def test_deep_unnormalized_raises_valueerror(self):
        """Message passing: the state overflows at large depth without
        normalization. Reachable in legacy mode too, so both are checked."""
        mols = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccc2c(c1)cccc2", "CCCCCCCCCC"] * 10
        for features in (None, FEATURES_CONTINUOUS):
            kw = dict(dimension=512, depth=14, seed=1, normalize=False, backend="rust")
            if features is not None:
                kw["features"] = features
            with pytest.raises(ValueError) as excinfo:
                Encoder(**kw).encode(mols)
            assert "non-finite" in str(excinfo.value)

    def test_error_is_catchable_and_state_survives(self):
        """`except Exception` must catch it, and the encoder must still work."""
        feats = {"element": {"kind": "categorical"},
                 "degree": {"kind": "continuous", "bandwidth": 1e-6}}
        bad = Encoder(dimension=64, depth=2, seed=0, features=feats, backend="rust")
        caught = 0
        for _ in range(3):
            try:
                bad.encode(["CCO"])
            except Exception:       # deliberately not BaseException
                caught += 1
        assert caught == 3

        # A healthy encoder in the same process is unaffected.
        good = Encoder(dimension=64, depth=2, seed=0,
                       features=FEATURES_CONTINUOUS, backend="rust")
        py = Encoder(dimension=64, depth=2, seed=0,
                     features=FEATURES_CONTINUOUS, backend="numpy")
        assert_backends_agree(good.encode(SWEEP_SMILES), py.encode(SWEEP_SMILES))

    def test_nonfinite_node_hv_rejected_at_the_boundary(self):
        """The low-level entry point guards itself, not just the Encoder."""
        from hyper_fingerprints._core import encode_batch_hv_rs

        hv = np.ones((2, 8))
        edge = np.array([[0, 1], [1, 0]], dtype=np.int64)
        batch = np.array([0, 0], dtype=np.int64)
        for bad in (np.inf, -np.inf, np.nan):
            poisoned = hv.copy()
            poisoned[0, 0] = bad
            with pytest.raises(ValueError):
                encode_batch_hv_rs(poisoned, edge, batch, 1, 1, "none", False)

        # Finite input still works.
        out, _ = encode_batch_hv_rs(hv, edge, batch, 1, 1, "none", False)
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("depth", [1, 2, 3, 4, 5, 6])
    def test_depth_up_to_six_is_the_supported_range(self, depth):
        """Agreement holds to ~1e-12 through depth 6 -- the supported claim.

        Measured on 100 real PubChem molecules across dims 64-512, both
        feature kinds and both `normalize` settings, the worst deviation
        through depth 6 is 4.6e-13. Beyond depth 6 conditioning degrades; see
        `test_deep_message_passing_degrades`.
        """
        categorical = {"element": {"kind": "categorical"},
                       "hydrogens": {"kind": "categorical"},
                       "aromatic": {"kind": "categorical"}}
        for features in (FEATURES_CONTINUOUS, categorical):
            for dim in (64, 256):
                for normalize in (True, False):
                    kw = dict(dimension=dim, depth=depth, seed=0,
                              normalize=normalize, features=features)
                    py = Encoder(**kw, backend="numpy").encode(SWEEP_SMILES)
                    rs = Encoder(**kw, backend="rust").encode(SWEEP_SMILES)
                    assert_backends_agree(rs, py)

    def test_deep_message_passing_degrades(self):
        """Pins the honest upper limit rather than implying there isn't one.

        Message passing is ill-conditioned at large depth: each layer binds
        the state with an aggregate of its neighbours, so rounding differences
        amplify. Past depth ~7 the backends can part company well beyond
        ROW_TOL -- ~1e-9 at depth 8 for an all-categorical configuration, and
        arbitrarily far once magnitudes approach the float64 ceiling (~1e186),
        where the result is meaningless on either backend anyway.

        This is conditioning, not a port defect: perturbing `node_hv` by one
        ULP moves the NumPy result by a comparable amount. The test asserts
        the *shape* of the limit -- deep results stay finite or fail cleanly,
        and shallow results remain tight -- so the documented boundary cannot
        silently drift.
        """
        categorical = {"element": {"kind": "categorical"},
                       "hydrogens": {"kind": "categorical"},
                       "aromatic": {"kind": "categorical"}}
        shallow = deep = None
        for dim in (256, 512):
            kw = dict(dimension=dim, seed=42, normalize=True, features=categorical)
            for depth, slot in ((3, "shallow"), (8, "deep")):
                py = Encoder(**kw, depth=depth, backend="numpy").encode(SWEEP_SMILES)
                rs = Encoder(**kw, depth=depth, backend="rust").encode(SWEEP_SMILES)
                dev = row_scaled_deviation(rs, py)
                if slot == "shallow":
                    shallow = max(shallow or 0.0, dev)
                else:
                    deep = max(deep or 0.0, dev)

        assert shallow < 1e-12, f"depth 3 should stay tight, got {shallow:.2e}"
        # Deep is allowed to be loose, but must not be catastrophic in this
        # normalized configuration -- that would signal a real defect.
        assert deep < 1e-6, f"depth 8 deviated {deep:.2e}, beyond conditioning"
        assert deep > shallow, "expected depth to degrade agreement"
