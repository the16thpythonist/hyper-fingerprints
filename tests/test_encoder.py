"""Tests for the Encoder class."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from hyper_fingerprints import Encoder
from hyper_fingerprints.encoder import _HAS_RUST


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


# ─────────────────────── Composite (dict features) mode ───────────────────────

from hyper_fingerprints import (  # noqa: E402
    FEATURES_CONTINUOUS,
    FEATURES_DEFAULT,
    FEATURES_GRAPH_HDC,
    GRAPH_FEATURES_GRAPH_HDC,
    cosine_similarity,
)

MOLS = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "C"]


class TestCompositeMode:

    def test_list_preset_is_legacy(self):
        enc = Encoder(features=FEATURES_DEFAULT, seed=0)
        assert enc.mode == "legacy"
        ref = Encoder(seed=0)
        assert np.array_equal(enc.encode(MOLS), ref.encode(MOLS))

    @pytest.mark.parametrize("preset", [FEATURES_CONTINUOUS, FEATURES_GRAPH_HDC])
    def test_presets_encode(self, preset):
        enc = Encoder(dimension=128, seed=0, features=preset, normalize=True)
        assert enc.mode == "composite"
        fps = enc.encode(MOLS)
        assert fps.shape == (4, 128)
        assert np.isfinite(fps).all()
        assert enc.encode_joint(MOLS).shape == (4, 256)

    def test_deterministic_and_seed_dependent(self):
        a = Encoder(dimension=128, seed=3, features=FEATURES_CONTINUOUS).encode(MOLS)
        b = Encoder(dimension=128, seed=3, features=FEATURES_CONTINUOUS).encode(MOLS)
        c = Encoder(dimension=128, seed=4, features=FEATURES_CONTINUOUS).encode(MOLS)
        assert np.array_equal(a, b)
        assert not np.allclose(a, c)

    def test_continuous_similarity_is_graded(self):
        # Single carbons differing only in hydrogen count (3 / 2 / 0): every
        # other feature is identical, so only the continuous encoding of
        # "hydrogens" can make CH3 resemble CH2. The legacy categorical encoder
        # gives near-orthogonal vectors for all three.
        mols = ["[CH3]", "[CH2]", "[C]"]
        cont = Encoder(dimension=1024, seed=0, features=FEATURES_CONTINUOUS, normalize=True)
        sim = cosine_similarity(cont.encode(mols), cont.encode(mols))
        assert sim[0, 1] > 0.5
        assert sim[0, 1] > sim[0, 2]
        legacy = Encoder(dimension=1024, seed=0, normalize=True)
        sim_legacy = cosine_similarity(legacy.encode(mols), legacy.encode(mols))
        assert abs(sim_legacy[0, 1]) < 0.2

    @pytest.mark.parametrize("mode,width", [("bundle", 1), ("bind", 1), ("concat", 2)])
    def test_graph_modes(self, mode, width):
        enc = Encoder(
            dimension=128, seed=0, features=FEATURES_GRAPH_HDC,
            graph_features=GRAPH_FEATURES_GRAPH_HDC, graph_mode=mode, normalize=True,
        )
        assert enc.encode(MOLS).shape == (4, 128 * width)
        assert enc.encode_joint(MOLS).shape == (4, 128 * (width + 1))
        plain = Encoder(dimension=128, seed=0, features=FEATURES_GRAPH_HDC, normalize=True)
        if mode == "concat":
            # first block is the unmodified structural embedding
            assert np.array_equal(enc.encode(MOLS)[:, :128], plain.encode(MOLS))
        else:
            assert not np.allclose(enc.encode(MOLS), plain.encode(MOLS))

    def test_bidirectional_flag(self):
        kw = dict(dimension=128, seed=0, features=FEATURES_CONTINUOUS, normalize=True)
        a = Encoder(bidirectional=True, **kw).encode("CCO")
        b = Encoder(bidirectional=False, **kw).encode("CCO")
        assert not np.allclose(a, b)

    def test_unknown_atoms_slot(self):
        with pytest.raises(ValueError):
            Encoder(dimension=64, seed=0, features=FEATURES_CONTINUOUS).encode("C[Si](C)C")
        enc = Encoder(dimension=64, seed=0, features=FEATURES_CONTINUOUS, unknown_atoms="slot")
        assert enc.encode("C[Si](C)C").shape == (1, 64)

    def test_invalid_options(self):
        with pytest.raises(ValueError):
            Encoder(features=FEATURES_CONTINUOUS, graph_mode="stack")
        with pytest.raises(ValueError):
            Encoder(features=FEATURES_CONTINUOUS, unknown_atoms="ignore")
        with pytest.raises(ValueError):
            Encoder(unknown_atoms="slot")  # legacy mode
        with pytest.raises(ValueError):
            Encoder(graph_features=GRAPH_FEATURES_GRAPH_HDC)  # legacy mode
        with pytest.raises(ValueError):
            Encoder(features=FEATURES_CONTINUOUS, graph_features={"graph_size": {"kind": "categorical"}})

    def test_rust_backend_accepted(self):
        """Composite mode runs its message passing on either backend.

        Cross-backend numerical parity is covered in tests/test_rust_backend.py;
        here we only check that the configuration is accepted and encodes.
        """
        kwargs = dict(dimension=64, seed=0, features=FEATURES_CONTINUOUS)
        expected = Encoder(**kwargs, backend="numpy").encode("CCO")
        for backend in ("auto", "rust") if _HAS_RUST else ("auto",):
            out = Encoder(**kwargs, backend=backend).encode("CCO")
            np.testing.assert_allclose(out, expected, atol=1e-10, rtol=1e-11)

    def test_save_load_round_trip(self, tmp_path: Path):
        # seed=None: the loader cannot regenerate anything, so equality proves
        # the arrays were actually written and read back.
        enc = Encoder(
            dimension=128, depth=2, seed=None, features=FEATURES_GRAPH_HDC,
            graph_features=GRAPH_FEATURES_GRAPH_HDC, graph_mode="concat",
            normalize=True, bidirectional=False, unknown_atoms="slot",
        )
        path = tmp_path / "composite.npz"
        enc.save(path)
        loaded = Encoder.load(path)
        assert loaded.mode == "composite"
        assert loaded.graph_mode == "concat"
        assert loaded.bidirectional is False
        assert loaded.unknown_atoms == "slot"
        for name, arr in enc._node_encoder.arrays().items():
            assert np.array_equal(loaded._node_encoder.arrays()[name], arr), name
        for name, arr in enc._graph_encoder.arrays().items():
            assert np.array_equal(loaded._graph_encoder.arrays()[name], arr), name
        assert np.array_equal(loaded.encode(MOLS), enc.encode(MOLS))

    def test_load_rejects_missing_arrays(self, tmp_path: Path):
        enc = Encoder(dimension=64, seed=1, features=FEATURES_CONTINUOUS)
        path = tmp_path / "composite.npz"
        enc.save(path)
        data = dict(np.load(path, allow_pickle=False))
        del data["node/degree"]
        stripped = tmp_path / "stripped.npz"
        np.savez(stripped, **data)
        with pytest.raises(ValueError, match="node/degree"):
            Encoder.load(stripped)

    def test_empty_input(self):
        for enc in (Encoder(dimension=64, seed=0, backend="numpy"),
                    Encoder(dimension=64, seed=0, features=FEATURES_CONTINUOUS)):
            assert enc.encode([]).shape == (0, 64)
            assert enc.encode_joint([]).shape == (0, 128)

    def test_legacy_file_still_loads(self, tmp_path: Path):
        enc = Encoder(dimension=64, seed=1)
        path = tmp_path / "legacy.npz"
        enc.save(path)
        loaded = Encoder.load(path)
        assert loaded.mode == "legacy"
        assert np.array_equal(loaded.encode(MOLS), enc.encode(MOLS))


# ─────────────────────── Batch invariance ───────────────────────


class TestBatchInvariance:
    """encode(x) should equal encode([x, ...])[0].

    Legacy mode violates this for edgeless (single-atom) molecules: an
    all-edgeless batch aggregates to the HRR identity, a mixed batch to zero,
    so the lone result is ``depth + 1`` copies of the node vector. That
    behaviour is pinned here (it is shared with the Rust backend and kept for
    output stability); composite mode is batch-invariant.
    """

    CASES = ["C", "O", "CC.O", "CCO", "c1ccccc1"]

    def test_composite_is_batch_invariant(self):
        enc = Encoder(dimension=64, seed=0, features=FEATURES_CONTINUOUS, normalize=True)
        batch = enc.encode(self.CASES)
        joint = enc.encode_joint(self.CASES)
        for i, smi in enumerate(self.CASES):
            np.testing.assert_allclose(enc.encode(smi), batch[i : i + 1], atol=1e-10, err_msg=smi)
            np.testing.assert_allclose(enc.encode_joint(smi), joint[i : i + 1], atol=1e-10, err_msg=smi)

    def test_legacy_single_atom_discrepancy_is_pinned(self):
        enc = Encoder(dimension=64, depth=3, seed=0, backend="numpy")
        alone = enc.encode("C")
        in_batch = enc.encode(["C", "CC"])[:1]
        node_vec = enc.encode_joint("C")[:, :64]  # order-0 term == node vector
        np.testing.assert_allclose(alone, 4 * node_vec, atol=1e-10)   # depth + 1 copies
        np.testing.assert_allclose(in_batch, node_vec, atol=1e-10)
        # bonded molecules are batch-invariant in legacy mode too
        for smi in ["CC.O", "CCO", "c1ccccc1"]:
            np.testing.assert_allclose(enc.encode(smi), enc.encode([smi, "CC"])[:1], atol=1e-10)


# ─────────────────── Node encoding deduplication ───────────────────

# The Rust node encoder is only a second `base` function for the same dedup
# wrapper, so both are swept where the extension is available.
USE_RUST = [False] + ([True] if _HAS_RUST else [])


class TestNodeEncodingDedup:
    """`encode_unique` computes each distinct feature row once.

    The claim is *exact* equality, not closeness: a hypervector depends on
    nothing but the atom's own feature row, so atoms sharing a row share an
    answer bitwise. Every assertion here is `array_equal`, never `allclose` --
    a tolerance would hide precisely the kind of bug worth catching.
    """

    SMILES = [
        "CCO", "c1ccccc1", "CC(=O)O", "CC(=O)Nc1ccc(O)cc1", "c1ccncc1",
        "CC(=O)Oc1ccccc1C(=O)O", "CS(C)=O", "[NH4+]", "CC(=O)[O-]",
        "C[N+](C)(C)C", "c1cc[nH]c1", "C", "O", "CC.OO",
    ]

    def _both(self, smiles, use_rust, **kwargs):
        enc = Encoder(**kwargs)
        x = enc._prepare_batch_from_list(smiles).x
        ce = enc._node_encoder
        base = ce.encode_rust(x) if use_rust else ce.encode(x)
        return base, ce.encode_unique(x, use_rust=use_rust)

    @pytest.mark.parametrize("use_rust", USE_RUST)
    @pytest.mark.parametrize("features", [FEATURES_CONTINUOUS, FEATURES_GRAPH_HDC])
    @pytest.mark.parametrize("dim", [64, 128, 256])
    @pytest.mark.parametrize("seed", [0, 42])
    def test_matches_row_by_row(self, use_rust, features, dim, seed):
        base, deduped = self._both(
            self.SMILES, use_rust, dimension=dim, depth=2, seed=seed, features=features
        )
        assert np.array_equal(deduped, base)

    @pytest.mark.parametrize("use_rust", USE_RUST)
    def test_unseen_feature_rows(self, use_rust):
        """Nothing is cached between calls, so there is no "unseen row" case.

        These elements appear in no ordinary organic batch and several fall
        through to the `unknown_atoms="slot"` row; their feature rows share
        nothing with the molecules above.
        """
        exotic = [
            "[Se]C(=[Se])[Se]", "[Si](C)(C)(C)C", "[B-](F)(F)(F)F", "C[As](C)C",
            "[Pt](Cl)(Cl)(N)N", "[U+4]", "C[Hg]Cl", "[Fe+3]", "[Te]",
        ]
        base, deduped = self._both(
            exotic, use_rust, dimension=128, depth=2, seed=42,
            features=FEATURES_CONTINUOUS, unknown_atoms="slot",
        )
        assert np.array_equal(deduped, base)

    @pytest.mark.parametrize("use_rust", USE_RUST)
    def test_all_rows_distinct(self, use_rust):
        """Worst case: dedup collapses nothing and must still be exact."""
        enc = Encoder(dimension=128, depth=1, seed=42, features=FEATURES_CONTINUOUS)
        ce = enc._node_encoder
        n = 500
        x = np.zeros((n, len(FEATURES_CONTINUOUS)))
        x[:, 0] = np.arange(n) % 9           # element index, in range
        x[:, 1] = np.arange(n) * 0.001       # continuous, all distinct
        x[:, 2] = np.arange(n) * 0.002       # continuous, all distinct
        assert len(np.unique(x, axis=0)) == n
        base = ce.encode_rust(x) if use_rust else ce.encode(x)
        assert np.array_equal(ce.encode_unique(x, use_rust=use_rust), base)

    @pytest.mark.parametrize("use_rust", USE_RUST)
    def test_all_rows_identical(self, use_rust):
        """Opposite extreme: one distinct row shared by every atom."""
        base, deduped = self._both(
            ["c1ccccc1"] * 50, use_rust, dimension=128, depth=1, seed=42,
            features=FEATURES_CONTINUOUS,
        )
        assert np.array_equal(deduped, base)
        assert len(np.unique(base, axis=0)) == 1

    @pytest.mark.parametrize("use_rust", USE_RUST)
    def test_degenerate_shapes(self, use_rust):
        enc = Encoder(dimension=64, depth=1, seed=42, features=FEATURES_CONTINUOUS)
        ce = enc._node_encoder
        f = len(FEATURES_CONTINUOUS)

        assert ce.encode_unique(np.zeros((0, f)), use_rust=use_rust).shape == (0, 64)

        one = enc._prepare_batch_from_list(["C"]).x
        base = ce.encode_rust(one) if use_rust else ce.encode(one)
        assert np.array_equal(ce.encode_unique(one, use_rust=use_rust), base)

        # 1-d input is promoted to a single row, as `encode` does.
        single = {"element": {"kind": "categorical"}}
        enc1 = Encoder(dimension=64, depth=1, seed=42, features=single)
        ce1 = enc1._node_encoder
        flat = np.array([2.0, 5.0, 2.0])
        base1 = ce1.encode_rust(flat) if use_rust else ce1.encode(flat)
        assert np.array_equal(ce1.encode_unique(flat, use_rust=use_rust), base1)

    @pytest.mark.parametrize("use_rust", USE_RUST)
    @pytest.mark.parametrize("features", [
        {"element": {"kind": "categorical"}},
        {"degree": {"kind": "continuous", "bandwidth": 2.0}},
        {"element": {"kind": "categorical"}, "charge": {"kind": "categorical"},
         "aromatic": {"kind": "categorical"}},
    ])
    def test_feature_shapes(self, use_rust, features):
        base, deduped = self._both(
            self.SMILES, use_rust, dimension=128, depth=1, seed=3, features=features
        )
        assert np.array_equal(deduped, base)

    def test_encoder_output_unchanged(self, monkeypatch):
        """End to end: dedup must not move `Encoder.encode` at all."""
        from hyper_fingerprints.codebook import CompositeEncoder

        kwargs = dict(dimension=256, depth=3, seed=42, features=FEATURES_GRAPH_HDC,
                      graph_features=GRAPH_FEATURES_GRAPH_HDC, backend="numpy")
        with_dedup = Encoder(**kwargs).encode(self.SMILES)

        # Bypass the wrapper so the comparison is against genuine row-by-row work.
        monkeypatch.setattr(
            CompositeEncoder, "encode_unique",
            lambda self, values, *, use_rust=False: self.encode(values),
        )
        without_dedup = Encoder(**kwargs).encode(self.SMILES)
        assert np.array_equal(with_dedup, without_dedup)
