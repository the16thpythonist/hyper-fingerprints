"""Unit-modulus (FHRR) codebooks, the default since 0.5.0.

Every sampled codebook vector keeps the phases of its random draw but has
Fourier magnitudes of one. These tests pin the properties that this buys and
that ``unit_modulus=False`` still reproduces the Gaussian codebooks of 0.4.0.
Agreement with graph_hdc is tested in ``test_graph_hdc_unit_parity.py``.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from hyper_fingerprints import Encoder
from hyper_fingerprints.codebook import (
    CategoricalCodebook,
    CompositeEncoder,
    ContinuousCodebook,
    generate_hermitian_spectrum,
    to_unit_modulus,
    unit_modulus_vectors,
)
from hyper_fingerprints.encoder import _HAS_RUST
from hyper_fingerprints.presets import (
    FEATURES_CONTINUOUS,
    FEATURES_GRAPH_HDC,
    GRAPH_FEATURES_GRAPH_HDC,
)
from hyper_fingerprints.utils import hrr_bind

FIXTURE_DIR = Path(__file__).parent / "fixtures"
DIMS = [1, 2, 3, 8, 64, 255, 256]
SEEDS = [0, 1, 42]
SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1", "c1ccc2[nH]ccc2c1",
    "CCN(CC)CCOC(=O)c1ccc(N)cc1", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "O=C(O)c1ccccc1O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "c1ccc(cc1)C(=O)c1ccccc1", "OCC(O)CO",
    "CCOC(=O)C1=C(C)NC(C)=C(C1c1ccccc1[N+](=O)[O-])C(=O)OC",
]
# Every way of building an encoder from a seed: legacy product codebook,
# graph_hdc preset (categorical + continuous node features, graph features)
# and the recommended continuous preset.
CONFIGS = {
    "legacy": dict(),
    "graph_hdc": dict(features=FEATURES_GRAPH_HDC, graph_features=GRAPH_FEATURES_GRAPH_HDC, normalize=True),
    "continuous": dict(features=FEATURES_CONTINUOUS, normalize=True),
}


def _magnitudes(arr: np.ndarray) -> np.ndarray:
    return np.abs(arr if np.iscomplexobj(arr) else np.fft.fft(arr, axis=-1))


def _sampled_arrays(enc: Encoder) -> dict[str, np.ndarray]:
    """All sampled base arrays of an encoder (rows or spectra)."""
    if enc.mode == "legacy":
        return {"codebook": enc._codebook}
    out = {f"node/{k}": v for k, v in enc._node_encoder.arrays().items()}
    if enc._graph_encoder is not None:
        out.update({f"graph/{k}": v for k, v in enc._graph_encoder.arrays().items()})
    return out


def _participation_ratio(x: np.ndarray) -> np.ndarray:
    """Effective number of Fourier components of each row (power-spectrum PR)."""
    p = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    return p.sum(-1) ** 2 / (p ** 2).sum(-1)


# ───────────────────── The transform ─────────────────────


class TestTransform:
    def test_to_unit_modulus_keeps_phase(self):
        z = np.array([3 + 4j, -2.0 + 0j, 0j, 1e-300 - 1e-300j])
        u = to_unit_modulus(z)
        np.testing.assert_allclose(np.abs(u), 1.0)
        np.testing.assert_allclose(u[:2], [0.6 + 0.8j, -1.0])
        assert u[2] == 1.0  # zero has no phase
        np.testing.assert_allclose(np.angle(u[3]), np.angle(z[3]))

    @pytest.mark.parametrize("dim", DIMS)
    def test_vectors_are_real_unit_norm_unit_modulus(self, dim):
        g = np.random.default_rng(0).standard_normal((5, dim))
        u = unit_modulus_vectors(g)
        assert u.dtype == np.float64 and u.shape == g.shape
        np.testing.assert_allclose(np.abs(np.fft.fft(u, axis=-1)), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(u, axis=-1), 1.0, atol=1e-12)
        # phases of the draw are kept
        gs, us = np.fft.fft(g, axis=-1), np.fft.fft(u, axis=-1)
        np.testing.assert_allclose(us, gs / np.abs(gs), atol=1e-12)

    def test_idempotent(self):
        u = unit_modulus_vectors(np.random.default_rng(3).standard_normal((4, 128)))
        np.testing.assert_allclose(unit_modulus_vectors(u), u, atol=1e-14)


# ───────────────────── Sampled codebooks ─────────────────────


class TestSampledCodebooks:
    @pytest.mark.parametrize("config", list(CONFIGS))
    @pytest.mark.parametrize("dim", DIMS)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_default_is_unit_modulus(self, config, dim, seed):
        enc = Encoder(dimension=dim, depth=2, seed=seed, **CONFIGS[config])
        assert enc.unit_modulus is True
        for name, arr in _sampled_arrays(enc).items():
            np.testing.assert_allclose(_magnitudes(arr), 1.0, atol=1e-12, err_msg=name)
            if not np.iscomplexobj(arr):
                np.testing.assert_allclose(np.linalg.norm(arr, axis=-1), 1.0, atol=1e-12)

    @pytest.mark.parametrize("config", list(CONFIGS))
    def test_gaussian_codebooks_are_not_unit_modulus(self, config):
        enc = Encoder(dimension=256, seed=0, unit_modulus=False, **CONFIGS[config])
        assert enc.unit_modulus is False
        for name, arr in _sampled_arrays(enc).items():
            assert np.abs(_magnitudes(arr) - 1.0).max() > 0.5, name

    @pytest.mark.parametrize("config", list(CONFIGS))
    @pytest.mark.parametrize("seed", SEEDS)
    def test_same_draw_same_phases(self, config, seed):
        """The flag changes the Fourier magnitudes only, never the random draw."""
        unit = _sampled_arrays(Encoder(dimension=128, seed=seed, **CONFIGS[config]))
        gauss = _sampled_arrays(Encoder(dimension=128, seed=seed, unit_modulus=False, **CONFIGS[config]))
        for name in unit:
            gs = gauss[name] if np.iscomplexobj(gauss[name]) else np.fft.fft(gauss[name], axis=-1)
            us = unit[name] if np.iscomplexobj(unit[name]) else np.fft.fft(unit[name], axis=-1)
            np.testing.assert_allclose(us, gs / np.abs(gs), atol=1e-12, err_msg=name)

    @pytest.mark.parametrize("dim", [8, 255, 256])
    def test_continuous_spectrum_stays_hermitian(self, dim):
        cb = ContinuousCodebook(dim, bandwidth=2.0, seed=3)
        k = np.arange(1, dim)
        np.testing.assert_allclose(cb.spectrum[k], np.conj(cb.spectrum[dim - k]), atol=1e-15)
        assert abs(cb.spectrum[0].imag) == 0.0 and abs(cb.spectrum[0]) == 1.0

    def test_codebook_classes_default_to_unit(self):
        assert CategoricalCodebook(64, 3, seed=0).unit_modulus is True
        assert ContinuousCodebook(64, 1.0, seed=0).unit_modulus is True
        cfgs = {"a": {"kind": "categorical", "num_categories": 2}}
        assert CompositeEncoder(64, cfgs, seed=0).unit_modulus is True

    def test_nearly_orthogonal(self):
        dim = 1024
        cb = Encoder(dimension=dim, seed=0)._codebook  # 9*6*3*4*2 = 1296 rows
        sims = cb[:200] @ cb[:200].T
        off = np.abs(sims[~np.eye(200, dtype=bool)])
        assert off.mean() < 1.5 / np.sqrt(dim)
        assert off.max() < 6.0 / np.sqrt(dim)

    def test_preset_codebooks_are_used_verbatim(self):
        """Passed-in arrays are never transformed, Gaussian or not."""
        gauss = Encoder(dimension=64, seed=1, unit_modulus=False)
        enc = Encoder(dimension=64, codebook=gauss._codebook)
        assert np.array_equal(enc._codebook, gauss._codebook)
        cg = Encoder(dimension=64, seed=1, unit_modulus=False, **CONFIGS["graph_hdc"])
        cc = Encoder(dimension=64, **CONFIGS["graph_hdc"],
                     node_codebooks=cg._node_encoder.arrays(),
                     graph_codebooks=cg._graph_encoder.arrays())
        for name, arr in cg._node_encoder.arrays().items():
            assert np.array_equal(cc._node_encoder.arrays()[name], arr)
        for name, arr in cg._graph_encoder.arrays().items():
            assert np.array_equal(cc._graph_encoder.arrays()[name], arr)

    def test_flag_describes_passed_in_arrays(self):
        gauss = Encoder(dimension=64, seed=1, unit_modulus=False)._codebook
        unit = Encoder(dimension=64, seed=1)._codebook
        assert Encoder(dimension=64, codebook=gauss).unit_modulus is False
        assert Encoder(dimension=64, codebook=unit, unit_modulus=False).unit_modulus is True
        cg = Encoder(dimension=64, seed=1, unit_modulus=False, **CONFIGS["graph_hdc"])
        cc = Encoder(dimension=64, **CONFIGS["graph_hdc"],
                     node_codebooks=cg._node_encoder.arrays(),
                     graph_codebooks=cg._graph_encoder.arrays())
        assert cc.unit_modulus is False
        # partially passed in: the missing codebooks are sampled, the flag stays
        partial = {"element": cg._node_encoder.arrays()["element"]}
        assert Encoder(dimension=64, seed=1, **CONFIGS["graph_hdc"],
                       node_codebooks=partial).unit_modulus is True


# ───────────────────── unit_modulus=False reproduces 0.4.0 ─────────────────────


class TestGaussianReproduces040:
    def test_seed_fixture_bit_exact(self):
        """seed_fixtures.npz was recorded with 0.4.0; False must reproduce it exactly."""
        path = FIXTURE_DIR / "seed_fixtures.npz"
        if not path.exists():
            pytest.skip("seed_fixtures.npz not found")
        raw = np.load(path, allow_pickle=False)
        cfg = json.loads(str(raw["config"]))
        legacy = Encoder(dimension=cfg["dimension"], seed=cfg["seed"], unit_modulus=False)
        assert np.array_equal(legacy._codebook, raw["legacy_codebook"])
        comp = Encoder(
            dimension=cfg["dimension"], depth=cfg["depth"], seed=cfg["seed"], normalize=True,
            features=cfg["features"], graph_features=cfg["graph_features"], unit_modulus=False,
        )
        for name, arr in _sampled_arrays(comp).items():
            assert np.array_equal(arr, raw[name]), name

    def test_default_differs_from_040(self):
        path = FIXTURE_DIR / "seed_fixtures.npz"
        if not path.exists():
            pytest.skip("seed_fixtures.npz not found")
        raw = np.load(path, allow_pickle=False)
        cfg = json.loads(str(raw["config"]))
        legacy = Encoder(dimension=cfg["dimension"], seed=cfg["seed"])
        assert not np.allclose(legacy._codebook, raw["legacy_codebook"])


# ───────────────────── What it buys ─────────────────────


class TestProperties:
    def test_binding_preserves_unit_modulus(self):
        """Binding two codebook vectors gives a unit-modulus, unit-norm vector."""
        cb = Encoder(dimension=256, seed=0)._codebook
        bound = hrr_bind(cb[:50], cb[50:100])
        np.testing.assert_allclose(np.abs(np.fft.fft(bound, axis=-1)), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(bound, axis=-1), 1.0, atol=1e-12)
        # Gaussian: the bound spectrum is the product of two random-magnitude
        # spectra, so a few frequencies dominate (compounded at every layer)
        gauss = Encoder(dimension=256, seed=0, unit_modulus=False)._codebook
        gmag = np.abs(np.fft.fft(hrr_bind(gauss[:50], gauss[50:100]), axis=-1))
        assert np.all(gmag.std(axis=-1) > 0.3)  # unit modulus: 0
        assert np.all(gmag.max(axis=-1) > 1.5)

    @pytest.mark.parametrize("dim", [64, 256, 257])
    def test_fpe_values_have_unit_norm(self, dim):
        # exact on every frequency; DC / Nyquist keep only cos(phase * v) of
        # their e^{i phase v}, so the squared norm lies in [(dim - 2) / dim, 1]
        cb = ContinuousCodebook(dim, bandwidth=2.0, seed=5)
        hv = cb.encode(np.linspace(-10.0, 50.0, 61))
        sq = np.sum(hv ** 2, axis=-1)
        assert np.all(sq <= 1.0 + 1e-12) and np.all(sq >= (dim - 2) / dim - 1e-12)
        g = ContinuousCodebook(dim, bandwidth=2.0, seed=5, unit_modulus=False)
        gsq = np.sum(g.encode(np.linspace(-10.0, 50.0, 61)) ** 2, axis=-1)
        assert gsq.max() / gsq.min() > 2.0  # Gaussian: the norm depends on the value

    def test_fpe_similarity_depends_on_difference_only(self):
        dim = 2048
        cb = ContinuousCodebook(dim, bandwidth=2.0, seed=0)
        for delta in (0.5, 1.0, 2.0, 4.0):
            base = np.array([0.0, 3.0, 7.5, 20.0])
            a, b = cb.encode(base), cb.encode(base + delta)
            sims = np.sum(a * b, axis=-1)
            assert np.ptp(sims) < 4.0 / dim

    @pytest.mark.parametrize("config", list(CONFIGS))
    def test_fingerprints_use_many_fourier_components(self, config):
        """The reason for the change: with Gaussian codebooks, repeated binding
        concentrates a fingerprint on a few Fourier components."""
        dim = 1024
        unit = np.concatenate([
            _participation_ratio(Encoder(dimension=dim, depth=2, seed=s, **CONFIGS[config]).encode(SMILES))
            for s in range(3)
        ])
        gauss = np.concatenate([
            _participation_ratio(
                Encoder(dimension=dim, depth=2, seed=s, unit_modulus=False, **CONFIGS[config]).encode(SMILES)
            )
            for s in range(3)
        ])
        assert unit.min() > dim / 20
        assert unit.mean() > 3 * gauss.mean()  # measured: 5x to 50x

    def test_tiny_bandwidth_stays_finite(self):
        """Unit modulus cannot overflow in the node encoding, whatever the bandwidth."""
        feats = {"element": {"kind": "categorical"},
                 "degree": {"kind": "continuous", "bandwidth": 1e-6}}
        enc = Encoder(dimension=64, depth=2, seed=0, features=feats, normalize=True, backend="numpy")
        out = enc.encode(["CCO", "c1ccccc1"])
        assert np.all(np.isfinite(out))
        if _HAS_RUST:
            rs = Encoder(dimension=64, depth=2, seed=0, features=feats, normalize=True,
                         backend="rust").encode(["CCO", "c1ccccc1"])
            assert np.all(np.isfinite(rs))


# ───────────────────── Backends, save / load, validation ─────────────────────


def _row_scaled_deviation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b).max(axis=-1, keepdims=True), 1e-300)))


@pytest.mark.skipif(not _HAS_RUST, reason="Rust extension not built")
@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("unit_modulus", [True, False])
@pytest.mark.parametrize("dim", [3, 64, 255])
@pytest.mark.parametrize("depth", [0, 2, 4])
def test_numpy_and_rust_agree(config, unit_modulus, dim, depth):
    kw = dict(dimension=dim, depth=depth, seed=7, unit_modulus=unit_modulus, **CONFIGS[config])
    py = Encoder(**kw, backend="numpy").encode(SMILES)
    rs = Encoder(**kw, backend="rust").encode(SMILES)
    assert _row_scaled_deviation(rs, py) < 1e-9


@pytest.mark.skipif(not _HAS_RUST, reason="Rust extension not built")
@pytest.mark.parametrize("unit_modulus", [True, False])
@pytest.mark.parametrize("graph_mode", ["bundle", "bind", "concat"])
@pytest.mark.parametrize("bidirectional", [True, False])
def test_numpy_and_rust_agree_graph_options(unit_modulus, graph_mode, bidirectional):
    kw = dict(dimension=128, depth=3, seed=3, unit_modulus=unit_modulus, graph_mode=graph_mode,
              bidirectional=bidirectional, unknown_atoms="slot", **CONFIGS["graph_hdc"])
    py = Encoder(**kw, backend="numpy").encode(SMILES)
    rs = Encoder(**kw, backend="rust").encode(SMILES)
    assert _row_scaled_deviation(rs, py) < 1e-9


class TestSaveLoad:
    @pytest.mark.parametrize("config", list(CONFIGS))
    @pytest.mark.parametrize("unit_modulus", [True, False])
    def test_roundtrip(self, config, unit_modulus, tmp_path):
        enc = Encoder(dimension=128, depth=2, seed=4, unit_modulus=unit_modulus, **CONFIGS[config])
        enc.save(tmp_path / "enc.npz")
        loaded = Encoder.load(tmp_path / "enc.npz")
        assert loaded.unit_modulus is unit_modulus
        assert np.array_equal(loaded.encode(SMILES), enc.encode(SMILES))

    @pytest.mark.parametrize("config", ["legacy", "graph_hdc"])
    def test_file_from_040_loads_as_gaussian(self, config, tmp_path):
        """A file saved before 0.5.0 has no unit_modulus key and Gaussian arrays."""
        enc = Encoder(dimension=64, depth=2, seed=4, unit_modulus=False, **CONFIGS[config])
        enc.save(tmp_path / "old.npz")
        data = dict(np.load(tmp_path / "old.npz", allow_pickle=False))
        cfg = json.loads(str(data["config"]))
        del cfg["unit_modulus"]
        data["config"] = np.array(json.dumps(cfg))
        np.savez(tmp_path / "old.npz", **data)

        loaded = Encoder.load(tmp_path / "old.npz")
        assert loaded.unit_modulus is False
        assert np.array_equal(loaded.encode(SMILES), enc.encode(SMILES))

    def test_wrong_flag_in_file_is_corrected_from_arrays(self, tmp_path):
        enc = Encoder(dimension=64, depth=2, seed=4, unit_modulus=False, **CONFIGS["graph_hdc"])
        enc.save(tmp_path / "f.npz")
        data = dict(np.load(tmp_path / "f.npz", allow_pickle=False))
        cfg = json.loads(str(data["config"]))
        cfg["unit_modulus"] = True
        data["config"] = np.array(json.dumps(cfg))
        np.savez(tmp_path / "f.npz", **data)
        loaded = Encoder.load(tmp_path / "f.npz")
        assert loaded.unit_modulus is False
        assert np.array_equal(loaded.encode(SMILES), enc.encode(SMILES))

    def test_old_graph_hdc_export_imports_as_gaussian(self):
        path = FIXTURE_DIR / "legacy_codebook_graph_hdc.npz"
        if not path.exists():
            pytest.skip("legacy codebook fixture not found")
        assert Encoder.from_legacy_codebook(path).unit_modulus is False


@pytest.mark.parametrize("bad", ["yes", 1, None])
def test_unit_modulus_must_be_bool(bad):
    with pytest.raises(ValueError, match="unit_modulus"):
        Encoder(dimension=64, unit_modulus=bad)
