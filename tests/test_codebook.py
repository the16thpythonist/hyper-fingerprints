"""Tests for the per-feature (composite) codebooks."""

import numpy as np
import pytest

from hyper_fingerprints.codebook import (
    CategoricalCodebook,
    CompositeEncoder,
    ContinuousCodebook,
    generate_hermitian_spectrum,
    to_unit_modulus,
)
from hyper_fingerprints.utils import hrr_bind

DIM = 256


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class TestHermitianSpectrum:
    @pytest.mark.parametrize("dim", [DIM, DIM + 1])
    def test_ifft_is_real_unit_norm(self, dim):
        spec = generate_hermitian_spectrum(dim, np.random.default_rng(0))
        sig = np.fft.ifft(spec)
        assert np.allclose(sig.imag, 0.0, atol=1e-12)
        assert np.isclose(np.linalg.norm(sig.real), 1.0)

    def test_seed_reproducible(self):
        a = generate_hermitian_spectrum(DIM, np.random.default_rng(7))
        b = generate_hermitian_spectrum(DIM, np.random.default_rng(7))
        c = generate_hermitian_spectrum(DIM, np.random.default_rng(8))
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)


class TestContinuousCodebook:
    def test_zero_is_identity(self):
        cb = ContinuousCodebook(DIM, bandwidth=2.0, seed=0)
        hv = cb.encode(np.array([0.0]))[0]
        expected = np.zeros(DIM)
        expected[0] = 1.0
        assert np.allclose(hv, expected)

    def test_similarity_decays_with_distance(self):
        cb = ContinuousCodebook(DIM, bandwidth=2.0, seed=0)
        hv = cb.encode(np.array([2.0, 2.5, 3.0, 4.0, 8.0]))
        sims = [_cos(hv[0], h) for h in hv[1:]]
        assert sims[0] > sims[1] > sims[2] > sims[3]
        assert sims[0] > 0.8
        assert abs(sims[3]) < 0.3

    def test_binding_adds_values(self):
        # Only approximately additive: the (real, possibly negative) DC and
        # Nyquist spectral components turn complex under fractional powers
        # and their imaginary parts are discarded -- inherited from graph_hdc.
        # The Gaussian spectrum damps that error by |r| ** (v / bandwidth) of
        # those two coefficients, which happens to be small for this seed.
        cb = ContinuousCodebook(DIM, bandwidth=1.0, seed=1, unit_modulus=False)
        a, b, ab = cb.encode(np.array([1.5, 2.25, 3.75]))
        assert np.allclose(hrr_bind(a[None], b[None])[0], ab, atol=1e-4)

    def test_binding_adds_values_unit_modulus(self):
        # Unit modulus: exactly additive on every frequency except DC and
        # Nyquist, whose discarded imaginary parts now have magnitude <= 1,
        # i.e. at most 1/DIM per entry each.
        cb = ContinuousCodebook(DIM, bandwidth=1.0, seed=1)
        a, b, ab = cb.encode(np.array([1.5, 2.25, 3.75]))
        bound = hrr_bind(a[None], b[None])[0]
        spec_bound, spec_ab = np.fft.fft(bound), np.fft.fft(ab)
        inner = np.ones(DIM, dtype=bool)
        inner[[0, DIM // 2]] = False
        assert np.allclose(spec_bound[inner], spec_ab[inner], atol=1e-10)
        assert np.max(np.abs(bound - ab)) <= 4.0 / DIM

    def test_preset_spectrum_roundtrip(self):
        cb = ContinuousCodebook(DIM, bandwidth=1.0, seed=1)
        cb2 = ContinuousCodebook(DIM, bandwidth=1.0, spectrum=cb.spectrum)
        vals = np.array([0.3, 1.7])
        assert np.array_equal(cb.encode(vals), cb2.encode(vals))

    def test_bad_spectrum_shape(self):
        with pytest.raises(ValueError):
            ContinuousCodebook(DIM, bandwidth=1.0, spectrum=np.ones(DIM + 1))


class TestCategoricalCodebook:
    def test_lookup_and_scale(self):
        cb = CategoricalCodebook(DIM, 5, seed=0)
        hv = cb.encode(np.array([0, 4, 0]))
        assert hv.shape == (3, DIM)
        assert np.array_equal(hv[0], hv[2])
        assert not np.array_equal(hv[0], hv[1])
        # Normal(0, 1/sqrt(dim)) rows have approximately unit norm
        assert np.allclose(np.linalg.norm(cb.vectors, axis=1), 1.0, atol=0.2)

    def test_bad_vectors_shape(self):
        with pytest.raises(ValueError):
            CategoricalCodebook(DIM, 5, vectors=np.ones((4, DIM)))


class TestCompositeEncoder:
    CFGS = {
        "element": {"kind": "categorical", "num_categories": 9},
        "degree": {"kind": "continuous", "bandwidth": 2.0},
        "aromatic": {"kind": "categorical", "num_categories": 2},
    }

    def test_matches_pairwise_bind(self):
        enc = CompositeEncoder(DIM, self.CFGS, seed=0)
        x = np.array([[1, 3.0, 0], [2, 1.0, 1]])
        out = enc.encode(x)
        cbs = list(enc.codebooks.values())
        expected = cbs[0].encode(x[:, 0])
        expected = hrr_bind(expected, cbs[1].encode(x[:, 1]))
        expected = hrr_bind(expected, cbs[2].encode(x[:, 2]))
        assert out.shape == (2, DIM)
        assert np.allclose(out, expected, atol=1e-10)

    def test_seed_offsets_and_reproducibility(self):
        a = CompositeEncoder(DIM, self.CFGS, seed=5)
        b = CompositeEncoder(DIM, self.CFGS, seed=5)
        for name in a.names:
            assert np.array_equal(a.codebooks[name].array, b.codebooks[name].array)
        # explicit offsets: continuous uses seed + offset + 1
        cfgs = {"degree": {"kind": "continuous", "bandwidth": 2.0, "seed_offset": 20}}
        c = CompositeEncoder(DIM, cfgs, seed=5)
        ref = generate_hermitian_spectrum(DIM, np.random.default_rng(5 + 20 + 1))
        assert np.array_equal(c.codebooks["degree"].spectrum, to_unit_modulus(ref))
        d = CompositeEncoder(DIM, cfgs, seed=5, unit_modulus=False)
        assert np.array_equal(d.codebooks["degree"].spectrum, ref)

    def test_arrays_roundtrip(self):
        a = CompositeEncoder(DIM, self.CFGS, seed=0)
        b = CompositeEncoder(DIM, self.CFGS, seed=None, codebooks=a.arrays())
        x = np.array([[0, 2.0, 1]])
        assert np.array_equal(a.encode(x), b.encode(x))

    def test_empty_input(self):
        enc = CompositeEncoder(DIM, self.CFGS, seed=0)
        assert enc.encode(np.zeros((0, 3))).shape == (0, DIM)

    def test_unknown_kind(self):
        with pytest.raises(ValueError):
            CompositeEncoder(DIM, {"x": {"kind": "fuzzy"}})
