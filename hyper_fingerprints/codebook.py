"""
Feature encoder: maps discrete feature tuples to hypervectors via codebooks.
"""

from __future__ import annotations

import numpy as np

from hyper_fingerprints.utils import TupleIndexer


# ───────────────────── Unit-modulus (FHRR) transform ─────────────────────


def to_unit_modulus(spectrum: np.ndarray) -> np.ndarray:
    """Set every complex Fourier coefficient to magnitude one, keeping its phase.

    Coefficients that are exactly zero have no phase and become one. A
    Hermitian-symmetric spectrum stays Hermitian-symmetric, so the
    corresponding time-domain vector stays real. Port of
    ``graph_hdc.utils.to_unit_modulus``.
    """
    spectrum = np.asarray(spectrum, dtype=np.complex128)
    magnitude = np.abs(spectrum)
    nonzero = magnitude > 0
    unit = spectrum / np.where(nonzero, magnitude, 1.0)
    return np.where(nonzero, unit, 1.0 + 0.0j)


def unit_modulus_vectors(vectors: np.ndarray) -> np.ndarray:
    """Real vectors ``[..., D]`` -> same phases, all Fourier magnitudes one.

    The result is a unitary HRR vector (FHRR seen through the FFT): its
    Euclidean norm is exactly one (Parseval), and binding with it by
    circular convolution preserves norms instead of concentrating the
    bound vector on a few Fourier components.
    """
    spectrum = np.fft.fft(np.asarray(vectors, dtype=np.float64), axis=-1)
    return np.fft.ifft(to_unit_modulus(spectrum), axis=-1).real


def is_unit_modulus(arr: np.ndarray, *, atol: float = 1e-9) -> bool:
    """True if every Fourier coefficient of ``arr`` has magnitude one.

    ``arr`` is either real vectors ``[..., D]`` (transformed along the last
    axis) or a complex spectrum, as stored by the codebooks.
    """
    arr = np.asarray(arr)
    spectrum = arr if np.iscomplexobj(arr) else np.fft.fft(arr, axis=-1)
    return bool(np.allclose(np.abs(spectrum), 1.0, rtol=0.0, atol=atol))


class FeatureEncoder:
    """Encodes multi-dimensional feature tuples into HRR hypervectors.

    With ``unit_modulus=True`` (default) every codebook row is the Gaussian
    draw with its Fourier magnitudes set to one (see
    :func:`unit_modulus_vectors`); ``False`` keeps the L2-normalized Gaussian
    rows of versions before 0.5.0. Both use the same random draw.
    """

    def __init__(
        self,
        dim: int,
        num_categories: int,
        indexer: TupleIndexer,
        *,
        seed: int | None = None,
        codebook: np.ndarray | None = None,
        unit_modulus: bool = True,
    ) -> None:
        self.dim = dim
        self.num_categories = num_categories
        self.indexer = indexer
        self.unit_modulus = unit_modulus

        if codebook is not None:
            self.codebook = codebook
        else:
            rng = np.random.default_rng(seed)
            self.codebook = self._generate_codebook(rng)

    def _generate_codebook(self, rng: np.random.Generator) -> np.ndarray:
        cb = rng.standard_normal((self.num_categories, self.dim))
        if self.unit_modulus:
            return unit_modulus_vectors(cb)
        norms = np.linalg.norm(cb, axis=-1, keepdims=True)
        cb = cb / norms
        return cb

    def encode_indices(self, data: np.ndarray) -> np.ndarray:
        """Map feature tuples ``[N, F]`` to flat codebook indices ``[N]``."""
        data = data.astype(np.int64)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        tup = list(map(tuple, data.tolist()))
        return np.array(self.indexer.get_idxs(tup), dtype=np.int64)

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode feature tuples ``[N, F]`` into hypervectors ``[N, D]``."""
        idxs = self.encode_indices(data)
        return self.codebook[idxs]


# ───────────────────── Per-feature codebooks (composite mode) ─────────────────────


def generate_hermitian_spectrum(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Random Hermitian-symmetric spectrum whose inverse FFT is real with unit norm.

    Port of ``graph_hdc.utils.generate_hermitian_symmetric_vector``: for even
    ``dim`` identical seeds produce identical spectra in both libraries. (The
    original mis-sizes the negative-frequency half for odd ``dim``; this
    version mirrors it correctly.)
    """
    half = dim // 2 + 1
    pos = rng.normal(size=half) + 1j * rng.normal(size=half)
    pos[0] = rng.normal(size=1)[0]
    if dim % 2 == 0:
        pos[-1] = rng.normal(size=1)[0]
        neg = np.conj(pos[-2:0:-1])
    else:
        neg = np.conj(pos[-1:0:-1])
    spectrum = np.zeros(dim, dtype=np.complex128)
    spectrum[:half] = pos
    spectrum[half:] = neg
    norm = np.linalg.norm(np.fft.ifft(spectrum))
    return spectrum / norm


class CategoricalCodebook:
    """One random unit-scale hypervector per category.

    Rows are drawn from ``Normal(0, 1/sqrt(dim))`` -- the same distribution
    graph_hdc uses for its categorical encoders. With ``unit_modulus=True``
    (default, as graph_hdc's ``AtomEncoder``) the Fourier magnitudes of each
    drawn row are then set to one; ``False`` keeps the Gaussian rows.
    """

    kind = "categorical"

    def __init__(
        self,
        dim: int,
        num_categories: int,
        *,
        seed: int | None = None,
        vectors: np.ndarray | None = None,
        unit_modulus: bool = True,
    ) -> None:
        self.dim = dim
        self.num_categories = num_categories
        self.unit_modulus = unit_modulus
        if vectors is not None:
            vectors = np.asarray(vectors, dtype=np.float64)
            if vectors.shape != (num_categories, dim):
                raise ValueError(
                    f"codebook shape {vectors.shape} != ({num_categories}, {dim})"
                )
            self.vectors = vectors
        else:
            rng = np.random.default_rng(seed)
            self.vectors = rng.normal(0.0, 1.0 / np.sqrt(dim), size=(num_categories, dim))
            if unit_modulus:
                self.vectors = unit_modulus_vectors(self.vectors)

    @property
    def array(self) -> np.ndarray:
        return self.vectors

    def encode(self, values: np.ndarray) -> np.ndarray:
        """Map integer category values ``[N]`` to hypervectors ``[N, D]``."""
        idx = np.asarray(values).astype(np.int64)
        return self.vectors[idx]


class ContinuousCodebook:
    """FHRR fractional-power encoder for real-valued features.

    A value ``v`` is encoded as ``ifft(spectrum ** (v / bandwidth)).real``.
    Nearby values yield similar hypervectors; ``encode(a) (*) encode(b) ==
    encode(a + b)`` under circular convolution.

    With ``unit_modulus=True`` (default, as graph_hdc's ``ContinuousEncoder``)
    every coefficient of the base spectrum has magnitude one and only its
    random phase is kept, so a value only rotates the phases: every encoded
    value has unit norm and the similarity of two values depends only on
    their difference (a sinc kernel in expectation). ``False`` keeps the
    complex Gaussian spectrum, whose random magnitudes are raised to the
    power ``v / bandwidth`` as well. Both use the same random draw.
    """

    kind = "continuous"

    def __init__(
        self,
        dim: int,
        bandwidth: float,
        *,
        seed: int | None = None,
        spectrum: np.ndarray | None = None,
        unit_modulus: bool = True,
    ) -> None:
        self.dim = dim
        self.bandwidth = float(bandwidth)
        self.unit_modulus = unit_modulus
        if spectrum is not None:
            spectrum = np.asarray(spectrum, dtype=np.complex128)
            if spectrum.shape != (dim,):
                raise ValueError(f"spectrum shape {spectrum.shape} != ({dim},)")
            self.spectrum = spectrum
        else:
            rng = np.random.default_rng(seed)
            self.spectrum = generate_hermitian_spectrum(dim, rng)
            if unit_modulus:
                self.spectrum = to_unit_modulus(self.spectrum)

    @property
    def array(self) -> np.ndarray:
        return self.spectrum

    def encode(self, values: np.ndarray) -> np.ndarray:
        """Map real values ``[N]`` to hypervectors ``[N, D]``."""
        exponent = (np.asarray(values, dtype=np.float64) / self.bandwidth)[:, np.newaxis]
        return np.fft.ifft(self.spectrum[np.newaxis, :] ** exponent).real


class CompositeEncoder:
    """Binds independent per-feature hypervectors into one vector per row.

    ``feature_cfgs`` is an ordered ``{name: cfg}`` mapping where ``cfg`` has
    ``kind`` (``"categorical"`` | ``"continuous"``), and for categorical
    ``num_categories``, for continuous ``bandwidth``. ``seed_offset`` is
    optional and defaults to ``10 * position``. Continuous codebooks are
    seeded with ``seed + offset + 1`` (mirroring graph_hdc's
    ``ContinuousEncoder``). ``unit_modulus`` is passed to every codebook that
    is sampled here; preset ``codebooks`` are always used as given.

    Encoding is a single FFT product over all feature hypervectors followed
    by one inverse FFT, numerically matching
    ``graph_hdc.binding._circular_convolution_fft``.
    """

    def __init__(
        self,
        dim: int,
        feature_cfgs: dict[str, dict],
        *,
        seed: int | None = None,
        codebooks: dict[str, np.ndarray] | None = None,
        unit_modulus: bool = True,
    ) -> None:
        self.dim = dim
        self.feature_cfgs = feature_cfgs
        self.unit_modulus = unit_modulus
        self.codebooks: dict[str, CategoricalCodebook | ContinuousCodebook] = {}

        for position, (name, cfg) in enumerate(feature_cfgs.items()):
            offset = int(cfg.get("seed_offset", 10 * position))
            preset = None if codebooks is None else codebooks.get(name)
            kind = cfg["kind"]
            if kind == "categorical":
                self.codebooks[name] = CategoricalCodebook(
                    dim,
                    int(cfg["num_categories"]),
                    seed=None if seed is None else seed + offset,
                    vectors=preset,
                    unit_modulus=unit_modulus,
                )
            elif kind == "continuous":
                self.codebooks[name] = ContinuousCodebook(
                    dim,
                    float(cfg["bandwidth"]),
                    seed=None if seed is None else seed + offset + 1,
                    spectrum=preset,
                    unit_modulus=unit_modulus,
                )
            else:
                raise ValueError(
                    f"Unknown feature kind {kind!r} for {name!r}; "
                    "expected 'categorical' or 'continuous'"
                )

    @property
    def names(self) -> list[str]:
        return list(self.codebooks)

    def arrays(self) -> dict[str, np.ndarray]:
        """Raw base arrays per feature (for save / export)."""
        return {name: cb.array for name, cb in self.codebooks.items()}

    def encode(self, values: np.ndarray) -> np.ndarray:
        """Encode feature matrix ``[N, F]`` into hypervectors ``[N, D]``."""
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, np.newaxis]
        n = values.shape[0]
        if n == 0:
            return np.zeros((0, self.dim), dtype=np.float64)
        spec = np.ones((n, self.dim), dtype=np.complex128)
        for col, cb in enumerate(self.codebooks.values()):
            spec *= np.fft.fft(cb.encode(values[:, col]), axis=-1)
        return np.fft.ifft(spec, axis=-1).real

    def encode_rust(self, values: np.ndarray) -> np.ndarray:
        """Rust-accelerated :meth:`encode`; identical result, same arithmetic.

        The codebooks are sampled here and passed across; nothing is drawn on
        the Rust side. Agreement with :meth:`encode` is ~1 ULP -- the only
        deviation is that Rust always evaluates ``z ** e`` as
        ``exp(e * ln z)``, whereas NumPy switches to repeated multiplication
        for real integer exponents with ``|e| < 100``. See
        ``tests/test_rust_backend.py::TestCompositeNodeEncoding``.
        """
        from hyper_fingerprints._core import composite_encode_rs

        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, np.newaxis]
        if values.shape[0] == 0:
            return np.zeros((0, self.dim), dtype=np.float64)

        kinds, categorical, spectra, bandwidths = [], [], [], []
        for cb in self.codebooks.values():
            if cb.kind == "categorical":
                kinds.append(0)
                categorical.append(np.ascontiguousarray(cb.vectors, dtype=np.float64))
            else:
                kinds.append(1)
                spectra.append(np.ascontiguousarray(cb.spectrum, dtype=np.complex128))
                bandwidths.append(float(cb.bandwidth))

        return composite_encode_rs(
            np.ascontiguousarray(values),
            kinds,
            categorical,
            spectra,
            bandwidths,
        )

    def encode_unique(self, values: np.ndarray, *, use_rust: bool = False) -> np.ndarray:
        """Encode ``[N, F]`` into ``[N, D]``, computing each distinct row once.

        Bit-identical to :meth:`encode` / :meth:`encode_rust`, not merely close:
        a row's hypervector depends on nothing but the row itself, so atoms that
        share a feature row share an answer exactly, and handing them a copy is
        the same arithmetic rather than an approximation of it.

        Molecules are built from a small alphabet of atom environments, so rows
        repeat heavily -- 9956 atoms of 512 PubChem molecules carry only 51
        distinct rows, and all six atoms of benzene share one. Nothing is cached
        between calls: ``np.unique`` runs over whatever rows are present, so a
        row never seen before is simply encoded from scratch. The number of
        distinct rows only affects speed, never the result; an input whose rows
        are all distinct costs ~4% more than :meth:`encode` for the sort.
        """
        base = self.encode_rust if use_rust else self.encode

        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, np.newaxis]
        if values.shape[0] < 2:
            return base(values)

        uniq, inv = np.unique(values, axis=0, return_inverse=True)
        # ravel() is load-bearing: NumPy 2.0 changed the shape of `inverse` for
        # axis-based calls, and this package supports numpy>=1.24. It is a no-op
        # on the 1-d result other versions return.
        return base(uniq)[inv.ravel()]
