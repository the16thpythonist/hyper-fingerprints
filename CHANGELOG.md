# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.4.0] - 2026-09-15

### Added

- The composite (dict-features) encoder now runs its message passing on the
  Rust backend. `backend="rust"` is accepted for every composite
  configuration — all feature kinds, `normalize`, `bidirectional`,
  `unknown_atoms="slot"` and all three `graph_mode`s — and `backend="auto"`
  uses Rust when the extension is available. There is no fallback path.
- New `_core.encode_batch_hv_rs`: the message-passing loop over pre-computed
  node hypervectors. `encode_batch_rs` now gathers its codebook rows and
  delegates to the same kernel, so legacy output is unchanged. The two
  semantics that differ between the modes are parameters: `norm_mode`
  (`"legacy"` = `x / (|x| + 1e-8)` vs `"l2"` = `x / max(|x|, 1e-12)`) and
  `empty_identity` (edgeless-batch scatter to the HRR identity vs to zeros).
- `tests/test_rust_backend.py::TestCompositeCrossBackend`: cross-backend parity
  across the composite configuration matrix, asserted at
  `atol=1e-10, rtol=1e-11` (unnormalized composite embeddings reach ~1e6 at
  depth 3, so an absolute-only tolerance is not meaningful there; observed
  worst-case relative deviation is ~6e-12).

- The composite node encoding (`CompositeEncoder.encode`) now also runs in Rust
  via `_core.composite_encode_rs`: per-feature categorical lookups and FHRR
  fractional powers, bound with one spectral product. All codebook sampling
  stays in NumPy — the sampled tables and spectra are passed across the
  boundary — so seeded reproducibility is unaffected.
- `tests/test_rust_backend.py::TestCompositeNodeEncoding`: compares node
  hypervectors directly (before message passing can smear a discrepancy)
  across both presets x 7 dimensions x 4 seeds, plus bandwidth sweeps,
  all-categorical and single-feature configurations, `unknown_atoms="slot"`,
  and out-of-range index handling.

- Composite node encoding computes each **distinct feature row** once
  (`CompositeEncoder.encode_unique`) rather than once per atom. A hypervector
  depends on nothing but the atom's own feature row, so atoms sharing a row
  share an answer bitwise — the output is `array_equal`, not merely close.
  Molecules are built from a small alphabet of atom environments: 9956 atoms
  of 512 PubChem molecules carry only 51 distinct rows, and all six atoms of
  benzene share one.
- `tests/test_rust_backend.py::TestCompositeSweep`: sweeps the axes the suite
  previously pinned -- every seed x dimension (including 3 and 8), depth out to
  6, batch sizes 1-200, bandwidths 0.25-1e6, and the cross product of
  `graph_mode` x `bidirectional` x `unknown_atoms` -- plus a deterministic
  pseudo-random sweep over the whole option space, which is the shape of test
  that found the crashes above.
- `tests/test_rust_backend.py::TestNumericalLimits`: pins overflow as a
  catchable `ValueError` at both sites, that a failed call leaves the process
  usable, and the two regimes where the backends genuinely part company --
  depth > 6 (message passing is ill-conditioned, so agreement degrades from
  ~1e-13 to ~1e-9 by depth 8 and becomes arbitrary near the float64 ceiling)
  and a bandwidth far below the value range (~5.6e-7, where magnitudes reach
  1e63 and the result is meaningless on either backend). Depth 6 is documented
  as the supported range for tight agreement; through it the worst measured
  deviation over 100 real PubChem molecules, dims 64-512, both feature kinds
  and both `normalize` settings is 4.6e-13.
- `tests/test_encoder.py::TestNodeEncodingDedup`: asserts exact equality (never
  a tolerance) across both presets x dimensions x seeds x both backends, plus
  exotic elements whose rows appear in no ordinary batch, the all-distinct and
  all-identical extremes, degenerate shapes, and an end-to-end check that
  `Encoder.encode` does not move.

### Fixed

- Overflow in the Rust backend raised `pyo3_runtime.PanicException` and printed
  a panic per affected row to stderr. That type derives from `BaseException`,
  not `Exception`, so a caller's `try: enc.encode(chunk) except Exception:`
  loop did not catch it and the whole job died. Both FFT sites now return an
  error that surfaces as a catchable `ValueError` naming the likely cause.
  Two ways in: a `bandwidth` far below the feature's value range (new with the
  composite node encoder), and large `depth` with `normalize=False` (reachable
  in legacy mode too, so pre-existing). The NumPy backend still returns
  inf/NaN in these situations; the asymmetry is documented rather than changed.

### Changed

- Composite cross-backend parity is now measured as deviation scaled by each
  **row's own magnitude**, replacing a global `atol=1e-10, rtol=1e-11`. The old
  pair was not a meaningful statement about agreement: unnormalized composite
  embeddings reach ~1e7 at depth 3 and ~1e15 at depth 4 and the magnitude moves
  with the seed, so it consumed 99% of its budget at seed 42 / dim 128 /
  depth 3 and would have **failed** at seed 7 (7.6x over) or seed 1 / depth 4
  (16.5x over). The suite was green by seed luck. Row-scaled deviation is
  stable: worst 4.1e-13 across dims {3,8,64,128,256,512} x seeds {0,1,7,42,123}
  x depth 0-3 x normalize {True,False}, so the new 1e-9 bound has ~400x
  headroom.
- `Cargo.toml`: `extension-module` is now an optional feature, on by default.
  `cargo test` could not link before this — the Rust unit tests now run with
  `cargo test --no-default-features` (with libpython on `LD_LIBRARY_PATH`).

### Notes

- Composite molecules are still prepared with RDKit: `prepare_batch_rs` only
  emits flat product-codebook indices, so the Rust SMILES parser covers the
  legacy feature set only.
- Measured on 512 PubChem molecules (9956 atoms, dim=256, depth=2). Node
  encoding: 489 ms -> 12.6 ms on NumPy by dedup alone (~39x), and 25.7 ms ->
  10.6 ms on Rust. Message passing: ~280 ms -> ~12 ms in Rust (~24x).
  End-to-end: 845 ms -> 370 ms (NumPy backend) and -> 111 ms (Rust backend),
  of which RDKit preparation (~81 ms, not ported) is now the largest part.
- Note that after deduplication the Rust node encoder buys little on typical
  molecular batches (12.6 ms NumPy vs 10.6 ms Rust) — with only 51 distinct
  rows the remaining cost is mostly the `np.unique` sort. It still matters
  when rows do not repeat: on a synthetic batch of 9956 all-distinct rows the
  split is 888 ms (NumPy) vs 39 ms (Rust).
- Rust evaluates `z ** e` as `exp(e * ln z)` throughout, while NumPy switches
  to repeated multiplication for real integer exponents with `|e| < 100`.
  Replicating that dispatch was measured and rejected as unnecessary: worst
  observed node-hypervector deviation across 56 configurations is 1.3e-15.
  The composite fixtures in `tests/test_regression.py` (`atol=1e-10, rtol=0`)
  and the graph_hdc reference in `tests/test_legacy_parity.py` pass unchanged
  through the Rust path.
- The continuous branch requires a *full complex* inverse FFT (`rustfft`, now a
  direct dependency), not `realfft`'s c2r: the spectrum's DC bin — and, for
  even `dim`, its Nyquist bin — is real and often negative, and raising a
  negative real to a non-integer power breaks the Hermitian symmetry that c2r
  assumes.
- Deduplication caches nothing between calls: `np.unique` runs over whatever
  rows are present, so a row never seen before is encoded from scratch and
  there is no "unseen data" failure mode. The number of distinct rows affects
  speed only — an input whose rows are all distinct costs ~4% more than
  encoding them directly, for the sort.

## [0.3.1] - 2026-09-11

### Fixed

- Invalid configurations now raise `ValueError` at construction with a message
  naming the parameter (`dimension < 1`, `depth < 0`, negative/non-int `seed`,
  non-bool flags, unknown `backend`, empty/duplicate/unknown `atom_types` or
  `features`, `bandwidth <= 0` or missing, unknown feature-config keys,
  continuous `element`, wrong-shaped `codebook`). Previously some surfaced
  late (`depth=-1` → `OverflowError` at encode), silently (`backend="foo"`
  meant NumPy, `bandwidth=0` produced NaNs) or as a Rust panic that killed
  the process (wrong-shaped `codebook`).
- Input handling is now identical across backends and modes: empty SMILES,
  zero-atom `Mol`s and unsanitized `Mol`s raise `ValueError`; explicit
  hydrogens (`[H]C([H])([H])[H]`) are normalized by RDKit before the Rust
  parser so they match the implicit form.
- `Encoder.load(path, backend=...)`: the backend is a runtime preference and
  is not stored in the file.

### Added

- `tests/test_public_api.py`: sweeps the exported API over parameter
  combinations (shape/dtype/finiteness, save/load identity, Rust == NumPy in
  legacy mode) and pins the error contract for invalid configurations and
  inputs.

### Notes

- Known difference: the Rust SMILES parser does not validate valences, so a
  chemically impossible SMILES like `C[C](C)(C)(C)(C)C` is accepted on the
  Rust path but rejected by RDKit. Pinned in `tests/test_public_api.py`.

## [0.3.0] - 2026-09-11

### Added

- **Composite (per-feature) encoder** with FHRR continuous features, selected
  by passing `features` as a `{name: cfg}` dict. Each feature gets its own
  codebook (`"categorical"` lookup or `"continuous"` fractional-power encoding
  with a `bandwidth`); the per-feature hypervectors are bound by circular
  convolution. Ported from `graph_hdc`; the legacy list-based product-codebook
  encoder and its outputs are unchanged.
- Presets: `FEATURES_DEFAULT`, `FEATURES_CONTINUOUS`, `FEATURES_GRAPH_HDC`,
  `GRAPH_FEATURES_GRAPH_HDC`.
- Graph-level features (`graph_size`, `graph_diameter`, `graph_total_charge`)
  via `graph_features=` with `graph_mode="bundle" | "bind" | "concat"`.
- New `Encoder` options `bidirectional` (default `True`) and
  `unknown_atoms="error" | "slot"`; new atom feature `implicit_hydrogens`.
- `Encoder.from_legacy_codebook(path)` imports a codebook exported by
  `graph_hdc.HyperNet.export_legacy_codebook`, reproducing graph_hdc
  embeddings to ~1e-15 (node features) / ~1e-8 (graph features, float32
  rounding on the graph_hdc side). Covered by `tests/test_legacy_parity.py`.
- `save()` / `load()` handle composite encoders (format version 2); legacy
  files still load.

### Fixed

- Rust backend: SMILES with Kekulé-form aromatic rings (`C1=CC=CC=C1`) were
  featurized as non-aromatic, giving a different fingerprint than `c1ccccc1`
  or the equivalent RDKit `Mol`. Such strings are now aromatized via RDKit
  before reaching the Rust parser.
- Legacy feature lists containing `implicit_hydrogens` failed for SMILES
  input under `backend="auto"`; unsupported features now take the RDKit
  preparation path.
- `encode([])` returned a spurious `(1, D)` row on the NumPy backend; both
  backends now return `(0, D)`.
- Composite mode is batch-invariant for single-atom molecules (a node with no
  incoming messages aggregates to zero, matching graph_hdc). Legacy mode keeps
  its historical behaviour (`encode("C")` differs from `encode(["C", "CC"])[0]`),
  now documented and pinned by a test.
- `Encoder.load` raises instead of silently regenerating codebooks when a
  composite file lacks an array for a declared feature.

### Notes

- The composite encoder currently runs on the NumPy backend only;
  `backend="rust"` raises, `"auto"` falls back silently.
- New `tests/fixtures/seed_fixtures.npz` protects seed→codebook generation for
  both modes and composite-mode outputs; `regression_data.npz` is unchanged.

## [0.2.2] - 2026-04-01

### Added

- `.zenodo.json` metadata file for Zenodo DOI archival
- `CITATION.cff` for GitHub's "Cite this repository" button

## [0.2.1] - 2026-04-01

### Fixed

- CI: pass `-i python3.11` to maturin in manylinux builds to produce correct
  `abi3` wheel tags (was producing `cp38` wheels due to container default Python)
- CI: use `macos-latest` for macOS x86_64 builds (`macos-13` runner unavailable)

## [0.2.0] - 2026-04-01

### Added

- **Rust extension backend** for ~22x end-to-end encoding speedup
  - `encode_batch_rs`: HRR message-passing via `realfft` (r2c/c2r FFT) with
    molecule-level rayon parallelism and row-level FFT parallelism
  - `prepare_batch_rs`: SMILES parsing and atom feature extraction via the
    `purr` crate, replacing RDKit for the Rust path (~23x faster than Python
    stages 1-4)
  - GIL released during all Rust computation (`py.detach()`)
  - Thread-local FFT scratch buffers to avoid allocation and contention
- **Configurable atom features** via `features` parameter on `Encoder`
  - `features=["element", "degree", "charge", "hydrogens", "aromatic"]` (default)
  - Any subset in any order; codebook size adapts automatically
  - Feature registry in `features.py` with named extractors
  - Works with both Python and Rust backends
- **Backend selection** via `backend` parameter on `Encoder`
  - `"auto"` (default): use Rust if available, else NumPy
  - `"rust"`: require Rust extension
  - `"numpy"`: force pure-Python path
- **Cross-backend test suite** (`tests/test_rust_backend.py`)
  - Parameterized regression tests across both backends
  - Edge case tests: single-atom, disconnected graphs, charged atoms,
    all supported elements, depth 0/1, large batches, normalize mode
  - 35 new tests (76 total Python + 9 Rust `#[cfg(test)]`)
- **Benchmarking experiments** (`experiments/`)
  - `benchmark_fps.py`: configurable sweep comparing Morgan, HDF-Rust, and
    HDF-Python across fingerprint sizes with console table, CSV, and plot output
  - `profile_hdf.py`: per-stage timing breakdown of the HDF pipeline for
    both backends with stacked bar chart
  - `molecules.smi`: 10,000 PubChem molecules for benchmarking
- **Build and release infrastructure**
  - `build.sh`: build release wheels and sdist
  - `release.sh`: bump version, test, tag, and push (CI publishes to PyPI)
  - `nox -s build_test`: build wheel, install in clean venv, run full test suite
  - GitHub Actions CI (`.github/workflows/ci.yml`): builds wheels for
    Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows (x64),
    runs tests, and publishes to PyPI on tag push via trusted publishing
- Type stubs for the Rust module (`hyper_fingerprints/_core.pyi`)
- `DEVELOP.md` with development setup, CI, and release documentation
- `RUST_EXTENSION.md` with design notes and implementation status

### Changed

- Build backend switched from hatchling to maturin (Rust toolchain required)
- `FeatureEncoder.encode()` refactored to use `encode_indices()` internally
- `features.py` refactored from hardcoded extraction to registry-based system
- `Encoder.save()` / `Encoder.load()` now persists the `features` list
  (backward compatible: old files without `features` key default to all 5)

## [0.1.0] - 2026-03-14

Initial release.

### Added

- `Encoder` class for encoding molecules into fixed-size HRR hypervectors
- `encode()` for order-N fingerprints via message passing
- `encode_joint()` for combined order-0 + order-N representations
- `Encoder.save()` / `Encoder.load()` for persisting encoders as `.npz` files
- `cosine_similarity()` helper for pairwise vector comparison
- Configurable atom vocabulary, dimension, depth, and normalization
- 5-feature atom scheme: atom type, degree, formal charge, total Hs, aromaticity
- Deterministic codebook generation via seeded RNG
- Batch encoding of multiple molecules in a single call
- Support for both SMILES strings and RDKit `Mol` objects as input
- Regression test suite for numerical stability
- Multi-version testing with nox (Python 3.9-3.13)
- Quickstart notebook (`examples/00_quickstart.ipynb`)
