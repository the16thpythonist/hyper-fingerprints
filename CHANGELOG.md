# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

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
