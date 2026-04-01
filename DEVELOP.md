# Development Guide

## Prerequisites

- Python 3.9+
- [Rust toolchain](https://rustup.rs/) >= 1.83
- [maturin](https://www.maturin.rs/): `pip install maturin`

## Local development setup

```bash
git clone https://github.com/the16thpythonist/hyper-fingerprints.git
cd hyper-fingerprints

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Build the Rust extension in release mode (with native SIMD)
RUSTFLAGS="-C target-cpu=native" maturin develop --release

# Install dev dependencies
pip install -e ".[dev]"

# Verify
python -c "from hyper_fingerprints._core import encode_batch_rs; print('Rust OK')"
```

## Running tests

```bash
# Full test suite (76 Python + 9 Rust tests)
pytest

# Rust-side unit tests only
cargo test

# Build a wheel, install in clean venv, run tests
nox -s build_test
```

## Building wheels

```bash
# Build release wheel + sdist for the current platform
./build.sh

# With native CPU optimizations
./build.sh --native

# Debug build (fast compile, slow runtime)
./build.sh --debug

# Artifacts are in target/wheels/
ls target/wheels/
```

## Project structure

```
hyper-fingerprints/
  Cargo.toml                        # Rust crate config
  src/lib.rs                        # Rust extension (encode_batch_rs, prepare_batch_rs)
  pyproject.toml                    # Python package config (maturin build backend)
  hyper_fingerprints/
    __init__.py                     # Public API
    encoder.py                      # Encoder class (routes to Rust or NumPy)
    features.py                     # Feature registry + RDKit extractors
    codebook.py                     # FeatureEncoder (codebook generation)
    utils.py                        # HRR algebra, graph data structures
    _core.abi3.so                   # Compiled Rust extension (built by maturin)
    _core.pyi                       # Type stubs for the Rust module
  tests/
    test_encoder.py                 # Encoder unit tests
    test_features.py                # Feature extraction tests
    test_regression.py              # Numerical stability regression tests
    test_rust_backend.py            # Cross-backend + edge case tests
  experiments/
    benchmark_fps.py                # Morgan vs HDF-Rust vs HDF-Python benchmark
    profile_hdf.py                  # Per-stage profiling of the HDF pipeline
    molecules.smi                   # 10k molecules from PubChem
```

## CI pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) builds pre-compiled
wheels for all platforms so users can `pip install` without a Rust toolchain.

### How it works

```
Trigger (push/PR/tag)
  │
  ├── linux job (x86_64 + aarch64)     ── manylinux Docker container
  ├── macos job (x86_64 + arm64)       ── native GitHub runners
  ├── windows job (x64)                ── native GitHub runner
  └── sdist job                        ── source distribution
        │
        └── test job                   ── installs linux wheel, runs pytest
              │
              └── release job          ── publishes to PyPI (tag pushes only)
```

### Triggers

| Event | Builds | Tests | Publishes |
|---|---|---|---|
| Push to `master` | yes | yes | no |
| Pull request | yes | yes | no |
| Push tag `v*` | yes | yes | **yes** |
| Manual dispatch | yes | yes | no |

### Platform matrix

| Platform | Runner | Target | Wheel type |
|---|---|---|---|
| Linux x86_64 | `ubuntu-22.04` | native | manylinux, abi3 |
| Linux aarch64 | `ubuntu-22.04` | cross (QEMU) | manylinux, abi3 |
| macOS x86_64 | `macos-13` | native | abi3 |
| macOS arm64 | `macos-latest` | native | abi3 |
| Windows x64 | `windows-latest` | native | abi3 |

All wheels use the Python stable ABI (`abi3-py39`), so one wheel per platform
covers all Python versions from 3.9 onward.

### Build caching

`sccache` caches Rust compilation artifacts between CI runs. PR builds reuse
the cache (~2-3 min). Tag builds skip the cache for clean reproducible
releases (~15 min).

## Publishing a release

### One-time PyPI setup

Register the GitHub repo as a trusted publisher on PyPI. This enables
passwordless publishing via OIDC (no API tokens needed).

1. Go to `https://pypi.org/manage/project/hyper-fingerprints/settings/publishing/`
2. Add a trusted publisher:
   - **Owner**: `the16thpythonist`
   - **Repository**: `hyper-fingerprints`
   - **Workflow**: `ci.yml`
   - **Environment**: `release`

### Release process

```bash
# 1. Bump version (updates VERSION file and pyproject.toml)
bump-my-version bump patch   # or: minor, major

# 2. Commit the version bump
git add -A && git commit -m "Bump version to $(cat hyper_fingerprints/VERSION)"

# 3. Tag and push
git tag v$(cat hyper_fingerprints/VERSION)
git push origin master --tags

# 4. CI automatically:
#    - Builds wheels for Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows (x64)
#    - Runs the test suite against the built wheel
#    - Publishes all wheels + sdist to PyPI
```

### What users get

```bash
# Pre-built wheel, no Rust toolchain needed
pip install hyper-fingerprints
```

## Experiments

### Benchmarking

Compare Morgan, HDF-Rust, and HDF-Python fingerprint computation:

```bash
python experiments/benchmark_fps.py --sizes 128 256 512 1024 2048
```

### Profiling

Per-stage timing breakdown of the HDF pipeline:

```bash
python experiments/profile_hdf.py --dim 256 --depth 2 --batch-size 512
```

## Architecture notes

See [RUST_EXTENSION.md](RUST_EXTENSION.md) for detailed documentation of the
Rust extension design, implementation decisions, and performance analysis.
