# Rust Extension for hyper_fingerprints

Research notes and implementation status for the Rust acceleration backend.

## Status: IMPLEMENTED

All planned phases are complete. The Rust extension provides a **~22x end-to-end speedup** over the pure-Python/NumPy path (measured on 512 molecules, dim=256, depth=2).

### What was built

| Component | Crate | What it does |
|---|---|---|
| `prepare_batch_rs` | `purr` | SMILES parsing + atom feature extraction + graph batching in one call |
| `encode_batch_rs` | `realfft` + `rayon` | Message-passing + graph readout with molecule-level parallelism |

### Performance (512 molecules, dim=256, depth=2)

| Stage | Python | Rust | Speedup |
|---|---|---|---|
| Stages 1-4 (parse + features + batch) | 105 ms | 4.6 ms | ~23x |
| Stage 5 (message-passing) | 464 ms | 21.5 ms | ~22x |
| **Total** | **569 ms** | **26 ms** | **~22x** |

### Key implementation decisions

- **Maturin as sole build backend** (Option B). Rust toolchain is required for installation. Pre-built wheels can be published for pip-install-without-Rust.
- **PyO3 0.28 + rust-numpy 0.28** (version-aligned, contrary to initial expert concern).
- **`py.detach()`** (PyO3 0.28's replacement for `allow_threads`) with `to_owned_array()` to satisfy lifetime constraints.
- **`realfft`** (r2c/c2r) from Phase 1, not deferred to Phase 5.
- **Molecule-level parallelism** via rayon with `batch_has_edges` flag to handle the known single-atom batch discrepancy.
- **Row-level FFT parallelism** also added within `hrr_bind_rows` as a secondary parallelism layer using thread-local scratch buffers.
- **`purr`** crate for SMILES parsing (replaces RDKit for the Rust path). Symbol capitalization fix needed for bracket aromatic atoms (`[nH]` → "N").
- **Configurable features** via `features=["element", "degree", ...]` parameter, driving both Python and Rust extraction paths.

### Test coverage

- 76 Python tests (41 original + 35 cross-backend/edge-case)
- 9 Rust `#[cfg(test)]` unit tests
- `nox -s build_test` builds a wheel and runs all tests in a clean venv

---

## Why Rust?

The current `_encode_batch` loop crosses the Python-C boundary many times per layer (array indexing, `scatter_hd`, `hrr_bind` each trigger separate NumPy calls). Moving the entire message-passing loop into Rust eliminates that overhead and enables SIMD + multi-threading.

## Toolchain: PyO3 + maturin

| Component | Role |
|---|---|
| **PyO3** | Rust crate providing CPython bindings (`#[pyfunction]`, `#[pymodule]`) |
| **maturin** (>=1.0, <2.0) | PEP 517 build backend that compiles Rust and packages wheels |
| **rust-numpy** | Zero-copy NumPy array access from Rust via `ndarray` views |

**Version compatibility**: PyO3 and rust-numpy must be version-aligned (rust-numpy tracks PyO3 releases). Check the [rust-numpy compatibility table](https://github.com/PyO3/rust-numpy#version-matrix) and pin accordingly. For example, if rust-numpy latest supports PyO3 0.23, use `pyo3 = "0.23"` and `numpy = "0.23"`. Do **not** mix incompatible versions -- the project will fail to compile.

### Key Rust crates

| Crate | Purpose |
|---|---|
| `pyo3` | Python bindings (pin to version compatible with rust-numpy) |
| `numpy` (rust-numpy) | `PyReadonlyArray2<f64>` / `PyArray2<f64>` zero-copy interop |
| `ndarray` | N-dimensional array types in Rust (version must match rust-numpy's `ndarray` dep) |
| `realfft` | Real-to-complex FFT (built on rustfft; ~50% faster than full complex FFT for real inputs) |
| `rustfft` | Full complex FFT (AVX/SSE4.1/Neon auto-detected; transitive dep of realfft) |
| `num-complex` | Complex number types for FFT |
| `rayon` | Data parallelism (parallelize across molecules) |

## Project structure

```
hyper-fingerprints/
  Cargo.toml                      # Rust project config
  src/
    lib.rs                        # Rust extension entry point
  pyproject.toml                  # maturin replaces hatchling as build backend
  hyper_fingerprints/
    _core.pyi                     # type stubs for the Rust module
    __init__.py
    encoder.py                    # calls _core.encode_batch_rs internally
    ...
```

### Cargo.toml

```toml
[package]
name = "hyper_fingerprints_rs"
version = "0.1.0"
edition = "2021"

[lib]
name = "_core"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.28", features = ["abi3-py39", "extension-module"] }
numpy = "0.28"
ndarray = "0.17"
purr = "0.9"
realfft = "3.5"
num-complex = "0.4"
rayon = "1.10"
```

- `abi3-py39` produces **one wheel** for all CPython >= 3.9 (no per-version builds).
- `crate-type = ["cdylib"]` produces a shared library Python can import.
- `realfft` wraps `rustfft` with r2c/c2r transforms -- preferred over raw `rustfft` since all our inputs are real-valued.

### Build backend strategy

Maturin and hatchling **cannot coexist** as PEP 517 build backends in the same package. Two options:

**Option A -- Separate Rust companion package** (recommended):
Keep `hyper_fingerprints` using hatchling. Create a separate `hyper-fingerprints-core` package with maturin that builds and installs the `_core` extension module. The main package has a soft dependency (try-import at runtime). Users without Rust get the pure-Python path.

**Option B -- Maturin as sole backend**:
Replace hatchling with maturin. Simpler setup but **requires a Rust toolchain for all installs** (including `pip install .`). No pure-Python fallback at build time.

For either option, the `[tool.maturin]` config is:
```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[tool.maturin]
module-name = "hyper_fingerprints._core"
```

Note: `features = ["pyo3/extension-module"]` in `[tool.maturin]` is redundant when already specified in `Cargo.toml` -- omit it to avoid confusion.

## NumPy interop and GIL release

**Important lifetime constraint**: `PyReadonlyArray::as_array()` returns an `ArrayView` that borrows from a Python object. This borrow is tied to the `'py` lifetime and is **not `Send`**, so it **cannot** be used inside `py.allow_threads(|| { ... })`. The closure must own all data it touches.

**Solution**: Copy inputs to owned `ndarray::Array` before entering the GIL-released closure. The copy cost is negligible (one-time per batch call, e.g. ~40 MB for a large codebook).

```rust
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArray2};
use ndarray::{Array1, Array2};
use pyo3::prelude::*;

#[pyfunction]
fn encode_batch_rs<'py>(
    py: Python<'py>,
    codebook: PyReadonlyArray2<'py, f64>,       // [num_categories, dim]
    feature_indices: PyReadonlyArray1<'py, i64>, // [N] flat codebook indices
    edge_index: PyReadonlyArray2<'py, i64>,      // [2, E]
    batch_indices: PyReadonlyArray1<'py, i64>,   // [N] graph membership
    num_graphs: usize,
    depth: usize,
    normalize: bool,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>) {
    // Copy to owned arrays -- required for allow_threads
    let codebook = codebook.to_owned_array();
    let feature_indices = feature_indices.to_owned_array();
    let edge_index = edge_index.to_owned_array();
    let batch_indices = batch_indices.to_owned_array();

    let (graph_emb, node_terms) = py.allow_threads(|| {
        // entire message-passing loop runs here, GIL released
        // all data is owned -- no Python references
        encode_batch_inner(&codebook, &feature_indices, &edge_index,
                           &batch_indices, num_graphs, depth, normalize)
    });

    // Convert back to Python arrays
    (graph_emb.into_pyarray(py), node_terms.into_pyarray(py))
}
```

Notes:
- `to_owned_array()` handles both contiguous and non-contiguous inputs (copies if needed).
- The inner function is pure Rust with no Python dependency -- easily testable with `#[cfg(test)]`.

## What moved to Rust

| In Rust | In Python |
|---|---|
| SMILES parsing (via `purr`) | RDKit Mol object handling (fallback path) |
| Atom feature extraction | Codebook generation (one-time cost) |
| Graph batching + codebook index computation | Save/load (.npz serialization) |
| `hrr_bind` (FFT circular convolution via `realfft`) | Public API surface (`encode`, `encode_joint`) |
| Message-passing loop + graph readout | Feature registry definition |
| Molecule-level parallelism (rayon) | |

When SMILES strings are passed and Rust is available, the entire pipeline from raw strings to fingerprint vectors runs in Rust. The Python side only does codebook generation (one-time) and the final `np.asarray` conversion.

## FFT details

`hrr_bind(a, b) = ifft(fft(a) * fft(b))` -- circular convolution.

Since all inputs are real-valued, use the `realfft` crate (r2c/c2r transforms) instead of full complex FFT. This produces only `D/2 + 1` complex values instead of `D`, giving ~40-50% speedup on the FFT path.

- `realfft` is built on `rustfft` and inherits its AVX/SSE4.1/Neon SIMD auto-detection.
- `RealFftPlanner` caches algorithm selection -- create once, reuse for all FFTs of the same `dimension`.
- Powers of 2 (128, 256, 512, 1024, 2048) are optimal.
- **Important**: neither rustfft nor realfft normalize. Divide by `D` after inverse FFT to match NumPy's `ifft`.
- The r2c output is conjugate-symmetric, so element-wise multiply only needs `D/2 + 1` complex muls.

## Parallelism

Parallelize at the **molecule level** using rayon. Split the batched graph into per-molecule subgraphs (using `batch_indices`), run each molecule's full message-passing loop independently on rayon workers, then scatter-bundle the per-molecule results into the final output.

Why molecule-level, not row-level:
- Individual FFT rows (D=256-2048) are too cheap -- rayon thread-scheduling overhead dominates.
- Molecule-level gives coarser work units and scales with batch size (512 molecules >> num cores).
- Eliminates synchronization between layers -- each worker runs the full depth independently.
- Row-level parallelism within `hrr_bind_rows` can be a secondary optimization for unusually large molecules, gated by a node-count threshold.

**Scatter operations must remain sequential** even after rayon is added. Floating-point addition is not associative -- parallel scatter with different accumulation orders would break determinism and make cross-backend regression tests unreliable.

GIL must be released (`py.allow_threads`) before rayon threads can run. Use `thread_local!` for per-thread `FftPlanner` and scratch buffers.

## Development workflow

```bash
# Install maturin
uv pip install maturin

# Development build (debug, fast compile)
maturin develop --uv

# Release build (optimized)
maturin develop --uv --release

# Build wheel
maturin build --release
```

To auto-rebuild when Rust source changes, add to `pyproject.toml`:

```toml
[tool.uv]
cache-keys = [
    { file = "pyproject.toml" },
    { file = "Cargo.toml" },
    { file = "src/**/*.rs" },
]
```

For maximum SIMD performance on the build machine:

```bash
RUSTFLAGS="-C target-cpu=native" maturin develop --uv --release
```

## Alternatives considered

| Approach | Verdict |
|---|---|
| **Cython** | Good for annotating existing Python but weaker parallelism (OpenMP), no memory safety guarantees on index operations |
| **Numba JIT** | Quick to write but limited to NumPy subset, no custom data structures, JIT warmup cost |
| **cffi / ctypes** | Better for wrapping existing C code, not writing new code |
| **pybind11 / nanobind** | C++ equivalent of PyO3; Rust is preferred for memory safety in index-heavy scatter operations |

PyO3 + maturin is the best fit because we are writing **new** performance-critical code with complex index manipulation where memory safety matters.

---

## Implementation plan

### Overview

The original goal was to replace `Encoder._encode_batch()` with a single Rust function. This was later expanded to also replace SMILES parsing and feature extraction with `purr`, and to add configurable feature sets.

The plan was split into phases. Each phase produced a working, testable state. **All phases are complete.**

### Phase 0 -- Project scaffolding [DONE]

Set up the mixed Python+Rust project so it compiles and can be imported, before writing any real logic.

1. **Decide build strategy**: choose between separate companion package or maturin-as-sole-backend (see "Build backend strategy" above). This decision affects all subsequent phases.
2. Install maturin: `uv pip install maturin`
3. Create `Cargo.toml` at repo root (as shown above). **Verify crate version compatibility** -- check rust-numpy's compatibility table and pin PyO3 + rust-numpy to matching versions.
4. Create `src/lib.rs` with a minimal `#[pymodule]` that exports a no-op function:
   ```rust
   #[pyfunction]
   fn hello() -> String { "hello from rust".to_string() }
   ```
5. Update `pyproject.toml` per chosen build strategy
6. Build: `maturin develop --uv`
7. Verify: `python -c "from hyper_fingerprints._core import hello; print(hello())"`

**Done when**: `hyper_fingerprints._core` is importable and the existing Python tests still pass (the Rust module is additive, nothing removed yet).

### Phase 1 -- Port `hrr_bind` to Rust [DONE]

The innermost hot operation. Port it first so it can be tested in isolation.

**What `hrr_bind` does** (utils.py:77-79):
```python
np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))
```
For inputs `a`, `b` of shape `[N, D]`, it does row-wise circular convolution.

**Rust implementation** (`src/lib.rs`):
1. Create a helper `fn hrr_bind_rows(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64>`:
   - Use `realfft::RealFftPlanner` to plan r2c and c2r transforms for length `D`
   - Allocate scratch buffers: two `Vec<f64>` of length `D`, two `Vec<Complex64>` of length `D/2 + 1`
   - For each row pair `(a[i], b[i])`:
     - Copy into real scratch buffers
     - `r2c.process(&mut real_buf_a, &mut complex_buf_a)` (produces `D/2 + 1` complex values)
     - `r2c.process(&mut real_buf_b, &mut complex_buf_b)`
     - Element-wise multiply: `complex_buf_a[k] *= complex_buf_b[k]` (only `D/2 + 1` muls)
     - `c2r.process(&mut complex_buf_a, &mut real_buf_a)` (back to `D` real values)
     - Divide by `D` (realfft doesn't normalize), write to output row
   - Return `Array2<f64>` of shape `[N, D]`
2. Expose as `#[pyfunction] fn hrr_bind_rs(...)` for cross-validation testing
3. Pre-allocate scratch buffers once and reuse across rows (avoid per-row allocation)

**Testing**: Call both Python `hrr_bind` and Rust `hrr_bind_rs` on the same random inputs, assert `np.allclose(atol=1e-12)`. Use tight tolerances matching the existing test suite -- not the default `atol=1e-8`.

**Also add Rust-side `#[cfg(test)]`** unit tests for the inner `hrr_bind_rows` function to verify FFT normalization, edge cases (D=1, identity binding), and numerical precision independently of the PyO3 boundary.

### Phase 2 -- Port `scatter_hd` to Rust [DONE]

**What `scatter_hd` does** (utils.py:90-127):
- `op="bundle"`: `out = zeros([dim_size, D]); np.add.at(out, index, src)`
- `op="bind"`: `out = identity([dim_size, D]); np.multiply.at(out, index, src)`
- Edge case: if `index` is empty, returns identity vectors

**Rust implementation**:
1. `fn scatter_bundle(src: &Array2<f64>, index: &Array1<i64>, dim_size: usize) -> Array2<f64>`:
   - Allocate `Array2::zeros((dim_size, d))`
   - For each `i in 0..N`: `out.row_mut(index[i]) += src.row(i)` (sequential loop)
   - **Must remain sequential** -- parallel accumulation breaks floating-point determinism
2. `fn scatter_bind(src: &Array2<f64>, index: &Array1<i64>, dim_size: usize) -> Array2<f64>`:
   - Allocate identity array (column 0 = 1.0, rest 0.0)
   - For each `i in 0..N`: `out.row_mut(index[i]) *= src.row(i)` element-wise
3. Expose as `#[pyfunction] fn scatter_hd_rs(...)` for cross-validation testing
4. Handle empty-index edge case (return identity)
5. Handle single-atom molecules (no edges, empty edge_index) -- the existing test suite has a known single-atom batch discrepancy; reproduce the same behavior

**Testing**: Compare against Python `scatter_hd` on known inputs with `atol=1e-12`. Test edge cases: empty index, single-element index, duplicate indices, disconnected graphs.

### Phase 3 -- Port the full `_encode_batch` loop [DONE]

Combine phases 1-2 into a single Rust function that replaces the entire `_encode_batch` method. This is where the real speedup comes from (eliminating per-layer Python overhead).

**Prerequisite -- Expose flat codebook indices from Python**:
`FeatureEncoder.encode()` (codebook.py:40-47) currently computes flat indices internally but only returns the gathered codebook rows. The `idxs` variable is local and never exposed. This must be changed:
- Add `FeatureEncoder.encode_indices(data: np.ndarray) -> np.ndarray` that returns the flat index array
- Or have `_encode_batch` in `encoder.py` compute indices directly via `self._indexer.get_idxs()` and pass them to Rust

**Rust function signature** (see "NumPy interop" section for full example with `to_owned_array`):
```rust
#[pyfunction]
fn encode_batch_rs<'py>(
    py: Python<'py>,
    codebook: PyReadonlyArray2<'py, f64>,        // [num_categories, D]
    feature_indices: PyReadonlyArray1<'py, i64>,  // [N] flat codebook indices
    edge_index: PyReadonlyArray2<'py, i64>,       // [2, E]
    batch_indices: PyReadonlyArray1<'py, i64>,    // [N] graph membership
    num_graphs: usize,
    depth: usize,
    normalize: bool,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)
// returns (graph_embedding [B,D], node_terms [B,D])
```

**Rust implementation** (all inside `py.allow_threads`, inputs copied to owned arrays first):
1. **Node feature lookup**: `node_hv[i] = codebook[feature_indices[i]]` -- gather rows from codebook into `Array2<f64>` of shape `[N, D]`
2. **Allocate stack**: `Vec<Array2<f64>>` of length `depth + 1`, each `[N, D]`. `stack[0] = node_hv`.
3. **Allocate reusable message buffer**: `Array2<f64>` of shape `[E, D]` -- reused across layers (E and D are constant). This avoids per-layer allocation.
4. **Message-passing loop** (`for layer in 0..depth`):
   - Extract `srcs` and `dsts` from `edge_index` (row 0 and row 1)
   - **Fused gather+scatter**: instead of materializing the full messages array, accumulate directly: `aggregated[src[e]] += stack[layer].row(dst[e])`. This saves one `[E, D]` allocation and one memory pass.
   - **HRR bind**: `hr = hrr_bind_rows(&stack[layer], &aggregated)` (uses `realfft` r2c/c2r)
   - **Optional normalize**: if `normalize`, divide each row by its L2 norm (+ 1e-8)
   - `stack[layer + 1] = hr`
5. **Graph readout**:
   - **Multibundle**: for each node, sum across all layers: `bundled[i] = sum(stack[0..=depth].row(i))`
   - **Order-0**: `node_terms = scatter_bundle(node_hv, batch_indices, num_graphs)`
   - **Order-N**: `graph_embedding = scatter_bundle(bundled, batch_indices, num_graphs)`
6. Convert outputs to `PyArray2` and return

**Python-side changes**:
- `codebook.py`: add `encode_indices()` method that returns flat index array
- `encoder.py`: add try-import of `_core.encode_batch_rs` with fallback:
  ```python
  try:
      from hyper_fingerprints._core import encode_batch_rs
      _HAS_RUST = True
  except ImportError:
      _HAS_RUST = False
  ```
- In `_encode_batch`, if `_HAS_RUST`, compute `feature_indices` and call Rust; otherwise use existing NumPy path

**Edge cases to handle**: empty edge_index (single-atom molecules), disconnected graphs (`.` in SMILES -- untested in current suite, add tests).

### Phase 4 -- Rayon parallelism [DONE]

After the single-threaded Rust version is correct, add molecule-level parallelism.

**Strategy**: Split the batched graph into per-molecule subgraphs and run each molecule's full message-passing loop independently on rayon workers.

1. Add `rayon` to `Cargo.toml`
2. Partition inputs by `batch_indices`:
   - For each molecule `g`, extract its node slice (contiguous in the batch) and edge subset
   - Run `encode_single_molecule(codebook, nodes_g, edges_g, depth, normalize)` per molecule
   - Each worker owns its own `FftPlanner` + scratch buffers via `thread_local!`
3. Collect per-molecule embeddings and stack into the output `[B, D]` arrays
4. **Scatter operations remain sequential** within each molecule (determinism guarantee)
5. **Do not parallelize** the final graph-readout scatter -- it's already O(N) with small constant and the index set is partitioned

**Secondary optimization** (only if profiling shows benefit): For batches containing unusually large molecules (>100 atoms), add row-level parallelism within `hrr_bind_rows`, gated by a node-count threshold.

**Testing**: Numerical results must be **identical** to single-threaded Rust path (add `assert_allclose(atol=1e-12)` regression test). Run the existing `tests/test_regression.py` fixture against both paths.

### Phase 5 -- Benchmarking and optimization [DONE]

1. Run `experiments/benchmark_fps.py` with the Rust backend and compare against NumPy
2. Add a new `RustHDFFP` method to the benchmark registry so both backends can be compared side-by-side
3. Profile with `cargo flamegraph` or `perf` to identify remaining bottlenecks
4. Potential micro-optimizations (after profiling -- don't prematurely optimize):
   - **Pre-sort edges by source** at the Python level so scatter writes are sequential in memory, improving cache locality
   - **`target-cpu=native`**: compile with `RUSTFLAGS="-C target-cpu=native"` for best SIMD on the build machine
   - **In-place operations**: reuse buffers across layers instead of allocating new arrays each iteration
   - **L2 normalization via SIMD**: `ndarray` can auto-vectorize norm computation, but explicit SIMD may help for small D

Note: r2c/c2r FFT and fused gather+scatter are already part of Phase 1 and Phase 3 respectively -- they are not afterthought optimizations.

### Phase 6 -- Testing and verification [DONE]

Ensuring cross-backend correctness is critical and deserves its own phase.

1. **Parameterized regression tests**: Run `tests/test_regression.py` against both backends by monkeypatching the import (force Python path vs. Rust path). Same fixtures, same tolerances (`atol=1e-12`).
2. **Edge case test suite**: Add tests for single-atom molecules, disconnected graphs (`.` in SMILES), empty batches, molecules with only the supported atom subset.
3. **Rust-side `#[cfg(test)]`**: Unit tests for `hrr_bind_rows`, `scatter_bundle`, FFT normalization. These run via `cargo test` without Python -- fast and isolated.
4. **Keep Python-facing `hrr_bind_rs` and `scatter_hd_rs`** as `#[pyfunction]` for cross-validation during development. Remove from the public Python API in Phase 7, but retain the underlying Rust functions with `#[cfg(test)]` coverage.

### Phase 7 -- Cleanup and CI [DONE]

1. Add `maturin` build to CI (e.g. `maturin build --release` in GitHub Actions)
2. If using companion package strategy: ensure `pip install hyper-fingerprints` works without Rust (pure-Python fallback)
3. Add type stubs `_core.pyi` for the Rust module
4. Update README with build instructions for the Rust extension
5. Remove `hrr_bind_rs` and `scatter_hd_rs` from the public Python API (keep as internal Rust functions with `#[cfg(test)]`)

### Additional work (beyond original plan)

**Configurable features** — Added `features=["element", "degree", ...]` parameter to `Encoder`. Feature registry in `features.py` with named extractors. Both Python and Rust paths respect the configured feature set. Codebook size adapts automatically.

**Rust SMILES parsing** — Added `purr` crate for SMILES parsing + feature extraction in Rust. `prepare_batch_rs` replaces Python stages 1-4 (SMILES → mol_to_data → batch → codebook indices) in a single call. ~23x faster than the Python/RDKit path.

**Build infrastructure** — `build.sh` for building release wheels + sdist. `nox -s build_test` session that builds a wheel, installs into clean venv, runs full test suite.

**Benchmark updates** — `experiments/benchmark_fps.py` with three methods: Morgan, HDF-Rust, HDF-Python. `experiments/profile_hdf.py` for per-stage timing breakdown.

### Summary of file changes

| File | Status |
|---|---|
| `Cargo.toml` | New — Rust dependencies |
| `src/lib.rs` | New — `encode_batch_rs`, `prepare_batch_rs`, HRR/scatter internals |
| `pyproject.toml` | Modified — maturin build backend |
| `hyper_fingerprints/encoder.py` | Modified — `backend`, `features` params, Rust routing |
| `hyper_fingerprints/features.py` | Modified — feature registry, configurable extractors |
| `hyper_fingerprints/codebook.py` | Modified — `encode_indices()` method |
| `hyper_fingerprints/__init__.py` | Modified — export `DEFAULT_FEATURES` |
| `hyper_fingerprints/_core.pyi` | New — type stubs |
| `tests/test_rust_backend.py` | New — 35 cross-backend + edge-case tests |
| `experiments/benchmark_fps.py` | Modified — Morgan / HDF-Rust / HDF-Python |
| `experiments/profile_hdf.py` | New — per-stage profiling |
| `build.sh` | New — release build script |
| `noxfile.py` | Modified — `build_test` session |
| `README.md` | Modified — install + Rust backend docs |

### Risks encountered and resolved

| Risk | What happened | Resolution |
|---|---|---|
| PyO3 / rust-numpy version mismatch | Expert review warned about it, but 0.28 versions are actually compatible | Verified via `cargo tree` — no issue |
| `ArrayView` in `allow_threads` | `allow_threads` renamed to `py.detach()` in PyO3 0.28 | Used `to_owned_array()` + `py.detach()` |
| FFT normalization | realfft doesn't normalize | Divide by `D` after c2r; verified at `atol=1e-12` |
| Single-atom batch discrepancy | Molecule-level parallelism changed no-edge behavior | Added `batch_has_edges` flag to match Python semantics exactly |
| Bracket aromatic symbols | purr returns lowercase `"n"` for `[nH]` | Capitalize first letter in `atom_symbol()` |
| Benchmark not using Rust SMILES path | `benchmark_fps.py` passed pre-parsed Mol objects | Threaded raw SMILES strings through benchmark pipeline |
