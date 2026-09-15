use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{s, Array1, Array2, Axis};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use purr::feature::{AtomKind, Charge};
use purr::graph::{Atom, Builder};
use purr::read::read;
use rayon::prelude::*;
use realfft::{RealFftPlanner, RealToComplex, ComplexToReal};
use rustfft::{Fft, FftPlanner};

// ---------------------------------------------------------------------------
// HRR bind (circular convolution via real FFT)
// ---------------------------------------------------------------------------

/// Per-thread scratch buffers for FFT operations.
struct FftScratch {
    buf_a: Vec<f64>,
    buf_b: Vec<f64>,
    spec_a: Vec<Complex64>,
    spec_b: Vec<Complex64>,
    scratch_fwd: Vec<Complex64>,
    scratch_inv: Vec<Complex64>,
}

impl FftScratch {
    fn new(r2c: &Arc<dyn RealToComplex<f64>>, c2r: &Arc<dyn ComplexToReal<f64>>, d: usize) -> Self {
        let complex_len = d / 2 + 1;
        Self {
            buf_a: vec![0.0; d],
            buf_b: vec![0.0; d],
            spec_a: vec![Complex64::new(0.0, 0.0); complex_len],
            spec_b: vec![Complex64::new(0.0, 0.0); complex_len],
            scratch_fwd: r2c.make_scratch_vec(),
            scratch_inv: c2r.make_scratch_vec(),
        }
    }
}

/// Row-wise circular convolution: for each row i, compute
/// `ifft(fft(a[i]) * fft(b[i]))`.
///
/// Uses r2c / c2r transforms since inputs are real-valued, giving ~50%
/// speedup over full complex FFT. Parallelized across rows with rayon.
fn hrr_bind_rows(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n_rows = a.nrows();
    let d = a.ncols();

    if d == 0 || n_rows == 0 {
        return Array2::zeros((n_rows, d));
    }

    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(d);
    let c2r = planner.plan_fft_inverse(d);

    let complex_len = d / 2 + 1;
    let inv_d = 1.0 / d as f64;

    // Thread-local scratch buffers
    let r2c_ref = Arc::clone(&r2c);
    let c2r_ref = Arc::clone(&c2r);
    thread_local! {
        static SCRATCH: RefCell<Option<(usize, FftScratch)>> = const { RefCell::new(None) };
    }

    // Collect output rows in parallel
    let rows: Vec<Vec<f64>> = (0..n_rows)
        .into_par_iter()
        .map(|i| {
            let a_row = a.row(i);
            let b_row = b.row(i);

            SCRATCH.with(|cell| {
                let mut borrow = cell.borrow_mut();
                // Re-create scratch if dimension changed
                if borrow.as_ref().map_or(true, |(sz, _)| *sz != d) {
                    *borrow = Some((d, FftScratch::new(&r2c_ref, &c2r_ref, d)));
                }
                let s = &mut borrow.as_mut().unwrap().1;

                s.buf_a.iter_mut().zip(a_row.iter()).for_each(|(dst, src)| *dst = *src);
                s.buf_b.iter_mut().zip(b_row.iter()).for_each(|(dst, src)| *dst = *src);

                r2c_ref
                    .process_with_scratch(&mut s.buf_a, &mut s.spec_a, &mut s.scratch_fwd)
                    .expect("r2c failed");
                r2c_ref
                    .process_with_scratch(&mut s.buf_b, &mut s.spec_b, &mut s.scratch_fwd)
                    .expect("r2c failed");

                for k in 0..complex_len {
                    s.spec_a[k] *= s.spec_b[k];
                }

                c2r_ref
                    .process_with_scratch(&mut s.spec_a, &mut s.buf_a, &mut s.scratch_inv)
                    .expect("c2r failed");

                s.buf_a.iter().map(|v| v * inv_d).collect::<Vec<f64>>()
            })
        })
        .collect();

    // Assemble into Array2
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n_rows, d), flat).unwrap()
}

// ---------------------------------------------------------------------------
// Scatter-reduce
// ---------------------------------------------------------------------------

/// HRR identity array: shape [n, d], with column 0 = 1.0, rest 0.0.
#[allow(dead_code)]
fn hrr_identity(n: usize, d: usize) -> Array2<f64> {
    let mut out = Array2::zeros((n, d));
    if d > 0 {
        for i in 0..n {
            out[[i, 0]] = 1.0;
        }
    }
    out
}

/// Scatter-reduce with element-wise sum (bundle).
///
/// For each i in 0..N: `out[index[i]] += src[i]`
///
/// Sequential to preserve floating-point determinism.
#[allow(dead_code)]
fn scatter_bundle(src: &Array2<f64>, index: &Array1<i64>, dim_size: usize) -> Array2<f64> {
    let d = src.ncols();

    if index.len() == 0 {
        return hrr_identity(if dim_size == 0 { 1 } else { dim_size }, d);
    }

    let mut out = Array2::zeros((dim_size, d));
    for i in 0..index.len() {
        let idx = index[i] as usize;
        let src_row = src.row(i);
        let mut out_row = out.row_mut(idx);
        out_row += &src_row;
    }
    out
}

/// Scatter-reduce with element-wise product (bind).
///
/// For each i in 0..N: `out[index[i]] *= src[i]`
///
/// Sequential to preserve floating-point determinism.
#[allow(dead_code)]
fn scatter_bind_op(src: &Array2<f64>, index: &Array1<i64>, dim_size: usize) -> Array2<f64> {
    let d = src.ncols();

    if index.len() == 0 {
        return hrr_identity(if dim_size == 0 { 1 } else { dim_size }, d);
    }

    let mut out = hrr_identity(dim_size, d);
    for i in 0..index.len() {
        let idx = index[i] as usize;
        let src_row = src.row(i);
        let mut out_row = out.row_mut(idx);
        out_row *= &src_row;
    }
    out
}

// ---------------------------------------------------------------------------
// Full encode_batch pipeline
// ---------------------------------------------------------------------------

/// Per-layer normalization variant.
///
/// The two encoder modes disagree on the formula: the legacy (product
/// codebook) pipeline divides by `‖x‖ + 1e-8`, while the composite pipeline
/// mirrors `torch.nn.functional.normalize` and divides by `max(‖x‖, 1e-12)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NormMode {
    /// No normalization between layers.
    None,
    /// Legacy formula: `x / (‖x‖ + 1e-8)`.
    LegacyEps,
    /// Composite formula: `x / max(‖x‖, 1e-12)`.
    L2Clamp,
}

impl NormMode {
    fn from_str(name: &str) -> PyResult<Self> {
        match name {
            "none" => Ok(NormMode::None),
            "legacy" => Ok(NormMode::LegacyEps),
            "l2" => Ok(NormMode::L2Clamp),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "norm_mode must be 'none', 'legacy' or 'l2', got {other:?}"
            ))),
        }
    }

    /// Reciprocal scale factor applied to a row with squared norm `sq_norm`.
    fn inv_scale(self, sq_norm: f64) -> f64 {
        match self {
            NormMode::None => 1.0,
            NormMode::LegacyEps => 1.0 / (sq_norm.sqrt() + 1e-8),
            NormMode::L2Clamp => 1.0 / sq_norm.sqrt().max(1e-12),
        }
    }
}

/// Encode a single molecule's message-passing pipeline.
///
/// `batch_has_edges`: whether the parent batch contains any edges at all.
/// Together with `empty_identity` this controls no-edge molecule behavior so
/// that it matches Python's batched `scatter_hd`:
/// - batch has edges → no-edge molecules get zeros aggregation → bind(x,0)=0
/// - batch has NO edges and `empty_identity` (legacy, `empty="identity"`)
///   → identity aggregation → bind(x,id)=x
/// - batch has NO edges and not `empty_identity` (composite, `empty="zeros"`)
///   → zeros aggregation → bind(x,0)=0
fn encode_single_molecule(
    node_hv: &Array2<f64>,
    local_srcs: &[usize],
    local_dsts: &[usize],
    depth: usize,
    norm_mode: NormMode,
    batch_has_edges: bool,
    empty_identity: bool,
) -> (Array1<f64>, Array1<f64>) {
    let n = node_hv.nrows();
    let d = node_hv.ncols();

    let mut current_layer = node_hv.clone();
    let mut multibundle_sum = node_hv.clone();

    if depth > 0 && !local_srcs.is_empty() {
        // Normal message-passing with edges
        let mut aggregated = Array2::zeros((n, d));

        for _layer in 0..depth {
            aggregated.fill(0.0);
            for e in 0..local_srcs.len() {
                let s = local_srcs[e];
                let dt = local_dsts[e];
                for j in 0..d {
                    aggregated[[s, j]] += current_layer[[dt, j]];
                }
            }

            let hr = hrr_bind_rows(&current_layer, &aggregated);

            if norm_mode != NormMode::None {
                current_layer = Array2::zeros((n, d));
                for i in 0..n {
                    let mut sq_norm = 0.0f64;
                    for j in 0..d {
                        sq_norm += hr[[i, j]] * hr[[i, j]];
                    }
                    let inv_norm = norm_mode.inv_scale(sq_norm);
                    for j in 0..d {
                        current_layer[[i, j]] = hr[[i, j]] * inv_norm;
                    }
                }
            } else {
                current_layer = hr;
            }

            multibundle_sum += &current_layer;
        }
    } else if depth > 0 && !batch_has_edges && empty_identity {
        // No edges in entire batch: Python scatter_hd(empty="identity")
        // returns identity. bind(x, identity) = x → all layers equal layer 0.
        // multibundle = (depth+1) * node_hv. Composite mode always scatters
        // with empty="zeros", so it passes empty_identity=false and takes the
        // zero-aggregation branch below instead.
        for _ in 0..depth {
            multibundle_sum += &current_layer;
        }
    }
    // else: no local edges (and either the batch has edges, or empty="zeros")
    // Python scatter starts from zeros, this node gets aggregated=0.
    // bind(x, 0) = 0, so layers 1..depth are zeros.
    // multibundle = node_hv + 0 + ... = node_hv (already the state).

    // Per-molecule readout: sum all node vectors
    let node_term = node_hv.sum_axis(Axis(0));
    let graph_emb = multibundle_sum.sum_axis(Axis(0));

    (graph_emb, node_term)
}

/// Pure-Rust implementation of the full HRR message-passing pipeline.
///
/// Parallelizes across molecules in the batch using rayon.
fn encode_batch_inner(
    codebook: &Array2<f64>,
    feature_indices: &Array1<i64>,
    edge_index: &Array2<i64>,
    batch_indices: &Array1<i64>,
    num_graphs: usize,
    depth: usize,
    norm_mode: NormMode,
) -> (Array2<f64>, Array2<f64>) {
    let n = feature_indices.len();
    let d = codebook.ncols();

    // Step 1: Gather node hypervectors from codebook
    let mut node_hv = Array2::zeros((n, d));
    for i in 0..n {
        let idx = feature_indices[i] as usize;
        node_hv.row_mut(i).assign(&codebook.row(idx));
    }

    encode_batch_from_hv(
        &node_hv,
        edge_index,
        batch_indices,
        num_graphs,
        depth,
        norm_mode,
        /* empty_identity = */ true,
    )
}

/// Message passing over pre-computed node hypervectors.
///
/// This is the shared core: the legacy pipeline reaches it through
/// `encode_batch_inner` (which gathers rows from a product codebook first),
/// while the composite pipeline hands in node hypervectors that Python already
/// built from per-feature codebooks.
///
/// Parallelizes across molecules in the batch using rayon.
fn encode_batch_from_hv(
    node_hv: &Array2<f64>,
    edge_index: &Array2<i64>,
    batch_indices: &Array1<i64>,
    num_graphs: usize,
    depth: usize,
    norm_mode: NormMode,
    empty_identity: bool,
) -> (Array2<f64>, Array2<f64>) {
    let n = node_hv.nrows();
    let d = node_hv.ncols();
    let batch_has_edges = edge_index.ncols() > 0;

    // Step 2: Partition nodes and edges by molecule
    let mut mol_node_start = vec![0usize; num_graphs + 1];
    for i in 0..n {
        mol_node_start[batch_indices[i] as usize + 1] += 1;
    }
    for g in 0..num_graphs {
        mol_node_start[g + 1] += mol_node_start[g];
    }

    let mut mol_edges: Vec<(Vec<usize>, Vec<usize>)> = vec![(vec![], vec![]); num_graphs];
    if batch_has_edges {
        let global_srcs = edge_index.row(0);
        let global_dsts = edge_index.row(1);
        for e in 0..global_srcs.len() {
            let s = global_srcs[e] as usize;
            let g = batch_indices[s] as usize;
            let offset = mol_node_start[g];
            mol_edges[g].0.push(s - offset);
            mol_edges[g].1.push(global_dsts[e] as usize - offset);
        }
    }

    // Step 3: Parallel message-passing per molecule
    let results: Vec<(Array1<f64>, Array1<f64>)> = (0..num_graphs)
        .into_par_iter()
        .map(|g| {
            let start = mol_node_start[g];
            let end = mol_node_start[g + 1];
            let mol_nodes = node_hv.slice(s![start..end, ..]).to_owned();
            let (ref local_srcs, ref local_dsts) = mol_edges[g];
            encode_single_molecule(
                &mol_nodes, local_srcs, local_dsts,
                depth, norm_mode, batch_has_edges, empty_identity,
            )
        })
        .collect();

    // Step 4: Assemble output
    let mut graph_embedding = Array2::zeros((num_graphs, d));
    let mut node_terms = Array2::zeros((num_graphs, d));
    for (g, (ge, nt)) in results.into_iter().enumerate() {
        graph_embedding.row_mut(g).assign(&ge);
        node_terms.row_mut(g).assign(&nt);
    }

    (graph_embedding, node_terms)
}

/// Full HRR message-passing encode. Replaces `Encoder._encode_batch()`.
///
/// Args:
///     codebook: [num_categories, D]
///     feature_indices: [N] flat codebook indices
///     edge_index: [2, E]
///     batch_indices: [N] graph membership
///     num_graphs: number of graphs in the batch
///     depth: number of message-passing layers
///     normalize: whether to L2-normalize after each layer
///
/// Returns:
///     (graph_embedding [B, D], node_terms [B, D])
#[pyfunction]
fn encode_batch_rs<'py>(
    py: Python<'py>,
    codebook: PyReadonlyArray2<'py, f64>,
    feature_indices: PyReadonlyArray1<'py, i64>,
    edge_index: PyReadonlyArray2<'py, i64>,
    batch_indices: PyReadonlyArray1<'py, i64>,
    num_graphs: usize,
    depth: usize,
    normalize: bool,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>) {
    let codebook = codebook.to_owned_array();
    let feature_indices = feature_indices.to_owned_array();
    let edge_index = edge_index.to_owned_array();
    let batch_indices = batch_indices.to_owned_array();
    let norm_mode = if normalize { NormMode::LegacyEps } else { NormMode::None };

    let (graph_emb, node_terms) = py.detach(|| {
        encode_batch_inner(
            &codebook,
            &feature_indices,
            &edge_index,
            &batch_indices,
            num_graphs,
            depth,
            norm_mode,
        )
    });

    (graph_emb.into_pyarray(py), node_terms.into_pyarray(py))
}

/// HRR message passing over pre-computed node hypervectors.
///
/// Used by the composite encoder, whose node hypervectors are built in Python
/// from per-feature codebooks (categorical lookups bound with FHRR continuous
/// encodings) and cannot be expressed as rows of a single product codebook.
///
/// Args:
///     node_hv: [N, D] node hypervectors
///     edge_index: [2, E]
///     batch_indices: [N] graph membership
///     num_graphs: number of graphs in the batch
///     depth: number of message-passing layers
///     norm_mode: "none", "legacy" (x / (‖x‖ + 1e-8)) or "l2" (x / max(‖x‖, 1e-12))
///     empty_identity: for an edgeless batch, whether aggregation yields the
///         HRR identity (legacy) or zeros (composite)
///
/// Returns:
///     (graph_embedding [B, D], node_terms [B, D])
#[pyfunction]
fn encode_batch_hv_rs<'py>(
    py: Python<'py>,
    node_hv: PyReadonlyArray2<'py, f64>,
    edge_index: PyReadonlyArray2<'py, i64>,
    batch_indices: PyReadonlyArray1<'py, i64>,
    num_graphs: usize,
    depth: usize,
    norm_mode: &str,
    empty_identity: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let norm_mode = NormMode::from_str(norm_mode)?;
    let node_hv = node_hv.to_owned_array();
    let edge_index = edge_index.to_owned_array();
    let batch_indices = batch_indices.to_owned_array();

    if batch_indices.len() != node_hv.nrows() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "batch_indices has length {} but node_hv has {} rows",
            batch_indices.len(),
            node_hv.nrows()
        )));
    }

    let (graph_emb, node_terms) = py.detach(|| {
        encode_batch_from_hv(
            &node_hv,
            &edge_index,
            &batch_indices,
            num_graphs,
            depth,
            norm_mode,
            empty_identity,
        )
    });

    Ok((graph_emb.into_pyarray(py), node_terms.into_pyarray(py)))
}

// ---------------------------------------------------------------------------
// Composite node encoding (per-feature codebooks + FHRR continuous features)
// ---------------------------------------------------------------------------

/// One feature's codebook, as sampled by NumPy.
///
/// All randomness stays on the Python side: this only ever holds arrays that
/// were handed across the FFI boundary.
enum FeatureCodebook {
    /// `[num_categories, D]` lookup table; the feature value is a row index.
    Categorical(Array2<f64>),
    /// FHRR fractional-power encoder. `ln_spectrum` is `ln(spectrum)`
    /// precomputed once — it does not depend on the atom, and hoisting it out
    /// of the per-atom loop is bit-identical while halving the transcendental
    /// work (`z^e` is otherwise `exp(e * ln z)` with `ln z` recomputed every
    /// time).
    Continuous {
        ln_spectrum: Array1<Complex64>,
        bandwidth: f64,
    },
}

/// Per-thread scratch for composite node encoding.
struct CompositeScratch {
    /// Full-length complex buffer for the continuous branch's inverse FFT.
    full: Vec<Complex64>,
    /// Scratch required by the rustfft plan.
    fft_scratch: Vec<Complex64>,
    /// Real-valued per-feature hypervector.
    real: Vec<f64>,
    /// Half-spectrum of the current feature.
    spec: Vec<Complex64>,
    /// Running product of half-spectra across features.
    acc: Vec<Complex64>,
    /// Scratch required by the realfft forward plan.
    r2c_scratch: Vec<Complex64>,
    /// Scratch required by the realfft inverse plan.
    c2r_scratch: Vec<Complex64>,
}

impl CompositeScratch {
    fn new(
        cfft: &Arc<dyn Fft<f64>>,
        r2c: &Arc<dyn RealToComplex<f64>>,
        c2r: &Arc<dyn ComplexToReal<f64>>,
        d: usize,
    ) -> Self {
        let complex_len = d / 2 + 1;
        Self {
            full: vec![Complex64::new(0.0, 0.0); d],
            fft_scratch: vec![Complex64::new(0.0, 0.0); cfft.get_inplace_scratch_len()],
            real: vec![0.0; d],
            spec: vec![Complex64::new(0.0, 0.0); complex_len],
            acc: vec![Complex64::new(0.0, 0.0); complex_len],
            r2c_scratch: r2c.make_scratch_vec(),
            c2r_scratch: c2r.make_scratch_vec(),
        }
    }
}

/// Encode a feature matrix `[N, F]` into node hypervectors `[N, D]`.
///
/// Mirrors `CompositeEncoder.encode` in `hyper_fingerprints/codebook.py`
/// operation for operation:
///
/// 1. per feature, build a real hypervector
///    - categorical: a codebook row
///    - continuous: `ifft(spectrum ** (value / bandwidth)).real`
/// 2. multiply the per-feature spectra together (circular convolution)
/// 3. one inverse transform back to the real domain
///
/// Two numerical points make the transform choice non-obvious:
///
/// - The **continuous branch needs a full complex inverse FFT**. The spectrum
///   is Hermitian-symmetric by construction, but its DC bin (and, for even
///   `d`, its Nyquist bin) is real and frequently *negative*; a negative real
///   raised to a non-integer power leaves the real axis, which breaks the
///   Hermitian symmetry. NumPy discards the resulting imaginary component via
///   `.real`, and a `c2r` transform — which assumes symmetry — would produce a
///   materially different vector, not merely a rounding difference.
/// - Everything *after* that `.real` is genuinely real, so its forward
///   transform is Hermitian and the remaining steps use the cheaper r2c / c2r
///   pair.
fn composite_encode_inner(
    values: &Array2<f64>,
    codebooks: &[FeatureCodebook],
) -> Array2<f64> {
    let n = values.nrows();
    let d = match codebooks.first() {
        Some(FeatureCodebook::Categorical(table)) => table.ncols(),
        Some(FeatureCodebook::Continuous { ln_spectrum, .. }) => ln_spectrum.len(),
        None => 0,
    };

    if n == 0 || d == 0 {
        return Array2::zeros((n, d));
    }

    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(d);
    let c2r = planner.plan_fft_inverse(d);
    let cfft = FftPlanner::<f64>::new().plan_fft_inverse(d);

    let complex_len = d / 2 + 1;
    let inv_d = 1.0 / d as f64;

    let cfft_ref = Arc::clone(&cfft);
    let r2c_ref = Arc::clone(&r2c);
    let c2r_ref = Arc::clone(&c2r);
    thread_local! {
        static SCRATCH: RefCell<Option<(usize, CompositeScratch)>> = const { RefCell::new(None) };
    }

    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            SCRATCH.with(|cell| {
                let mut borrow = cell.borrow_mut();
                if borrow.as_ref().map_or(true, |(sz, _)| *sz != d) {
                    *borrow = Some((
                        d,
                        CompositeScratch::new(&cfft_ref, &r2c_ref, &c2r_ref, d),
                    ));
                }
                let s = &mut borrow.as_mut().unwrap().1;

                s.acc.fill(Complex64::new(1.0, 0.0));

                for (f, cb) in codebooks.iter().enumerate() {
                    let value = values[[i, f]];

                    match cb {
                        FeatureCodebook::Categorical(table) => {
                            let idx = value as usize;
                            let row = table.row(idx);
                            s.real
                                .iter_mut()
                                .zip(row.iter())
                                .for_each(|(dst, src)| *dst = *src);
                        }
                        FeatureCodebook::Continuous {
                            ln_spectrum,
                            bandwidth,
                        } => {
                            // spectrum ** e == exp(e * ln spectrum)
                            let e = value / bandwidth;
                            for k in 0..d {
                                s.full[k] = (ln_spectrum[k] * e).exp();
                            }
                            // Full complex inverse FFT; rustfft is unnormalized.
                            cfft_ref.process_with_scratch(&mut s.full, &mut s.fft_scratch);
                            for k in 0..d {
                                s.real[k] = s.full[k].re * inv_d;
                            }
                        }
                    }

                    r2c_ref
                        .process_with_scratch(&mut s.real, &mut s.spec, &mut s.r2c_scratch)
                        .expect("r2c failed");
                    for k in 0..complex_len {
                        s.acc[k] *= s.spec[k];
                    }
                }

                c2r_ref
                    .process_with_scratch(&mut s.acc, &mut s.real, &mut s.c2r_scratch)
                    .expect("c2r failed");
                s.real.iter().map(|v| v * inv_d).collect::<Vec<f64>>()
            })
        })
        .collect();

    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n, d), flat).unwrap()
}

/// Composite node encoding: feature values `[N, F]` -> hypervectors `[N, D]`.
///
/// Codebooks are sampled in NumPy and passed in; nothing is drawn here.
///
/// Args:
///     values: [N, F] raw feature values (categorical ones are row indices)
///     kinds: per feature, 0 = categorical, 1 = continuous
///     categorical: [C_f, D] tables, in the order the categorical features appear
///     spectra: [D] complex spectra, in the order the continuous features appear
///     bandwidths: one per continuous feature, same order as `spectra`
///
/// Returns:
///     node hypervectors [N, D]
#[pyfunction]
fn composite_encode_rs<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    kinds: Vec<u8>,
    categorical: Vec<PyReadonlyArray2<'py, f64>>,
    spectra: Vec<PyReadonlyArray1<'py, Complex64>>,
    bandwidths: Vec<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let values = values.to_owned_array();

    if values.ncols() != kinds.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "values has {} columns but {} feature kinds were given",
            values.ncols(),
            kinds.len()
        )));
    }
    if spectra.len() != bandwidths.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{} spectra but {} bandwidths",
            spectra.len(),
            bandwidths.len()
        )));
    }

    // Consume the categorical tables and continuous spectra in feature order.
    let mut cat_iter = categorical.into_iter();
    let mut cont_iter = spectra.into_iter().zip(bandwidths);
    let mut codebooks: Vec<FeatureCodebook> = Vec::with_capacity(kinds.len());
    let mut dim: Option<usize> = None;

    for (f, kind) in kinds.iter().enumerate() {
        match kind {
            0 => {
                let table = cat_iter.next().ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "feature {f} is categorical but no codebook was given for it"
                    ))
                })?;
                let table = table.to_owned_array();
                if table.nrows() == 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "feature {f}: categorical codebook has no rows"
                    )));
                }
                codebooks.push(FeatureCodebook::Categorical(table));
            }
            1 => {
                let (spectrum, bandwidth) = cont_iter.next().ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "feature {f} is continuous but no spectrum was given for it"
                    ))
                })?;
                if bandwidth == 0.0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "feature {f}: bandwidth must be non-zero"
                    )));
                }
                let ln_spectrum = spectrum.to_owned_array().mapv(|z| z.ln());
                codebooks.push(FeatureCodebook::Continuous {
                    ln_spectrum,
                    bandwidth,
                });
            }
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "feature {f}: kind must be 0 (categorical) or 1 (continuous), got {other}"
                )))
            }
        }

        let d = match codebooks.last().unwrap() {
            FeatureCodebook::Categorical(t) => t.ncols(),
            FeatureCodebook::Continuous { ln_spectrum, .. } => ln_spectrum.len(),
        };
        match dim {
            None => dim = Some(d),
            Some(expected) if expected != d => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "feature {f} has dimension {d}, expected {expected}"
                )))
            }
            _ => {}
        }
    }

    // Bounds-check categorical indices here, where a clear Python exception can
    // still be raised (the kernel itself runs without the GIL).
    for (f, cb) in codebooks.iter().enumerate() {
        if let FeatureCodebook::Categorical(table) = cb {
            let rows = table.nrows() as f64;
            for i in 0..values.nrows() {
                let v = values[[i, f]];
                if !(v >= 0.0 && v < rows) || v.fract() != 0.0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "feature {f}: categorical value {v} at row {i} is not a valid index \
                         into a codebook with {} rows",
                        table.nrows()
                    )));
                }
            }
        }
    }

    let out = py.detach(|| composite_encode_inner(&values, &codebooks));
    Ok(out.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// SMILES parsing + feature extraction (purr-based)
// ---------------------------------------------------------------------------

/// Get the element symbol string for a purr atom (always uppercase first letter).
fn atom_symbol(atom: &Atom) -> Option<String> {
    let raw = match &atom.kind {
        AtomKind::Star => return None,
        AtomKind::Aliphatic(a) => a.to_string(),
        AtomKind::Aromatic(a) => a.to_string(),
        AtomKind::Bracket { symbol, .. } => symbol.to_string(),
    };
    // Capitalize first letter (aromatic Display gives lowercase: "n" -> "N")
    let mut chars = raw.chars();
    match chars.next() {
        Some(first) => Some(first.to_uppercase().to_string() + chars.as_str()),
        None => Some(raw),
    }
}

/// Convert purr Charge enum to i8.
fn charge_to_i8(charge: &Option<Charge>) -> i8 {
    match charge {
        None => 0,
        Some(c) => {
            let i: i8 = c.into();
            i
        }
    }
}

/// Extract a single feature value from a purr atom given a feature name.
/// Returns None if the atom can't provide this feature (e.g., unsupported element).
fn extract_feature(
    atom: &Atom,
    feature: &str,
    atom_to_idx: &HashMap<String, usize>,
) -> Result<i64, String> {
    match feature {
        "element" => {
            let sym = atom_symbol(atom)
                .ok_or_else(|| "Star atoms not supported".to_string())?;
            atom_to_idx.get(&sym)
                .map(|&idx| idx as i64)
                .ok_or_else(|| format!("Atom '{}' not in supported atom types", sym))
        }
        "degree" => {
            let deg = atom.bonds.len().min(5);
            Ok(deg as i64)
        }
        "charge" => {
            let c = match &atom.kind {
                AtomKind::Bracket { charge, .. } => charge_to_i8(charge),
                _ => 0,
            };
            Ok(if c == 0 { 0 } else if c > 0 { 1 } else { 2 })
        }
        "hydrogens" => {
            let h = atom.suppressed_hydrogens().min(3);
            Ok(h as i64)
        }
        "aromatic" => {
            Ok(if atom.is_aromatic() { 1 } else { 0 })
        }
        _ => Err(format!("Unknown feature: {}", feature)),
    }
}

/// Parse SMILES and extract batched graph data.
///
/// This replaces Python stages 1-4 (SMILES parsing, mol_to_data,
/// batch_from_data_list, codebook index computation) in a single Rust call.
fn prepare_batch_inner(
    smiles_list: &[String],
    atom_to_idx: &HashMap<String, usize>,
    feature_names: &[String],
    feature_bins: &[usize],
) -> Result<(Array1<i64>, Array2<i64>, Array1<i64>, usize), String> {
    // Parse all molecules
    let mut all_feature_indices: Vec<i64> = Vec::new();
    let mut all_srcs: Vec<i64> = Vec::new();
    let mut all_dsts: Vec<i64> = Vec::new();
    let mut all_batch: Vec<i64> = Vec::new();
    let mut node_offset: i64 = 0;

    for (mol_idx, smi) in smiles_list.iter().enumerate() {
        let mut builder = Builder::new();
        read(smi, &mut builder, None)
            .map_err(|e| format!("Invalid SMILES: {:?} ({:?})", smi, e))?;
        let atoms = builder.build()
            .map_err(|e| format!("Failed to build graph for '{}': {:?}", smi, e))?;

        let n_atoms = atoms.len();

        for atom in &atoms {
            // Extract features and compute flat codebook index
            let mut flat_idx: i64 = 0;
            let mut stride: i64 = 1;
            // Compute flat index using mixed-radix encoding (reverse order for TupleIndexer compat)
            for (f_idx, fname) in feature_names.iter().enumerate().rev() {
                let val = extract_feature(atom, fname, atom_to_idx)?;
                flat_idx += val * stride;
                stride *= feature_bins[f_idx] as i64;
            }
            all_feature_indices.push(flat_idx);
            all_batch.push(mol_idx as i64);
        }

        // Extract edges (bidirectional)
        for atom_idx in 0..n_atoms {
            for bond in &atoms[atom_idx].bonds {
                let target = bond.tid;
                // Only add each direction once: atom_idx -> target
                // purr stores bonds bidirectionally in the adjacency list,
                // so only add when atom_idx < target to avoid duplicates,
                // then add both directions.
                if atom_idx < target {
                    all_srcs.push(node_offset + atom_idx as i64);
                    all_dsts.push(node_offset + target as i64);
                    all_srcs.push(node_offset + target as i64);
                    all_dsts.push(node_offset + atom_idx as i64);
                }
            }
        }

        node_offset += n_atoms as i64;
    }

    let n_edges = all_srcs.len();
    let num_graphs = smiles_list.len();

    let feature_indices = Array1::from_vec(all_feature_indices);
    let batch_indices = Array1::from_vec(all_batch);

    let edge_index = if n_edges > 0 {
        let mut ei = Array2::zeros((2, n_edges));
        for e in 0..n_edges {
            ei[[0, e]] = all_srcs[e];
            ei[[1, e]] = all_dsts[e];
        }
        ei
    } else {
        Array2::zeros((2, 0))
    };

    Ok((feature_indices, edge_index, batch_indices, num_graphs))
}

/// Parse SMILES strings and extract batched graph data for encoding.
///
/// Replaces Python stages 1-4 in a single Rust call.
///
/// Args:
///     smiles_list: list of SMILES strings
///     atom_to_idx: mapping of atom symbol -> index
///     feature_names: list of feature names (e.g., ["element", "degree", ...])
///     feature_bins: list of bin counts per feature
///
/// Returns:
///     (feature_indices [N], edge_index [2, E], batch_indices [N], num_graphs)
#[pyfunction]
fn prepare_batch_rs<'py>(
    py: Python<'py>,
    smiles_list: Vec<String>,
    atom_to_idx: HashMap<String, usize>,
    feature_names: Vec<String>,
    feature_bins: Vec<usize>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<i64>>,
    usize,
)> {
    let (feat_idx, edge_idx, batch_idx, num_graphs) = py.detach(|| {
        prepare_batch_inner(&smiles_list, &atom_to_idx, &feature_names, &feature_bins)
    }).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    Ok((
        feat_idx.into_pyarray(py),
        edge_idx.into_pyarray(py),
        batch_idx.into_pyarray(py),
        num_graphs,
    ))
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Native acceleration module for hyper_fingerprints.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_batch_rs, m)?)?;
    m.add_function(wrap_pyfunction!(encode_batch_hv_rs, m)?)?;
    m.add_function(wrap_pyfunction!(composite_encode_rs, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_batch_rs, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Rust-side tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Binding with identity vector [1, 0, 0, ...] should return the other vector.
    #[test]
    fn test_bind_with_identity() {
        let d = 8;
        let mut identity = Array2::zeros((1, d));
        identity[[0, 0]] = 1.0;

        let v = Array2::from_shape_vec((1, d), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();

        let result = hrr_bind_rows(&v, &identity);

        for j in 0..d {
            assert!(
                (result[[0, j]] - v[[0, j]]).abs() < 1e-12,
                "Mismatch at col {}: got {}, expected {}",
                j,
                result[[0, j]],
                v[[0, j]]
            );
        }
    }

    /// Binding is commutative: bind(a, b) == bind(b, a).
    #[test]
    fn test_bind_commutative() {
        let a = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let b = array![[0.5, -1.0, 0.3, 2.0], [-0.1, 0.7, 1.2, -0.5]];

        let ab = hrr_bind_rows(&a, &b);
        let ba = hrr_bind_rows(&b, &a);

        for i in 0..2 {
            for j in 0..4 {
                assert!(
                    (ab[[i, j]] - ba[[i, j]]).abs() < 1e-12,
                    "Not commutative at [{}, {}]: {} != {}",
                    i,
                    j,
                    ab[[i, j]],
                    ba[[i, j]]
                );
            }
        }
    }

    /// Empty inputs should return empty output.
    #[test]
    fn test_bind_empty() {
        let a = Array2::<f64>::zeros((0, 4));
        let b = Array2::<f64>::zeros((0, 4));
        let result = hrr_bind_rows(&a, &b);
        assert_eq!(result.shape(), &[0, 4]);
    }

    /// Single-element dimension (D=1): circular convolution is just multiplication.
    #[test]
    fn test_bind_d1() {
        let a = array![[3.0]];
        let b = array![[5.0]];
        let result = hrr_bind_rows(&a, &b);
        assert!((result[[0, 0]] - 15.0).abs() < 1e-12);
    }

    // -- scatter tests --

    /// Bundle: accumulate rows by index.
    #[test]
    fn test_scatter_bundle_basic() {
        // Two source rows mapping to the same bucket
        let src = array![[1.0, 2.0], [3.0, 4.0]];
        let index = Array1::from_vec(vec![0_i64, 0]);
        let result = scatter_bundle(&src, &index, 1);
        assert_eq!(result.shape(), &[1, 2]);
        assert!((result[[0, 0]] - 4.0).abs() < 1e-12);
        assert!((result[[0, 1]] - 6.0).abs() < 1e-12);
    }

    /// Bundle: rows into different buckets.
    #[test]
    fn test_scatter_bundle_multi_bucket() {
        let src = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let index = Array1::from_vec(vec![0_i64, 1, 0]);
        let result = scatter_bundle(&src, &index, 2);
        assert_eq!(result.shape(), &[2, 2]);
        // bucket 0: [1,2] + [5,6] = [6,8]
        assert!((result[[0, 0]] - 6.0).abs() < 1e-12);
        assert!((result[[0, 1]] - 8.0).abs() < 1e-12);
        // bucket 1: [3,4]
        assert!((result[[1, 0]] - 3.0).abs() < 1e-12);
        assert!((result[[1, 1]] - 4.0).abs() < 1e-12);
    }

    /// Bundle: empty index returns identity.
    #[test]
    fn test_scatter_bundle_empty() {
        let src = Array2::<f64>::zeros((0, 4));
        let index = Array1::<i64>::from_vec(vec![]);
        let result = scatter_bundle(&src, &index, 2);
        assert_eq!(result.shape(), &[2, 4]);
        // Should be identity: col 0 = 1, rest = 0
        assert!((result[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((result[[0, 1]] - 0.0).abs() < 1e-12);
        assert!((result[[1, 0]] - 1.0).abs() < 1e-12);
    }

    /// Bind: element-wise product by index.
    #[test]
    fn test_scatter_bind_basic() {
        let src = array![[2.0, 3.0], [4.0, 5.0]];
        let index = Array1::from_vec(vec![0_i64, 0]);
        let result = scatter_bind_op(&src, &index, 1);
        assert_eq!(result.shape(), &[1, 2]);
        // identity col 0 starts at 1.0: 1.0 * 2.0 * 4.0 = 8.0
        assert!((result[[0, 0]] - 8.0).abs() < 1e-12);
        // identity col 1 starts at 0.0: 0.0 * 3.0 * 5.0 = 0.0
        assert!((result[[0, 1]] - 0.0).abs() < 1e-12);
    }

    /// Bind: empty index returns identity.
    #[test]
    fn test_scatter_bind_empty() {
        let src = Array2::<f64>::zeros((0, 4));
        let index = Array1::<i64>::from_vec(vec![]);
        let result = scatter_bind_op(&src, &index, 3);
        assert_eq!(result.shape(), &[3, 4]);
        assert!((result[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((result[[2, 0]] - 1.0).abs() < 1e-12);
        assert!((result[[0, 1]] - 0.0).abs() < 1e-12);
    }

    // ---- encode_batch_from_hv / NormMode ----

    /// Helper: two-node, two-molecule edgeless batch of unit-ish hypervectors.
    fn edgeless_batch() -> (Array2<f64>, Array2<i64>, Array1<i64>) {
        let node_hv = array![[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 0.25]];
        let edge_index = Array2::<i64>::zeros((2, 0));
        let batch = Array1::from_vec(vec![0_i64, 1]);
        (node_hv, edge_index, batch)
    }

    /// empty_identity = false (composite): layers 1..depth aggregate to zero,
    /// so the multibundle is just the layer-0 hypervector.
    #[test]
    fn test_from_hv_edgeless_zeros() {
        let (node_hv, edge_index, batch) = edgeless_batch();
        let (graph_emb, node_terms) =
            encode_batch_from_hv(&node_hv, &edge_index, &batch, 2, 3, NormMode::None, false);
        for g in 0..2 {
            for j in 0..4 {
                assert!((graph_emb[[g, j]] - node_hv[[g, j]]).abs() < 1e-12);
                assert!((node_terms[[g, j]] - node_hv[[g, j]]).abs() < 1e-12);
            }
        }
    }

    /// empty_identity = true (legacy): an edgeless batch aggregates to the HRR
    /// identity, so every layer equals layer 0 and the multibundle scales by
    /// depth + 1.
    #[test]
    fn test_from_hv_edgeless_identity() {
        let (node_hv, edge_index, batch) = edgeless_batch();
        let (graph_emb, _) =
            encode_batch_from_hv(&node_hv, &edge_index, &batch, 2, 3, NormMode::None, true);
        for g in 0..2 {
            for j in 0..4 {
                assert!((graph_emb[[g, j]] - 4.0 * node_hv[[g, j]]).abs() < 1e-12);
            }
        }
    }

    /// Gathering from a codebook and passing the gathered rows directly must
    /// give identical results.
    #[test]
    fn test_from_hv_matches_codebook_gather() {
        let codebook = array![[0.1, 0.2, 0.3, 0.4], [-0.5, 0.6, 0.7, -0.8]];
        let feature_indices = Array1::from_vec(vec![0_i64, 1, 1]);
        let edge_index = array![[0_i64, 1, 1, 2], [1, 0, 2, 1]];
        let batch = Array1::from_vec(vec![0_i64, 0, 0]);

        let (via_codebook, nt_a) =
            encode_batch_inner(&codebook, &feature_indices, &edge_index, &batch, 1, 2, NormMode::LegacyEps);

        let mut node_hv = Array2::zeros((3, 4));
        for i in 0..3 {
            node_hv.row_mut(i).assign(&codebook.row(feature_indices[i] as usize));
        }
        let (via_hv, nt_b) =
            encode_batch_from_hv(&node_hv, &edge_index, &batch, 1, 2, NormMode::LegacyEps, true);

        for j in 0..4 {
            assert_eq!(via_codebook[[0, j]], via_hv[[0, j]]);
            assert_eq!(nt_a[[0, j]], nt_b[[0, j]]);
        }
    }

    /// The two normalization formulas agree for well-scaled vectors but differ
    /// in their epsilon handling; both produce unit-norm rows here.
    #[test]
    fn test_norm_mode_scales() {
        assert_eq!(NormMode::None.inv_scale(4.0), 1.0);
        assert!((NormMode::LegacyEps.inv_scale(4.0) - 1.0 / (2.0 + 1e-8)).abs() < 1e-18);
        assert!((NormMode::L2Clamp.inv_scale(4.0) - 0.5).abs() < 1e-18);
        // Zero vector: legacy blows up to 1e8, l2 clamps at 1e12.
        assert!((NormMode::LegacyEps.inv_scale(0.0) - 1e8).abs() < 1.0);
        assert!((NormMode::L2Clamp.inv_scale(0.0) - 1e12).abs() < 1.0);
    }

    #[test]
    fn test_norm_mode_from_str() {
        assert_eq!(NormMode::from_str("none").unwrap(), NormMode::None);
        assert_eq!(NormMode::from_str("legacy").unwrap(), NormMode::LegacyEps);
        assert_eq!(NormMode::from_str("l2").unwrap(), NormMode::L2Clamp);
        assert!(NormMode::from_str("bogus").is_err());
    }

    /// D = 1 and a zero-molecule batch must not panic.
    #[test]
    fn test_from_hv_degenerate_shapes() {
        let node_hv = array![[2.0], [3.0]];
        let edge_index = array![[0_i64, 1], [1, 0]];
        let batch = Array1::from_vec(vec![0_i64, 0]);
        let (graph_emb, _) =
            encode_batch_from_hv(&node_hv, &edge_index, &batch, 1, 1, NormMode::None, false);
        assert_eq!(graph_emb.shape(), &[1, 1]);

        let empty = Array2::<f64>::zeros((0, 8));
        let (ge, nt) = encode_batch_from_hv(
            &empty,
            &Array2::<i64>::zeros((2, 0)),
            &Array1::<i64>::from_vec(vec![]),
            0,
            2,
            NormMode::L2Clamp,
            false,
        );
        assert_eq!(ge.shape(), &[0, 8]);
        assert_eq!(nt.shape(), &[0, 8]);
    }

    // ---- composite_encode_inner ----

    /// A single categorical feature must reproduce its codebook row exactly:
    /// binding nothing else onto it is the identity.
    #[test]
    fn test_composite_single_categorical() {
        let table = array![[0.1, 0.2, 0.3, 0.4], [-0.5, 0.6, 0.7, -0.8]];
        let codebooks = vec![FeatureCodebook::Categorical(table.clone())];
        let values = array![[1.0], [0.0], [1.0]];
        let out = composite_encode_inner(&values, &codebooks);
        assert_eq!(out.shape(), &[3, 4]);
        for j in 0..4 {
            assert!((out[[0, j]] - table[[1, j]]).abs() < 1e-12);
            assert!((out[[1, j]] - table[[0, j]]).abs() < 1e-12);
            assert!((out[[2, j]] - table[[1, j]]).abs() < 1e-12);
        }
    }

    /// Two categorical features must bind to their circular convolution.
    #[test]
    fn test_composite_two_categorical_is_convolution() {
        let a = array![[1.0, 2.0, 3.0, 4.0]];
        let b = array![[0.5, -1.0, 2.0, 0.25]];
        let codebooks = vec![
            FeatureCodebook::Categorical(a.clone()),
            FeatureCodebook::Categorical(b.clone()),
        ];
        let out = composite_encode_inner(&array![[0.0, 0.0]], &codebooks);
        let expected = hrr_bind_rows(&a, &b);
        for j in 0..4 {
            assert!((out[[0, j]] - expected[[0, j]]).abs() < 1e-12);
        }
    }

    /// Exponent 0 makes the powered spectrum all-ones, whose inverse FFT is
    /// the HRR identity [1, 0, 0, ...].
    #[test]
    fn test_composite_continuous_zero_exponent_is_identity() {
        let d = 8;
        let ln_spectrum = Array1::from_shape_fn(d, |k| {
            Complex64::new(0.3 + k as f64 * 0.1, 0.2 - k as f64 * 0.05).ln()
        });
        let codebooks = vec![FeatureCodebook::Continuous {
            ln_spectrum,
            bandwidth: 2.0,
        }];
        let out = composite_encode_inner(&array![[0.0]], &codebooks);
        assert!((out[[0, 0]] - 1.0).abs() < 1e-12);
        for j in 1..d {
            assert!(out[[0, j]].abs() < 1e-12, "bin {j} = {}", out[[0, j]]);
        }
    }

    /// A continuous feature must survive a negative-real DC bin: raising it to
    /// a non-integer power breaks Hermitian symmetry, and the real part of the
    /// full complex inverse FFT is what NumPy keeps. A c2r transform would
    /// give a different answer here, so this pins the branch.
    #[test]
    fn test_composite_continuous_negative_dc_bin() {
        let d = 8;
        // Hermitian spectrum with a deliberately negative real DC bin.
        let mut spectrum = vec![Complex64::new(0.0, 0.0); d];
        spectrum[0] = Complex64::new(-0.75, 0.0);
        spectrum[4] = Complex64::new(-1.25, 0.0);
        for k in 1..4 {
            let z = Complex64::new(0.4 * k as f64, 0.3 - 0.1 * k as f64);
            spectrum[k] = z;
            spectrum[d - k] = z.conj();
        }
        let ln_spectrum = Array1::from_shape_fn(d, |k| spectrum[k].ln());
        let codebooks = vec![FeatureCodebook::Continuous {
            ln_spectrum,
            bandwidth: 2.0,
        }];
        // exponent 1.5 — the case that leaves the real axis.
        let out = composite_encode_inner(&array![[3.0]], &codebooks);
        assert_eq!(out.shape(), &[1, d]);
        assert!(out.iter().all(|v| v.is_finite()));
        // A Hermitian-assuming transform would drop the discarded imaginary
        // energy; the real inverse must not be all-zero or symmetric-trivial.
        assert!(out.iter().any(|v| v.abs() > 1e-6));
    }

    /// Degenerate shapes must not panic.
    #[test]
    fn test_composite_degenerate_shapes() {
        let table = array![[1.0, 2.0]];
        let codebooks = vec![FeatureCodebook::Categorical(table)];
        let out = composite_encode_inner(&Array2::<f64>::zeros((0, 1)), &codebooks);
        assert_eq!(out.shape(), &[0, 2]);

        let d1 = vec![FeatureCodebook::Categorical(array![[3.0]])];
        let out = composite_encode_inner(&array![[0.0]], &d1);
        assert_eq!(out.shape(), &[1, 1]);
        assert!((out[[0, 0]] - 3.0).abs() < 1e-12);

        let out = composite_encode_inner(&Array2::<f64>::zeros((2, 0)), &[]);
        assert_eq!(out.shape(), &[2, 0]);
    }
}
