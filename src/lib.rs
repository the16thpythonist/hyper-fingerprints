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

/// Encode a single molecule's message-passing pipeline.
///
/// `batch_has_edges`: whether the parent batch contains any edges at all.
/// This controls no-edge molecule behavior to match Python's batched scatter_hd:
/// - batch has edges → no-edge molecules get zeros aggregation → bind(x,0)=0
/// - batch has NO edges → identity aggregation → bind(x,id)=x
fn encode_single_molecule(
    node_hv: &Array2<f64>,
    local_srcs: &[usize],
    local_dsts: &[usize],
    depth: usize,
    normalize: bool,
    batch_has_edges: bool,
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

            if normalize {
                current_layer = Array2::zeros((n, d));
                for i in 0..n {
                    let mut norm = 0.0f64;
                    for j in 0..d {
                        norm += hr[[i, j]] * hr[[i, j]];
                    }
                    let inv_norm = 1.0 / (norm.sqrt() + 1e-8);
                    for j in 0..d {
                        current_layer[[i, j]] = hr[[i, j]] * inv_norm;
                    }
                }
            } else {
                current_layer = hr;
            }

            multibundle_sum += &current_layer;
        }
    } else if depth > 0 && !batch_has_edges {
        // No edges in entire batch: Python scatter_hd returns identity.
        // bind(x, identity) = x → all layers equal layer 0.
        // multibundle = (depth+1) * node_hv.
        for _ in 0..depth {
            multibundle_sum += &current_layer;
        }
    }
    // else: depth > 0 && batch_has_edges && no local edges
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
    normalize: bool,
) -> (Array2<f64>, Array2<f64>) {
    let n = feature_indices.len();
    let d = codebook.ncols();
    let batch_has_edges = edge_index.ncols() > 0;

    // Step 1: Gather node hypervectors from codebook
    let mut node_hv = Array2::zeros((n, d));
    for i in 0..n {
        let idx = feature_indices[i] as usize;
        node_hv.row_mut(i).assign(&codebook.row(idx));
    }

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
                depth, normalize, batch_has_edges,
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

    let (graph_emb, node_terms) = py.detach(|| {
        encode_batch_inner(
            &codebook,
            &feature_indices,
            &edge_index,
            &batch_indices,
            num_graphs,
            depth,
            normalize,
        )
    });

    (graph_emb.into_pyarray(py), node_terms.into_pyarray(py))
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
}
