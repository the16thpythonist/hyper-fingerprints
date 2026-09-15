"""Type stubs for the Rust extension module."""

import numpy as np
import numpy.typing as npt

def prepare_batch_rs(
    smiles_list: list[str],
    atom_to_idx: dict[str, int],
    feature_names: list[str],
    feature_bins: list[int],
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    int,
]:
    """Parse SMILES and extract batched graph data (Rust-accelerated).

    Replaces Python stages 1-4 (SMILES parsing, feature extraction,
    batching, codebook index computation) in a single call.

    Parameters
    ----------
    smiles_list
        List of SMILES strings.
    atom_to_idx
        Atom symbol to index mapping.
    feature_names
        Feature names (e.g., ``["element", "degree", "charge"]``).
    feature_bins
        Bin count per feature.

    Returns
    -------
    tuple
        ``(feature_indices [N], edge_index [2, E], batch_indices [N], num_graphs)``
    """
    ...

def encode_batch_rs(
    codebook: npt.NDArray[np.float64],
    feature_indices: npt.NDArray[np.int64],
    edge_index: npt.NDArray[np.int64],
    batch_indices: npt.NDArray[np.int64],
    num_graphs: int,
    depth: int,
    normalize: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Run the HRR message-passing pipeline (Rust-accelerated).

    Parameters
    ----------
    codebook
        Shape ``(num_categories, D)``.
    feature_indices
        Shape ``(N,)`` — flat codebook indices per node.
    edge_index
        Shape ``(2, E)`` — source and destination node indices.
    batch_indices
        Shape ``(N,)`` — graph membership per node.
    num_graphs
        Number of graphs in the batch.
    depth
        Number of message-passing layers.
    normalize
        Whether to L2-normalize after each layer.

    Returns
    -------
    tuple[ndarray, ndarray]
        ``(graph_embedding, node_terms)``, each shape ``(num_graphs, D)``.
    """
    ...

def encode_batch_hv_rs(
    node_hv: npt.NDArray[np.float64],
    edge_index: npt.NDArray[np.int64],
    batch_indices: npt.NDArray[np.int64],
    num_graphs: int,
    depth: int,
    norm_mode: str,
    empty_identity: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Run HRR message passing over pre-computed node hypervectors.

    Used by the composite encoder, whose node hypervectors are built in Python
    from per-feature codebooks and cannot be expressed as rows of a single
    product codebook.

    Parameters
    ----------
    node_hv
        Shape ``(N, D)`` — one hypervector per node.
    edge_index
        Shape ``(2, E)`` — source and destination node indices.
    batch_indices
        Shape ``(N,)`` — graph membership per node.
    num_graphs
        Number of graphs in the batch.
    depth
        Number of message-passing layers.
    norm_mode
        ``"none"``, ``"legacy"`` (``x / (|x| + 1e-8)``) or ``"l2"``
        (``x / max(|x|, 1e-12)``).
    empty_identity
        For an edgeless batch, whether aggregation yields the HRR identity
        (legacy ``scatter_hd(empty="identity")``) or zeros (composite).

    Returns
    -------
    tuple[ndarray, ndarray]
        ``(graph_embedding, node_terms)``, each shape ``(num_graphs, D)``.
    """
    ...

def composite_encode_rs(
    values: npt.NDArray[np.float64],
    kinds: list[int],
    categorical: list[npt.NDArray[np.float64]],
    spectra: list[npt.NDArray[np.complex128]],
    bandwidths: list[float],
) -> npt.NDArray[np.float64]:
    """Composite node encoding (Rust-accelerated).

    Binds one hypervector per feature into a single hypervector per row:
    categorical features are codebook lookups, continuous features are FHRR
    fractional powers ``ifft(spectrum ** (value / bandwidth)).real``.

    All codebooks are sampled in NumPy and passed in; nothing is drawn here.

    Parameters
    ----------
    values
        Shape ``(N, F)`` — raw feature values; categorical ones are row indices.
    kinds
        Per feature, ``0`` = categorical, ``1`` = continuous.
    categorical
        ``(C_f, D)`` tables, in the order the categorical features appear.
    spectra
        ``(D,)`` complex spectra, in the order the continuous features appear.
    bandwidths
        One per continuous feature, in the same order as ``spectra``.

    Returns
    -------
    ndarray
        Node hypervectors, shape ``(N, D)``.
    """
    ...
