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
