#!/usr/bin/env python
"""Export a unit-modulus reference fixture from graph_hdc.

graph_hdc samples unit-modulus codebooks by default (``unit_modulus=True`` in
``AtomEncoder`` / ``ContinuousEncoder`` since commit d740800). This script
records what graph_hdc itself produces, so that ``test_graph_hdc_unit_parity.py``
can check hyper_fingerprints against it without torch:

- ``node/*`` / ``graph/*`` -- the exact unit-modulus base vectors / spectra of a
  continuous-mode HyperNet (same layout as ``legacy_codebook_graph_hdc.npz``,
  readable with ``Encoder.from_legacy_codebook``),
- ``ref/<case>`` / ``ref0/<case>`` -- HyperNet graph embeddings and layer-0
  node terms for a fixed SMILES list under eight configurations,
- ``seeded/d<D>_s<seed>/...`` -- the FPE base spectra graph_hdc samples for a
  given seed, which hyper_fingerprints must reproduce seed for seed,
- ``atoms_gaussian`` / ``atoms_unit`` -- one AtomEncoder draw with and without
  the unit-modulus transform (graph_hdc draws categorical vectors with torch,
  so only the transform itself can be compared).

It must run in an environment with graph_hdc (and torch) installed, e.g.:

    /path/to/graph_hdc/.venv/bin/python tests/generate_graph_hdc_unit_fixture.py
"""

import json
import subprocess
from pathlib import Path

import numpy as np
from rdkit import Chem

import graph_hdc
from graph_hdc.models import HyperNet
from graph_hdc.special.molecules import (
    AtomEncoder,
    graph_dict_from_mol,
    make_molecule_graph_encoder_map_cont,
    make_molecule_node_encoder_map_cont,
)

OUT_PATH = Path(__file__).parent / "fixtures" / "graph_hdc_unit_codebook.npz"

SEED = 0
DIMENSION = 256
SEEDED = [(64, 0), (64, 7), (256, 0), (256, 7)]

# Same molecules as legacy_codebook_graph_hdc.npz
SMILES = [
    "CC", "C=C", "c1ccccc1", "CCO", "CC(=O)O", "CC(=O)Oc1ccccc1C(=O)O", "CN",
    "c1ccncc1", "c1ccc(N)cc1", "c1ccc2[nH]ccc2c1", "CC(=O)Nc1ccc(O)cc1", "CF",
    "c1ccc(F)cc1", "FC(F)(F)c1ccccc1", "CCl", "c1ccc(Cl)cc1", "ClC(Cl)Cl", "CBr",
    "c1ccc(Br)cc1", "CI", "c1ccc(I)cc1", "CSC", "c1ccsc1", "CS(C)=O", "CP(C)C",
    "OP(O)(O)=O", "c1cc(F)c(Cl)cc1Br", "c1cnc2ccccc2n1", "CC(=O)SC", "ClC(=O)c1ccc(F)cc1",
]

# graph_hdc encoder name -> hyper_fingerprints feature name
NODE_NAMES = {"node_atoms": "element", "node_degrees": "degree", "node_valences": "implicit_hydrogens"}
GRAPH_NAMES = {"graph_size": "graph_size", "graph_diameter": "graph_diameter"}


def _graph_hdc_commit() -> tuple[str, bool]:
    root = Path(graph_hdc.__file__).resolve().parent.parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    # Only tracked library files matter: experiment scripts are not imported,
    # and committed code cannot import a file that was never committed.
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no", "--",
         "graph_hdc", ":(exclude)graph_hdc/experiments"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    return commit, bool(dirty)


def _arr(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main() -> None:
    commit, dirty = _graph_hdc_commit()
    if dirty:
        raise SystemExit("graph_hdc library has uncommitted changes; refusing to record a fixture")

    node_map = make_molecule_node_encoder_map_cont(dim=DIMENSION, seed=SEED)
    graph_map = make_molecule_graph_encoder_map_cont(dim=DIMENSION, seed=SEED)
    for enc in [*node_map.values(), *graph_map.values()]:
        assert enc.unit_modulus, "this graph_hdc version does not default to unit-modulus codebooks"

    atom_enc = node_map["node_atoms"]
    table = Chem.GetPeriodicTable()
    atom_types = [table.GetElementSymbol(int(z)) for z in atom_enc.atoms]

    save = {
        "node/element": _arr(atom_enc.embeddings).astype(np.float64),
        "node/degree": _arr(node_map["node_degrees"].matrix),
        "node/implicit_hydrogens": _arr(node_map["node_valences"].matrix),
        "graph/graph_size": _arr(graph_map["graph_size"].matrix),
        "graph/graph_diameter": _arr(graph_map["graph_diameter"].matrix),
    }

    graphs = [graph_dict_from_mol(Chem.MolFromSmiles(s)) for s in SMILES]
    for depth in (0, 2):
        for use_graph in (False, True):
            for bidir in (False, True):
                net = HyperNet(
                    hidden_dim=DIMENSION, depth=depth, node_encoder_map=node_map,
                    graph_encoder_map=graph_map if use_graph else {},
                    pooling="sum", normalize_all=False, bidirectional=bidir, seed=SEED,
                )
                results = net.forward_graphs(graphs)
                case = f"d{depth}_g{int(use_graph)}_b{int(bidir)}"
                save[f"ref/{case}"] = np.stack([r["graph_embedding"] for r in results]).astype(np.float64)
                save[f"ref0/{case}"] = np.stack([r["graph_hv_stack"][0] for r in results]).astype(np.float64)

    for dim, seed in SEEDED:
        nm = make_molecule_node_encoder_map_cont(dim=dim, seed=seed)
        gm = make_molecule_graph_encoder_map_cont(dim=dim, seed=seed)
        for gname, name in NODE_NAMES.items():
            if name != "element":
                save[f"seeded/d{dim}_s{seed}/node/{name}"] = _arr(nm[gname].matrix)
        for gname, name in GRAPH_NAMES.items():
            save[f"seeded/d{dim}_s{seed}/graph/{name}"] = _arr(gm[gname].matrix)

    atoms = list(atom_enc.atoms)
    save["atoms_gaussian"] = _arr(AtomEncoder(DIMENSION, atoms, seed=SEED + 10, unit_modulus=False).embeddings)
    save["atoms_unit"] = _arr(AtomEncoder(DIMENSION, atoms, seed=SEED + 10, unit_modulus=True).embeddings)

    deg, val = node_map["node_degrees"], node_map["node_valences"]
    size, diam = graph_map["graph_size"], graph_map["graph_diameter"]
    config = {
        "format_version": 2,
        "source": "graph_hdc",
        "graph_hdc_commit": commit,
        "unit_modulus": True,
        "mode": "composite",
        "dimension": DIMENSION,
        "depth": 2,
        "seed": SEED,
        "normalize": True,
        "normalize_all": False,
        "bidirectional": False,
        "pooling": "sum",
        "atom_types": atom_types,
        "unknown_atoms": "slot",
        "features": {
            "element": {"kind": "categorical"},
            "degree": {"kind": "continuous", "bandwidth": deg.bandwidth, "size": deg.size},
            "implicit_hydrogens": {"kind": "continuous", "bandwidth": val.bandwidth, "size": val.size},
        },
        "graph_features": {
            "graph_size": {"kind": "continuous", "bandwidth": size.bandwidth, "size": size.size},
            "graph_diameter": {"kind": "continuous", "bandwidth": diam.bandwidth, "size": diam.size},
        },
        "graph_mode": "bundle",
        "seeded": [list(x) for x in SEEDED],
    }
    save["config"] = np.array(json.dumps(config))
    save["smiles"] = np.array(json.dumps(SMILES))

    np.savez_compressed(OUT_PATH, **save)
    print(f"Saved {OUT_PATH} (graph_hdc {commit[:7]}, {len(save)} arrays)")


if __name__ == "__main__":
    main()
