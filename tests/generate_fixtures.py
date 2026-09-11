#!/usr/bin/env python
"""Generate regression fixture tensors for the encoder.

Run this script once to record the current encoder outputs. The resulting
.npz file is loaded by test_regression.py to verify that future refactors
preserve identical results.

Two fixture files are written (paths relative to the repository root):

- ``tests/fixtures/regression_data.npz`` -- legacy encoder outputs with the
  codebook stored alongside and re-injected by the tests (protects the
  message-passing pipeline). Regenerating this changes the codebook.
- ``tests/fixtures/seed_fixtures.npz`` -- codebooks generated from a seed for
  both encoder modes plus composite-mode outputs (protects seed -> codebook
  RNG paths and composite-mode stability).

Usage:
    python -m tests.generate_fixtures            # both files
    python -m tests.generate_fixtures seed       # seed_fixtures.npz only
"""

import json
import sys
from pathlib import Path

import numpy as np

from hyper_fingerprints.encoder import Encoder
from hyper_fingerprints.presets import FEATURES_GRAPH_HDC, GRAPH_FEATURES_GRAPH_HDC

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# ── Encoder configuration ──────────────────────────────────────────────
SEED = 42
DIMENSION = 512
DEPTH = 3  # default

# ── Molecule set ────────────────────────────────────────────────────────
# Chosen to cover all 9 default atom types (Br, C, Cl, F, I, N, O, P, S),
# a range of sizes, and a mix of aromatic / non-aromatic structures.
SMILES = [
    # --- simple / C-only ---
    "C",                            # methane
    "CC",                           # ethane
    "C=C",                          # ethylene
    "c1ccccc1",                     # benzene
    # --- O ---
    "CCO",                          # ethanol
    "CC(=O)O",                      # acetic acid
    "CC(=O)Oc1ccccc1C(=O)O",       # aspirin
    "O",                            # water
    # --- N ---
    "CN",                           # methylamine
    "c1ccncc1",                     # pyridine
    "c1ccc(N)cc1",                  # aniline
    "c1ccc2[nH]ccc2c1",            # indole
    "CC(=O)Nc1ccc(O)cc1",          # acetaminophen (C, N, O)
    # --- F ---
    "CF",                           # fluoromethane
    "c1ccc(F)cc1",                  # fluorobenzene
    "FC(F)(F)c1ccccc1",            # benzotrifluoride
    # --- Cl ---
    "CCl",                          # chloromethane
    "c1ccc(Cl)cc1",                # chlorobenzene
    "ClC(Cl)Cl",                   # chloroform
    # --- Br ---
    "CBr",                          # bromomethane
    "c1ccc(Br)cc1",                # bromobenzene
    # --- I ---
    "CI",                           # iodomethane
    "c1ccc(I)cc1",                 # iodobenzene
    # --- S ---
    "CSC",                          # dimethyl sulfide
    "c1ccsc1",                      # thiophene
    "CS(C)=O",                     # dimethyl sulfoxide
    # --- P ---
    "CP(C)C",                       # trimethylphosphine
    "OP(O)(O)=O",                  # phosphoric acid
    # --- multi-heteroatom ---
    "c1cc(F)c(Cl)cc1Br",          # 1-bromo-2-chloro-4-fluorobenzene
    "c1cnc2ccccc2n1",              # quinoxaline (C, N, aromatic)
    "CC(=O)SC",                    # thioacetic acid S-methyl ester (C, O, S)
    "ClC(=O)c1ccc(F)cc1",         # 4-fluorobenzoyl chloride (C, O, F, Cl)
]


def main() -> None:
    encoder = Encoder(dimension=DIMENSION, depth=DEPTH, seed=SEED)

    # ── Export codebook ─────────────────────────────────────────────────
    codebook = encoder._codebook

    # ── Encode each molecule individually ───────────────────────────────
    individual_encode = {}
    individual_encode_joint = {}
    for smi in SMILES:
        individual_encode[smi] = encoder.encode(smi)
        individual_encode_joint[smi] = encoder.encode_joint(smi)

    # ── Encode all molecules in a single batch ──────────────────────────
    batch_encode = encoder.encode(SMILES)
    batch_encode_joint = encoder.encode_joint(SMILES)

    # ── Build save dict ─────────────────────────────────────────────────
    # np.savez stores arrays by string key. We store the SMILES list and
    # config as JSON strings, individual results keyed by "ind_encode_{i}"
    # and "ind_joint_{i}", and the batch results directly.
    save_dict = {
        "config": np.void(json.dumps({
            "seed": SEED,
            "dimension": DIMENSION,
            "depth": DEPTH,
        }).encode()),
        "smiles": np.void(json.dumps(SMILES).encode()),
        "codebook": codebook,
        "batch_encode": batch_encode,
        "batch_encode_joint": batch_encode_joint,
    }
    for i, smi in enumerate(SMILES):
        save_dict[f"ind_encode_{i}"] = individual_encode[smi]
        save_dict[f"ind_joint_{i}"] = individual_encode_joint[smi]

    out_path = FIXTURE_DIR / "regression_data.npz"
    np.savez(out_path, **save_dict)
    print(f"Saved regression fixtures to {out_path}")
    print(f"  {len(SMILES)} molecules")
    print(f"  codebook shape:     {codebook.shape}")
    print(f"  encode shape:       {batch_encode.shape}")
    print(f"  encode_joint shape: {batch_encode_joint.shape}")


def generate_seed_fixtures() -> None:
    """Record seed-generated codebooks and composite-mode outputs."""
    seed, dimension, depth = 7, 64, 2
    composite_smiles = ["CCO", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1", "C", "CC.O"]

    legacy = Encoder(dimension=dimension, seed=seed)
    composite = Encoder(
        dimension=dimension, depth=depth, seed=seed, normalize=True,
        features=FEATURES_GRAPH_HDC, graph_features=GRAPH_FEATURES_GRAPH_HDC,
    )
    config = {
        "seed": seed, "dimension": dimension, "depth": depth,
        "features": FEATURES_GRAPH_HDC, "graph_features": GRAPH_FEATURES_GRAPH_HDC,
    }
    save_dict = {
        "config": np.array(json.dumps(config)),
        "legacy_codebook": legacy._codebook,
        "composite_smiles": np.array(json.dumps(composite_smiles)),
        "composite_encode": composite.encode(composite_smiles),
        "composite_encode_joint": composite.encode_joint(composite_smiles),
    }
    save_dict.update({f"node/{k}": v for k, v in composite._node_encoder.arrays().items()})
    save_dict.update({f"graph/{k}": v for k, v in composite._graph_encoder.arrays().items()})

    out_path = FIXTURE_DIR / "seed_fixtures.npz"
    np.savez(out_path, **save_dict)
    print(f"Saved seed fixtures to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "seed":
        generate_seed_fixtures()
    else:
        main()
        generate_seed_fixtures()
