#!/usr/bin/env python
"""Generate regression fixture tensors for the encoder.

Run this script once to record the current encoder outputs. The resulting
.npz file is loaded by test_regression.py to verify that future refactors
preserve identical results.

The codebook is exported as a plain numpy array so that fixtures remain
valid even after replacing torch/torchhd with a different backend.

Usage:
    python -m tests.generate_fixtures
"""

import json

import numpy as np

from hyper_fingerprints.encoder import Encoder

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

    out_path = "tests/fixtures/regression_data.npz"
    np.savez(out_path, **save_dict)
    print(f"Saved regression fixtures to {out_path}")
    print(f"  {len(SMILES)} molecules")
    print(f"  codebook shape:     {codebook.shape}")
    print(f"  encode shape:       {batch_encode.shape}")
    print(f"  encode_joint shape: {batch_encode_joint.shape}")


if __name__ == "__main__":
    main()
