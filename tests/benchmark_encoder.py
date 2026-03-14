#!/usr/bin/env python
"""Benchmark encoder timing for torch vs numpy comparison.

Usage:
    python -m tests.benchmark_encoder
"""

import json
import time
from pathlib import Path

from hyper_fingerprints.encoder import Encoder

SMILES = [
    "C", "CC", "C=C", "c1ccccc1",
    "CCO", "CC(=O)O", "CC(=O)Oc1ccccc1C(=O)O", "O",
    "CN", "c1ccncc1", "c1ccc(N)cc1", "c1ccc2[nH]ccc2c1",
    "CC(=O)Nc1ccc(O)cc1",
    "CF", "c1ccc(F)cc1", "FC(F)(F)c1ccccc1",
    "CCl", "c1ccc(Cl)cc1", "ClC(Cl)Cl",
    "CBr", "c1ccc(Br)cc1",
    "CI", "c1ccc(I)cc1",
    "CSC", "c1ccsc1", "CS(C)=O",
    "CP(C)C", "OP(O)(O)=O",
    "c1cc(F)c(Cl)cc1Br", "c1cnc2ccccc2n1",
    "CC(=O)SC", "ClC(=O)c1ccc(F)cc1",
]

ITERATIONS = 100


def bench(func, *args, n=ITERATIONS):
    # warmup
    func(*args)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)
    return times


def main():
    encoder = Encoder(dimension=512, depth=3, seed=42)

    results = {}

    # Individual encode
    for smi in SMILES:
        ts = bench(encoder.encode, smi)
        results[f"encode_individual_{smi}"] = {
            "mean": sum(ts) / len(ts),
            "std": (sum((t - sum(ts)/len(ts))**2 for t in ts) / len(ts)) ** 0.5,
        }

    # Individual encode_joint
    for smi in SMILES:
        ts = bench(encoder.encode_joint, smi)
        results[f"encode_joint_individual_{smi}"] = {
            "mean": sum(ts) / len(ts),
            "std": (sum((t - sum(ts)/len(ts))**2 for t in ts) / len(ts)) ** 0.5,
        }

    # Batch encode
    ts = bench(encoder.encode, SMILES)
    results["encode_batch_all"] = {
        "mean": sum(ts) / len(ts),
        "std": (sum((t - sum(ts)/len(ts))**2 for t in ts) / len(ts)) ** 0.5,
    }

    # Batch encode_joint
    ts = bench(encoder.encode_joint, SMILES)
    results["encode_joint_batch_all"] = {
        "mean": sum(ts) / len(ts),
        "std": (sum((t - sum(ts)/len(ts))**2 for t in ts) / len(ts)) ** 0.5,
    }

    out_path = Path(__file__).parent / "fixtures" / "benchmark_torch.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark results to {out_path}")
    print(f"  Batch encode mean: {results['encode_batch_all']['mean']*1000:.2f} ms")
    print(f"  Batch encode_joint mean: {results['encode_joint_batch_all']['mean']*1000:.2f} ms")


if __name__ == "__main__":
    main()
