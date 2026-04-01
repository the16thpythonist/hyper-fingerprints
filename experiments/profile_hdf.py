#!/usr/bin/env python
"""Profile HDF encoding pipeline stage-by-stage.

Breaks the encode pipeline into 5 stages and times each independently
for both Rust and Python backends side-by-side.

Stages:
  1. SMILES parsing       (MolFromSmiles)
  2. mol_to_data           (feature extraction + edge index)
  3. batch_from_data_list  (concatenate graphs)
  4. Feature encoding      (codebook index lookup)
  5. Message-passing       (encode_batch — Rust or NumPy)

Usage:
    .venv/bin/python experiments/profile_hdf.py
    .venv/bin/python experiments/profile_hdf.py --input molecules.smi --dim 512 --depth 3
    .venv/bin/python experiments/profile_hdf.py --repeats 20 --batch-size 256
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from hyper_fingerprints.encoder import Encoder, _HAS_RUST
from hyper_fingerprints.features import atom_type_map, mol_to_data
from hyper_fingerprints.utils import batch_from_data_list

if _HAS_RUST:
    from hyper_fingerprints._core import encode_batch_rs


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

STAGE_NAMES_PYTHON = [
    "1_smiles_parse",
    "2_mol_to_data",
    "3_batch_build",
    "4_feature_enc",
    "5_msg_pass",
]

STAGE_NAMES_RUST = [
    "1-4_prepare_batch",
    "5_msg_pass",
]


def run_stages_python(
    smiles: list[str],
    encoder: Encoder,
) -> dict[str, float]:
    """Run the full pipeline with Python backend, timing each stage."""
    atom_to_idx = atom_type_map(encoder.atom_types)

    # Stage 1: SMILES parsing
    t0 = time.perf_counter()
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    t_parse = time.perf_counter() - t0

    # Stage 2: mol_to_data
    t0 = time.perf_counter()
    data_list = [mol_to_data(m, atom_to_idx) for m in mols]
    t_features = time.perf_counter() - t0

    # Stage 3: batch_from_data_list
    t0 = time.perf_counter()
    batch = batch_from_data_list(data_list)
    t_batch = time.perf_counter() - t0

    # Stage 4: Feature encoding (codebook lookup)
    t0 = time.perf_counter()
    _node_hv = encoder._encode_node_features(batch.x)
    t_encode_feat = time.perf_counter() - t0

    # Stage 5: Message-passing + readout (NumPy)
    t0 = time.perf_counter()
    encoder._encode_batch_numpy(batch)
    t_msg_pass = time.perf_counter() - t0

    return dict(zip(STAGE_NAMES_PYTHON, [t_parse, t_features, t_batch, t_encode_feat, t_msg_pass]))


def run_stages_rust(
    smiles: list[str],
    encoder: Encoder,
) -> dict[str, float]:
    """Run the full pipeline with Rust backend, timing each stage."""
    from hyper_fingerprints._core import prepare_batch_rs, encode_batch_rs

    # Stages 1-4: prepare_batch_rs (SMILES parsing + features + batching)
    t0 = time.perf_counter()
    feature_indices, edge_index, batch_indices, num_graphs = prepare_batch_rs(
        smiles,
        encoder._atom_to_idx,
        encoder.features,
        encoder._feature_bins,
    )
    t_prepare = time.perf_counter() - t0

    # Stage 5: Message-passing + readout (Rust)
    feature_indices = np.asarray(feature_indices)
    edge_index = np.asarray(edge_index)
    batch_indices = np.asarray(batch_indices)
    t0 = time.perf_counter()
    encode_batch_rs(
        encoder._codebook,
        feature_indices,
        edge_index,
        batch_indices,
        num_graphs,
        encoder.depth,
        encoder.normalize,
    )
    t_msg_pass = time.perf_counter() - t0

    return dict(zip(STAGE_NAMES_RUST, [t_prepare, t_msg_pass]))


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


def load_smiles(path: Path) -> list[str]:
    """Read a .smi file and return SMILES strings."""
    out: list[str] = []
    with open(path) as fh:
        for line in fh:
            smi = line.strip().split()[0] if line.strip() else ""
            if smi:
                out.append(smi)
    return out


def profile_backend(
    name: str,
    runner,
    smiles: list[str],
    encoder: Encoder,
    repeats: int,
    warmup: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Profile a backend, returning per-stage timing rows."""
    # Use only batch_size molecules
    smiles_batch = smiles[:batch_size]

    # Warmup
    for _ in range(warmup):
        runner(smiles_batch, encoder)

    # Discover stage names from first run
    first_result = runner(smiles_batch, encoder)
    stage_names = list(first_result.keys())

    # Collect timings
    all_timings: dict[str, list[float]] = {s: [first_result[s]] for s in stage_names}
    for _ in range(repeats - 1):
        timings = runner(smiles_batch, encoder)
        for stage, t in timings.items():
            all_timings[stage].append(t)

    rows = []
    for stage in stage_names:
        ts = all_timings[stage]
        mean = statistics.mean(ts)
        std = statistics.stdev(ts) if len(ts) > 1 else 0.0
        rows.append({
            "backend": name,
            "stage": stage,
            "mean_ms": mean * 1000,
            "std_ms": std * 1000,
            "mean_s": mean,
            "std_s": std,
            "n_mols": len(smiles_batch),
            "repeats": repeats,
        })
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_table(results: list[dict[str, Any]]) -> None:
    """Print per-backend stage breakdown tables."""
    backends = sorted(set(r["backend"] for r in results))
    lookup = {(r["backend"], r["stage"]): r for r in results}

    for b in backends:
        stages = [r["stage"] for r in results if r["backend"] == b]
        total = sum(lookup[(b, s)]["mean_ms"] for s in stages)

        print(f"\n  {b} backend:")
        print(f"  {'Stage':>20s} {'Mean (ms)':>12s} {'Std (ms)':>10s} {'%':>8s}")
        print(f"  {'-'*54}")
        for stage in stages:
            r = lookup[(b, stage)]
            pct = r["mean_ms"] / total * 100 if total > 0 else 0
            print(f"  {stage:>20s} {r['mean_ms']:>12.2f} {r['std_ms']:>10.2f} {pct:>7.1f}%")
        print(f"  {'-'*54}")
        print(f"  {'TOTAL':>20s} {total:>12.2f}")

    # Summary comparison
    if len(backends) == 2:
        totals = {}
        for b in backends:
            stages = [r["stage"] for r in results if r["backend"] == b]
            totals[b] = sum(lookup[(b, s)]["mean_ms"] for s in stages)
        b0, b1 = backends
        speedup = totals[b0] / totals[b1] if totals[b1] > 0 else float("inf")
        print(f"\n  Speedup: {b0} / {b1} = {speedup:.1f}x")
    print()


def save_csv(results: list[dict[str, Any]], path: Path) -> None:
    """Write results to CSV."""
    fields = ["backend", "stage", "mean_ms", "std_ms", "mean_s", "std_s", "n_mols", "repeats"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {path}")


def plot_results(results: list[dict[str, Any]], path: Path) -> None:
    """Generate a stacked bar chart of stage timings per backend."""
    import matplotlib.pyplot as plt

    backends = sorted(set(r["backend"] for r in results))
    lookup = {(r["backend"], r["stage"]): r for r in results}

    # Collect all unique stages across backends
    all_stages = []
    for r in results:
        if r["stage"] not in all_stages:
            all_stages.append(r["stage"])

    x = np.arange(len(backends))
    width = 0.5

    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = np.zeros(len(backends))

    for stage in all_stages:
        means = [lookup.get((b, stage), {}).get("mean_ms", 0) for b in backends]
        ax.bar(x, means, width, bottom=bottom, label=stage)
        bottom += np.array(means)

    ax.set_ylabel("Time (ms)")
    ax.set_title("HDF encoding pipeline breakdown by stage")
    ax.set_xticks(x)
    ax.set_xticklabels(backends)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_smi = Path(__file__).parent / "molecules.smi"
    parser = argparse.ArgumentParser(
        description="Profile HDF encoding pipeline stage-by-stage.",
    )
    parser.add_argument("--input", "-i", type=Path, default=default_smi,
                        help="Path to .smi file.")
    parser.add_argument("--dim", type=int, default=256,
                        help="Fingerprint dimension (default: 256).")
    parser.add_argument("--depth", type=int, default=2,
                        help="Message-passing depth (default: 2).")
    parser.add_argument("--repeats", "-r", type=int, default=5,
                        help="Timed repetitions (default: 5).")
    parser.add_argument("--warmup", "-w", type=int, default=1,
                        help="Warmup runs (default: 1).")
    parser.add_argument("--batch-size", "-b", type=int, default=512,
                        help="Number of molecules per batch (default: 512).")
    parser.add_argument("--output", "-o", type=Path, default=Path("experiments/results"),
                        help="Output directory (default: experiments/results).")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip generating the plot.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print(f"Loading molecules from {args.input} ...")
    all_smiles = load_smiles(args.input)
    print(f"  {len(all_smiles)} SMILES loaded (using first {min(args.batch_size, len(all_smiles))})")
    print(f"  dim={args.dim}  depth={args.depth}  repeats={args.repeats}  warmup={args.warmup}")
    print(f"  Rust backend available: {_HAS_RUST}")
    print()

    results: list[dict[str, Any]] = []

    # Python backend
    enc_py = Encoder(dimension=args.dim, depth=args.depth, seed=42, backend="numpy")
    print("Profiling Python backend ...")
    results.extend(profile_backend(
        "Python", run_stages_python, all_smiles, enc_py,
        args.repeats, args.warmup, args.batch_size,
    ))

    # Rust backend
    if _HAS_RUST:
        enc_rs = Encoder(dimension=args.dim, depth=args.depth, seed=42, backend="rust")
        print("Profiling Rust backend ...")
        results.extend(profile_backend(
            "Rust", run_stages_rust, all_smiles, enc_rs,
            args.repeats, args.warmup, args.batch_size,
        ))
    else:
        print("Rust backend not available, skipping.")

    print_table(results)

    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(results, args.output / "profile_hdf.csv")

    if not args.no_plot:
        plot_results(results, args.output / "profile_hdf.png")


if __name__ == "__main__":
    main()
