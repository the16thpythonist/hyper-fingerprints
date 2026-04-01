#!/usr/bin/env python
"""Benchmark fingerprint computation speed across methods and sizes.

Sweeps over configurable fingerprint dimensions and compares timing
across registered fingerprint methods (e.g., Morgan vs HDF).

Usage:
    .venv/bin/python experiments/benchmark_fps.py --input molecules.smi --sizes 128 256 512 1024 2048
    .venv/bin/python experiments/benchmark_fps.py --input molecules.smi --repeats 10 --methods Morgan HDF
    .venv/bin/python experiments/benchmark_fps.py --input molecules.smi --no-plot
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[FingerprintMethod]] = {}


def register(cls: type[FingerprintMethod]) -> type[FingerprintMethod]:
    """Class decorator that adds a FingerprintMethod subclass to the registry."""
    _REGISTRY[cls.name] = cls
    return cls


def available_methods() -> list[str]:
    """Return names of all registered fingerprint methods."""
    return list(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class FingerprintMethod(ABC):
    """Interface that every fingerprint method must implement."""

    name: str = ""

    @abstractmethod
    def encode(self, mols: list[Chem.Mol], size: int, smiles: list[str] | None = None) -> np.ndarray:
        """Compute fingerprints for *mols* at the given *size*.

        Parameters
        ----------
        mols
            Pre-parsed RDKit ``Mol`` objects.
        size
            Fingerprint dimensionality (nBits for Morgan, dimension for HDF).
        smiles
            Raw SMILES strings (same order as *mols*). Used by Rust backend.

        Returns
        -------
        np.ndarray
            ``(len(mols), size)``
        """

    def setup(self, mols: list[Chem.Mol], size: int, params: dict[str, Any] | None = None) -> None:
        """Optional per-(method, size) setup called once before timing.

        Parameters
        ----------
        params
            Extra CLI parameters that methods can consume (e.g. ``depth``).
        """


# ---------------------------------------------------------------------------
# Built-in methods
# ---------------------------------------------------------------------------


@register
class MorganFP(FingerprintMethod):
    name = "Morgan"

    def __init__(self) -> None:
        self.radius = 2
        self._gen: Any = None
        self._current_size: int | None = None

    def setup(self, mols: list[Chem.Mol], size: int, params: dict[str, Any] | None = None) -> None:
        if self._current_size != size:
            self._gen = AllChem.GetMorganGenerator(radius=self.radius, fpSize=size)
            self._current_size = size

    def encode(self, mols: list[Chem.Mol], size: int, smiles: list[str] | None = None) -> np.ndarray:
        fps = self._gen.GetFingerprints(mols)
        out = np.zeros((len(mols), size), dtype=np.int8)
        for i, fp in enumerate(fps):
            DataStructs.ConvertToNumpyArray(fp, out[i])
        return out


@register
class HDFRustFP(FingerprintMethod):
    name = "HDF-Rust"

    def __init__(self) -> None:
        self._encoder = None
        self._current_key: tuple[int, int] | None = None

    def setup(self, mols: list[Chem.Mol], size: int, params: dict[str, Any] | None = None) -> None:
        from hyper_fingerprints import Encoder

        depth = (params or {}).get("depth", 2)
        key = (size, depth)
        if self._current_key != key:
            self._encoder = Encoder(dimension=size, depth=depth, seed=42, backend="rust")
            self._current_key = key

    def encode(self, mols: list[Chem.Mol], size: int, smiles: list[str] | None = None) -> np.ndarray:
        if smiles is not None:
            return self._encoder.encode(smiles)
        return self._encoder.encode(mols)


@register
class HDFNumpyFP(FingerprintMethod):
    name = "HDF-Python"

    def __init__(self) -> None:
        self._encoder = None
        self._current_key: tuple[int, int] | None = None

    def setup(self, mols: list[Chem.Mol], size: int, params: dict[str, Any] | None = None) -> None:
        from hyper_fingerprints import Encoder

        depth = (params or {}).get("depth", 2)
        key = (size, depth)
        if self._current_key != key:
            self._encoder = Encoder(dimension=size, depth=depth, seed=42, backend="numpy")
            self._current_key = key

    def encode(self, mols: list[Chem.Mol], size: int, smiles: list[str] | None = None) -> np.ndarray:
        return self._encoder.encode(mols)


# ---------------------------------------------------------------------------
# Benchmarking logic
# ---------------------------------------------------------------------------


def load_smiles(path: Path) -> tuple[list[Chem.Mol], list[str]]:
    """Read a .smi file and return parsed Mols and raw SMILES strings."""
    mols: list[Chem.Mol] = []
    smiles: list[str] = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            smi = line.strip().split()[0] if line.strip() else ""
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"WARNING: skipping invalid SMILES on line {lineno}: {smi!r}", file=sys.stderr)
                continue
            mols.append(mol)
            smiles.append(smi)
    if not mols:
        print(f"ERROR: no valid molecules found in {path}", file=sys.stderr)
        sys.exit(1)
    return mols, smiles


def _encode_batched(
    method: FingerprintMethod,
    mols: list[Chem.Mol],
    size: int,
    batch_size: int,
    smiles: list[str] | None = None,
) -> np.ndarray:
    """Encode molecules in chunks of *batch_size*."""
    parts = []
    for start in range(0, len(mols), batch_size):
        smi_chunk = smiles[start : start + batch_size] if smiles is not None else None
        parts.append(method.encode(mols[start : start + batch_size], size, smi_chunk))
    return np.concatenate(parts)


def bench_one(
    method: FingerprintMethod,
    mols: list[Chem.Mol],
    size: int,
    repeats: int,
    warmup: int,
    params: dict[str, Any] | None = None,
    smiles: list[str] | None = None,
) -> dict[str, Any]:
    """Time a single (method, size) combination."""
    method.setup(mols, size, params)
    batch_size = (params or {}).get("batch_size", len(mols))

    # warmup runs
    for _ in range(warmup):
        _encode_batched(method, mols, size, batch_size, smiles)

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _encode_batched(method, mols, size, batch_size, smiles)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return {
        "method": method.name,
        "size": size,
        "mean_s": mean,
        "std_s": std,
        "repeats": repeats,
        "n_mols": len(mols),
        "mean_ms": mean * 1000,
        "std_ms": std * 1000,
    }


def run_benchmark(
    mols: list[Chem.Mol],
    method_names: list[str],
    sizes: list[int],
    repeats: int,
    warmup: int,
    params: dict[str, Any] | None = None,
    smiles: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run the full sweep and return a list of result dicts."""
    results: list[dict[str, Any]] = []
    for method_name in method_names:
        cls = _REGISTRY[method_name]
        method = cls()
        for size in sizes:
            print(f"  {method_name:>10s}  dim={size:<6d} ... ", end="", flush=True)
            row = bench_one(method, mols, size, repeats, warmup, params, smiles)
            print(f"{row['mean_ms']:8.2f} ms  (std {row['std_ms']:.2f} ms)")
            results.append(row)
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_table(results: list[dict[str, Any]]) -> None:
    """Print a formatted results table to stdout."""
    header = f"{'Method':>10s} {'Size':>8s} {'Mean (ms)':>12s} {'Std (ms)':>10s} {'Mols':>6s} {'Repeats':>8s}"
    sep = "-" * len(header)
    print()
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['method']:>10s} {r['size']:>8d} {r['mean_ms']:>12.2f} {r['std_ms']:>10.2f} "
            f"{r['n_mols']:>6d} {r['repeats']:>8d}"
        )
    print(sep)


def save_csv(results: list[dict[str, Any]], path: Path) -> None:
    """Write results to a CSV file."""
    fields = ["method", "size", "mean_ms", "std_ms", "mean_s", "std_s", "n_mols", "repeats"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {path}")


def plot_results(results: list[dict[str, Any]], path: Path) -> None:
    """Generate a line plot comparing methods across sizes."""
    import matplotlib.pyplot as plt

    methods = sorted(set(r["method"] for r in results))
    sizes = sorted(set(r["size"] for r in results))

    lookup = {(r["method"], r["size"]): r for r in results}

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in methods:
        means = [lookup.get((method, s), {}).get("mean_ms", 0) for s in sizes]
        stds = [lookup.get((method, s), {}).get("std_ms", 0) for s in sizes]
        ax.errorbar(sizes, means, yerr=stds, label=method, marker="x", capsize=3)

    ax.set_xlabel("Fingerprint size")
    ax.set_ylabel("Time (ms)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_title("Fingerprint computation time by method and size")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fingerprint methods across different sizes.",
    )
    default_smi = Path(__file__).parent / "molecules.smi"
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=default_smi,
        help=f"Path to a .smi file (one SMILES per line). Default: {default_smi}",
    )
    parser.add_argument(
        "--sizes", "-s",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help=f"Fingerprint sizes to sweep (default: {DEFAULT_SIZES}).",
    )
    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        default=None,
        help=f"Methods to benchmark (default: all registered: {available_methods()}).",
    )
    parser.add_argument(
        "--repeats", "-r",
        type=int,
        default=5,
        help="Number of timed repetitions per configuration (default: 5).",
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=1,
        help="Number of warmup runs before timing (default: 1).",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=512,
        help="Number of molecules per batch (default: 512).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("experiments/results"),
        help="Output directory for CSV and plot (default: experiments/results).",
    )
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=2,
        help="Message-passing depth for HDF (default: 2).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the plot (avoids matplotlib dependency).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    method_names = args.methods or available_methods()
    for name in method_names:
        if name not in _REGISTRY:
            print(f"ERROR: unknown method {name!r}. Available: {available_methods()}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading molecules from {args.input} ...")
    mols, smiles = load_smiles(args.input)
    print(f"  {len(mols)} molecules loaded.")
    params = {"depth": args.depth, "batch_size": args.batch_size}

    print(f"Methods: {method_names}")
    print(f"Sizes:   {args.sizes}")
    print(f"Repeats: {args.repeats}  Warmup: {args.warmup}  Depth: {args.depth}  Batch: {args.batch_size}")
    print()

    results = run_benchmark(mols, method_names, args.sizes, args.repeats, args.warmup, params, smiles)
    print_table(results)

    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(results, args.output / "benchmark.csv")

    if not args.no_plot:
        plot_results(results, args.output / "benchmark.png")


if __name__ == "__main__":
    main()
