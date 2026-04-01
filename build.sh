#!/usr/bin/env bash
# Build release wheels for hyper_fingerprints.
#
# Usage:
#   ./build.sh              # build wheel + sdist
#   ./build.sh --release    # same (default)
#   ./build.sh --debug      # debug build (faster compile, slower runtime)
#
# Output:
#   target/wheels/*.whl     # pre-compiled wheel (abi3, linux x86_64)
#   target/wheels/*.tar.gz  # source distribution
#
# Requirements:
#   - Rust toolchain (rustup.rs), >= 1.83
#   - maturin: pip install maturin
#   - Python 3.9+

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse args
PROFILE="--release"
RUSTFLAGS_EXTRA=""
for arg in "$@"; do
    case "$arg" in
        --debug) PROFILE="" ;;
        --release) PROFILE="--release" ;;
        --native) RUSTFLAGS_EXTRA="-C target-cpu=native" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# Check prerequisites
if ! command -v rustc &>/dev/null; then
    echo "ERROR: Rust toolchain not found. Install from https://rustup.rs/"
    exit 1
fi

if ! command -v maturin &>/dev/null; then
    # Try in current venv
    if [ -f ".venv/bin/maturin" ]; then
        MATURIN=".venv/bin/maturin"
    else
        echo "ERROR: maturin not found. Install with: pip install maturin"
        exit 1
    fi
else
    MATURIN="maturin"
fi

echo "=== hyper_fingerprints build ==="
echo "  Rust:    $(rustc --version)"
echo "  Maturin: $($MATURIN --version)"
echo "  Profile: ${PROFILE:-debug}"
echo

# Build wheel
echo "Building wheel..."
RUSTFLAGS="${RUSTFLAGS_EXTRA}" $MATURIN build $PROFILE --out target/wheels

# Build sdist
echo "Building source distribution..."
$MATURIN sdist --out target/wheels

echo
echo "=== Build complete ==="
echo "Artifacts:"
ls -lh target/wheels/
echo
echo "To install:"
echo "  pip install target/wheels/hyper_fingerprints-*.whl"
