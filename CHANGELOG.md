# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-03-14

Initial release.

### Added

- `Encoder` class for encoding molecules into fixed-size HRR hypervectors
- `encode()` for order-N fingerprints via message passing
- `encode_joint()` for combined order-0 + order-N representations
- `Encoder.save()` / `Encoder.load()` for persisting encoders as `.npz` files
- `cosine_similarity()` helper for pairwise vector comparison
- Configurable atom vocabulary, dimension, depth, and normalization
- 5-feature atom scheme: atom type, degree, formal charge, total Hs, aromaticity
- Deterministic codebook generation via seeded RNG
- Batch encoding of multiple molecules in a single call
- Support for both SMILES strings and RDKit `Mol` objects as input
- Regression test suite for numerical stability
- Multi-version testing with nox (Python 3.9-3.13)
- Quickstart notebook (`examples/00_quickstart.ipynb`)
