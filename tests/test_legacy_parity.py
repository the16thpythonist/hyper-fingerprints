"""Parity tests against graph_hdc's HyperNet.

``fixtures/legacy_codebook_graph_hdc.npz`` is produced by
``graph_hdc/_scripts/export_legacy_codebook_fixture.py``: it contains the exact
base vectors of a continuous-mode HyperNet (via ``HyperNet.export_legacy_codebook``)
plus reference embeddings for a fixed SMILES list under several configurations.
Loading it with ``Encoder.from_legacy_codebook`` must reproduce those embeddings,
which validates the composite encoding, message passing, normalization and
graph-property readout paths without needing torch in this test environment.

Tolerances: the node-only cases agree to floating-point precision. Cases with
graph features differ by ~1e-8 because graph_hdc feeds ``graph_size`` /
``graph_diameter`` as float32 tensors before dividing by the bandwidth.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from hyper_fingerprints import Encoder

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "legacy_codebook_graph_hdc.npz"
CASES = [
    (depth, graph, bidir)
    for depth in (0, 2)
    for graph in (False, True)
    for bidir in (False, True)
]


@pytest.fixture(scope="module")
def fixture():
    if not FIXTURE_PATH.exists():
        pytest.skip("legacy codebook fixture not found")
    raw = np.load(FIXTURE_PATH, allow_pickle=False)
    return raw, json.loads(str(raw["smiles"]))


def test_import_reads_config(fixture):
    enc = Encoder.from_legacy_codebook(FIXTURE_PATH)
    assert enc.mode == "composite"
    assert enc.dimension == 256
    assert enc.depth == 2
    assert enc.normalize is True
    assert enc.bidirectional is False
    assert enc.unknown_atoms == "slot"
    assert enc.graph_mode == "bundle"
    assert list(enc.features) == ["element", "degree", "implicit_hydrogens"]
    assert list(enc.graph_features) == ["graph_size", "graph_diameter"]
    assert "C" in enc.atom_types and "Si" in enc.atom_types


@pytest.mark.parametrize("depth,graph,bidir", CASES)
def test_embeddings_match_graph_hdc(fixture, depth, graph, bidir):
    raw, smiles = fixture
    enc = Encoder.from_legacy_codebook(
        FIXTURE_PATH,
        depth=depth,
        bidirectional=bidir,
        graph_features=None if graph else False,
    )
    case = f"d{depth}_g{int(graph)}_b{int(bidir)}"
    result = enc._encode_full(smiles)

    np.testing.assert_allclose(result["node_terms"], raw[f"ref0/{case}"], atol=1e-10, rtol=0)
    atol = 1e-6 if graph else 1e-10
    np.testing.assert_allclose(result["graph_embedding"], raw[f"ref/{case}"], atol=atol, rtol=0)


def test_save_load_preserves_imported_codebook(fixture, tmp_path):
    _, smiles = fixture
    enc = Encoder.from_legacy_codebook(FIXTURE_PATH)
    enc.save(tmp_path / "roundtrip.npz")
    loaded = Encoder.load(tmp_path / "roundtrip.npz")
    assert np.array_equal(loaded.encode(smiles[:5]), enc.encode(smiles[:5]))
