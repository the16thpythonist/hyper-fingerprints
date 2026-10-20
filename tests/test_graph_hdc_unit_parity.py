"""Parity tests against graph_hdc's unit-modulus (FHRR) codebooks.

``fixtures/graph_hdc_unit_codebook.npz`` is produced by
``tests/generate_graph_hdc_unit_fixture.py`` with graph_hdc itself (the commit
is recorded in the fixture config). graph_hdc samples unit-modulus codebooks by
default, and so does hyper_fingerprints since 0.5.0. The tests check

- that the fixture really holds unit-modulus vectors (guards against a fixture
  recorded with an older graph_hdc),
- that the FPE base spectra hyper_fingerprints samples for a seed equal the
  ones graph_hdc samples for the same seed,
- that the unit-modulus transform of the categorical vectors is the same
  (graph_hdc draws them with torch, so the draws themselves differ),
- that graph_hdc's unit-modulus codebook, imported with
  ``Encoder.from_legacy_codebook``, reproduces graph_hdc's embeddings on both
  backends.

Tolerances as in ``test_legacy_parity.py``: graph_hdc feeds ``graph_size`` /
``graph_diameter`` as float32, so the cases with graph features agree to ~1e-8.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from hyper_fingerprints import Encoder
from hyper_fingerprints.codebook import unit_modulus_vectors
from hyper_fingerprints.encoder import _HAS_RUST
from hyper_fingerprints.presets import FEATURES_GRAPH_HDC, GRAPH_FEATURES_GRAPH_HDC

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "graph_hdc_unit_codebook.npz"
CASES = [
    (depth, graph, bidir)
    for depth in (0, 2)
    for graph in (False, True)
    for bidir in (False, True)
]
BACKENDS = ["numpy"] + (["rust"] if _HAS_RUST else [])


@pytest.fixture(scope="module")
def fixture():
    if not FIXTURE_PATH.exists():
        pytest.skip("graph_hdc unit-modulus fixture not found")
    raw = np.load(FIXTURE_PATH, allow_pickle=False)
    return raw, json.loads(str(raw["config"])), json.loads(str(raw["smiles"]))


def test_fixture_holds_unit_modulus_codebooks(fixture):
    raw, cfg, _ = fixture
    assert cfg["unit_modulus"] is True
    assert cfg["source"] == "graph_hdc" and len(cfg["graph_hdc_commit"]) == 40
    for key in raw.files:
        if not key.startswith(("node/", "graph/", "seeded/")):
            continue
        arr = raw[key]
        spectrum = arr if np.iscomplexobj(arr) else np.fft.fft(arr, axis=-1)
        np.testing.assert_allclose(np.abs(spectrum), 1.0, atol=1e-12, err_msg=key)


def test_import_is_flagged_unit_modulus(fixture):
    enc = Encoder.from_legacy_codebook(FIXTURE_PATH)
    assert enc.mode == "composite"
    assert enc.unit_modulus is True
    assert list(enc.features) == ["element", "degree", "implicit_hydrogens"]


def test_seeded_spectra_match_graph_hdc(fixture):
    """Same seed -> same FPE base spectra as graph_hdc (node and graph features)."""
    raw, cfg, _ = fixture
    for dim, seed in cfg["seeded"]:
        enc = Encoder(
            dimension=dim, seed=seed,
            features=FEATURES_GRAPH_HDC, graph_features=GRAPH_FEATURES_GRAPH_HDC,
        )
        checked = 0
        for kind, composite in (("node", enc._node_encoder), ("graph", enc._graph_encoder)):
            for name, arr in composite.arrays().items():
                key = f"seeded/d{dim}_s{seed}/{kind}/{name}"
                if key not in raw.files:
                    continue  # categorical: drawn with torch in graph_hdc
                np.testing.assert_allclose(arr, raw[key], atol=1e-14, rtol=0, err_msg=key)
                checked += 1
        assert checked == 4


def test_categorical_transform_matches_graph_hdc(fixture):
    """graph_hdc's Gaussian AtomEncoder draw -> our transform -> graph_hdc's unit vectors."""
    raw, _, _ = fixture
    np.testing.assert_allclose(
        unit_modulus_vectors(raw["atoms_gaussian"]), raw["atoms_unit"], atol=1e-14, rtol=0,
    )
    # the transform changes the vectors (the Gaussian draw is not unit modulus)
    assert np.abs(raw["atoms_gaussian"] - raw["atoms_unit"]).max() > 1e-2


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("depth,graph,bidir", CASES)
def test_embeddings_match_graph_hdc(fixture, depth, graph, bidir, backend):
    raw, _, smiles = fixture
    enc = Encoder.from_legacy_codebook(
        FIXTURE_PATH,
        depth=depth,
        bidirectional=bidir,
        graph_features=None if graph else False,
    )
    enc.backend = backend
    case = f"d{depth}_g{int(graph)}_b{int(bidir)}"
    result = enc._encode_full(smiles)

    np.testing.assert_allclose(result["node_terms"], raw[f"ref0/{case}"], atol=1e-10, rtol=0)
    atol = 1e-6 if graph else 1e-10
    np.testing.assert_allclose(result["graph_embedding"], raw[f"ref/{case}"], atol=atol, rtol=0)


def test_save_load_keeps_imported_unit_codebook(fixture, tmp_path):
    _, _, smiles = fixture
    enc = Encoder.from_legacy_codebook(FIXTURE_PATH)
    enc.save(tmp_path / "roundtrip.npz")
    loaded = Encoder.load(tmp_path / "roundtrip.npz")
    assert loaded.unit_modulus is True
    for name, arr in enc._node_encoder.arrays().items():
        assert np.array_equal(loaded._node_encoder.arrays()[name], arr)
    assert np.array_equal(loaded.encode(smiles[:5]), enc.encode(smiles[:5]))
