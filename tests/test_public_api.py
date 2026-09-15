"""Public-API matrix: every exported entry point across parameter combinations.

For each valid configuration the contract is:

- ``encode`` / ``encode_joint`` return float64 arrays of the documented shape
  with only finite values,
- ``save`` -> ``load`` reproduces the outputs bit-for-bit,
- legacy mode: Rust and NumPy backends agree to 1e-10.

Invalid configurations raise ``ValueError`` at construction (never at encode
time); invalid molecule inputs raise ``ValueError``/``TypeError`` at encode
time, identically on every backend and in every mode.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from rdkit import Chem

import hyper_fingerprints
from hyper_fingerprints import (
    DEFAULT_FEATURES,
    FEATURES_CONTINUOUS,
    FEATURES_DEFAULT,
    FEATURES_GRAPH_HDC,
    GRAPH_FEATURES_GRAPH_HDC,
    Encoder,
    cosine_similarity,
)
from hyper_fingerprints.encoder import _HAS_RUST

DIM = 64
ATOL = 1e-10

# Covers charged, disconnected, single-atom, hypervalent and Kekulé cases.
MOLS = [
    "CCO", "c1ccccc1", "CC(=O)[O-]", "C[N+](C)(C)C", "CC.O", "C",
    "C1=CC=CC=C1", "FS(F)(F)(F)(F)F", "c1cc[nH]c1",
]
CUSTOM_ATOMS = ["C", "N", "O", "F", "S"]


def _check_outputs(enc: Encoder, mols, width_factor: int):
    fp = enc.encode(mols)
    joint = enc.encode_joint(mols)
    assert fp.dtype == np.float64 and joint.dtype == np.float64
    assert fp.shape == (len(mols), width_factor * DIM)
    assert joint.shape == (len(mols), (width_factor + 1) * DIM)
    assert np.isfinite(fp).all() and np.isfinite(joint).all()
    return fp, joint


def _check_round_trip(enc: Encoder, mols, tmp_path, fp, joint):
    path = tmp_path / "enc.npz"
    enc.save(path)
    loaded = Encoder.load(path, backend=enc.backend)
    np.testing.assert_array_equal(loaded.encode(mols), fp)
    np.testing.assert_array_equal(loaded.encode_joint(mols), joint)


# ─────────────────────── Legacy (list features) matrix ───────────────────────

LEGACY_CONFIGS = [
    dict(depth=depth, normalize=normalize, features=features, atom_types=atom_types, seed=seed)
    for depth, normalize, features, atom_types, seed in itertools.product(
        (0, 2),
        (False, True),
        (None, ["element", "degree"], ["element", "implicit_hydrogens", "aromatic"]),
        (None, CUSTOM_ATOMS),
        (0, None),
    )
]


def _legacy_id(cfg):
    feats = "default" if cfg["features"] is None else "+".join(f[:3] for f in cfg["features"])
    atoms = "default" if cfg["atom_types"] is None else "custom"
    return f"d{cfg['depth']}-n{int(cfg['normalize'])}-{feats}-{atoms}-s{cfg['seed']}"


@pytest.mark.parametrize("cfg", LEGACY_CONFIGS, ids=_legacy_id)
def test_legacy_matrix(cfg, tmp_path):
    mols = MOLS
    enc = Encoder(dimension=DIM, backend="numpy", **cfg)
    assert enc.mode == "legacy"
    fp, joint = _check_outputs(enc, mols, 1)
    _check_round_trip(enc, mols, tmp_path, fp, joint)

    if _HAS_RUST:
        # Same codebook, Rust pipeline: SMILES input (Rust parser where
        # supported) and Mol input (RDKit preparation) must both agree.
        rs = Encoder(dimension=DIM, backend="rust", codebook=enc._codebook,
                     **{k: v for k, v in cfg.items() if k != "seed"})
        np.testing.assert_allclose(rs.encode(mols), fp, atol=ATOL, rtol=0)
        np.testing.assert_allclose(rs.encode_joint(mols), joint, atol=ATOL, rtol=0)
        as_mols = [Chem.MolFromSmiles(m) for m in mols]
        np.testing.assert_allclose(rs.encode(as_mols), fp, atol=ATOL, rtol=0)


# ─────────────────────── Composite (dict features) matrix ───────────────────────

def _composite_configs():
    for features, depth, normalize, bidirectional, unknown_atoms, seed in itertools.product(
        (FEATURES_CONTINUOUS, FEATURES_GRAPH_HDC),
        (0, 2),
        (False, True),
        (True, False),
        ("error", "slot"),
        (0, None),
    ):
        base = dict(features=features, depth=depth, normalize=normalize,
                    bidirectional=bidirectional, unknown_atoms=unknown_atoms, seed=seed)
        yield dict(base, graph_features=None, graph_mode="bundle")
        for graph_mode in ("bundle", "bind", "concat"):
            yield dict(base, graph_features=GRAPH_FEATURES_GRAPH_HDC, graph_mode=graph_mode)


COMPOSITE_CONFIGS = list(_composite_configs())


def _composite_id(cfg):
    feats = "cont" if cfg["features"] is FEATURES_CONTINUOUS else "ghdc"
    graph = cfg["graph_mode"] if cfg["graph_features"] else "nograph"
    return (f"{feats}-d{cfg['depth']}-n{int(cfg['normalize'])}-b{int(cfg['bidirectional'])}"
            f"-{cfg['unknown_atoms']}-{graph}-s{cfg['seed']}")


@pytest.mark.parametrize("cfg", COMPOSITE_CONFIGS, ids=_composite_id)
def test_composite_matrix(cfg, tmp_path):
    mols = MOLS + (["C[Si](C)C"] if cfg["unknown_atoms"] == "slot" else [])
    enc = Encoder(dimension=DIM, **cfg)
    assert enc.mode == "composite"
    width = 2 if (cfg["graph_features"] and cfg["graph_mode"] == "concat") else 1
    fp, joint = _check_outputs(enc, mols, width)
    _check_round_trip(enc, mols, tmp_path, fp, joint)


def test_presets_are_not_mutated_by_encoder():
    snapshot = {k: dict(v) for k, v in FEATURES_CONTINUOUS.items()}
    graph_snapshot = {k: dict(v) for k, v in GRAPH_FEATURES_GRAPH_HDC.items()}
    Encoder(dimension=DIM, seed=0, features=FEATURES_CONTINUOUS,
            graph_features=GRAPH_FEATURES_GRAPH_HDC, unknown_atoms="slot").encode("CCO")
    assert FEATURES_CONTINUOUS == snapshot
    assert GRAPH_FEATURES_GRAPH_HDC == graph_snapshot
    assert FEATURES_DEFAULT == DEFAULT_FEATURES


# ─────────────────────── Input handling ───────────────────────

def _encoders():
    encs = [
        ("legacy-numpy", Encoder(dimension=DIM, seed=0, backend="numpy")),
        ("composite", Encoder(dimension=DIM, seed=0, features=FEATURES_CONTINUOUS, normalize=True)),
    ]
    if _HAS_RUST:
        encs.append(("legacy-rust", Encoder(dimension=DIM, seed=0, backend="rust")))
    return encs


@pytest.mark.parametrize("name,enc", _encoders(), ids=lambda x: x if isinstance(x, str) else "")
class TestInputs:

    def test_input_forms_are_equivalent(self, name, enc):
        mol = Chem.MolFromSmiles("CCO")
        single_str = enc.encode("CCO")
        assert single_str.shape == (1, DIM)
        np.testing.assert_allclose(enc.encode(mol), single_str, atol=ATOL, rtol=0)
        np.testing.assert_allclose(enc.encode(["CCO"]), single_str, atol=ATOL, rtol=0)
        mixed = enc.encode(["CCO", mol, "c1ccccc1"])
        np.testing.assert_allclose(mixed[:2], np.vstack([single_str, single_str]), atol=ATOL, rtol=0)

    def test_empty_list(self, name, enc):
        assert enc.encode([]).shape == (0, DIM)
        assert enc.encode_joint([]).shape == (0, 2 * DIM)

    @pytest.mark.parametrize("bad", [None, 5, ("CCO",), np.array(["CCO"]), ["CCO", None], [5]],
                             ids=["None", "int", "tuple", "ndarray", "list-with-None", "list-of-int"])
    def test_wrong_input_type_raises(self, name, enc, bad):
        with pytest.raises(TypeError):
            enc.encode(bad)

    @pytest.mark.parametrize("bad", ["", " ", "C(C", "[Na+].[Cl-]", "[H][H]", "[2H]C"],
                             ids=["empty", "blank", "syntax", "unknown-element", "H2", "deuterium"])
    def test_invalid_molecule_raises_value_error(self, name, enc, bad):
        with pytest.raises(ValueError):
            enc.encode(bad)

    def test_chemically_invalid_valence(self, name, enc):
        # Known difference: the Rust SMILES parser does not check valences,
        # so a hexavalent carbon is accepted there but rejected by RDKit.
        smiles = "C[C](C)(C)(C)(C)C"
        if name == "legacy-rust":
            assert enc.encode(smiles).shape == (1, DIM)
        else:
            with pytest.raises(ValueError):
                enc.encode(smiles)

    def test_bad_mol_objects_raise(self, name, enc):
        with pytest.raises(ValueError, match="no atoms"):
            enc.encode(Chem.Mol())
        with pytest.raises(ValueError, match="sanitized"):
            enc.encode(Chem.MolFromSmiles("c1ccc1", sanitize=False))

    def test_explicit_hydrogens_are_normalized(self, name, enc):
        # RDKit drops explicit [H]; every path must agree with the implicit form.
        np.testing.assert_allclose(enc.encode("[H]C([H])([H])[H]"), enc.encode("C"), atol=ATOL, rtol=0)


# ─────────────────────── Invalid configurations ───────────────────────

INVALID = [
    (dict(dimension=0), "dimension"),
    (dict(dimension=2.5), "dimension"),
    (dict(depth=-1), "depth"),
    (dict(depth=1.0), "depth"),
    (dict(seed=-1), "seed"),
    (dict(seed=1.5), "seed"),
    (dict(normalize="yes"), "normalize"),
    (dict(bidirectional=1), "bidirectional"),
    (dict(backend="torch"), "backend"),
    (dict(graph_mode="stack", features=FEATURES_CONTINUOUS), "graph_mode"),
    (dict(unknown_atoms="ignore", features=FEATURES_CONTINUOUS), "unknown_atoms"),
    (dict(atom_types=[]), "atom_types"),
    (dict(atom_types=["C", "C"]), "duplicates"),
    (dict(atom_types=["C", "Xx"]), "unknown element"),
    (dict(features=[]), "at least one"),
    (dict(features={}), "at least one"),
    (dict(features=["element", "element"]), "duplicates"),
    (dict(features=["element", "mass"]), "Unknown feature"),
    (dict(features="element"), "features must be"),
    (dict(codebook=np.zeros((3, 3))), "codebook has shape"),
    (dict(codebook=np.zeros((3, 3)), features=FEATURES_CONTINUOUS), "legacy"),
    (dict(unknown_atoms="slot"), "composite"),
    (dict(graph_features=GRAPH_FEATURES_GRAPH_HDC), "composite"),
    (dict(features={"element": {}, "degree": {"kind": "continuous"}}), "bandwidth"),
    (dict(features={"element": {}, "degree": {"kind": "continuous", "bandwidth": 0}}), "bandwidth"),
    (dict(features={"element": {}, "degree": {"kind": "continuous", "bandwidth": -1.0}}), "bandwidth"),
    (dict(features={"element": {"kind": "continuous", "bandwidth": 1.0}}), "element"),
    (dict(features={"element": {"kind": "fuzzy"}}), "kind"),
    (dict(features={"element": {"bandwith": 2.0}}), "unknown keys"),
    (dict(features={"element": {"clip": -1}}), "clip"),
    (dict(features={"element": "categorical"}), "must be a dict"),
    (dict(features=FEATURES_CONTINUOUS, graph_features={"graph_size": {"kind": "categorical"}}), "continuous"),
    (dict(features=FEATURES_CONTINUOUS, graph_features={"graph_volume": {"kind": "continuous", "bandwidth": 1}}), "Unknown graph feature"),
]


@pytest.mark.parametrize("kwargs,match", INVALID, ids=[m for _, m in INVALID])
def test_invalid_config_raises_at_construction(kwargs, match):
    kwargs = {"dimension": DIM, **kwargs}
    with pytest.raises(ValueError, match=match):
        Encoder(**kwargs)


def test_lenient_but_valid_values():
    # odd / tiny dimension, single-element vocabulary, numpy integer types
    assert Encoder(dimension=3, seed=0, backend="numpy").encode("CC").shape == (1, 3)
    assert Encoder(dimension=DIM, seed=0, atom_types=["C"]).encode("CC").shape == (1, DIM)
    enc = Encoder(dimension=np.int64(DIM), depth=np.int64(1), seed=np.int64(3))
    assert enc.encode("CCO").shape == (1, DIM)
    assert Encoder(dimension=DIM, features=("element", "degree")).mode == "legacy"


# ─────────────────────── cosine_similarity ───────────────────────

class TestCosineSimilarity:

    def test_shapes_and_self_similarity(self):
        enc = Encoder(dimension=DIM, seed=0)
        fps = enc.encode(MOLS[:4])
        sim = cosine_similarity(fps, fps)
        assert sim.shape == (4, 4)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-12)
        np.testing.assert_allclose(sim, sim.T, atol=1e-12)
        assert (np.abs(sim) <= 1 + 1e-12).all()

    def test_1d_inputs_and_rectangular(self):
        enc = Encoder(dimension=DIM, seed=0)
        a, b = enc.encode(["CCO", "CCC"])
        assert cosine_similarity(a, b).shape == (1, 1)
        assert cosine_similarity(a, enc.encode(MOLS[:3])).shape == (1, 3)
        np.testing.assert_allclose(cosine_similarity(a, a), 1.0, atol=1e-12)


# ─────────────────────── Package surface ───────────────────────

def test_all_exports_exist():
    for name in hyper_fingerprints.__all__:
        assert hasattr(hyper_fingerprints, name), name
    for preset in (FEATURES_CONTINUOUS, FEATURES_GRAPH_HDC):
        Encoder(dimension=DIM, features=preset)  # every preset is a valid config
    Encoder(dimension=DIM, features=FEATURES_GRAPH_HDC, graph_features=GRAPH_FEATURES_GRAPH_HDC)
    Encoder(dimension=DIM, features=FEATURES_DEFAULT)
