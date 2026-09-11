"""Known-good feature configurations.

``FEATURES_DEFAULT`` is the list form that selects the original product-codebook
encoder. The dict-form presets select the per-feature (composite) encoder in
which each feature is either a categorical lookup or an FHRR continuous
encoding with a ``bandwidth`` controlling how quickly similarity decays with
value distance.
"""

from __future__ import annotations

from hyper_fingerprints.features import DEFAULT_FEATURES

FEATURES_DEFAULT: list[str] = list(DEFAULT_FEATURES)

# Mirrors graph_hdc.special.molecules.make_molecule_node_encoder_map_cont
FEATURES_GRAPH_HDC: dict[str, dict] = {
    "element": {"kind": "categorical", "seed_offset": 10},
    "degree": {"kind": "continuous", "bandwidth": 2.0, "size": 10.0, "seed_offset": 20},
    "implicit_hydrogens": {"kind": "continuous", "bandwidth": 2.0, "size": 10.0, "seed_offset": 30},
}

# Mirrors graph_hdc.special.molecules.make_molecule_graph_encoder_map_cont defaults
GRAPH_FEATURES_GRAPH_HDC: dict[str, dict] = {
    "graph_size": {
        "kind": "continuous",
        "bandwidth": max(3.0, 130.0 / 7.0),
        "size": 130.0,
        "seed_offset": 0,
    },
    "graph_diameter": {
        "kind": "continuous",
        "bandwidth": max(2.0, 20.0 / 5.0),
        "size": 20.0,
        "seed_offset": 10,
    },
}

# hyper-fingerprints recommendation: continuous counts, categorical everything else
FEATURES_CONTINUOUS: dict[str, dict] = {
    "element": {"kind": "categorical"},
    "degree": {"kind": "continuous", "bandwidth": 2.0},
    "hydrogens": {"kind": "continuous", "bandwidth": 2.0},
    "charge": {"kind": "categorical"},
    "aromatic": {"kind": "categorical"},
}

__all__ = [
    "FEATURES_DEFAULT",
    "FEATURES_GRAPH_HDC",
    "GRAPH_FEATURES_GRAPH_HDC",
    "FEATURES_CONTINUOUS",
]
