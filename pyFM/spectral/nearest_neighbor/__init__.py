"""Nearest-neighbour queries for the NumPy backend.

A kd-tree is the right tool at low dims only

    from pyFM.spectral.nearest_neighbor import knn_query, set_config

    matches = knn_query(X, Y)                  # backend chosen automatically
    matches = knn_query(X, Y, method="brute")  # or forced
    set_config(kdtree_max_dim=12)              # or retuned
"""

from .backends import brute_query, compute_sqdistmat, tree_query
from .config import DEFAULTS, get_config, reset_config, set_config
from .dispatch import choose_backend, knn_query

__all__ = [
    "DEFAULTS",
    "brute_query",
    "choose_backend",
    "compute_sqdistmat",
    "get_config",
    "knn_query",
    "reset_config",
    "set_config",
    "tree_query",
]
