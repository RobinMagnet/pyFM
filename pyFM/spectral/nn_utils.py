"""Deprecated alias for :mod:`pyFM.spectral.nearest_neighbor`."""

import warnings

from .nearest_neighbor import knn_query

__all__ = ["knn_query"]

warnings.warn(
    "pyFM.spectral.nn_utils is deprecated, use pyFM.spectral.nearest_neighbor instead",
    DeprecationWarning,
    stacklevel=2,
)
