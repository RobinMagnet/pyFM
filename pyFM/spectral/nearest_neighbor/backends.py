"""The two nearest-neighbour backends, plus the dense distance matrix helper.

Shape conventions used throughout: n1 is the number of reference points (X), n2 the number of query points (Y), p the embedding dimension and k the
number of neighbours.
"""

import numpy as np
import sklearn
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

from .config import get_config


def _resolve_workers(n_ref, n_query, n_jobs):
    """Turn n_jobs into a cKDTree workers count.

    None means decide automatically.

    Parameters
    ----------
    n_ref : int
        Number of reference points, n1.
    n_query : int
        Number of query points, n2.
    n_jobs : int or None
        Requested worker count, or None to decide from the problem size.

    Returns
    -------
    workers : int
        Worker count to hand to cKDTree.query.
    """
    if n_jobs is not None:
        return n_jobs
    if n_ref * n_query < get_config("parallel_min_work"):
        return 1
    return -1


def tree_query(X, Y, k=1, return_distance=False, n_jobs=None, leaf_size=None, **_):
    """Nearest neighbours via a kd-tree. Best in low dimension.

    Trailing ``**_`` is to ignore any extra arguments passed by the dispatcher.

    Parameters
    ----------
    X : np.ndarray
        (n1, p). Reference points, the set being searched.
    Y : np.ndarray
        (n2, p). Query points.
    k : int
        Number of neighbours.
    return_distance : bool
        Whether to also return distances.
    n_jobs : int or None
        -1 uses all cores, 1 forces serial. None (default) uses all cores when
        n1 * n2 >= parallel_min_work, serial below it.
    leaf_size : int, optional
        cKDTree leafsize. Defaults to the configured value.

    Returns
    -------
    dists : np.ndarray, optional
        (n2,) if k = 1 else (n2, k). Distance to each neighbour. Only if return_distance.
    matches : np.ndarray
        (n2,) if k = 1 else (n2, k). Index in X of each neighbour.
    """
    leaf_size = get_config("leaf_size") if leaf_size is None else leaf_size
    workers = _resolve_workers(X.shape[0], Y.shape[0], n_jobs)

    tree = cKDTree(X, leafsize=leaf_size)
    # scipy already squeezes the last axis when k == 1
    dists, matches = tree.query(Y, k=k, workers=workers)  # (n2,) or (n2, k)

    if return_distance:
        return dists, matches
    return matches


def brute_query(X, Y, k=1, return_distance=False, n_jobs=None, working_memory=None, **_):
    r"""Nearest neighbours by brute force. Best above a handful of dimensions.

    Scikit learn uses nice mixed precision with memory handling.

    Parameters
    ----------
    X : np.ndarray
        (n1, p). Reference points, the set being searched.
    Y : np.ndarray
        (n2, p). Query points.
    k : int
        Number of neighbours.
    return_distance : bool
        Whether to also return distances.
    n_jobs : int
        Passed through to scikit-learn.
    working_memory : int, optional
        Chunking budget in MB. Defaults to the configured value.

    Returns
    -------
    dists : np.ndarray, optional
        (n2,) if k = 1 else (n2, k). Distance to each neighbour. Only if return_distance.
    matches : np.ndarray
        (n2,) if k = 1 else (n2, k). Index in X of each neighbour.
    """
    if working_memory is None:
        working_memory = get_config("working_memory_mb")

    with sklearn.config_context(working_memory=working_memory):
        tree = NearestNeighbors(n_neighbors=k, algorithm="brute", n_jobs=n_jobs)
        tree.fit(X)
        dists, matches = tree.kneighbors(Y)  # (n2, k)

    if k == 1:
        dists = dists.squeeze(-1)  # (n2,)
        matches = matches.squeeze(-1)  # (n2,)

    if return_distance:
        return dists, matches
    return matches


def compute_sqdistmat(X, Y, normalized=False):
    """Pairwise squared Euclidean distance matrix between two sets of points X and Y.

    Parameters
    ----------
    X : np.ndarray
        (n1, p) The first set of points.
    Y : np.ndarray
        (n2, p) The second set of points.
    normalized : bool
        Whether the points already have unit norm. If so the squared distance reduces
        to 2 - 2 X.Y, which skips the two norm computations.

    Returns
    -------
    distmat : np.ndarray
        (n1, n2). Squared Euclidean distance between each pair.
    """

    if not normalized:
        # (n1, 1) + (1, n2) -> (n1, n2)
        return (
            np.square(X).sum(-1, keepdims=True)
            + np.square(Y).sum(-1, keepdims=True).T
            - 2 * (X @ Y.T)
        )
    else:
        return 2 - 2 * X @ Y.T  # (n1, n2)
