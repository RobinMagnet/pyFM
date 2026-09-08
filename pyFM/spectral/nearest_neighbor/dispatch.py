"""Backend selection (and batching).

Shape conventions n1 is they number of keys (X), n2 the number of queries (Y), p the embedding dimension,
k the number of neighbours, and B the batch size.
"""

from .backends import brute_query, tree_query
from .config import get_config

_BACKENDS = {"tree": tree_query, "brute": brute_query}


def choose_backend(n_keys, dim, method="auto"):
    """Pick the backend for a query of n_keys reference points in dim dimensions.

    KDtree is only useful in low dim with enough points to search.

    Parameters
    ----------
    n_keys : int
        Number of reference points, n1 (the set being searched).
    dim : int
        Dimension p of the embedding.
    method : str
        "auto", "tree" or "brute". Anything but "auto" forces that backend.

    Returns
    -------
    backend : callable
        One of :func:`~.backends.tree_query` or :func:`~.backends.brute_query`.
    """
    if method != "auto":
        if method not in _BACKENDS:
            raise ValueError(f"method must be 'auto', 'tree' or 'brute', got {method!r}")
        return _BACKENDS[method]

    if dim <= get_config("kdtree_max_dim") and n_keys >= get_config("tree_min_points"):
        return tree_query
    return brute_query


def knn_query(
    X,
    Y,
    k=1,
    return_distance=False,
    n_jobs=None,
    *,
    method="auto",
    leaf_size=None,
    working_memory=None,
):
    """Query the k nearest neighbours in X of each point of Y.

    The backend is chosen according to :func:`choose_backend`.

    Parameters
    ----------
    X : np.ndarray
        (n1, p). Reference points, the set being searched.
    Y : np.ndarray
        (n2, p). Query points.
    k : int
        Number of neighbours
    return_distance : bool
        Whether to also return the nearest neighbour distances.
    n_jobs : int or None
        Number of parallel jobs. -1 uses all processes, 1 forces single thread. None (default)
        uses all cores when n1 * n2 >= parallel_min_work (see :mod:`.config`).
        Only affects the tree backend.
    method : str
        "auto" (default), or "tree" / "brute" to force a backend.
    leaf_size : int, optional
        Override the configured cKDTree leafsize.
    working_memory : int, optional
        Override the configured chunking budget in MB. Ignored by the tree backend.

    Returns
    -------
    dists : np.ndarray, optional
        (n2,) if k = 1 else (n2, k). Distance to each
        neighbour. Only if return_distance is True.
    matches : np.ndarray
        (n2,) if k = 1 else (n2, k). Index in X of each
        neighbour.
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must both be 2D arrays")

    backend = choose_backend(X.shape[0], X.shape[1], method)
    return backend(
        X,
        Y,
        k=k,
        return_distance=return_distance,
        n_jobs=n_jobs,
        leaf_size=leaf_size,
        working_memory=working_memory,
    )
