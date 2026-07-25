from sklearn.neighbors import NearestNeighbors

__all__ = ["knn_query"]


def knn_query(X, Y, k=1, return_distance=False, n_jobs=1):
    """
    Query nearest neighbors.

    Parameters
    ----------
    X : (n1, p) np.ndarray
        First collection.
    Y : (n2, p) np.ndarray
        Second collection.
    k : int, optional
        Number of neighbors to look for.
    return_distance : bool, optional
        Whether to return the nearest neighbor distance.
    n_jobs : int, optional
        Number of parallel jobs. Set to -1 to use all processes.

    Returns
    -------
    dists : (n2, k) or (n2,) np.ndarray
        Nearest neighbor distance ((n2,) if k=1). Returned ONLY if return_distance is True.
    matches : (n2, k) or (n2,) np.ndarray
        Nearest neighbor indices ((n2,) if k=1).
    """
    tree = NearestNeighbors(n_neighbors=k, leaf_size=40, algorithm="kd_tree", n_jobs=n_jobs)
    tree.fit(X)
    dists, matches = tree.kneighbors(Y)

    if k == 1:
        dists = dists.squeeze()
        matches = matches.squeeze()

    if return_distance:
        return dists, matches
    return matches
