from sklearn.neighbors import NearestNeighbors
import numpy as np

try:
    import pynndescent
    index = pynndescent.NNDescent(np.random.random((100, 3)), n_jobs=2)
    del index
    ANN = True
except ImportError:
    ANN = False


def knn_query(X, Y, k=1, return_distance=False, use_ANN=False, n_jobs=1):
    """
    Query nearest neighbors.

    Parameters
    -------------------------------
    X : (n1,p) first collection
    Y : (n2,p) second collection
    k : int - number of neighbors to look for
    return_distance : whether to return the nearest neighbor distance
    use_ANN         : use Approximate Nearest Neighbors
    n_jobs          : number of parallel jobs. Set to -1 to use all processes

    Output
    -------------------------------
    dists   : (n2,k) or (n2,) if k=1 - ONLY if return_distance is False. Nearest neighbor distance.
    matches : (n2,k) or (n2,) if k=1 - nearest neighbor
    """
    if use_ANN and ANN:
        index = pynndescent.NNDescent(X, n_jobs=n_jobs)
        matches, dists = index.query(Y, k=k)  # (n2,1)
    else:
        tree = NearestNeighbors(n_neighbors=k, leaf_size=40, algorithm="kd_tree", n_jobs=n_jobs)
        tree.fit(X)
        dists, matches = tree.kneighbors(Y)

    if k == 1:
        dists = dists.squeeze()
        matches = matches.squeeze()

    if return_distance:
        return dists, matches
    return matches
