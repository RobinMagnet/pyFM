import numpy as np

__all__ = ["accuracy", "continuity", "coverage"]


def accuracy(p2p, gt_p2p, D1_geod, return_all=False, sqrt_area=None):
    """Compute the geodesic accuracy of a vertex to vertex map.

    The map goes from the target shape to the source shape.

    Parameters
    ----------
    p2p : (n2,) np.ndarray
        Vertex to vertex map giving the index of the matched vertex on the source
        shape for each vertex on the target shape (from a functional map point of view).
    gt_p2p : (n2,) np.ndarray
        Ground truth mapping between the pairs.
    D1_geod : (n1, n1) np.ndarray
        Geodesic distance between pairs of vertices on the source mesh.
    return_all : bool, optional
        Whether to return all the distances or only the average geodesic distance.
    sqrt_area : float, optional
        If provided, the square root of the mesh area used to normalize the distances.

    Returns
    -------
    acc : float
        Average accuracy of the vertex to vertex map.
    dists : (n2,) np.ndarray
        Only if return_all is True - all the pairwise distances.
    """

    dists = D1_geod[(p2p, gt_p2p)]
    if sqrt_area is not None:
        dists /= sqrt_area

    if return_all:
        return dists.mean(), dists

    return dists.mean()


def continuity(p2p, D1_geod, D2_geod, edges):
    """Compute the continuity of a vertex to vertex map.

    The map goes from the target shape to the source shape.

    Parameters
    ----------
    p2p : (n2,) np.ndarray
        Vertex to vertex map giving the index of the matched vertex on the source
        shape for each vertex on the target shape (from a functional map point of view).
    D1_geod : (n1, n1) np.ndarray
        Geodesic distance between pairs of vertices on the source mesh.
    D2_geod : (n2, n2) np.ndarray
        Geodesic distance between pairs of vertices on the target mesh.
    edges : (n_edges, 2) np.ndarray
        Edges on the target shape.

    Returns
    -------
    continuity : float
        Average continuity of the vertex to vertex map.
    """
    source_len = D2_geod[(edges[:, 0], edges[:, 1])]
    target_len = D1_geod[(p2p[edges[:, 0]], p2p[edges[:, 1]])]

    continuity = np.mean(target_len / source_len)

    return continuity


def coverage(p2p, A):
    """Compute the coverage of a vertex to vertex map.

    The map goes from the target shape to the source shape.

    Parameters
    ----------
    p2p : (n2,) np.ndarray
        Vertex to vertex map giving the index of the matched vertex on the source
        shape for each vertex on the target shape (from a functional map point of view).
    A : (n1, n1) or (n1,) np.ndarray
        Area matrix on the source shape or array of per-vertex areas.

    Returns
    -------
    coverage : float
        Coverage of the vertex to vertex map.
    """
    if len(A.shape) == 2:
        vert_area = np.asarray(A.sum(1)).flatten()
    else:
        vert_area = np.asarray(A).flatten()
    coverage = vert_area[np.unique(p2p)].sum() / vert_area.sum()

    return coverage
