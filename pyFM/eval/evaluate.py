import numpy as np


def accuracy(p2p, gt_p2p, D1_geod, return_all=False, sqrt_area=None):
    """
    Computes the geodesic accuracy of a vertex to vertex map. The map goes from
    the target shape to the source shape.

    Parameters
    ----------------------
    p2p        : (n2,) - vertex to vertex map giving the index of the matched vertex on the source shape
                 for each vertex on the target shape (from a functional map point of view)
    gt_p2p     : (n2,) - ground truth mapping between the pairs
    D1_geod    : (n1,n1) - geodesic distance between pairs of vertices on the source mesh
    return_all : bool - whether to return all the distances or only the average geodesic distance

    Output
    -----------------------
    acc   : float - average accuracy of the vertex to vertex map
    dists : (n2,) - if return_all is True, returns all the pairwise distances
    """

    dists = D1_geod[(p2p,gt_p2p)]
    if sqrt_area is not None:
        dists /= sqrt_area

    if return_all:
        return dists.mean(), dists

    return dists.mean()


def continuity(p2p, D1_geod, D2_geod, edges):
    """
    Computes continuity of a vertex to vertex map. The map goes from
    the target shape to the source shape.

    Parameters
    ----------------------
    p2p     : (n2,) - vertex to vertex map giving the index of the matched vertex on the source shape
                 for each vertex on the target shape (from a functional map point of view)
    gt_p2p  : (n2,) - ground truth mapping between the pairs
    D1_geod : (n1,n1) - geodesic distance between pairs of vertices on the source mesh
    D2_geod : (n1,n1) - geodesic distance between pairs of vertices on the target mesh
    edges   : (n2,2) edges on the target shape

    Output
    -----------------------
    continuity : float - average continuity of the vertex to vertex map
    """
    source_len = D2_geod[(edges[:,0], edges[:,1])]
    target_len = D1_geod[(p2p[edges[:,0]], p2p[edges[:,1]])]

    continuity = np.mean(target_len / source_len)

    return continuity


def coverage(p2p, A):
    """
    Computes coverage of a vertex to vertex map. The map goes from
    the target shape to the source shape.

    Parameters
    ----------------------
    p2p : (n2,) - vertex to vertex map giving the index of the matched vertex on the source shape
                 for each vertex on the target shape (from a functional map point of view)
    A   : (n1,n1) or (n1,) - area matrix on the source shape or array of per-vertex areas.

    Output
    -----------------------
    coverage : float - coverage of the vertex to vertex map
    """
    if len(A.shape) == 2:
        vert_area = np.asarray(A.sum(1)).flatten()
    coverage = vert_area[np.unique(p2p)].sum() / vert_area.sum()

    return coverage
