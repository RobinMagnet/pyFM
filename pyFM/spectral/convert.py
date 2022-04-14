import scipy.linalg
import numpy as np

from . import projection_utils as pju
from .nn_utils import knn_query


def p2p_to_FM(p2p_21, evects1, evects2, A2=None):
    """
    Compute a Functional Map from a vertex to vertex maps (with possible subsampling).
    Can compute with the pseudo inverse of eigenvectors (if no subsampling) or least square.

    Parameters
    ------------------------------
    p2p_21    : (n2,) vertex to vertex map from target to source.
                For each vertex on the target shape, gives the index of the corresponding vertex on mesh 1.
                Can also be presented as a (n2,n1) sparse matrix.
    eigvects1 : (n1,k1) eigenvectors on source mesh. Possibly subsampled on the first dimension.
    eigvects2 : (n2,k2) eigenvectors on target mesh. Possibly subsampled on the first dimension.
    A2        : (n2,n2) area matrix of the target mesh. If specified, the eigenvectors can't be subsampled

    Outputs
    -------------------------------
    FM_12       : (k2,k1) functional map corresponding to the p2p map given.
                  Solved with pseudo inverse if A2 is given, else using least square.
    """
    # Pulled back eigenvectors
    evects1_pb = evects1[p2p_21, :] if np.asarray(p2p_21).ndim == 1 else p2p_21 @ evects1

    if A2 is not None:
        if A2.shape[0] != evects2.shape[0]:
            raise ValueError("Can't compute exact pseudo inverse with subsampled eigenvectors")

        if A2.ndim == 1:
            return evects2.T @ (A2[:, None] * evects1_pb)  # (k2,k1)

        return evects2.T @ A2 @ evects1_pb  # (k2,k1)

    # Solve with least square
    return scipy.linalg.lstsq(evects2, evects1_pb)[0]  # (k2,k1)


def mesh_p2p_to_FM(p2p_21, mesh1, mesh2, dims=None, subsample=None):
    """
    Compute a Functional Map from a vertex to vertex maps (with possible subsampling).

    Parameters
    ------------------------------
    p2p_21    : (n2,) vertex to vertex map from target to source.
                For each vertex on the target shape, gives the index of the corresponding vertex on mesh 1.
                Can also be presented as a (n2,n1) sparse matrix.
    mesh1     : source mesh for the functional map. Requires enough processed eigenvectors.
    mesh2     : target mesh for the functional map. Requires enough processed eigenvectors.
    dims      : int, or 2-uple of int. Dimension of the functional map to return.
                If None uses all the processed eigenvectors.
                If single int k , returns a (k,k) functional map
                If 2-uple of int (k1,k2), returns a (k2,k1) functional map
    subsample : None or size 2 iterable ((n1',), (n2',)).
                Subsample of vertices for both mesh.
                If specified the p2p map is between the two subsamples.

    Outputs
    -------------------------------
    FM_12       : (k2,k1) functional map corresponding to the p2p map given.
    """
    if dims is None:
        k1, k2 = len(mesh1.eigenvalues), len(mesh2.eigenvalues)
    elif np.issubdtype(type(dims), np.integer):
        k1 = dims
        k2 = dims
    else:
        k1, k2 = dims

    if subsample is None:
        return p2p_to_FM(p2p_21, mesh1.eigenvectors[:, :k1], mesh2.eigenvectors[:, :k2], A2=mesh2.A)

    sub1, sub2 = subsample
    return p2p_to_FM(p2p_21, mesh1.eigenvectors[sub1, :k1], mesh2.eigenvectors[sub2, :k2], A2=None)


def FM_to_p2p(FM_12, evects1, evects2, use_adj=False, use_ANN=False, n_jobs=1):
    """
    Obtain a point to point map from a functional map C.
    Compares embeddings of dirac functions on the second mesh Phi_2.T with embeddings
    of dirac functions of the first mesh Phi_1.T

    Either one can transport the first diracs with the functional map or the second ones with
    the adjoint, which leads to different results (adjoint is the mathematically correct way)

    Parameters
    --------------------------
    FM_12     : (k2,k1) functional map from mesh1 to mesh2 in reduced basis
    eigvects1 : (n1,k1') first k' eigenvectors of the first basis  (k1'>k1).
                First dimension can be subsampled.
    eigvects2 : (n2,k2') first k' eigenvectors of the second basis (k2'>k2)
                First dimension can be subsampled.
    use_adj   : use the adjoint method
    use_ANN   : Whether to use approximate nearest neighbors
    n_jobs    : number of parallel jobs. Use -1 to use all processes


    Outputs:
    --------------------------
    p2p_21     : (n2,) match vertex i on shape 2 to vertex p2p_21[i] on shape 1,
                 or equivalent result if the eigenvectors are subsampled.
    """
    k2, k1 = FM_12.shape

    assert k1 <= evects1.shape[1], \
        f'At least {k1} should be provided, here only {evects1.shape[1]} are given'
    assert k2 <= evects2.shape[1], \
        f'At least {k2} should be provided, here only {evects2.shape[1]} are given'

    if use_adj:
        emb1 = evects1[:, :k1]
        emb2 = evects2[:, :k2] @ FM_12

    else:
        emb1 = evects1[:, :k1] @ FM_12.T
        emb2 = evects2[:, :k2]

    p2p_21 = knn_query(emb1, emb2,  k=1,
                       use_ANN=use_ANN, n_jobs=n_jobs)
    return p2p_21  # (n2,)


def mesh_FM_to_p2p(FM_12, mesh1, mesh2, use_adj=False, subsample=None, use_ANN=False, n_jobs=1):
    """
    Wrapper for `FM_to_p2p` using TriMesh class

    Parameters
    --------------------------
    FM_12     : (k2,k1) functional map in reduced basis
    mesh1     : TriMesh - source mesh for the functional map
    mesh2     : TriMesh - target mesh for the functional map
    use_ANN   : bool - whether to use approximate nearest neighbors
    use_adj   : bool - whether to use the adjoint map.
    subsample : None or size 2 iterable ((n1',), (n2',)).
                Subsample of vertices for both mesh.
                If specified the p2p map is between the two subsamples.
    n_jobs    : number of parallel jobs. Use -1 to use all processes

    Outputs:
    --------------------------
    p2p_21     : (n2,) match vertex i on shape 2 to vertex p2p_21[i] on shape 1
    """
    k2, k1 = FM_12.shape
    if subsample is None:
        p2p_21 = FM_to_p2p(FM_12, mesh1.eigenvectors[:, :k1], mesh2.eigenvectors[:, :k2],
                           use_adj=use_adj, use_ANN=use_ANN, n_jobs=n_jobs)

    else:
        sub1, sub2 = subsample
        p2p_21 = FM_to_p2p(FM_12, mesh1.eigenvectors[sub1, :k1], mesh2.eigenvectors[sub2, :k2],
                           use_adj=use_adj, use_ANN=use_ANN, n_jobs=n_jobs)

    return p2p_21


def mesh_FM_to_p2p_precise(FM_12, mesh1, mesh2, precompute_dmin=True, use_adj=True, batch_size=None,
                           n_jobs=1, verbose=False):
    """
    Computes a precise pointwise map between two meshes, that is for each vertex in mesh2, gives
    barycentric coordinates of its image on mesh1.
    See [1] for details on notations.

    [1] - "Deblurring and Denoising of Maps between Shapes", by Danielle Ezuz and Mirela Ben-Chen.

    Parameters
    ----------------------------
    FM_12           : (k2,k1) Functional map from mesh1 to mesh2
    mesh1           : Source mesh (for the functional map) with n1 vertices
    mesh2           : Target mesh (for the functional map) with n2 vertices
    precompute_dmin : Whether to precompute all the values of delta_min.
                      Faster but heavier in memory
    use_adj         : use the adjoint method
    batch_size      : If precompute_dmin is False, projects batches of points on the surface
    n_jobs          : number of parallel process for nearest neighbor precomputation

    Output
    ----------------------------
    P_21 : (n2,n1) - precise point to point map from mesh2 to mesh1
    """
    k2, k1 = FM_12.shape

    if use_adj:
        emb1 = mesh1.eigenvectors[:, :k1]
        emb2 = mesh2.eigenvectors[:, :k2] @ FM_12
    else:
        emb1 = mesh1.eigenvectors[:, :k1] @ FM_12.T
        emb2 = mesh2.eigenvectors[:, :k2]

    P_21 = pju.project_pc_to_triangles(emb1, mesh1.facelist, emb2, precompute_dmin=precompute_dmin,
                                       batch_size=batch_size, n_jobs=n_jobs, verbose=verbose)

    return P_21
