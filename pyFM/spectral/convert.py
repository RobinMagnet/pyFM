"""
Python implementation of:

[1] - "Deblurring and Denoising of Maps between Shapes", by Danielle Ezuz and Mirela Ben-Chen.
"""

import numpy as np
import scipy.linalg

from . import projection_utils as pju
from .nearest_neighbor import knn_query

__all__ = [
    "p2p_to_FM",
    "mesh_p2p_to_FM",
    "FM_to_p2p",
    "mesh_FM_to_p2p",
    "mesh_FM_to_p2p_precise",
]


def p2p_to_FM(p2p_21, evects1, evects2, A2=None):
    """
    Compute a functional map from a vertex to vertex map (with possible subsampling).

    Can be computed with the pseudo inverse of the eigenvectors (if no subsampling)
    or with a least square solve.

    Parameters
    ----------
    p2p_21 : (n2,) np.ndarray
        Vertex to vertex map from target to source. For each vertex on the target
        shape, gives the index of the corresponding vertex on mesh 1. Can also be
        given as a (n2, n1) sparse matrix.
    evects1 : (n1, k1) np.ndarray
        Eigenvectors on the source mesh. Possibly subsampled on the first dimension.
    evects2 : (n2, k2) np.ndarray
        Eigenvectors on the target mesh. Possibly subsampled on the first dimension.
    A2 : (n2, n2) np.ndarray or (n2,) np.ndarray, optional
        Area matrix of the target mesh. If specified, the eigenvectors can't be subsampled.

    Returns
    -------
    FM_12 : (k2, k1) np.ndarray
        Functional map corresponding to the p2p map given. Solved with the pseudo
        inverse if A2 is given, else using a least square solve.
    """
    # Pulled back eigenvectors
    evects1_pb = evects1[p2p_21, :] if np.asarray(p2p_21).ndim == 1 else p2p_21 @ evects1

    if A2 is not None:
        if A2.shape[0] != evects2.shape[0]:
            raise ValueError("Can't compute exact pseudo inverse with subsampled eigenvectors")

        if A2.ndim == 1:
            return evects2.T @ (A2[:, None] * evects1_pb)  # (k2,k1)

        return evects2.T @ (A2 @ evects1_pb)  # (k2,k1)

    # Solve with least square
    return scipy.linalg.lstsq(evects2, evects1_pb)[0]  # (k2,k1)


def mesh_p2p_to_FM(p2p_21, mesh1, mesh2, dims=None, subsample=None):
    """
    Compute a functional map from a vertex to vertex map (with possible subsampling).

    Parameters
    ----------
    p2p_21 : (n2,) np.ndarray
        Vertex to vertex map from target to source. For each vertex on the target
        shape, gives the index of the corresponding vertex on mesh 1. Can also be
        given as a (n2, n1) sparse matrix.
    mesh1 : TriMesh
        Source mesh for the functional map. Requires enough processed eigenvectors.
    mesh2 : TriMesh
        Target mesh for the functional map. Requires enough processed eigenvectors.
    dims : int or (int, int), optional
        Dimension of the functional map to return. If None uses all the processed
        eigenvectors. If a single int k, returns a (k, k) functional map. If a
        2-uple of int (k1, k2), returns a (k2, k1) functional map.
    subsample : (2,) iterable, optional
        None or size 2 iterable ((n1',), (n2',)). Subsample of vertices for both
        meshes. If specified the p2p map is between the two subsamples.

    Returns
    -------
    FM_12 : (k2, k1) np.ndarray
        Functional map corresponding to the p2p map given.
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


def FM_to_p2p(FM_12, evects1, evects2, use_adj=False, n_jobs=None):
    """
    Obtain a point to point map from a functional map C.

    Compares embeddings of dirac functions on the second mesh Phi_2.T with embeddings
    of dirac functions of the first mesh Phi_1.T.

    Either one can transport the first diracs with the functional map or the second
    ones with the adjoint, which leads to different results (the adjoint is the
    mathematically correct way).

    Parameters
    ----------
    FM_12 : (k2, k1) np.ndarray
        Functional map from mesh1 to mesh2 in reduced basis.
    evects1 : (n1, k1') np.ndarray
        First k1' eigenvectors of the first basis (k1' > k1). First dimension can
        be subsampled.
    evects2 : (n2, k2') np.ndarray
        First k2' eigenvectors of the second basis (k2' > k2). First dimension can
        be subsampled.
    use_adj : bool, optional
        Use the adjoint method.
    n_jobs : int, optional
        Number of parallel jobs. None (default) decides automatically, -1 uses all processes.

    Returns
    -------
    p2p_21 : (n2,) np.ndarray
        Match vertex i on shape 2 to vertex p2p_21[i] on shape 1, or equivalent
        result if the eigenvectors are subsampled.
    """
    k2, k1 = FM_12.shape

    if k1 > evects1.shape[1]:
        raise ValueError(
            f"At least {k1} eigenvectors should be provided on the source, "
            f"here only {evects1.shape[1]} are given"
        )
    if k2 > evects2.shape[1]:
        raise ValueError(
            f"At least {k2} eigenvectors should be provided on the target, "
            f"here only {evects2.shape[1]} are given"
        )

    if use_adj:
        emb1 = evects1[:, :k1]
        emb2 = evects2[:, :k2] @ FM_12

    else:
        emb1 = evects1[:, :k1] @ FM_12.T
        emb2 = evects2[:, :k2]

    p2p_21 = knn_query(emb1, emb2, k=1, n_jobs=n_jobs)
    return p2p_21  # (n2,)


def mesh_FM_to_p2p(FM_12, mesh1, mesh2, use_adj=False, subsample=None, n_jobs=None):
    """
    Wrapper for `FM_to_p2p` using the TriMesh class.

    Parameters
    ----------
    FM_12 : (k2, k1) np.ndarray
        Functional map in reduced basis.
    mesh1 : TriMesh
        Source mesh for the functional map.
    mesh2 : TriMesh
        Target mesh for the functional map.
    use_adj : bool, optional
        Whether to use the adjoint map.
    subsample : (2,) iterable, optional
        None or size 2 iterable ((n1',), (n2',)). Subsample of vertices for both
        meshes. If specified the p2p map is between the two subsamples.
    n_jobs : int, optional
        Number of parallel jobs. None (default) decides automatically, -1 uses all processes.

    Returns
    -------
    p2p_21 : (n2,) np.ndarray
        Match vertex i on shape 2 to vertex p2p_21[i] on shape 1.
    """
    k2, k1 = FM_12.shape
    if subsample is None:
        p2p_21 = FM_to_p2p(
            FM_12,
            mesh1.eigenvectors[:, :k1],
            mesh2.eigenvectors[:, :k2],
            use_adj=use_adj,
            n_jobs=n_jobs,
        )

    else:
        sub1, sub2 = subsample
        p2p_21 = FM_to_p2p(
            FM_12,
            mesh1.eigenvectors[sub1, :k1],
            mesh2.eigenvectors[sub2, :k2],
            use_adj=use_adj,
            n_jobs=n_jobs,
        )

    return p2p_21


def mesh_FM_to_p2p_precise(
    FM_12,
    mesh1,
    mesh2,
    precompute_dmin=True,
    use_adj=True,
    batch_size=None,
    n_jobs=None,
    verbose=False,
):
    """
    Compute a precise pointwise map between two meshes.

    For each vertex in mesh2, gives barycentric coordinates of its image on mesh1.
    See [1] for details on notations.

    [1] - "Deblurring and Denoising of Maps between Shapes", by Danielle Ezuz and Mirela Ben-Chen.

    Parameters
    ----------
    FM_12 : (k2, k1) np.ndarray
        Functional map from mesh1 to mesh2.
    mesh1 : TriMesh
        Source mesh (for the functional map) with n1 vertices.
    mesh2 : TriMesh
        Target mesh (for the functional map) with n2 vertices.
    precompute_dmin : bool, optional
        Whether to precompute all the values of delta_min. Faster but heavier in memory.
    use_adj : bool, optional
        Use the adjoint method.
    batch_size : int, optional
        If precompute_dmin is False, projects batches of points on the surface.
    n_jobs : int, optional
        Number of parallel jobs. None (default) decides automatically.
    verbose : bool, optional
        Whether to display progress information.

    Returns
    -------
    P_21 : scipy.sparse.csr_matrix
        (n2, n1) precise point to point map from mesh2 to mesh1.
    """
    k2, k1 = FM_12.shape

    if use_adj:
        emb1 = mesh1.eigenvectors[:, :k1]
        emb2 = mesh2.eigenvectors[:, :k2] @ FM_12
    else:
        emb1 = mesh1.eigenvectors[:, :k1] @ FM_12.T
        emb2 = mesh2.eigenvectors[:, :k2]

    P_21 = pju.project_pc_to_triangles(
        emb1,
        mesh1.faces,
        emb2,
        precompute_dmin=precompute_dmin,
        batch_size=batch_size,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    return P_21
