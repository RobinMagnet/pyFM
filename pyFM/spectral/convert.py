from sklearn.neighbors import KDTree
import scipy.linalg
import numpy as np

try:
    import pynndescent
    index = pynndescent.NNDescent(np.random.random((100, 3)), n_jobs=2)
    del index
    ANN = True
except ImportError:
    ANN = False


def p2p_to_FM(p2p, eigvects1, eigvects2, A2=None):
    """
    Compute a Functional Map from a vertex to vertex maps (with possible subsampling).
    Can compute with the pseudo inverse of eigenvectors (if no subsampling) or least square.

    Parameters
    ------------------------------
    p2p       : (n2,) vertex to vertex map from target to source (for the functional map).
                For each vertex on the target shape, gives the index of the corresponding vertex on mesh 1.
    eigvects1 : (n1,k1) eigenvectors on source mesh. Possibly subsampled on the first dimension.
    eigvects2 : (n2,k2) eigenvectors on target mesh. Possibly subsampled on the first dimension.
    A2        : (n2,n2) area matrix of the target mesh. If specified, the eigenvectors can't be subsampled

    Outputs
    -------------------------------
    FM          : (k2,k1) functional map corresponding to the p2p map given.
                  Solved with pseudo inverse if A2 is given, else using least square.
    """
    if A2 is not None:
        if A2.shape[0] != eigvects2.shape[0]:
            raise ValueError("Can't compute pseudo inverse with subsampled eigenvectors")

        if len(A2.shape) == 1:
            return eigvects2.T @ (A2[:, None] * eigvects1[p2p, :])  # (k2,k1)

        return eigvects2.T @ A2 @ eigvects1[p2p, :]  # (k2,k1)

    # Solve with least square
    return scipy.linalg.lstsq(eigvects2, eigvects1[p2p, :])[0]  # (k2,k1)


def mesh_p2p_to_FM(p2p, mesh1, mesh2, dims=None, subsample=None):
    """
    Compute a Functional Map from a vertex to vertex maps (with possible subsampling).

    Parameters
    ------------------------------
    p2p       : (n2,) or (n2',) vertex to vertex map from mesh2 to mesh1.
                For each vertex on mesh2 gives the index of the corresponding vertex on mesh 1.
                If subsample is specified, gives a index-to-index map between the subsamples.
    mesh1     : source mesh for the functional map. Requires enough processed eigenvectors.
    mesh2     : target mesh for the functional map. Requires enough processed eigenvectors.
    dims      : int, or 2-uple of int. Dimension of the functional map to return.
                If None uses all the processed eigenvectors.
                If single int k , returns a (k,k) functional map
                If 2-uple of int (k1,k2), returns a (k2,k1) functional map
    subsample : None or size 2 iterable ((n1',), (n2',)).
                Subsample of vertices for both mesh.
                If specified the p2p map is between the two subsamples.
    """
    if dims is None:
        k1,k2 = len(mesh1.eigenvalues),len(mesh2.eigenvalues)
    elif type(dims) is int:
        k1 = dims
        k2 = dims
    else:
        k1,k2 = dims

    if subsample is None:
        return p2p_to_FM(p2p, mesh1.eigenvectors[:, :k1], mesh2.eigenvectors[:, :k2], A2=mesh2.A)

    sub1, sub2 = subsample
    A2 = None
    # if type(mesh2.A) == scipy.sparse.dia.dia_matrix:
    #     A2 = np.array(mesh2.A.sum(1)).flatten()[sub2]
    return p2p_to_FM(p2p, mesh1.eigenvectors[sub1, :k1], mesh2.eigenvectors[sub2, :k2], A2=A2)


def FM_to_p2p(FM, eigvects1, eigvects2, use_ANN=False):
    """
    Obtain a point to point map from a functional map C.
    Compares embeddings of dirac functions on the second mesh Phi_2.T with embeddings
    of dirac functions of the first mesh (transported with the functional map) C@Phi_1.T

    Parameters
    --------------------------
    FM        : (k2,k1) functional map in reduced basis
    eigvects1 : (n1,k1') first k' eigenvectors of the first basis  (k1'>k1).
                First dimension can be subsampled.
    eigvects2 : (n2,k2') first k' eigenvectors of the second basis (k2'>k2)
                First dimension can be subsampled.
    use_ANN   : Whether to use approximate nearest neighbors

    Outputs:
    --------------------------
    p2p       : (n2,) match vertex i on shape 2 to vertex p2p[i] on shape 1,
                or equivalent result if the eigenvectors are subsampled.
    """
    if use_ANN and not ANN:
        raise ValueError('Please install pydescent to achieve Approximate Nearest Neighbor')

    k2,k1 = FM.shape

    assert k1 <= eigvects1.shape[1], \
        f'At least {k1} should be provided, here only {eigvects1.shape[1]} are given'
    assert k2 <= eigvects2.shape[1], \
        f'At least {k2} should be provided, here only {eigvects2.shape[1]} are given'

    if use_ANN:
        index = pynndescent.NNDescent(eigvects1[:, :k1] @ FM.T, n_jobs=8)
        matches,_ = index.query(eigvects2[:, :k2], k=1)  # (n2,1)
        matches = matches.flatten()  # (n2,)
    else:
        tree = KDTree(eigvects1[:, :k1] @ FM.T)  # Tree on (n1,k2)
        matches = tree.query(eigvects2[:, :k2], k=1, return_distance=False).flatten()  # (n2,)

    return matches  # (n2,)


def FM_to_p2p_adj(FM, eigvects1, eigvects2, use_ANN=False):
    """
    Obtain a point to point map from the adjoint functional map.
    Compares ebeddings of dirac functions on the second mesh transported using the **adjoint**
    functional map C.T@Phi_2.T with embeddings of dirac functions of the first mesh Phi_1.T

    Parameters
    --------------------------
    FM        : (k2,k1) functional map in reduced basis
    eigvects1 : (n1,k1') first k' eigenvectors of the first basis  (k1'>k1).
                First dimension can be subsampled.
    eigvects2 : (n2,k2') first k' eigenvectors of the second basis (k2'>k2)
                First dimension can be subsampled.
    use_ANN   : Whether to use approximate nearest neighbors

    Outputs:
    --------------------------
    p2p       : (n2,) match vertex i on shape 2 to vertex p2p[i] on shape 1,
                or equivalent result if the eigenvectors are subsampled.
    """
    if use_ANN and not ANN:
        raise ValueError('Please install pydescent to achieve Approximate Nearest Neighbor')

    k2,k1 = FM.shape

    assert k1 <= eigvects1.shape[1], \
        f'At least {k1} should be provided, here only {eigvects1.shape[1]} are given'
    assert k2 <= eigvects2.shape[1], \
        f'At least {k2} should be provided, here only {eigvects2.shape[1]} are given'

    if use_ANN:
        index = pynndescent.NNDescent(eigvects1[:, :k1], n_jobs=8)
        matches,_ = index.query(eigvects2[:, :k2] @ FM, k=1)  # (n2,1)
        matches = matches.flatten()  # (n2,)
    else:
        tree = KDTree(eigvects1[:, :k1])  # Tree on (n1,k1)
        matches = tree.query(eigvects2[:, :k2] @ FM, k=1, return_distance=False).flatten()  # (n2,)

    return matches  # (n2,)


def mesh_FM_to_p2p(FM, mesh1, mesh2, use_ANN=False, use_adj=False, subsample=False):
    """
    Obtain a point to point map from a functional map C.
    Uses either the functional map or its adjoint for computation.
    The functional map goes from mesh1 to mesh2, and the pointwise map the other way.

    Parameters
    --------------------------
    FM        : (k2,k1) functional map in reduced basis
    mesh1     : TriMesh - source mesh for the functional map
    mesh2     : TriMesh - target mesh for the functional map
    use_ANN   : bool - whether to use approximate nearest neighbors
    use_adj   : bool - whether to use the adjoint map.
    subsample : None or size 2 iterable ((n1',), (n2',)).
                Subsample of vertices for both mesh.
                If specified the p2p map is between the two subsamples.

    Outputs:
    --------------------------
    p2p       : (n2,) match vertex i on shape 2 to vertex p2p[i] on shape 1
    """
    k2,k1 = FM.shape
    if subsample is None:
        if use_adj:
            return FM_to_p2p_adj(FM, mesh1.eigenvectors[:, :k1], mesh2.eigenvectors[:, :k2], use_ANN=use_ANN)
        return FM_to_p2p(FM, mesh1.eigenvectors[:, :k1], mesh2.eigenvectors[:, :k2], use_ANN=use_ANN)

    sub1, sub2 = subsample
    if use_adj:
        return FM_to_p2p_adj(FM, mesh1.eigenvectors[sub1, :k1], mesh2.eigenvectors[sub2, :k2], use_ANN=use_ANN)
    return FM_to_p2p(FM, mesh1.eigenvectors[sub1, :k1], mesh2.eigenvectors[sub2, :k2], use_ANN=use_ANN)
