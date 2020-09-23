import numpy as np
from sklearn.neighbors import KDTree
import scipy.linalg


def get_P2P(FM, eigvects1, eigvects2):
    """
    Obtain a point to point map from a functional map

    Parameters
    --------------------------
    FM : (k2,k1) functional map in reduced basis
    eigvects1 : (n1,k1') first k' eigenvectors of the first basis (k1'>k1)
    eigvects2 : (n2,k2') first k' eigenvectors of the second basis (k2'>k2)

    Outputs:
    --------------------------
    p2p       : (n2,) match vertex i on shape 2 to vertex p2p[i] on shape 1
    """
    k2,k1 = FM.shape

    assert k1 <= eigvects1.shape[1], f'At least {k1} should be provided, here only {eigvects1.shape[1]} are given'
    assert k2 <= eigvects2.shape[1], f'At least {k2} should be provided, here only {eigvects2.shape[1]} are given'

    tree = KDTree((FM@eigvects1[:,:k1].T).T)  # Tree on (n1,K2)
    matches = tree.query(eigvects2[:,:k2],k=1,return_distance=False).flatten()  # (n2,)

    return matches  # (n2,)


def get_SD(mesh1, mesh2, k1=None, k2=None, mapping=None, SD_type='spectral'):
    """
    Computes shape difference operators from a vertex to vertex map.

    Parameterss
    -----------------------------
    mesh1   : pyFM.mesh.TriMesh object with computed eigenvectors. Source mesh
    mesh2   : pyFM.mesh.TriMesh object with computed eigenvectors. Target mesh
    k1      : Dimension to use on the source basis. If None, use all the computed eigenvectors
    k2      : Dimension to use on the source basis if SD_type is 'spectral'.
              If None and SD_type is spectral, uses 3*k1
    mapping : (n2,n1) vertex to vertex matrix between the two shapes.
              If None, set to diagonal matrix.
    SD_type : 'spectral' | 'semican' : first option uses the LB basis on the target shape.
              Second option uses the canonical basis on the target shape

    Output
    ----------------------------
    SD_a,SD_c : (k1,k1), (k1,k1) Area and conformal shape difference operators on the reduced basis.
    """
    assert SD_type in ['spectral', 'semican'], f"Problem with type of SD type : {SD_type}"

    if k1 is None:
        k1 = len(mesh1.eigenvalues)

    if k2 is None:
        k2 = 3*k1

    if mapping is None:
        mapping = np.eye(mesh1.n_vertices)

    if SD_type == 'spectral':
        FM = mesh2.eigenvectors[:,:k2].T @ mesh2.A @ mapping @ mesh1.eigenvectors[:,:k1]  # (K2,K1)
        SD_a = FM.T @ FM  # (K1,K1)
        SD_c = np.linalg.pinv(np.diag(mesh1.eigenvalues[:k1])) @ FM.T @ np.diag(mesh2.eigenvalues[:k2]) @ FM  # (K1,K1)

    elif SD_type == 'semican':
        FM = mapping@mesh1.eigenvectors[:,:k1]  # (n2,K1)
        SD_a = FM.T @ mesh2.A @ FM  # (K1,K1)
        SD_c = np.linalg.pinv(np.diag(mesh1.eigenvalues[:k1])) @ FM.T @ mesh2.W @ FM  # (K1,K1)

    return SD_a, SD_c


def p2p_to_FM(p2p, mesh1, mesh2, dims=None, subsample=None):
    """
    Compute a Functional Map from a vertex to vertex maps (with possible subsampling).

    Parameters
    ------------------------------
    p2p       : (n2,) or (n2',) vertex to vertex map from mesh2 to mesh1.
                For each vertex on mesh2 gives the index of the corresponding vertex on mesh 1.
                Can work for the arguments of subsample
    mesh1     : source mesh for the functional map. Requires enough processed eigenvectors.
    mesh2     : target mesh for the functional map. Requires enough processed eigenvectors.
    dims      : None, int, or 2-uple of int. Dimension of the functional map.
                If None uses all the processed eigenvectors.
                If single int k , returns a (k,k) functional map
                If 2-uple of int (k1,k2), returns a (k2,k1) functional map
    subsample : None or size 2 iterable. Subsample of vertices for both mesh. If not None,
                then the p2p map is between the two subsamples.
    """
    if dims is None:
        k1,k2 = len(mesh1.eigenvalues),len(mesh2.eigenvalues)
    elif type(dims) is int:
        k1 = dims
        k2 = dims
    else:
        k1,k2 = dims

    if subsample is None:
        mapping = np.zeros((mesh2.n_vertices,mesh1.n_vertices))
        mapping[np.arange(mesh2.n_vertices),p2p] = 1

        # FM = mesh2.eigenvectors[:,:k2].T @ mesh2.A @ mapping @ mesh1.eigenvectors[:,:k1]
        FM = mesh2.eigenvectors[:,:k2].T @ mesh2.A @ mesh1.eigenvectors[p2p,:k1]

    else:
        sub1,sub2 = subsample
        # mapping = np.zeros((len(sub2),len(sub1)))
        # mapping[np.arange(len(sub2)),p2p] = 1

        FM = scipy.linalg.lstsq(mesh2.eigenvectors[sub2,:k2],mesh1.eigenvectors[sub1[p2p],:k1])[0]
        # FM = mesh2.eigenvectors[sub2,:k2].T @ mesh2.A[sub2[:,None],sub2] @ mapping@ mesh1.eigenvectors[sub1,:k1]

    return FM
