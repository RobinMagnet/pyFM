import numpy as np
import scipy.sparse
from sklearn.neighbors import KDTree

try:
    import pynndescent
    index = pynndescent.NNDescent(np.random.random((100,3)),n_jobs=2)
    del index
    ANN = True
except ImportError:
    ANN = False


def zoomout_refine(L1, L2, A2, C, nit, step=1, subsample=None, use_ANN=False, return_p2p=False):
    """
    Refine a functional map with ZoomOut.
    Supports subsampling for each mesh, different step size, and approximate nearest neighbor.

    Parameters
    --------------------
    L1         : (n1,k1) eigenvectors on source shape with k1 >= K + nit
    L2         : (n2,k2) eigenvectors on target shape with k2 >= K + nit
    A2         : (n2,n2) sparse area matrix on target mesh
    C          : (K,K) Functional map from from L1[:,:K] to L2[:,:K]
    nit        : int - number of iteration of zoomout
    step       : increase in dimension at each Zoomout Iteration
    subsample  : tuple or iterable of size 2. Each gives indices of vertices so sample
                 for faster optimization. If not specified, no subsampling is done.
    use_ANN    : bool - whether to use approximate nearest neighbor.
                 Only trigger once dimension 90 is reached.
    return_p2p : bool - if True returns the vertex to vertex map.

    Output
    --------------------
    C_zo : zoomout-refined functional map
    """
    if use_ANN and not ANN:
        raise ValueError('Please install pydescent to achieve Approximate Nearest Neighbor')
    assert C.shape[0] == C.shape[1], f"Zoomout input should be square not {C.shape}"

    if subsample is not None:
        L1_zo = L1[subsample[0]]
        L2_zo = L2[subsample[1]]
        A2_zo = A2[subsample[1][:,None],subsample[1]]
    else:
        L1_zo = L1
        L2_zo = L2
        A2_zo = A2

    kinit = C.shape[0]

    C_zo = C.copy()

    if L1.shape[1] >= kinit+nit*step:
        raise ValueError(f"Enough eigenvectors should be provided on the source mesh, "
                         f"here {kinit+step*nit} are needed when {L1.shape[1]} are provided")
    if L2.shape[1] >= kinit+nit*step:
        raise ValueError(f"Enough eigenvectors should be provided on the target mesh, "
                         f"here {kinit+step*nit} are needed when {L2.shape[1]} are provided")

    ANN_adventage = False
    for k in [kinit + i*step for i in range(nit)]:

        if use_ANN and k > 90:
            ANN_adventage = True

        C_zo = zoomout_iteration(L1_zo, L2_zo, A2_zo, C_zo, k+step, use_ANN=ANN_adventage)

    return zoomout_iteration(L1, L2, A2, C_zo, kinit+nit*step, use_ANN=False, return_p2p=return_p2p)


def zoomout_iteration(L1,L2,A2,C,k_end,use_ANN=False, return_p2p=False):
    """
    Performs an iteration of zoomout.

    Parameters
    --------------------
    L1         : (n1',k1) eigenvectors on source shape with k1 >= K + nit.
                 Can be a subsample of the original ones.
    L2         : (n2',k2) eigenvectors on target shape with k2 >= K + nit.
                 Can be a subsample of the original ones.
    A2         : (n2',n2') sparse area matrix on target mesh.
                 Can be a subsample from the original matrix
    C          : (K,K) Functional map from from L1[:,:K] to L2[:,:K]
    k_end      : int - dimension at the end of the iteration
    use_ANN    : bool - whether to use approximate nearest neighbor
    return_p2p : bool - if True returns the vertex to vertex map.

    Output
    --------------------
    C_zo : zoomout-refined functional map
    p2p  : If return_p2p is True, (n2',) vertex to vertex mapping.
    """
    n1 = L1.shape[0]
    n2 = L2.shape[0]
    k_init = C.shape[0]

    if use_ANN:
        index = pynndescent.NNDescent((C@L1[:,:k_init].T).T,n_jobs=2)
        matches,_ = index.query(L2[:,:k_init],k=1)  # (n2',1)
        matches = matches.flatten()  # (n2',)
    else:
        tree = KDTree((C@L1[:,:k_init].T).T)  # Tree on (n1',K2)
        matches = tree.query(L2[:,:k_init],k=1,return_distance=False).flatten()  # (n2',)

    P = scipy.sparse.coo_matrix((np.ones(n2),(np.arange(n2),matches)), shape=(n2,n1)).tocsc()  # (n2',n1') point2point

    C = L2[:,:k_end].T @ A2 @ P @ L1[:,:k_end]  # (k_end,k_end)

    if return_p2p:
        return C,matches
    else:
        return C
