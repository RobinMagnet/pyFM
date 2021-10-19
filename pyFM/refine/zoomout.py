import numpy as np
from tqdm import tqdm

import pyFM.spectral as spectral


def zoomout_iteration(eigvects1, eigvects2, FM, step=1, A2=None, use_ANN=False, n_jobs=1):
    """
    Performs an iteration of ZoomOut on a functional map. That is iven a

    Parameters
    --------------------
    eigvects1  : (n1,k1') eigenvectors on source shape with k1' >= k1 + step.
                 Can be a subsample of the original ones on the first dimension.
    eigvects2  : (n2,k2') eigenvectors on target shape with k2' >= k2 + step.
                 Can be a subsample of the original ones on the first dimension.
    FM         : (k2,k1) Functional map from from eigvects1[:,:k1] to eigvects2[:,:k2]
    step       : int or (int,int) - step of increase of dimension.
    A2         : (n2,n2) sparse area matrix on target mesh, for vertex to vertex computation.
                 If specified, the eigenvectors can't be subsampled !
    use_ANN    : bool - if True, uses approximate nearest neighbor
    n_jobs     : number of parallel jobs. Use -1 to use all processes

    Output
    --------------------
    FM_zo : (k2+step2, k1+step1) zoomout-refined functional map
    """
    k2, k1 = FM.shape
    try:
        step1, step2 = step
    except TypeError:
        step1 = step
        step2 = step
    new_k1, new_k2 = k1 + step1, k2 + step2

    p2p = spectral.FM_to_p2p(FM, eigvects1, eigvects2, use_ANN=use_ANN, n_jobs=n_jobs)  # (n2,)
    FM_zo = spectral.p2p_to_FM(p2p, eigvects1[:, :new_k1], eigvects2[:, :new_k2], A2=A2)  # (k2+step, k1+step)

    return FM_zo


def zoomout_refine(eigvects1, eigvects2, FM, nit=10, step=1, A2=None, subsample=None, use_ANN=False,
                   return_p2p=False, n_jobs=1, verbose=False):
    """
    Refine a functional map with ZoomOut.
    Supports subsampling for each mesh, different step size, and approximate nearest neighbor.

    Parameters
    --------------------
    eigvects1  : (n1,k1) eigenvectors on source shape with k1 >= K + nit
    eigvects2  : (n2,k2) eigenvectors on target shape with k2 >= K + nit
    FM         : (K,K) Functional map from from L1[:,:K] to L2[:,:K]
    nit        : int - number of iteration of zoomout
    step       : int or (int,int) - increase in dimension at each Zoomout Iteration
    A2         : (n2,n2) sparse area matrix on target mesh.
    subsample  : tuple or iterable of size 2. Each gives indices of vertices to sample
                 for faster optimization. If not specified, no subsampling is done.
    use_ANN    : bool - whether to use approximate nearest neighbor.
                 Only trigger once dimension 90 is reached.
    return_p2p : bool - if True returns the vertex to vertex map.
    n_jobs     : number of parallel jobs. Use -1 to use all processes

    Output
    --------------------
    FM_zo : zoomout-refined functional map
    p2p   : only if return_p2p is set to True - the refined pointwise map
    """
    k2_0, k1_0 = FM.shape
    try:
        step1, step2 = step
    except TypeError:
        step1 = step
        step2 = step

    assert k1_0 + nit*step1 <= eigvects1.shape[1], \
        f"Not enough eigenvectors on source : \
        {k1_0 + nit*step1} are needed when {eigvects1.shape[1]} are provided"
    assert k2_0 + nit*step2 <= eigvects2.shape[1], \
        f"Not enough eigenvectors on target : \
        {k2_0 + nit*step2} are needed when {eigvects2.shape[1]} are provided"

    use_subsample = False
    if subsample is not None:
        use_subsample = True
        sub1, sub2 = subsample

    FM_zo = FM.copy()

    ANN_adventage = False
    iterable = range(nit) if not verbose else tqdm(range(nit))
    for it in iterable:
        ANN_adventage = use_ANN and (FM_zo.shape[0] > 90) and (FM_zo.shape[1] > 90)  # Not so sure...

        if use_subsample:
            FM_zo = zoomout_iteration(eigvects1[sub1], eigvects2[sub2], FM_zo, A2=None,
                                      step=step, use_ANN=ANN_adventage, n_jobs=n_jobs)

        else:
            FM_zo = zoomout_iteration(eigvects1, eigvects2, FM_zo, A2=A2,
                                      step=step, use_ANN=ANN_adventage, n_jobs=n_jobs)

    if return_p2p:
        p2p_zo = spectral.FM_to_p2p(FM_zo, eigvects1, eigvects2, use_ANN=False, n_jobs=n_jobs)  # (n2,)
        return FM_zo, p2p_zo

    return FM_zo


def mesh_zoomout_refine(mesh1, mesh2, FM, nit=10, step=1, subsample=None, use_ANN=False,
                        return_p2p=False, n_jobs=1, verbose=False):
    """
    Refine a functional map between meshes with ZoomOut.
    Supports subsampling for each mesh, different step size, and approximate nearest neighbor.

    Parameters
    --------------------
    mesh1      : TriMesh - Source mesh
    mesh2      : TriMesh - Target mesh
    FM         : (K,K) Functional map between
    nit        : int - number of iteration of zoomout
    step       : increase in dimension at each Zoomout Iteration
    A2         : (n2,n2) sparse area matrix on target mesh.
    subsample  : int or tuple or iterable of size 2. Each gives indices of vertices so sample
                 for faster optimization. If not specified, no subsampling is done.
    use_ANN    : bool - whether to use approximate nearest neighbor.
                 Only trigger once dimension 90 is reached.
    return_p2p : bool - if True returns the vertex to vertex map.

    Output
    --------------------
    FM_zo : zoomout-refined functional map
    p2p   : only if return_p2p is set to True - the refined pointwise map
    """

    if np.issubdtype(type(subsample), np.integer):
        if verbose:
            print(f'Computing farthest point sampling of size {subsample}')
        sub1 = mesh1.extract_fps(subsample)
        sub2 = mesh2.extract_fps(subsample)
        subsample = (sub1,sub2)

    result = zoomout_refine(mesh1.eigenvectors, mesh2.eigenvectors, FM, nit,
                            step=step, A2=mesh2.A, subsample=subsample,
                            use_ANN=use_ANN, return_p2p=return_p2p,
                            n_jobs=n_jobs, verbose=verbose)

    return result
