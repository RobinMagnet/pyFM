import numpy as np
from tqdm.auto import tqdm
import scipy.sparse

import pyFM.spectral as spectral


def zoomout_iteration(FM_12, evects1, evects2, step=1, A2=None, use_ANN=False, n_jobs=1):
    """
    Performs an iteration of ZoomOut.

    Parameters
    --------------------
    FM_12    : (k2,k1) Functional map from evects1[:,:k1] to evects2[:,:k2]
    evects1  : (n1,k1') eigenvectors on source shape with k1' >= k1 + step.
                 Can be a subsample of the original ones on the first dimension.
    evects2  : (n2,k2') eigenvectors on target shape with k2' >= k2 + step.
                 Can be a subsample of the original ones on the first dimension.
    step     : int - step of increase of dimension.
    A2       : (n2,n2) sparse area matrix on target mesh, for vertex to vertex computation.
                 If specified, the eigenvectors can't be subsampled !
    use_ANN  : bool - if True, uses approximate nearest neighbor

    Output
    --------------------
    FM_zo : zoomout-refined functional map
    """
    k2, k1 = FM_12.shape
    try:
        step1, step2 = step
    except TypeError:
        step1 = step
        step2 = step
    new_k1, new_k2 = k1 + step1, k2 + step2

    p2p_21 = spectral.FM_to_p2p(FM_12, evects1, evects2, use_ANN=use_ANN, n_jobs=n_jobs)  # (n2,)
    # Compute the (k2+step, k1+step) FM
    FM_zo = spectral.p2p_to_FM(p2p_21, evects1[:, :new_k1], evects2[:, :new_k2], A2=A2)

    return FM_zo


def zoomout_refine(FM_12, evects1, evects2, nit=10, step=1, A2=None, subsample=None, use_ANN=False,
                   return_p2p=False, n_jobs=1, verbose=False):
    """
    Refine a functional map with ZoomOut.
    Supports subsampling for each mesh, different step size, and approximate nearest neighbor.

    Parameters
    --------------------
    eigvects1  : (n1,k1) eigenvectors on source shape with k1 >= K + nit
    eigvects2  : (n2,k2) eigenvectors on target shape with k2 >= K + nit
    FM_12      : (K,K) Functional map from from shape 1 to shape 2
    nit        : int - number of iteration of zoomout
    step       : increase in dimension at each Zoomout Iteration
    A2         : (n2,n2) sparse area matrix on target mesh.
    subsample  : tuple or iterable of size 2. Each gives indices of vertices to sample
                 for faster optimization. If not specified, no subsampling is done.
    use_ANN    : bool - whether to use approximate nearest neighbor.
                 Only trigger once dimension 90 is reached.
    return_p2p : bool - if True returns the vertex to vertex map.

    Output
    --------------------
    FM_12_zo  : zoomout-refined functional map from basis 1 to 2
    p2p_21_zo : only if return_p2p is set to True - the refined pointwise map from basis 2 to basis 1
    """
    k2_0, k1_0 = FM_12.shape
    try:
        step1, step2 = step
    except TypeError:
        step1 = step
        step2 = step

    assert k1_0 + nit*step1 <= evects1.shape[1], \
        f"Not enough eigenvectors on source : \
        {k1_0 + nit*step1} are needed when {evects1.shape[1]} are provided"
    assert k2_0 + nit*step2 <= evects2.shape[1], \
        f"Not enough eigenvectors on target : \
        {k2_0 + nit*step2} are needed when {evects2.shape[1]} are provided"

    use_subsample = False
    if subsample is not None:
        use_subsample = True
        sub1, sub2 = subsample

    FM_12_zo = FM_12.copy()

    ANN_adventage = False
    iterable = range(nit) if not verbose else tqdm(range(nit))
    for it in iterable:
        ANN_adventage = use_ANN and (FM_12_zo.shape[0] > 90) and (FM_12_zo.shape[1] > 90)  # Not so sure...

        if use_subsample:
            FM_12_zo = zoomout_iteration(FM_12_zo, evects1[sub1], evects2[sub2], A2=None,
                                         step=step, use_ANN=ANN_adventage, n_jobs=n_jobs)

        else:
            FM_12_zo = zoomout_iteration(FM_12_zo, evects1, evects2, A2=A2,
                                         step=step, use_ANN=ANN_adventage, n_jobs=n_jobs)

    if return_p2p:
        p2p_21_zo = spectral.FM_to_p2p(FM_12_zo, evects1, evects2, use_ANN=False, n_jobs=n_jobs)  # (n2,)
        return FM_12_zo, p2p_21_zo

    return FM_12_zo


def mesh_zoomout_refine(FM_12, mesh1, mesh2, nit=10, step=1, subsample=None, use_ANN=False,
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
        subsample = (sub1, sub2)

    result = zoomout_refine(FM_12, mesh1.eigenvectors, mesh2.eigenvectors, nit,
                            step=step, A2=mesh2.A, subsample=subsample,
                            use_ANN=use_ANN, return_p2p=return_p2p,
                            n_jobs=n_jobs, verbose=verbose)

    return result


def mesh_zoomout_refine_p2p(p2p_21, mesh1, mesh2, k_init, nit=10, step=1, subsample=None, use_ANN=False,
                            return_p2p=False, n_jobs=1, p2p_on_sub=False, verbose=False):

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
        if p2p_on_sub:
            raise ValueError("P2P can't be defined on undefined subsample")
        if verbose:
            print(f'Computing farthest point sampling of size {subsample}')
        sub1 = mesh1.extract_fps(subsample)
        sub2 = mesh2.extract_fps(subsample)
        subsample = (sub1, sub2)

    if p2p_on_sub:
        FM_12_init = spectral.mesh_p2p_to_FM(p2p_21, mesh1, mesh2, dims=k_init, subsample=subsample)
    else:
        FM_12_init = spectral.mesh_p2p_to_FM(p2p_21, mesh1, mesh2, dims=k_init, subsample=None)

    result = zoomout_refine(FM_12_init, mesh1.eigenvectors, mesh2.eigenvectors, nit,
                            step=step, A2=mesh2.A, subsample=subsample,
                            use_ANN=use_ANN, return_p2p=return_p2p,
                            n_jobs=n_jobs, verbose=verbose)

    return result
