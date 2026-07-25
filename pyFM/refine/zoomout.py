import numpy as np
from tqdm.auto import tqdm

from .. import spectral


def zoomout_iteration(FM_12, evects1, evects2, step=1, A2=None, n_jobs=1):
    """Perform an iteration of ZoomOut.

    Parameters
    ----------
    FM_12 : (k2, k1) np.ndarray
        Functional map from evects1[:, :k1] to evects2[:, :k2].
    evects1 : (n1, k1') np.ndarray
        Eigenvectors on source shape with k1' >= k1 + step.
        Can be a subsample of the original ones on the first dimension.
    evects2 : (n2, k2') np.ndarray
        Eigenvectors on target shape with k2' >= k2 + step.
        Can be a subsample of the original ones on the first dimension.
    step : int or tuple, optional
        Step of increase of dimension. A tuple gives separate steps for each shape.
    A2 : (n2, n2) scipy.sparse, optional
        Area matrix on target mesh, for vertex to vertex computation.
        If specified, the eigenvectors can't be subsampled !
    n_jobs : int, optional
        Number of parallel jobs. Use -1 to use all processes.

    Returns
    -------
    FM_zo : (k2 + step, k1 + step) np.ndarray
        ZoomOut-refined functional map.
    """
    k2, k1 = FM_12.shape
    try:
        step1, step2 = step
    except TypeError:
        step1 = step
        step2 = step
    new_k1, new_k2 = k1 + step1, k2 + step2

    p2p_21 = spectral.FM_to_p2p(FM_12, evects1, evects2, n_jobs=n_jobs)  # (n2,)
    # Compute the (k2+step, k1+step) FM
    FM_zo = spectral.p2p_to_FM(p2p_21, evects1[:, :new_k1], evects2[:, :new_k2], A2=A2)

    return FM_zo


def zoomout_refine(
    FM_12,
    evects1,
    evects2,
    nit=10,
    step=1,
    A2=None,
    subsample=None,
    return_p2p=False,
    n_jobs=1,
    verbose=False,
):
    """Refine a functional map with ZoomOut.

    Supports subsampling for each mesh, different step size, and approximate
    nearest neighbor.

    Parameters
    ----------
    FM_12 : (k2, k1) np.ndarray
        Functional map from shape 1 to shape 2.
    evects1 : (n1, k1') np.ndarray
        Eigenvectors on source shape with k1' >= k1 + nit * step.
    evects2 : (n2, k2') np.ndarray
        Eigenvectors on target shape with k2' >= k2 + nit * step.
    nit : int, optional
        Number of iterations of ZoomOut.
    step : int or tuple, optional
        Increase in dimension at each ZoomOut iteration. A tuple gives separate
        steps for each shape.
    A2 : (n2, n2) scipy.sparse, optional
        Area matrix on target mesh.
    subsample : tuple or iterable of size 2, optional
        Each element gives indices of vertices to sample for faster optimization.
        If not specified, no subsampling is done.
    return_p2p : bool, optional
        If True, also return the vertex to vertex map.
    n_jobs : int, optional
        Number of parallel jobs. Use -1 to use all processes.
    verbose : bool, optional
        Whether to display a progress bar.

    Returns
    -------
    FM_12_zo : (k2 + nit * step, k1 + nit * step) np.ndarray
        ZoomOut-refined functional map from basis 1 to 2.
    p2p_21_zo : (n2,) np.ndarray
        Only if return_p2p is set to True - the refined pointwise map
        from basis 2 to basis 1.
    """
    k2_0, k1_0 = FM_12.shape
    try:
        step1, step2 = step
    except TypeError:
        step1 = step
        step2 = step

    assert k1_0 + nit * step1 <= evects1.shape[1], (
        f"Not enough eigenvectors on source : \
        {k1_0 + nit * step1} are needed when {evects1.shape[1]} are provided"
    )
    assert k2_0 + nit * step2 <= evects2.shape[1], (
        f"Not enough eigenvectors on target : \
        {k2_0 + nit * step2} are needed when {evects2.shape[1]} are provided"
    )

    use_subsample = False
    if subsample is not None:
        use_subsample = True
        sub1, sub2 = subsample

    FM_12_zo = FM_12.copy()

    iterable = range(nit) if not verbose else tqdm(range(nit))
    for it in iterable:
        if use_subsample:
            FM_12_zo = zoomout_iteration(
                FM_12_zo,
                evects1[sub1],
                evects2[sub2],
                A2=None,
                step=step,
                n_jobs=n_jobs,
            )

        else:
            FM_12_zo = zoomout_iteration(
                FM_12_zo, evects1, evects2, A2=A2, step=step, n_jobs=n_jobs
            )

    if return_p2p:
        p2p_21_zo = spectral.FM_to_p2p(FM_12_zo, evects1, evects2, n_jobs=n_jobs)  # (n2,)
        return FM_12_zo, p2p_21_zo

    return FM_12_zo


def mesh_zoomout_refine(
    FM_12,
    mesh1,
    mesh2,
    nit=10,
    step=1,
    subsample=None,
    return_p2p=False,
    n_jobs=1,
    verbose=False,
):
    """Refine a functional map between meshes with ZoomOut.

    Supports subsampling for each mesh, different step size, and approximate
    nearest neighbor.

    Parameters
    ----------
    FM_12 : (k2, k1) np.ndarray
        Functional map from mesh1 to mesh2.
    mesh1 : TriMesh
        Source mesh.
    mesh2 : TriMesh
        Target mesh.
    nit : int, optional
        Number of iterations of ZoomOut.
    step : int or tuple, optional
        Increase in dimension at each ZoomOut iteration. A tuple gives separate
        steps for each shape.
    subsample : int or tuple or iterable of size 2, optional
        If an int, size of the farthest point sampling to compute on each mesh.
        Otherwise, each element gives indices of vertices to sample for faster
        optimization. If not specified, no subsampling is done.
    return_p2p : bool, optional
        If True, also return the vertex to vertex map.
    n_jobs : int, optional
        Number of parallel jobs. Use -1 to use all processes.
    verbose : bool, optional
        Whether to display progress.

    Returns
    -------
    FM_zo : np.ndarray
        ZoomOut-refined functional map.
    p2p : np.ndarray
        Only if return_p2p is set to True - the refined pointwise map.
    """

    if np.issubdtype(type(subsample), np.integer):
        if verbose:
            print(f"Computing farthest point sampling of size {subsample}")
        sub1 = mesh1.extract_fps(subsample)
        sub2 = mesh2.extract_fps(subsample)
        subsample = (sub1, sub2)

    result = zoomout_refine(
        FM_12,
        mesh1.eigenvectors,
        mesh2.eigenvectors,
        nit,
        step=step,
        A2=mesh2.A,
        subsample=subsample,
        return_p2p=return_p2p,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    return result


def mesh_zoomout_refine_p2p(
    p2p_21,
    mesh1,
    mesh2,
    k_init,
    nit=10,
    step=1,
    subsample=None,
    return_p2p=False,
    n_jobs=1,
    p2p_on_sub=False,
    verbose=False,
):
    """Refine a functional map between meshes with ZoomOut, starting from a p2p map.

    This algorithm starts from an initial pointwise map, which it first converts
    to a functional map before running ZoomOut. Supports subsampling for each mesh,
    different step size, and approximate nearest neighbor.

    Parameters
    ----------
    p2p_21 : (n2,) np.ndarray
        Initial pointwise map from mesh2 to mesh1.
    mesh1 : TriMesh
        Source mesh.
    mesh2 : TriMesh
        Target mesh.
    k_init : int
        Initial number of eigenvectors to use for the functional map.
    nit : int, optional
        Number of iterations of ZoomOut.
    step : int or tuple, optional
        Increase in dimension at each ZoomOut iteration. A tuple gives separate
        steps for each shape.
    subsample : int or tuple or iterable of size 2, optional
        If an int, size of the farthest point sampling to compute on each mesh.
        Otherwise, each element gives indices of vertices to sample for faster
        optimization. If not specified, no subsampling is done.
    return_p2p : bool, optional
        If True, also return the vertex to vertex map.
    n_jobs : int, optional
        Number of parallel jobs. Use -1 to use all processes.
    p2p_on_sub : bool, optional
        Whether the initial p2p map is defined on the subsampled vertices.
    verbose : bool, optional
        Whether to display progress.

    Returns
    -------
    FM_zo : np.ndarray
        ZoomOut-refined functional map.
    p2p : np.ndarray
        Only if return_p2p is set to True - the refined pointwise map.
    """

    if np.issubdtype(type(subsample), np.integer):
        if p2p_on_sub:
            raise ValueError("P2P can't be defined on undefined subsample")
        if verbose:
            print(f"Computing farthest point sampling of size {subsample}")
        sub1 = mesh1.extract_fps(subsample)
        sub2 = mesh2.extract_fps(subsample)
        subsample = (sub1, sub2)

    if p2p_on_sub:
        FM_12_init = spectral.mesh_p2p_to_FM(p2p_21, mesh1, mesh2, dims=k_init, subsample=subsample)
    else:
        FM_12_init = spectral.mesh_p2p_to_FM(p2p_21, mesh1, mesh2, dims=k_init, subsample=None)

    result = zoomout_refine(
        FM_12_init,
        mesh1.eigenvectors,
        mesh2.eigenvectors,
        nit,
        step=step,
        A2=mesh2.A,
        subsample=subsample,
        return_p2p=return_p2p,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    return result
