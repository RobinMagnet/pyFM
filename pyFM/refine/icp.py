import time

import numpy as np
import scipy.linalg
from tqdm.auto import tqdm

from .. import spectral


def icp_iteration(FM_12, evects1, evects2, use_adj=False, n_jobs=1):
    """Perform an iteration of ICP.

    Conversion from a functional map to a pointwise map is done by comparing
    embeddings of dirac functions on the second mesh Phi_2.T with embeddings
    of dirac functions of the first mesh Phi_1.T.
    The diracs are transposed using the functional map or its adjoint.

    Parameters
    ----------
    FM_12 : (k2, k1) np.ndarray
        Functional map in reduced basis.
    evects1 : (n1, k1') np.ndarray
        First k1' eigenvectors of the first basis (k1' > k1).
    evects2 : (n2, k2') np.ndarray
        First k2' eigenvectors of the second basis (k2' > k2).
    use_adj : bool, optional
        Use the adjoint method.
    n_jobs : int, optional
        Number of parallel jobs. Use -1 to use all processes.

    Returns
    -------
    FM_refined : (k2, k1) np.ndarray
        An orthogonal functional map after one step of refinement.
    """
    k2, k1 = FM_12.shape
    p2p_21 = spectral.FM_to_p2p(FM_12, evects1, evects2, use_adj=use_adj, n_jobs=n_jobs)
    FM_icp = spectral.p2p_to_FM(p2p_21, evects1[:, :k1], evects2[:, :k2])
    U, _, VT = scipy.linalg.svd(FM_icp)
    return U @ np.eye(k2, k1) @ VT


def icp_refine(
    FM_12,
    evects1,
    evects2,
    nit=10,
    tol=1e-10,
    use_adj=False,
    return_p2p=False,
    n_jobs=1,
    verbose=False,
):
    """Refine a functional map using the standard ICP algorithm.

    One can use the adjoint instead of the functional map for pointwise map computation.

    Parameters
    ----------
    FM_12 : (k2, k1) np.ndarray
        Functional map from first to second basis.
    evects1 : (n1, k1') np.ndarray
        First k1' eigenvectors of the first basis (k1' > k1).
    evects2 : (n2, k2') np.ndarray
        First k2' eigenvectors of the second basis (k2' > k2).
    nit : int, optional
        Number of iterations to perform. If not specified, uses the tol parameter.
    tol : float, optional
        Maximum change in a functional map to stop refinement
        (only used if nit is not specified).
    use_adj : bool, optional
        Use the adjoint method.
    return_p2p : bool, optional
        If True, also return the vertex to vertex map from mesh2 to mesh1.
    n_jobs : int, optional
        Number of parallel jobs. Use -1 to use all processes.
    verbose : bool, optional
        Whether to display progress and per-iteration diagnostics.

    Returns
    -------
    FM_12_icp : (k2, k1) np.ndarray
        ICP-refined functional map.
    p2p_21_icp : (n2,) np.ndarray
        Only if return_p2p is set to True - the refined pointwise map
        from basis 2 to basis 1.
    """
    FM_12_curr = FM_12.copy()
    if verbose:
        start_time = time.time()

    # If nit is not given (or 0), iterate until the map stops changing (tol mode).
    use_tol = nit is None or nit == 0
    myrange = range(10000) if use_tol else range(nit)

    # In tol mode we print per-iteration diagnostics instead of a progress bar.
    n_iter = 0
    for i in tqdm(myrange, disable=not verbose or use_tol):
        n_iter = i + 1
        FM_12_icp = icp_iteration(
            FM_12_curr, evects1, evects2, use_adj=use_adj, n_jobs=n_jobs
        )

        if use_tol:
            if verbose:
                print(
                    f"iteration : {n_iter} - mean : {np.square(FM_12_curr - FM_12_icp).mean():.2e}"
                    f" - max : {np.max(np.abs(FM_12_curr - FM_12_icp)):.2e}"
                )
            if np.max(np.abs(FM_12_curr - FM_12_icp)) <= tol:
                break

        FM_12_curr = FM_12_icp.copy()

    if use_tol and verbose:
        run_time = time.time() - start_time
        print(f"ICP done with {n_iter:d} iterations - {run_time:.2f} s")

    if return_p2p:
        p2p_21_icp = spectral.FM_to_p2p(
            FM_12_icp, evects1, evects2, use_adj=use_adj, n_jobs=n_jobs
        )  # (n2,)
        return FM_12_icp, p2p_21_icp

    return FM_12_icp


def mesh_icp_refine(
    FM_12,
    mesh1,
    mesh2,
    nit=10,
    tol=1e-10,
    use_adj=False,
    return_p2p=False,
    n_jobs=1,
    verbose=False,
):
    """Refine a functional map between meshes using the ICP algorithm.

    Parameters
    ----------
    FM_12 : (k2, k1) np.ndarray
        Functional map from mesh1 to mesh2.
    mesh1 : TriMesh
        Source mesh.
    mesh2 : TriMesh
        Target mesh.
    nit : int, optional
        Number of iterations to perform. If not specified, uses the tol parameter.
    tol : float, optional
        Maximum change in a functional map to stop refinement
        (only used if nit is not specified).
    use_adj : bool, optional
        Use the adjoint method.
    return_p2p : bool, optional
        If True, also return the vertex to vertex map from mesh2 to mesh1.
    n_jobs : int, optional
        Number of parallel jobs. Use -1 to use all processes.
    verbose : bool, optional
        Whether to display progress and per-iteration diagnostics.

    Returns
    -------
    FM_12_icp : (k2, k1) np.ndarray
        ICP-refined functional map.
    p2p_21_icp : (n2,) np.ndarray
        Only if return_p2p is set to True - the refined pointwise map
        from basis 2 to basis 1.
    """
    k2, k1 = FM_12.shape

    result = icp_refine(
        FM_12,
        mesh1.eigenvectors[:, :k1],
        mesh2.eigenvectors[:, :k2],
        nit=nit,
        tol=tol,
        use_adj=use_adj,
        return_p2p=return_p2p,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    return result


def mesh_icp_refine_p2p(
    p2p_21,
    mesh1,
    mesh2,
    k_init,
    nit=10,
    tol=1e-10,
    use_adj=False,
    return_p2p=False,
    n_jobs=1,
    verbose=False,
):
    """Refine a functional map starting from an initial pointwise map.

    This algorithm starts from an initial pointwise map instead of a functional map,
    which it first converts to a functional map before running ICP.

    Parameters
    ----------
    p2p_21 : (n2,) np.ndarray
        Initial pointwise map from mesh2 to mesh1.
    mesh1 : TriMesh
        Source mesh.
    mesh2 : TriMesh
        Target mesh.
    k_init : int
        Initial number of eigenvectors to use.
    nit : int, optional
        Number of iterations to perform. If not specified, uses the tol parameter.
    tol : float, optional
        Maximum change in a functional map to stop refinement
        (only used if nit is not specified).
    use_adj : bool, optional
        Use the adjoint method.
    return_p2p : bool, optional
        If True, also return the vertex to vertex map from mesh2 to mesh1.
    n_jobs : int, optional
        Number of parallel jobs. Use -1 to use all processes.
    verbose : bool, optional
        Whether to display progress and per-iteration diagnostics.

    Returns
    -------
    FM_12_icp : (k2, k1) np.ndarray
        ICP-refined functional map.
    p2p_21_icp : (n2,) np.ndarray
        Only if return_p2p is set to True - the refined pointwise map
        from basis 2 to basis 1.
    """

    FM_12_init = spectral.mesh_p2p_to_FM(
        p2p_21, mesh1, mesh2, dims=k_init, subsample=None
    )

    result = mesh_icp_refine(
        FM_12_init,
        mesh1,
        mesh2,
        nit=nit,
        tol=tol,
        use_adj=use_adj,
        return_p2p=return_p2p,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    return result
