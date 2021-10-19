import time
from tqdm import tqdm

import numpy as np
import scipy.linalg

import pyFM.spectral as spectral


def icp_iteration(eigvects1, eigvects2, FM, use_adj=False, n_jobs=1):
    """
    Performs an iteration of ICP.
    Conversion from a functional map to a pointwise map is done by comparing
    embeddings of dirac functions on the second mesh Phi_2.T with embeddings
    of dirac functions of the first mesh Phi_1.T.
    The diracs are transposed using the functional map or its adjoint.

    Parameters
    -------------------------
    FM        : (k2,k1) functional map in reduced basis
    eigvects1 : (n1,k1') first k' eigenvectors of the first basis  (k1'>k1).
    eigvects2 : (n2,k2') first k' eigenvectors of the second basis (k2'>k2)
    use_adj   : use the adjoint method
    n_jobs    : number of parallel jobs. Use -1 to use all processes

    Output
    --------------------------
    FM_refined : (k2,k1) An orthogonal functional map after one step of refinement
    """
    k2, k1 = FM.shape
    p2p = spectral.FM_to_p2p(FM, eigvects1, eigvects2, use_adj=use_adj, n_jobs=n_jobs)
    FM_icp = spectral.p2p_to_FM(p2p, eigvects1[:, :k1], eigvects2[:, :k2])
    U, _, VT = scipy.linalg.svd(FM_icp)
    return U @ np.eye(k2, k1) @ VT


def icp_refine(eigvects1, eigvects2, FM, nit=10, tol=1e-10, use_adj=False, n_jobs=1, verbose=False):
    """
    Refine a functional map using the standard ICP algorithm.
    One can use the adjoint instead of the functional map for pointwise map computation.

    Parameters
    --------------------------
    eigvects1 : (n1,k1') first k' eigenvectors of the first basis  (k1'>k1).
    eigvects2 : (n2,k2') first k' eigenvectors of the second basis (k2'>k2)
    FM        : (k2,k1) functional map in reduced basis
    nit       : int - Number of iterations to perform. If not specified, uses the tol parameter
    tol       : float - Maximum change in a functional map to stop refinement
                (only used if nit is not specified)
    use_adj   : use the adjoint method
    n_jobs    : number of parallel jobs. Use -1 to use all processes

    Output
    ---------------------------
    FM_icp    : ICP-refined functional map
    """
    current_FM = FM.copy()
    iteration = 1
    if verbose:
        start_time = time.time()

    if nit is not None and nit > 0:
        myrange = tqdm(range(nit)) if verbose else range(nit)
    else:
        myrange = range(10000)

    for i in myrange:
        FM_icp = icp_iteration(eigvects1, eigvects2, current_FM, use_adj=use_adj, n_jobs=n_jobs)

        if nit is None or nit == 0:
            if verbose:
                print(f'iteration : {1+i} - mean : {np.square(current_FM - FM_icp).mean():.2e}'
                      f' - max : {np.max(np.abs(current_FM - FM_icp)):.2e}')
            if np.max(np.abs(current_FM - FM_icp)) <= tol:
                break

        current_FM = FM_icp.copy()

    if nit is None or nit == 0 and verbose:
        run_time = time.time() - start_time
        print(f'ICP done with {iteration:d} iterations - {run_time:.2f} s')

    return current_FM


def mesh_icp_refine(mesh1, mesh2, FM, nit=10, tol=1e-10, use_adj=False, n_jobs=1, verbose=False):
    """
    Refine a functional map using the auxiliar ICP algorithm (different conversion
    from functional map to vertex-to-vertex)

    Parameters
    --------------------------
    mesh1   : TriMesh - Source mesh
    mesh2   : TriMesh - Target mesh
    FM      : (k2,k1) functional map in reduced basis
    nit     : int - Number of iterations to perform. If not specified, uses the tol parameter
    tol     : float - Maximum change in a functional map to stop refinement
              (only used if nit is not specified)
    use_adj : use the adjoint method
    n_jobs  : number of parallel jobs. Use -1 to use all processes

    Output
    ---------------------------
    FM_icp : ICP-refined functional map
    """
    k2, k1 = FM.shape

    result = icp_refine(mesh1.eigenvectors[:,:k1], mesh2.eigenvectors[:,:k2],
                        FM, nit=nit, tol=tol, use_adj=use_adj, n_jobs=n_jobs, verbose=verbose)

    return result
