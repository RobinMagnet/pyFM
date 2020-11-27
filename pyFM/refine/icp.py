import time

import numpy as np
import scipy.linalg

import pyFM.spectral as spectral


def icp_iteration(eigvects1, eigvects2, FM):
    """
    Performs an iteration of ICP.
    The nearest neighbors are computed so that for each row in Phi2,
    we look for the nearest row in Phi1 @ C.T

    Parameters
    -------------------------
    FM        : (k2,k1) functional map in reduced basis
    eigvects1 : (n1,k1') first k' eigenvectors of the first basis  (k1'>k1).
    eigvects2 : (n2,k2') first k' eigenvectors of the second basis (k2'>k2)

    Output
    --------------------------
    FM_refined : (k2,k1) An orthogonal functional map after one step of refinement
    """
    k2, k1 = FM.shape
    p2p = spectral.FM_to_p2p(FM, eigvects1, eigvects2)
    FM_icp = spectral.p2p_to_FM(p2p, eigvects1[:, :k1], eigvects2[:, :k2])
    U, _, VT = scipy.linalg.svd(FM_icp)
    return U @ np.eye(k2, k1) @ VT


def icp_iteration_aux(eigvects1, eigvects2, FM):
    """
    Performs an iteration of ICP with another type of query for nearest neighbor.
    The nearest neighbors are computed so that for each row in Phi2 @ C,
    we look for the nearest row in Phi1

    Parameters
    -------------------------
    FM        : (k2,k1) functional map in reduced basis
    eigvects1 : (n1,k1') first k' eigenvectors of the first basis  (k1'>k1).
    eigvects2 : (n2,k2') first k' eigenvectors of the second basis (k2'>k2)

    Output
    --------------------------
    FM_refined : (k2,k1) An orthogonal functional map after one step of refinement
    """
    k2, k1 = FM.shape
    p2p = spectral.FM_to_p2p_aux(FM, eigvects1, eigvects2)
    FM_icp = spectral.p2p_to_FM(p2p, eigvects1[:, :k1], eigvects2[:, :k2])
    U, _, VT = scipy.linalg.svd(FM_icp)
    return U @ np.eye(k2, k1) @ VT


def icp_refine(eigvects1, eigvects2, FM, nit=None, tol=1e-10, verbose=False):
    """
    Refine a functional map using the standard ICP algorithm.

    Parameters
    --------------------------
    eigvects1 : (n1,k1') first k' eigenvectors of the first basis  (k1'>k1).
    eigvects2 : (n2,k2') first k' eigenvectors of the second basis (k2'>k2)
    FM        : (k2,k1) functional map in reduced basis
    nit       : int - Number of iterations to perform. If not specified, uses the tol parameter
    tol       : float - Maximum change in a functional map to stop refinement
                (only used if nit is not specified)

    Output
    ---------------------------
    FM_icp    : ICP-refined functional map
    """
    current_FM = FM.copy()
    continue_icp = True
    iteration = 0
    if verbose:
        start_time = time.time()

    while continue_icp:
        iteration += 1
        FM_icp = icp_iteration(eigvects1, eigvects2, current_FM)

        if nit is None or nit == 0:
            continue_icp = np.max(np.abs(current_FM-FM_icp)) > tol
        else:
            continue_icp = iteration < nit

        if verbose:
            print(f'iteration : {iteration} - mean : {np.square(current_FM - FM_icp).mean():.2e}'
                  f' - max : {np.max(np.abs(current_FM - FM_icp)):.2e}')

        current_FM = FM_icp.copy()

    if verbose:
        run_time = time.time() - start_time
        print(f'ICP done with {iteration:d} iterations - {run_time:.2f} s')

    return current_FM


def icp_refine_aux(eigvects1, eigvects2, FM, nit=None, tol=1e-10, verbose=False):
    """
    Refine a functional map using the auxiliar ICP algorithm (different conversion
    from functional map to vertex-to-vertex)

    Parameters
    --------------------------
    eigvects1 : (n1,k1') first k' eigenvectors of the first basis  (k1'>k1).
    eigvects2 : (n2,k2') first k' eigenvectors of the second basis (k2'>k2)
    FM        : (k2,k1) functional map in reduced basis
    nit       : int - Number of iterations to perform. If not specified, uses the tol parameter
    tol       : float - Maximum change in a functional map to stop refinement
                (only used if nit is not specified)

    Output
    ---------------------------
    FM_icp    : ICP-refined functional map
    """
    current_FM = FM.copy()
    continue_icp = True
    iteration = 1
    if verbose:
        start_time = time.time()

    while continue_icp:
        FM_icp = icp_iteration_aux(eigvects1, eigvects2, current_FM)

        if nit is None or nit == 0:
            continue_icp = np.max(np.abs(current_FM - FM_icp)) > tol
        else:
            continue_icp = iteration < nit
        if verbose:
            print(f'iteration : {iteration} - mean : {np.square(current_FM - FM_icp).mean():.2e}'
                  f' - max : {np.max(np.abs(current_FM - FM_icp)):.2e}')
        iteration += 1
        current_FM = FM_icp.copy()

    if verbose:
        run_time = time.time() - start_time
        print(f'ICP done with {iteration:d} iterations - {run_time:.2f} s')

    return current_FM


def mesh_icp_refine(mesh1, mesh2, FM, nit=None, tol=1e-10, use_aux=False, verbose=False):
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
    use_aux : whether to use the second algorithm instead of the original one.

    Output
    ---------------------------
    FM_icp : ICP-refined functional map
    """
    k2,k1 = FM.shape
    if use_aux:
        result = icp_refine_aux(mesh1.eigenvectors[:,:k1], mesh2.eigenvectors[:,:k2],
                                FM, nit=nit, tol=tol, verbose=verbose)

    else:
        result = icp_refine(mesh1.eigenvectors[:,:k1], mesh2.eigenvectors[:,:k2],
                            FM, nit=nit, tol=tol, verbose=verbose)

    return result
