import numpy as np


def HKS(evals, evects, time_list,scaled=False):
    """
    Returns the Heat Kernel Signature for num_T different values.
    The values of the time are interpolated in logscale between the limits
    given in the HKS paper. These limits only depends on the eigenvalues.

    Parameters
    ------------------------
    evals     : (K,) array of the K eigenvalues
    evecs     : (N,K) array with the K eigenvectors
    time_list : (num_T,) Time values to use
    scaled    : (bool) whether to scale for each time value

    Output
    ------------------------
    HKS : (N,num_T) array where each line is the HKS for a given t
    """
    evals_s = np.asarray(evals).flatten()
    t_list = np.asarray(time_list).flatten()

    coefs = np.exp(-np.outer(t_list, evals_s))  # (num_T,K)
    weighted_evects = evects[None, :, :] * coefs[:, None,:]  # (num_T,N,K)
    natural_HKS = np.einsum('tnk,nk->nt', weighted_evects, evects)

    if scaled:
        inv_scaling = coefs.sum(1)  # (num_T)
        return (1/inv_scaling)[None,:] * natural_HKS

    else:
        return natural_HKS


def lm_HKS(evals, evects, landmarks, time_list, scaled=False):
    """
    Returns the Heat Kernel Signature for some landmarks and time values.


    Parameters
    ------------------------
    evects      : (N,K) array with the K eigenvectors of the Laplace Beltrami operator
    evals       : (K,) array of the K corresponding eigenvalues
    landmarks   : (p,) indices of landmarks to compute
    time_list   : (num_T,) values of t to use

    Output
    ------------------------
    landmarks_HKS : (N,num_E*p) array where each column is the HKS for a given t for some landmark
    """

    evals_s = np.asarray(evals).flatten()
    t_list = np.asarray(time_list).flatten()

    coefs = np.exp(-np.outer(t_list, evals_s))  # (num_T,K)
    weighted_evects = evects[None, landmarks, :] * coefs[:,None,:]  # (num_T,p,K)

    landmarks_HKS = np.einsum('tpk,nk->ptn', weighted_evects, evects)  # (p,num_T,N)

    if scaled:
        inv_scaling = coefs.sum(1)  # (num_T,)
        landmarks_HKS = (1/inv_scaling)[None,:,None] * landmarks_HKS

    return landmarks_HKS.reshape(-1, evects.shape[0]).T  # (N,p*num_E)


def auto_HKS(evals, evects, num_T, landmarks=None, scaled=True):
    """
    Compute HKS with an automatic choice of tile values

    Parameters
    ------------------------
    evals       : (K,) array of  K eigenvalues
    evects      : (N,K) array with K eigenvectors
    landmarks   : (p,) if not None, indices of landmarks to compute.
    num_T       : (int) number values of t to use
    Output
    ------------------------
    HKS or lm_HKS : (N,num_E) or (N,p*num_E)  array where each column is the WKS for a given e
                    for some landmark
    """

    abs_ev = sorted(np.abs(evals))
    t_list = np.geomspace(4*np.log(10)/abs_ev[-1], 4*np.log(10)/abs_ev[1], num_T)

    if landmarks is None:
        return HKS(abs_ev, evects, t_list, scaled=scaled)
    else:
        return lm_HKS(abs_ev, evects, landmarks, t_list, scaled=scaled)


def mesh_HKS(mesh, num_T, landmarks=None, k=None):
    assert mesh.eigenvalues is not None, "Eigenvalues should be processed"

    if k is None:
        k = len(mesh.eigenvalues)
    else:
        assert len(mesh.eigenvalues >= k), f"At least ${k}$ eigenvalues should be computed, not {len(mesh.eigenvalues)}"

    return auto_HKS(mesh.eigenvalues[:k], mesh.eigenvectors[:,:k], num_T, landmarks=landmarks, scaled=True)
