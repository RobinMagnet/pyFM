import numpy as np


def WKS(evals, evects, energy_list, sigma, scaled=False):
    """
    Returns the Wave Kernel Signature for some energy values.

    Parameters
    ------------------------
    evects      : (N,K) array with the K eigenvectors of the Laplace Beltrami operator
    evals       : (K,) array of the K corresponding eigenvalues
    energy_list : (num_E,) values of e to use
    sigma       : (float) [positive] standard deviation to use
    scaled      : (bool) Whether to scale each energy level

    Output
    ------------------------
    WKS : (N,num_E) array where each column is the WKS for a given e
    """
    assert sigma > 0, f"Sigma should be positive ! Given value : {sigma}"

    evals = np.asarray(evals).flatten()
    indices = np.where(evals > 1e-5)[0].flatten()
    evals = evals[indices]
    evects = evects[:, indices]

    e_list = np.asarray(energy_list)
    coefs = np.exp(-np.square(e_list[:,None] - np.log(np.abs(evals))[None,:])/(2*sigma**2))  # (num_E,K)

    weighted_evects = evects[None, :, :] * coefs[:,None, :]  # (num_E,N,K)

    natural_WKS = np.einsum('tnk,nk->nt', weighted_evects, evects)  # (N,num_E)

    if scaled:
        inv_scaling = coefs.sum(1)  # (num_E)
        return (1/inv_scaling)[None,:] * natural_WKS

    else:
        return natural_WKS


def lm_WKS(evals, evects, landmarks, energy_list, sigma, scaled=False):
    """
    Returns the Wave Kernel Signature for some landmarks and energy values.


    Parameters
    ------------------------
    evects      : (N,K) array with the K eigenvectors of the Laplace Beltrami operator
    evals       : (K,) array of the K corresponding eigenvalues
    landmarks   : (p,) indices of landmarks to compute
    energy_list : (num_E,) values of e to use
    sigma       : int - standard deviation

    Output
    ------------------------
    landmarks_WKS : (N,num_E*p) array where each column is the WKS for a given e for some landmark
    """
    assert sigma > 0, f"Sigma should be positive ! Given value : {sigma}"

    evals = np.asarray(evals).flatten()
    indices = np.where(evals > 1e-2)[0].flatten()
    evals = evals[indices]
    evects = evects[:,indices]

    e_list = np.asarray(energy_list)
    coefs = np.exp(-np.square(e_list[:, None] - np.log(np.abs(evals))[None, :]) / (2*sigma**2))  # (num_E,K)
    weighted_evects = evects[None, landmarks, :] * coefs[:,None,:]  # (num_E,p,K)

    landmarks_WKS = np.einsum('tpk,nk->ptn', weighted_evects, evects)  # (p,num_E,N)

    if scaled:
        inv_scaling = coefs.sum(1)  # (num_E,)
        landmarks_WKS = ((1/inv_scaling)[None,:,None] * landmarks_WKS)

    return landmarks_WKS.reshape(-1,evects.shape[0]).T  # (N,p*num_E)


def auto_WKS(evals, evects, num_E, landmarks=None, scaled=True):
    """
    Compute WKS with an automatic choice of scale and energy

    Parameters
    ------------------------
    evals       : (K,) array of  K eigenvalues
    evects      : (N,K) array with K eigenvectors
    landmarks   : (p,) If not None, indices of landmarks to compute.
    num_E       : (int) number values of e to use
    Output
    ------------------------
    WKS or lm_WKS : (N,num_E) or (N,p*num_E)  array where each column is the WKS for a given e
                    and possibly for some landmarks
    """
    abs_ev = sorted(np.abs(evals))

    e_min,e_max = np.log(abs_ev[1]),np.log(abs_ev[-1])
    sigma = 7*(e_max-e_min)/num_E

    e_min += 2*sigma
    e_max -= 2*sigma

    energy_list = np.linspace(e_min,e_max,num_E)

    if landmarks is None:
        return WKS(abs_ev, evects, energy_list, sigma, scaled=scaled)
    else:
        return lm_WKS(abs_ev, evects, landmarks, energy_list, sigma, scaled=scaled)


def mesh_WKS(mesh, num_E, landmarks=None, k=None):
    assert mesh.eigenvalues is not None, "Eigenvalues should be processed"

    if k is None:
        k = len(mesh.eigenvalues)
    else:
        assert len(mesh.eigenvalues >= k), f"At least ${k}$ eigenvalues should be computed, not {len(mesh.eigenvalues)}"

    return auto_WKS(mesh.eigenvalues[:k], mesh.eigenvectors[:, :k], num_E, landmarks=landmarks, scaled=True)
