import numpy as np

__all__ = ["HKS", "lm_HKS", "auto_HKS", "mesh_HKS"]


def HKS(evals, evects, time_list, scaled=False):
    """Return the Heat Kernel Signature for num_T different time values.

    The time values are interpolated in logscale between the limits
    given in the HKS paper. These limits only depend on the eigenvalues.

    Parameters
    ----------
    evals : (K,) np.ndarray
        The K eigenvalues.
    evects : (N, K) np.ndarray
        The K eigenvectors.
    time_list : (num_T,) np.ndarray
        Time values to use.
    scaled : bool, optional
        Whether to scale for each time value.

    Returns
    -------
    HKS : (N, num_T) np.ndarray
        Array where each line is the HKS for a given t.
    """
    evals_s = np.asarray(evals).flatten()
    t_list = np.asarray(time_list).flatten()

    coefs = np.exp(-np.outer(t_list, evals_s))  # (num_T,K)
    natural_HKS = np.einsum("tk,nk->nt", coefs, np.square(evects))

    if scaled:
        inv_scaling = coefs.sum(1)  # (num_T)
        return (1 / inv_scaling)[None, :] * natural_HKS

    else:
        return natural_HKS


def lm_HKS(evals, evects, landmarks, time_list, scaled=False):
    """Return the Heat Kernel Signature for some landmarks and time values.

    Parameters
    ----------
    evals : (K,) np.ndarray
        The K eigenvalues of the Laplace Beltrami operator.
    evects : (N, K) np.ndarray
        The K eigenvectors of the Laplace Beltrami operator.
    landmarks : (p,) np.ndarray
        Indices of landmarks to compute.
    time_list : (num_T,) np.ndarray
        Time values to use.
    scaled : bool, optional
        Whether to scale for each time value.

    Returns
    -------
    landmarks_HKS : (N, num_T * p) np.ndarray
        Array where each column is the HKS for a given t for some landmark.
    """

    evals_s = np.asarray(evals).flatten()
    t_list = np.asarray(time_list).flatten()

    coefs = np.exp(-np.outer(t_list, evals_s))  # (num_T,K)
    weighted_evects = evects[None, landmarks, :] * coefs[:, None, :]  # (num_T,p,K)

    landmarks_HKS = np.einsum("tpk,nk->ptn", weighted_evects, evects)  # (p,num_T,N)

    if scaled:
        inv_scaling = coefs.sum(1)  # (num_T,)
        landmarks_HKS = (1 / inv_scaling)[None, :, None] * landmarks_HKS

    return landmarks_HKS.reshape(-1, evects.shape[0]).T  # (N,p*num_E)


def auto_HKS(evals, evects, num_T, landmarks=None, scaled=True):
    """Compute HKS with an automatic choice of time values.

    Parameters
    ----------
    evals : (K,) np.ndarray
        The K eigenvalues.
    evects : (N, K) np.ndarray
        The K eigenvectors.
    num_T : int
        Number of time values to use.
    landmarks : (p,) np.ndarray, optional
        If not None, indices of landmarks to compute.
    scaled : bool, optional
        Whether to scale for each time value.

    Returns
    -------
    HKS : (N, num_T) or (N, p * num_T) np.ndarray
        Array where each column is the HKS for a given t, possibly for some landmark.
    """

    abs_ev = sorted(np.abs(evals))
    t_list = np.geomspace(4 * np.log(10) / abs_ev[-1], 4 * np.log(10) / abs_ev[1], num_T)

    if landmarks is None:
        return HKS(abs_ev, evects, t_list, scaled=scaled)
    else:
        return lm_HKS(abs_ev, evects, landmarks, t_list, scaled=scaled)


def mesh_HKS(mesh, num_T, landmarks=None, k=None):
    """Compute the Heat Kernel Signature for a mesh.

    Parameters
    ----------
    mesh : TriMesh
        Mesh on which to compute the HKS.
    num_T : int
        Number of time values to use.
    landmarks : (p,) np.ndarray, optional
        Indices of landmarks to use.
    k : int, optional
        Number of eigenvalues to use.

    Returns
    -------
    HKS : (N, num_T) np.ndarray
        Array where each line is the HKS for a given t.
    """

    assert mesh.eigenvalues is not None, "Eigenvalues should be processed"

    if k is None:
        k = len(mesh.eigenvalues)
    else:
        assert len(mesh.eigenvalues) >= k, (
            f"At least {k} eigenvalues should be computed, not {len(mesh.eigenvalues)}"
        )

    return auto_HKS(
        mesh.eigenvalues[:k],
        mesh.eigenvectors[:, :k],
        num_T,
        landmarks=landmarks,
        scaled=True,
    )
