import numpy as np

from . import convert


def area_SD(FM):
    """
    Return the area shape difference computed from a functional map.

    Parameters
    ---------------------------
    FM : (k2,k1) functional map between two meshes

    Output
    ----------------------------
    SD : (k1,k1) - Area based shape difference operator
    """
    SD = FM.T @ FM
    return SD


def conformal_SD(FM, evals1, evals2):
    """
    Return the conformal shape difference operator computed from a functional map.

    Parameters
    ---------------------------
    FM     : (k2,k1) functional map between two meshes
    evals1 : eigenvalues of the LBO on the source mesh (at least k1)
    evals2 : eigenvalues of the LBO on the target mesh (at least k2)

    Output
    ----------------------------
    SD : (k1,k1) - Conformal shape difference operator
    """
    k2,k1 = FM.shape

    SD = np.linalg.pinv(np.diag(evals1[:k1])) @ FM.T @ (evals2[:k2,None] * FM)
    return SD


def compute_SD(mesh1, mesh2, k1=None, k2=None, p2p=None, SD_type='spectral'):
    """
    Computes shape difference operators from a vertex to vertex map.

    Parameterss
    -----------------------------
    mesh1   : pyFM.mesh.TriMesh object with computed eigenvectors. Source mesh
    mesh2   : pyFM.mesh.TriMesh object with computed eigenvectors. Target mesh
    k1      : Dimension to use on the source basis. If None, use all the computed eigenvectors
    k2      : Dimension to use on the source basis if SD_type is 'spectral'.
              If None and SD_type is spectral, uses 3*k1
    p2p     : (n2,) vertex to vertex map between the two meshes.
              If None, set to the identity mapping
    SD_type : 'spectral' | 'semican' : first option uses the LB basis on the target shape.
              Second option uses the canonical basis on the target shape

    Output
    ----------------------------
    SD_a, SD_c : (k1,k1), (k1,k1) Area and conformal shape difference operators on the reduced basis
    """
    assert SD_type in ['spectral', 'semican'], f"Problem with type of SD type : {SD_type}"

    if k1 is None:
        k1 = len(mesh1.eigenvalues)

    if k2 is None:
        k2 = 3*k1

    if p2p is None:
        p2p = np.arange(mesh2.n_vertices)

    if SD_type == 'spectral':
        FM = convert.mesh_p2p_to_FM(p2p, mesh1, mesh2, dims=(k1, k2))  # (K2,K1)
        SD_a = area_SD(FM)  # (K1,K1)
        SD_c = conformal_SD(FM, mesh1.eigenvalues, mesh2.eigenvalues)  # (K1,K1)

    elif SD_type == 'semican':
        FM = mesh1.eigenvectors[p2p,:k1]  # (n2,K1)
        SD_a = FM.T @ mesh2.A @ FM  # (K1,K1)
        SD_c = np.linalg.pinv(np.diag(mesh1.eigenvalues[:k1])) @ FM.T @ mesh2.W @ FM  # (K1,K1)

    return SD_a, SD_c
