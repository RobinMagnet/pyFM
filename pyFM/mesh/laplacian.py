import numpy as np
import scipy.sparse as sparse


def dia_area_mat(vertices, faces, faces_areas=None):
    """
    Compute the diagonal matrix of lumped vertex areas for the mesh Laplacian.

    Entry i on the diagonal is the area of vertex i, approximated as one third
    of the sum of the areas of its adjacent triangles.

    Parameters
    ----------
    vertices : (n, 3) np.ndarray
        Coordinates of the mesh vertices.
    faces : (m, 3) np.ndarray
        Vertex indices defining the faces.
    faces_areas : (m,) np.ndarray, optional
        Per-face areas. Computed from the mesh if not provided.

    Returns
    -------
    A : (n, n) scipy.sparse.dia_matrix
        Sparse diagonal matrix of vertex areas in dia format.
    """
    N = vertices.shape[0]

    # Compute face area
    if faces_areas is None:
        v1 = vertices[faces[:, 0]]  # (m,3)
        v2 = vertices[faces[:, 1]]  # (m,3)
        v3 = vertices[faces[:, 2]]  # (m,3)
        faces_areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)  # (m,)

    # Accumulate one third of each face area onto its three vertices.
    vertex_areas = np.zeros(N)
    np.add.at(vertex_areas, faces.flatten(), np.repeat(faces_areas / 3, 3))

    A = sparse.dia_matrix((vertex_areas, 0), shape=(N, N))
    return A


def fem_area_mat(vertices, faces, faces_areas=None):
    """
    Compute the area matrix for the mesh Laplacian using the finite elements method.

    Entry (i, i) is 1/6 of the sum of the areas of the surrounding triangles.
    Entry (i, j) is 1/12 of the sum of the areas of the triangles using edge (i, j).

    Parameters
    ----------
    vertices : (n, 3) np.ndarray
        Coordinates of the mesh vertices.
    faces : (m, 3) np.ndarray
        Vertex indices defining the faces.
    faces_areas : (m,) np.ndarray, optional
        Per-face areas. Computed from the mesh if not provided.

    Returns
    -------
    A : (n, n) scipy.sparse.csc_matrix
        Sparse area matrix in csc format.
    """
    N = vertices.shape[0]

    # Compute face area
    if faces_areas is None:
        v1 = vertices[faces[:, 0]]  # (m,3)
        v2 = vertices[faces[:, 1]]  # (m,3)
        v3 = vertices[faces[:, 2]]  # (m,3)
        faces_areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)  # (m,)

    # Use similar construction as cotangent weights
    I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])  # (3m,)
    J = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])  # (3m,)
    S = np.concatenate([faces_areas, faces_areas, faces_areas])  # (3m,)

    In = np.concatenate([I, J, I])  # (9m,)
    Jn = np.concatenate([J, I, I])  # (9m,)
    Sn = 1 / 12 * np.concatenate([S, S, 2 * S])  # (9m,)

    A = sparse.coo_matrix((Sn, (In, Jn)), shape=(N, N)).tocsc()
    return A


def cotangent_weights(vertices, faces):
    """
    Compute the cotangent weight (stiffness) matrix W for the mesh Laplacian.

    Off-diagonal entry (i, j) accumulates half the sum of the cotangents of the
    angles opposite edge (i, j), and each diagonal entry holds the negative sum
    of the off-diagonal weights on its row.

    Parameters
    ----------
    vertices : (n, 3) np.ndarray
        Coordinates of the mesh vertices.
    faces : (m, 3) np.ndarray
        Vertex indices defining the faces.

    Returns
    -------
    W : (n, n) scipy.sparse.csc_matrix
        Sparse cotangent weight matrix in csc format.
    """
    N = vertices.shape[0]

    v1 = vertices[faces[:, 0]]  # (m,3)
    v2 = vertices[faces[:, 1]]  # (m,3)
    v3 = vertices[faces[:, 2]]  # (m,3)

    # Edge lengths indexed by opposite vertex
    u1 = v3 - v2
    u2 = v1 - v3
    u3 = v2 - v1

    L1 = np.linalg.norm(u1, axis=1)  # (m,)
    L2 = np.linalg.norm(u2, axis=1)  # (m,)
    L3 = np.linalg.norm(u3, axis=1)  # (m,)

    # Compute cosine of angles
    A1 = np.einsum("ij,ij->i", -u2, u3) / (L2 * L3)  # (m,)
    A2 = np.einsum("ij,ij->i", u1, -u3) / (L1 * L3)  # (m,)
    A3 = np.einsum("ij,ij->i", -u1, u2) / (L1 * L2)  # (m,)

    # Use cot(arccos(x)) = x/sqrt(1-x^2)
    I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    J = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    S = np.concatenate([A3, A1, A2])
    S = 0.5 * S / np.sqrt(1 - S**2)

    In = np.concatenate([I, J, I, J])
    Jn = np.concatenate([J, I, I, J])
    Sn = np.concatenate([-S, -S, S, S])

    W = sparse.coo_matrix((Sn, (In, Jn)), shape=(N, N)).tocsc()
    return W


def laplacian_spectrum(W, A, spectrum_size=200):
    """
    Solve the generalized eigenvalue problem W @ x = lambda * A @ x.

    Change solver if necessary.

    Parameters
    ----------
    W : (n, n) scipy.sparse
        Sparse matrix of cotangent weights.
    A : (n, n) scipy.sparse
        Sparse matrix of area weights.
    spectrum_size : int
        Number of eigenvalues to compute.

    Returns
    -------
    eigenvalues : (spectrum_size,) np.ndarray
        Array of eigenvalues.
    eigenvectors : (n, spectrum_size) np.ndarray
        Array of eigenvectors.
    """
    try:
        eigenvalues, eigenvectors = sparse.linalg.eigsh(
            W, k=spectrum_size, M=A, sigma=-0.01
        )

    except RuntimeError:
        # raise ValueError('Matrices are not positive semidefinite')
        # Initial eigenvector values:
        print("Problem during LBO decomposition ! Please check")
        init_eigenvecs = np.random.default_rng().random((A.shape[0], spectrum_size))
        eigenvalues, eigenvectors = sparse.linalg.lobpcg(
            W, init_eigenvecs, B=A, largest=False, maxiter=40
        )

        eigenvalues = np.real(eigenvalues)
        sorting_arr = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorting_arr]
        eigenvectors = eigenvectors[:, sorting_arr]

    return eigenvalues, eigenvectors
