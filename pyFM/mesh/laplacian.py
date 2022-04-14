import numpy as np
import scipy.sparse as sparse


def dia_area_mat(vertices, faces, faces_areas=None):
    """
    Compute the diagonal matrix of lumped vertex area for mesh laplacian.
    Entry i on the diagonal is the area of vertex i, approximated as one third
    of adjacent triangles

    Parameters
    -----------------------------
    vertices   : (n,3) array of vertices coordinates
    faces      : (m,3) array of vertex indices defining faces
    faces_area : (m,) - Optional, array of per-face area

    Output
    -----------------------------
    A : (n,n) sparse diagonal matrix of vertex areas
    """
    N = vertices.shape[0]

    # Compute face area
    if faces_areas is None:
        v1 = vertices[faces[:,0]]  # (m,3)
        v2 = vertices[faces[:,1]]  # (m,3)
        v3 = vertices[faces[:,2]]  # (m,3)
        faces_areas = 0.5 * np.linalg.norm(np.cross(v2-v1,v3-v1),axis=1)  # (m,)

    I = np.concatenate([faces[:,0], faces[:,1], faces[:,2]])
    J = np.zeros_like(I)
    V = np.concatenate([faces_areas, faces_areas, faces_areas])/3

    # Get array of vertex areas
    vertex_areas = np.array(sparse.coo_matrix((V, (I, J)), shape=(N, 1)).todense()).flatten()

    A = sparse.dia_matrix((vertex_areas,0), shape=(N, N))
    return A


def fem_area_mat(vertices, faces, faces_areas=None):
    """
    Compute the area matrix for mesh laplacian using finite elements method.

    Entry (i,i) is 1/6 of the sum of the area of surrounding triangles
    Entry (i,j) is 1/12 of the sum of the area of triangles using edge (i,j)

    Parameters
    -----------------------------
    vertices   : (n,3) array of vertices coordinates
    faces      : (m,3) array of vertex indices defining faces
    faces_area : (m,) - Optional, array of per-face area

    Output
    -----------------------------
    A : (n,n) sparse area matrix
    """
    N = vertices.shape[0]

    # Compute face area
    if faces_areas is None:
        v1 = vertices[faces[:,0]]  # (m,3)
        v2 = vertices[faces[:,1]]  # (m,3)
        v3 = vertices[faces[:,2]]  # (m,3)
        faces_areas = 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1), axis=1)  # (m,)

    # Use similar construction as cotangent weights
    I = np.concatenate([faces[:,0], faces[:,1], faces[:,2]])  # (3m,)
    J = np.concatenate([faces[:,1], faces[:,2], faces[:,0]])  # (3m,)
    S = np.concatenate([faces_areas,faces_areas,faces_areas])  # (3m,)

    In = np.concatenate([I, J, I])  # (9m,)
    Jn = np.concatenate([J, I, I])  # (9m,)
    Sn = 1/12 * np.concatenate([S, S, 2*S])  # (9m,)

    A = sparse.coo_matrix((Sn, (In, Jn)), shape=(N, N)).tocsc()
    return A


def cotangent_weights(vertices, faces):
    """
    Compute the cotengenant weights matrix for mesh laplacian.

    Entry (i,i) is 1/6 of the sum of the area of surrounding triangles
    Entry (i,j) is 1/12 of the sum of the area of triangles using edge (i,j)

    Parameters
    -----------------------------
    vertices   : (n,3) array of vertices coordinates
    faces      : (m,3) array of vertex indices defining faces
    faces_area : (m,) - Optional, array of per-face area

    Output
    -----------------------------
    A : (n,n) sparse area matrix
    """
    N = vertices.shape[0]

    v1 = vertices[faces[:,0]]  # (m,3)
    v2 = vertices[faces[:,1]]  # (m,3)
    v3 = vertices[faces[:,2]]  # (m,3)

    # Edge lengths indexed by opposite vertex
    u1 = v3 - v2
    u2 = v1 - v3
    u3 = v2 - v1

    L1 = np.linalg.norm(u1,axis=1)  # (m,)
    L2 = np.linalg.norm(u2,axis=1)  # (m,)
    L3 = np.linalg.norm(u3,axis=1)  # (m,)

    # Compute cosine of angles
    A1 = np.einsum('ij,ij->i', -u2, u3) / (L2*L3)  # (m,)
    A2 = np.einsum('ij,ij->i', u1, -u3) / (L1*L3)  # (m,)
    A3 = np.einsum('ij,ij->i', -u1, u2) / (L1*L2)  # (m,)

    # Use cot(arccos(x)) = x/sqrt(1-x^2)
    I = np.concatenate([faces[:,0], faces[:,1], faces[:,2]])
    J = np.concatenate([faces[:,1], faces[:,2], faces[:,0]])
    S = np.concatenate([A3,A1,A2])
    S = 0.5 * S / np.sqrt(1-S**2)

    In = np.concatenate([I, J, I, J])
    Jn = np.concatenate([J, I, I, J])
    Sn = np.concatenate([-S, -S, S, S])

    W = sparse.coo_matrix((Sn, (In, Jn)), shape=(N, N)).tocsc()
    return W


def laplacian_spectrum(W, A, spectrum_size=200):
    """
    Solves the generalized eigenvalue problem.
    Change solver if necessary

    Parameters
    -----------------------------
    W             : (n,n) - sparse matrix of cotangent weights
    A             : (n,n) - sparse matrix of area weights
    spectrum_size : int - number of eigenvalues to compute
    """
    try:
        eigenvalues, eigenvectors = sparse.linalg.eigsh(W, k=spectrum_size, M=A,
                                                        sigma=-0.01)

    except RuntimeError:
        # raise ValueError('Matrices are not positive semidefinite')
        # Initial eigenvector values:
        print('Problem during LBO decomposition ! Please check')
        init_eigenvecs = np.random.random((A.shape[0], spectrum_size))
        eigenvalues, eigenvectors = sparse.linalg.lobpcg(W, init_eigenvecs,
                                                         B=A, largest=False, maxiter=40)

        eigenvalues = np.real(eigenvalues)
        sorting_arr = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorting_arr]
        eigenvectors = eigenvectors[sorting_arr]

    return eigenvalues, eigenvectors
