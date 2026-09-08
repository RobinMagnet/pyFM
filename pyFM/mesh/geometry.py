import numpy as np
import potpourri3d as pp3d
import scipy.sparse as sparse
from tqdm.auto import tqdm


def edges_from_faces(faces):
    """
    Compute all edges in the mesh.

    Parameters
    ----------
    faces : (m, 3) np.ndarray
        Array defining faces with vertex indices.

    Returns
    -------
    edges : (p, 2) np.ndarray
        Array of all edges defined by vertex indices, in no particular order.
    """
    # Number of vertices
    N = 1 + np.max(faces)

    # Use a sparse matrix and find non-zero elements
    # This is way faster than a np.unique somehow
    I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    J = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])

    # Orienting each edge (lo, hi) makes the two half-edges of an interior edge
    # collide, so building the matrix already deduplicates them.
    lo, hi = np.minimum(I, J), np.maximum(I, J)

    M = sparse.csr_matrix((np.ones(lo.size, dtype=bool), (lo, hi)), shape=(N, N)).tocoo()
    return np.stack([M.row, M.col], axis=1)


def compute_faces_areas(vertices, faces):
    """
    Compute per-face areas of a triangular mesh.

    Parameters
    ----------
    vertices : (n, 3) np.ndarray
        Coordinates of the mesh vertices.
    faces : (m, 3) np.ndarray
        Vertex indices defining the faces.

    Returns
    -------
    faces_areas : (m,) np.ndarray
        Array of per-face areas.
    """

    v1 = vertices[faces[:, 0]]  # (m,3)
    v2 = vertices[faces[:, 1]]  # (m,3)
    v3 = vertices[faces[:, 2]]  # (m,3)
    faces_areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)  # (m,)

    return faces_areas


def compute_vertex_areas(vertices, faces, faces_areas=None):
    """
    Compute per-vertex areas of a triangular mesh.

    The area of a vertex is approximated as one third of the sum of the areas
    of its adjacent triangles.

    Parameters
    ----------
    vertices : (n, 3) np.ndarray
        Coordinates of the mesh vertices.
    faces : (m, 3) np.ndarray or None
        Vertex indices defining the faces. If None (point cloud), uniform unit
        areas are returned.
    faces_areas : (m,) np.ndarray, optional
        Per-face areas. Computed from the mesh if not provided.

    Returns
    -------
    vert_areas : (n,) np.ndarray
        Array of per-vertex areas.
    """
    N = vertices.shape[0]

    if faces is None:
        # Point cloud: no connectivity, hence no meaningful face-based areas.
        # Fall back to uniform unit areas (one per vertex).
        return np.ones(N)

    if faces_areas is None:
        faces_areas = compute_faces_areas(vertices, faces)  # (m,)

    # Accumulate one third of each face area onto its three vertices.
    vertex_areas = np.zeros(N)
    np.add.at(vertex_areas, faces.flatten(), np.repeat(faces_areas / 3, 3))

    return vertex_areas


def compute_normals(vertices, faces):
    """
    Compute face normals of a triangular mesh.

    Parameters
    ----------
    vertices : (n, 3) np.ndarray
        Coordinates of the mesh vertices.
    faces : (m, 3) np.ndarray
        Vertex indices defining the faces.

    Returns
    -------
    normals : (m, 3) np.ndarray
        Array of normalized per-face normals.
    """
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]

    normals = np.cross(v2 - v1, v3 - v1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    return normals


def per_vertex_normal(vertices, faces, face_normals=None, weighting="uniform"):
    """
    Compute per-vertex normals of a triangular mesh, with a chosen weighting scheme.

    Parameters
    ----------
    vertices : (n, 3) np.ndarray
        Coordinates of the mesh vertices.
    faces : (m, 3) np.ndarray
        Vertex indices defining the faces.
    face_normals : (m, 3) np.ndarray, optional
        Per-face normals.
    weighting : str, optional
        Weighting scheme, either 'area' or 'uniform'.

    Returns
    -------
    vert_normals : (n, 3) np.ndarray
        Array of per-vertex normals.
    """
    if weighting.lower() == "uniform":
        vert_normals = per_vertex_normal_uniform(vertices, faces, face_normals=face_normals)

    elif weighting.lower() == "area":
        vert_normals = per_vertex_normal_area(vertices, faces)

    else:
        raise ValueError("Did not implement other weighting scheme for vertex-normals")

    return vert_normals


def per_vertex_normal_area(vertices, faces):
    """
    Compute per-vertex normals of a triangular mesh, weighted by the area of adjacent faces.

    Parameters
    ----------
    vertices : (n, 3) np.ndarray
        Coordinates of the mesh vertices.
    faces : (m, 3) np.ndarray
        Vertex indices defining the faces.

    Returns
    -------
    vert_normals : (n, 3) np.ndarray
        Array of per-vertex normals.
    """

    n_vertices = vertices.shape[0]

    v1 = vertices[faces[:, 0]]  # (m,3)
    v2 = vertices[faces[:, 1]]  # (m,3)
    v3 = vertices[faces[:, 2]]  # (m,3)

    # That is 2* A(T) n(T) with A(T) area of face T
    face_normals_weighted = np.cross(1e3 * (v2 - v1), 1e3 * (v3 - v1))  # (m,3)

    vert_normals = np.zeros((n_vertices, 3))
    np.add.at(vert_normals, faces.flatten(), np.repeat(face_normals_weighted, 3, axis=0))
    vert_normals /= 1e-8 + np.linalg.norm(vert_normals, axis=1, keepdims=True)

    return vert_normals


def per_vertex_normal_uniform(vertices, faces, face_normals=None):
    """
    Compute per-vertex normals of a triangular mesh, with uniform weights across adjacent faces.

    Parameters
    ----------
    vertices : (n, 3) np.ndarray
        Coordinates of the mesh vertices.
    faces : (m, 3) np.ndarray
        Vertex indices defining the faces.
    face_normals : (m, 3) np.ndarray, optional
        Per-face normals. Computed from the mesh if not provided.

    Returns
    -------
    vert_normals : (n, 3) np.ndarray
        Array of per-vertex normals.
    """

    n_vertices = vertices.shape[0]

    v1 = vertices[faces[:, 0]]  # (m,3)
    v2 = vertices[faces[:, 1]]  # (m,3)
    v3 = vertices[faces[:, 2]]  # (m,3)

    if face_normals is None:
        face_normals = np.cross(1e3 * (v2 - v1), 1e3 * (v3 - v1))  # (m,3)
        face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)

    vert_normals = np.zeros((n_vertices, 3))
    np.add.at(vert_normals, faces.flatten(), np.repeat(face_normals, 3, axis=0))
    vert_normals /= 1e-8 + np.linalg.norm(vert_normals, axis=1, keepdims=True)

    return vert_normals


def neigh_faces(faces):
    """
    Return the indices of neighbor faces for each vertex.

    This assumes all vertices appear in the face list.

    Parameters
    ----------
    faces : (m, 3) np.ndarray
        Vertex indices defining the faces.

    Returns
    -------
    neighbors : list
        Length-n list, where each entry is the list of indices of the neighbor
        faces of the corresponding vertex.
    """
    n_vertices = 1 + faces.max()

    neighbors = [set() for i in range(n_vertices)]

    for face_ind, (i, j, k) in enumerate(faces):
        neighbors[i].add(face_ind)
        neighbors[j].add(face_ind)
        neighbors[k].add(face_ind)

    neighbors = [list(x) for x in neighbors]

    return neighbors


def _get_grad_dir(vertices, faces, normals, face_areas=None):
    """
    Compute the gradient directions for each face for the hat basis.

    Parameters
    ----------
    vertices : (n, 3) np.ndarray
        Coordinates of the vertices.
    faces : (m, 3) np.ndarray
        Vertex indices for each face.
    normals : (m, 3) np.ndarray
        Normal coordinates for each face.
    face_areas : (m,) np.ndarray, optional
        Per-face areas, for faster computation. Computed from the mesh if not provided.

    Returns
    -------
    grads : (3, m, 3) np.ndarray
        Array of per-face gradient directions.
    """

    v1 = vertices[faces[:, 0]]  # (m,3)
    v2 = vertices[faces[:, 1]]  # (m,3)
    v3 = vertices[faces[:, 2]]  # (m,3)

    if face_areas is None:
        face_areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)  # (m,)

    grad1 = np.cross(normals, v3 - v2) / (2 * face_areas[:, None])  # (m,3)
    grad2 = np.cross(normals, v1 - v3) / (2 * face_areas[:, None])  # (m,3)
    grad3 = np.cross(normals, v2 - v1) / (2 * face_areas[:, None])  # (m,3)

    return np.asarray([grad1, grad2, grad3])


def grad_mat(vertices, faces, normals=None, face_areas=None, order_style="C"):
    """
    Return the gradient operator as a (3 * m, n) matrix G.

    This is a 'flattened' version of the gradient. Given a function f of shape
    (n,), the gradient is given by (G @ f).reshape(order=order_style).

    Parameters
    ----------
    vertices : (n, 3) np.ndarray
        Coordinates of the vertices.
    faces : (m, 3) np.ndarray
        Vertex indices for each face.
    normals : (m, 3) np.ndarray, optional
        Normal coordinates for each face. Computed from the mesh if not provided.
    face_areas : (m,) np.ndarray, optional
        Per-face areas, for faster computation. Computed from the mesh if not provided.
    order_style : str, optional
        Order style to use for reshape, either 'C' or 'F'.

    Returns
    -------
    G : (3 * m, n) scipy.sparse.csr_matrix
        Gradient matrix.
    """
    assert order_style in ["F", "C"], "Only C or F are implemented order styles"
    n_faces = faces.shape[0]
    n_vertices = vertices.shape[0]

    if normals is None:
        normals = compute_normals(vertices, faces)

    grad_dir = _get_grad_dir(vertices, faces, normals, face_areas=face_areas)

    I = np.repeat(np.arange(n_faces), 3)
    J = faces.flatten()

    if order_style == "F":
        In = np.concatenate([I, I + n_faces, I + 2 * n_faces])
        Jn = np.tile(J, 3)
        Vn = grad_dir.flatten(order="F")

    else:
        In = np.concatenate([3 * I, 3 * I + 1, 3 * I + 2])
        Jn = np.tile(J, 3)
        Vn = grad_dir.flatten(order="F")

    Gmat = sparse.csr_matrix((Vn, (In, Jn)), shape=(3 * n_faces, n_vertices))

    return Gmat


def grad_f(f, vertices, faces, normals, face_areas=None, use_sym=False, grads=None):
    """
    Compute the gradient of one or multiple functions on a mesh.

    Takes a function defined on each vertex and returns a per-face gradient.

    Parameters
    ----------
    f : (n, p) or (n,) np.ndarray
        Function values on each vertex.
    vertices : (n, 3) np.ndarray
        Coordinates of the vertices.
    faces : (m, 3) np.ndarray
        Vertex indices for each face.
    normals : (m, 3) np.ndarray
        Normal coordinates for each face.
    face_areas : (m,) np.ndarray, optional
        Per-face areas, for faster computation. Computed from the mesh if not provided.
    use_sym : bool, optional
        If True, uses the (slower but) symmetric expression of the gradient.
    grads : iterable, optional
        Iterable of size 3 containing arrays of size (m, 3) giving gradient
        directions for all faces (see function ``_get_grad_dir``).

    Returns
    -------
    gradient : (m, 3) or (m, p, 3) np.ndarray
        Gradient of f on the mesh.
    """

    if grads is not None:
        grad1, grad2, grad3 = grads[0], grads[1], grads[2]

    v1 = vertices[faces[:, 0]]  # (m,3)
    v2 = vertices[faces[:, 1]]  # (m,3)
    v3 = vertices[faces[:, 2]]  # (m,3)

    f1 = f[faces[:, 0]]  # (m,p) or (m,)
    f2 = f[faces[:, 1]]  # (m,p) or (m,)
    f3 = f[faces[:, 2]]  # (m,p) or (m,)

    # Compute area for each face
    if face_areas is None:
        face_areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)  # (m,)

    if not use_sym:
        if grads is None:
            grad2 = np.cross(normals, v1 - v3) / (2 * face_areas[:, None])  # (m,3)
            grad3 = np.cross(normals, v2 - v1) / (2 * face_areas[:, None])  # (m,3)

        if f.ndim == 1:
            gradient = (f2 - f1)[:, None] * grad2 + (f3 - f1)[:, None] * grad3  # (m,3)
        else:
            gradient = (f2 - f1)[:, :, None] * grad2[:, None, :] + (f3 - f1)[:, :, None] * grad3[
                :, None, :
            ]  # (m,3)

    else:
        if grads is None:
            grad1 = np.cross(normals, v3 - v2) / (2 * face_areas[:, None])  # (m,3)
            grad2 = np.cross(normals, v1 - v3) / (2 * face_areas[:, None])  # (m,3)
            grad3 = np.cross(normals, v2 - v1) / (2 * face_areas[:, None])  # (m,3)

        if f.ndim == 1:
            gradient = f1[:, None] * grad1 + f2[:, None] * grad2 + f3[:, None] * grad3  # (m,3)
        else:
            gradient = (
                f1[:, :, None] * grad1[:, None, :]
                + f2[:, :, None] * grad2[:, None, :]
                + f3[:, :, None] * grad3[:, None, :]
            )  # (m,p,3)

    return gradient


def div_f(f, vertices, faces, normals, vert_areas=None, grads=None, face_areas=None):
    """
    Compute the divergence of a vector field on a mesh.

    Takes a per-face vector field and returns a per-vertex divergence.

    Parameters
    ----------
    f : (m, 3) np.ndarray
        Vector field on each face.
    vertices : (n, 3) np.ndarray
        Coordinates of the vertices.
    faces : (m, 3) np.ndarray
        Vertex indices for each face.
    normals : (m, 3) np.ndarray
        Normal coordinates for each face.
    vert_areas : (n,) np.ndarray, optional
        Per-vertex areas, for faster computation. Computed from the mesh if not provided.
    grads : iterable, optional
        Iterable of size 3 containing arrays of size (m, 3) giving gradient
        directions for all faces.
    face_areas : (m,) np.ndarray, optional
        Per-face areas, for faster computation. Only used if grads is given.

    Returns
    -------
    divergence : (n,) np.ndarray
        Divergence of f on the mesh.
    """
    n_vertices = vertices.shape[0]

    v1 = vertices[faces[:, 0]]  # (m,3)
    v2 = vertices[faces[:, 1]]  # (m,3)
    v3 = vertices[faces[:, 2]]  # (m,3)

    # Compute area for each face
    if vert_areas is None:
        vert_areas = compute_vertex_areas(vertices, faces, faces_areas=None)  # (n,)

    # Compute gradient direction non normalized by face areas/
    if grads is None:
        grad1_n = np.cross(normals, v3 - v2) / 2
        grad2_n = np.cross(normals, v1 - v3) / 2
        grad3_n = np.cross(normals, v2 - v1) / 2
    else:
        if face_areas is None:
            face_areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)  # (m,)
        grad1_n = face_areas[:, None] * grads[0]
        grad2_n = face_areas[:, None] * grads[1]
        grad3_n = face_areas[:, None] * grads[2]

    # Check if a single gradient field is given or multiple
    if f.ndim == 2:
        grad1 = np.einsum("ij,ij->i", grad1_n, f)  # (m,)
        grad2 = np.einsum("ij,ij->i", grad2_n, f)  # (m,)
        grad3 = np.einsum("ij,ij->i", grad3_n, f)  # (m,)

        div_val = np.zeros(n_vertices)  # (n,)
        np.add.at(div_val, faces[:, 0], grad1)
        np.add.at(div_val, faces[:, 1], grad2)
        np.add.at(div_val, faces[:, 2], grad3)

        div_val /= vert_areas  # (n,)

    else:
        grad1 = np.einsum("ij,ipj->ip", grad1_n, f)  # (m,p)
        grad2 = np.einsum("ij,ipj->ip", grad2_n, f)  # (m,p)
        grad3 = np.einsum("ij,ipj->ip", grad3_n, f)  # (m,p)

        div_val = np.zeros((n_vertices, f.shape[1]))  # (n,p)
        np.add.at(div_val, faces[:, 0], grad1)  # (n,p)
        np.add.at(div_val, faces[:, 1], grad2)  # (n,p)
        np.add.at(div_val, faces[:, 2], grad3)  # (n,p)

        div_val /= vert_areas[:, None]  # (n,p)

    return div_val


def build_dijkstra_graph(vertices, faces):
    """
    Build the sparse symmetric edge-weighted graph used for Dijkstra-based
    geodesic distance computation.

    Edges are weighted by their length.

    Parameters
    ----------
    vertices : (n, 3) np.ndarray
        Coordinates of the vertices.
    faces : (m, 3) np.ndarray
        Vertex indices for each face.

    Returns
    -------
    graph : (n, n) scipy.sparse.csc_matrix
        Symmetric sparse graph with edge-length weights.
    """
    N = vertices.shape[0]
    edges = edges_from_faces(faces)

    I = edges[:, 0]  # (p,)
    J = edges[:, 1]  # (p,)
    V = np.linalg.norm(vertices[J] - vertices[I], axis=1)  # (p,)

    In = np.concatenate([I, J])
    Jn = np.concatenate([J, I])
    Vn = np.concatenate([V, V])

    graph = sparse.coo_matrix((Vn, (In, Jn)), shape=(N, N)).tocsc()

    return graph


def geodesic_distmat_dijkstra(vertices, faces):
    """
    Compute the geodesic distance matrix using Dijkstra's algorithm.

    Not very efficient, but works.

    Parameters
    ----------
    vertices : (n, 3) np.ndarray
        Coordinates of the vertices.
    faces : (m, 3) np.ndarray
        Vertex indices for each face.

    Returns
    -------
    geod_dist : (n, n) np.ndarray
        Geodesic distance matrix.
    """
    graph = build_dijkstra_graph(vertices, faces)

    geod_dist = sparse.csgraph.dijkstra(graph)

    return geod_dist


def dijkstra_from(inds, graph):
    """
    Compute geodesic distances from one or several source vertices to all
    vertices, using Dijkstra's algorithm on a precomputed edge-weighted graph.

    Parameters
    ----------
    inds : int or (p,) np.ndarray
        Index (or indices) of the source vertex/vertices.
    graph : (n, n) scipy.sparse.csc_matrix
        Sparse graph as built by build_dijkstra_graph.

    Returns
    -------
    geod_dist : np.ndarray
        Geodesic distance from each source index to every vertex. Shape (n,) if
        inds is a single int, or (n, p) if inds is a sequence of length p.
    """
    single = np.issubdtype(type(inds), np.integer)
    indices = [inds] if single else list(inds)

    dist = sparse.csgraph.dijkstra(graph, indices=indices)  # (p,n)
    dist = dist.T  # (n,p)

    return dist.squeeze(axis=1) if single else dist


def geodesic_distmat_fast_marching(vertices, faces):
    """
    Compute the geodesic distance matrix using the Fast Marching algorithm.

    Parameters
    ----------
    vertices : (n, 3) np.ndarray
        Coordinates of the vertices.
    faces : (m, 3) np.ndarray
        Vertex indices for each face.

    Returns
    -------
    geod_dist : (n, n) np.ndarray
        Geodesic distance matrix.
    """

    n_vertices = vertices.shape[0]
    distmat = np.zeros((n_vertices, n_vertices))

    solver = pp3d.MeshFastMarchingDistanceSolver(vertices, faces)

    for vertind in range(n_vertices):
        distmat[vertind] = solver.compute_distance([[(vertind, [])]])

    return distmat


def heat_geodmat_robust(vertices, faces, verbose=False):
    """
    Compute the geodesic distance matrix using the Heat Method, with robust computation.

    Parameters
    ----------
    vertices : (n, 3) np.ndarray
        Coordinates of the vertices.
    faces : (m, 3) np.ndarray
        Vertex indices for each face.
    verbose : bool, optional
        Whether to display a progress bar.

    Returns
    -------
    distmat : (n, n) np.ndarray
        Geodesic distance matrix.
    """
    n_vertices = vertices.shape[0]
    distmat = np.zeros((n_vertices, n_vertices))

    solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)
    iterable = tqdm(range(n_vertices)) if verbose else range(n_vertices)

    for vertind in iterable:
        distmat[vertind] = np.maximum(solver.compute_distance(vertind), 0)

    return distmat


def heat_geodesic_from(
    inds,
    vertices,
    faces,
    normals,
    A,
    W=None,
    t=1e-3,
    face_areas=None,
    vert_areas=None,
    grads=None,
    solver_heat=None,
    solver_lap=None,
):
    """
    Compute geodesic distances between vertices of index inds and all other
    vertices using the Heat Method.

    Parameters
    ----------
    inds : int or (p,) np.ndarray
        Index of the source vertex (or vertices).
    vertices : (n, 3) np.ndarray
        Coordinates of the vertices.
    faces : (m, 3) np.ndarray
        Triangular faces defined by 3 vertex indices.
    normals : (m, 3) np.ndarray
        Per-face normals.
    A : (n, n) scipy.sparse
        Area matrix of the mesh, so that the Laplacian L = A^-1 W.
    W : (n, n) scipy.sparse, optional
        Stiffness matrix, so that the Laplacian L = A^-1 W. Optional if solvers
        are given.
    t : float, optional
        Time parameter for which to solve the heat equation.
    face_areas : (m,) np.ndarray, optional
        Per-face areas, for faster computation.
    vert_areas : (n,) np.ndarray, optional
        Per-vertex areas, for faster computation.
    grads : list, optional
        List of size 3, each giving per-face gradient directions (output of
        _get_grad_dir()).
    solver_heat : callable, optional
        Solver for (A + tW) x = b given b.
    solver_lap : callable, optional
        Solver for W x = b given b.

    Returns
    -------
    geod_dist : (n,) or (n, p) np.ndarray
        Geodesic distance for each vertex in inds.
    """
    n_vertices = vertices.shape[0]
    n_inds = len(inds) if type(inds) in [np.ndarray, list] else 1

    if face_areas is None:
        face_areas = compute_faces_areas(vertices, faces)
    if vert_areas is None:
        vert_areas = compute_vertex_areas(vertices, faces)

    if grads is None:
        grads = _get_grad_dir(vertices, faces, normals, face_areas=face_areas)  # (3,m,3)
    # grads = None

    # Define the dirac function  d on the given index. Not that the area normalization
    # will be simplified later on so this is actually A*d with A the area matrix
    delta = np.zeros((n_vertices, n_inds))  # (n,p)
    delta[(inds, np.arange(n_inds))] = 1  # works even if inds is an int
    delta = delta.squeeze()  # (n,) if n_inds is 1

    # Solve (I + tL)u = d. Actually (A + tW)u = Ad
    if solver_heat is not None:
        u = solver_heat(delta)
    else:
        u = sparse.linalg.spsolve(A + t * W, delta)  # (n,) or (n,p)

    # Compute and normalize the gradient of the solution
    g = grad_f(u, vertices, faces, normals, face_areas=face_areas, grads=grads)  # (m,3) or (m,p,3)
    h = -g / np.linalg.norm(g, axis=-1, keepdims=True)  # (m,3) or (m,p,3)

    # Solve L*phi = div(h). Actually W*phi = A*div(h)
    div_h = div_f(h, vertices, faces, normals, vert_areas=vert_areas, grads=grads)  # (n,) or (n,p)

    if solver_lap is not None:
        phi = solver_lap(A @ div_h)  # (n,) or (n,p)
    else:
        phi = sparse.linalg.spsolve(W, A @ div_h)  # (n,) or (n,p)

    # Phi is defined up to an additive constant. Minimum distance is 0
    phi -= np.min(phi, axis=0, keepdims=True)  # (n,) or (n,p)

    if n_inds > 1:
        phi[(inds, np.arange(n_inds))] = 0
    else:
        phi[inds] = 0

    return phi.squeeze()


def heat_geodmat(
    vertices,
    faces,
    normals,
    A,
    W,
    t=1e-3,
    face_areas=None,
    vert_areas=None,
    batch_size=None,
    verbose=False,
):
    """
    Compute geodesic distances between all pairs of vertices using the Heat Method.

    Parameters
    ----------
    vertices : (n, 3) np.ndarray
        Coordinates of the vertices.
    faces : (m, 3) np.ndarray
        Triangular faces defined by 3 vertex indices.
    normals : (m, 3) np.ndarray
        Per-face normals.
    A : (n, n) scipy.sparse
        Area matrix of the mesh, so that the Laplacian L = A^-1 W.
    W : (n, n) scipy.sparse
        Stiffness matrix, so that the Laplacian L = A^-1 W.
    t : float, optional
        Time parameter for which to solve the heat equation.
    face_areas : (m,) np.ndarray, optional
        Per-face areas, for faster computation.
    vert_areas : (n,) np.ndarray, optional
        Per-vertex areas, for faster computation.
    batch_size : int, optional
        Size of batches to use for computation. None means the full shape.
    verbose : bool, optional
        Whether to display a progress bar.

    Returns
    -------
    distmat : (n, n) np.ndarray
        Geodesic distance matrix.
    """
    n_vertices = vertices.shape[0]

    if face_areas is None:
        face_areas = compute_faces_areas(vertices, faces)
    if vert_areas is None:
        vert_areas = compute_vertex_areas(vertices, faces, face_areas)

    # Prefactor linear systems
    solver_heat = sparse.linalg.factorized(A.tocsc() + t * W.tocsc())
    solver_lap = sparse.linalg.factorized(W.tocsc())

    # Precompute gradient directions for each shapes
    grads = _get_grad_dir(vertices, faces, normals, face_areas=face_areas)  # (3,m,3)

    batch_size = n_vertices if batch_size is None else batch_size
    n_batches = n_vertices // batch_size + int(n_vertices % batch_size > 0)

    distmat = np.zeros((n_vertices, n_vertices))

    if verbose:
        ind_list = tqdm(range(n_batches))
    else:
        ind_list = range(n_batches)

    for batch_ind in ind_list:
        # Handle batch size of 1 (and possibly the last batcg of size 1)
        if batch_size > 1:
            batch = np.arange(batch_ind * batch_size, min(n_vertices, (1 + batch_ind) * batch_size))
        else:
            batch = batch_ind
        if batch_ind == n_batches - 1 and n_vertices % batch_size == 1:
            batch = batch[0]
        distmat[:, batch] = heat_geodesic_from(
            batch,
            vertices,
            faces,
            normals,
            A,
            W=None,
            t=t,
            face_areas=face_areas,
            vert_areas=vert_areas,
            grads=grads,
            solver_heat=solver_heat,
            solver_lap=solver_lap,
        )

    return distmat


def farthest_point_sampling(d, k, random_init=True, n_points=None, verbose=False):
    """
    Sample points using farthest point sampling.

    Uses either a complete distance matrix or a function giving distances to a
    given index i.

    Parameters
    ----------
    d : (n, n) np.ndarray or callable
        Either a distance matrix between points, or a function computing
        geodesic distance from a given index.
    k : int
        Number of points to sample.
    random_init : bool, optional
        Whether to sample the first point randomly or to take the one furthest
        away from all the others. Only used if d is a distance matrix.
    n_points : int, optional
        In the case where d is callable, specifies the size of the output.
    verbose : bool, optional
        Whether to display a progress bar.

    Returns
    -------
    fps : (k,) np.ndarray
        Array of indices of sampled points.
    """

    if callable(d):
        return farthest_point_sampling_call(d, k, n_points=n_points, verbose=verbose)

    else:
        if d.shape[0] != d.shape[1]:
            raise ValueError(f"D should be a n x n matrix not a {d.shape[0]} x {d.shape[1]}")

        return farthest_point_sampling_distmat(d, k, random_init=random_init, verbose=verbose)


def farthest_point_sampling_distmat(D, k, random_init=True, verbose=False):
    """
    Sample points using farthest point sampling from a complete distance matrix.

    Parameters
    ----------
    D : (n, n) np.ndarray
        Distance matrix between points.
    k : int
        Number of points to sample.
    random_init : bool, optional
        Whether to sample the first point randomly or to take the one furthest
        away from all the others.
    verbose : bool, optional
        Whether to display a progress bar.

    Returns
    -------
    fps : (k,) np.ndarray
        Array of indices of sampled points.
    """
    if random_init:
        rng = np.random.default_rng()
        inds = [rng.integers(D.shape[0]).item()]
    else:
        inds = [np.argmax(D.sum(1))]

    dists = D[inds[0]]

    iterable = range(k - 1) if not verbose else tqdm(range(k))
    for i in iterable:
        if i == k - 1:
            continue
        newid = np.argmax(dists)
        inds.append(newid)
        dists = np.minimum(dists, D[newid])

    return np.asarray(inds)


def farthest_point_sampling_call(d_func, k, n_points=None, verbose=False):
    """
    Sample points using farthest point sampling, initialized randomly.

    Parameters
    ----------
    d_func : callable
        For index i, d_func(i) is a (n_points,) array of geodesic distances to
        other points.
    k : int
        Number of points to sample.
    n_points : int, optional
        Number of points. If not specified, checks d_func(0).
    verbose : bool, optional
        Whether to display a progress bar.

    Returns
    -------
    fps : (k,) np.ndarray
        Array of indices of sampled points.
    """
    rng = np.random.default_rng()

    if n_points is None:
        n_points = d_func(0).shape

    else:
        assert n_points > 0

    inds = [rng.integers(n_points).item(0)]
    dists = d_func(inds[0])

    iterable = range(k - 1) if not verbose else tqdm(range(k))
    for i in iterable:
        if i == k - 1:
            continue
        newid = np.argmax(dists)
        inds.append(newid)
        dists = np.minimum(dists, d_func(newid))

    # print(inds)
    return np.asarray(inds)


def farthest_point_sampling_call_sub(
    d_func, k, sub_points, return_sub_inds=False, random_init=True, verbose=False
):
    """
    Sample points using farthest point sampling on a mesh, restricted to a set
    of samples.

    Parameters
    ----------
    d_func : callable
        For index i, d_func(i) is a (n_points,) array of geodesic distances to
        other points.
    k : int
        Number of points to sample.
    sub_points : (m,) np.ndarray
        Indices of vertices in the subsample.
    return_sub_inds : bool, optional
        Whether to return indices of the fps inside the subsample.
    random_init : bool, optional
        Whether to sample the first point randomly or to take the one furthest away.
    verbose : bool, optional
        Whether to display a progress bar.

    Returns
    -------
    fps : (k,) np.ndarray
        Array of indices of sampled points (as seen from the full set of points).
    fps_sub : (k,) np.ndarray, optional
        Returned only if return_sub_inds is True. Array of indices of sampled
        points (as seen from inside sub_points).
    """
    rng = np.random.default_rng()

    sub_points, first_index = np.unique(sub_points, return_index=True)

    n_points = sub_points.size

    start_subid = rng.integers(n_points)

    inds_sub = [first_index[start_subid]]
    inds = [sub_points[start_subid]]
    dists = d_func(inds[0])[sub_points]  # n_sub

    if not random_init:
        new_subid = np.argmax(dists)
        newid_sub = first_index[new_subid]
        newid = sub_points[new_subid]

        inds_sub[0] = newid_sub
        inds[0] = newid
        dists = d_func(inds[0])[sub_points]

    iterable = range(k - 1) if not verbose else tqdm(range(k))
    for i in iterable:
        if i == k - 1:
            continue

        new_subid = np.argmax(dists)

        newid_sub = first_index[new_subid]
        newid = sub_points[new_subid]

        inds_sub.append(newid_sub)
        inds.append(newid)
        dists = np.minimum(dists, d_func(newid)[sub_points])

    if return_sub_inds:
        return np.asarray(inds), np.asarray(inds_sub)
    return np.asarray(inds)


def get_orientation_op(grad_field, vertices, faces, normals, per_vert_area, rotated=False):
    """
    Compute the linear orientation operator associated to a gradient field grad(f).

    This operator computes g -> < grad(f) x grad(g), n> (given at each vertex)
    for any function g. In practice, we compute < n x grad(f), grad(g) > for
    simpler computation.

    Parameters
    ----------
    grad_field : (n_f, 3) np.ndarray
        Gradient field on the mesh.
    vertices : (n_v, 3) np.ndarray
        Coordinates of the vertices.
    faces : (n_f, 3) np.ndarray
        Vertex indices for each face.
    normals : (n_f, 3) np.ndarray
        Normal coordinates for each face.
    per_vert_area : (n_v,) np.ndarray
        Voronoi area for each vertex.
    rotated : bool, optional
        Whether the gradient field is already rotated by n x grad(f).

    Returns
    -------
    operator : (n_v, n_v) scipy.sparse.csc_matrix
        Orientation operator.
    """
    n_vertices = per_vert_area.shape[0]
    per_vert_area = np.asarray(per_vert_area)

    v1 = vertices[faces[:, 0]]  # (n_f,3)
    v2 = vertices[faces[:, 1]]  # (n_f,3)
    v3 = vertices[faces[:, 2]]  # (n_f,3)

    # Define (normalized) gradient directions for each barycentric coordinate on each face
    # Remove normalization since it will disappear later on after multiplication
    Jc1 = np.cross(normals, v3 - v2) / 2
    Jc2 = np.cross(normals, v1 - v3) / 2
    Jc3 = np.cross(normals, v2 - v1) / 2

    # Rotate the gradient field
    if rotated:
        rot_field = grad_field
    else:
        rot_field = np.cross(normals, grad_field)  # (n_f,3)

    I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    J = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])

    # Compute pairwise dot products between the gradient directions
    # and the gradient field
    Sij = (
        1
        / 3
        * np.concatenate(
            [
                np.einsum("ij,ij->i", Jc2, rot_field),
                np.einsum("ij,ij->i", Jc3, rot_field),
                np.einsum("ij,ij->i", Jc1, rot_field),
            ]
        )
    )

    Sji = (
        1
        / 3
        * np.concatenate(
            [
                np.einsum("ij,ij->i", Jc1, rot_field),
                np.einsum("ij,ij->i", Jc2, rot_field),
                np.einsum("ij,ij->i", Jc3, rot_field),
            ]
        )
    )

    In = np.concatenate([I, J, I, J])
    Jn = np.concatenate([J, I, I, J])
    Sn = np.concatenate([Sij, Sji, -Sij, -Sji])

    W = sparse.coo_matrix((Sn, (In, Jn)), shape=(n_vertices, n_vertices)).tocsc()
    inv_area = sparse.diags(1 / per_vert_area, shape=(n_vertices, n_vertices), format="csc")

    return inv_area @ W
