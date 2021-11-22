import numpy as np
import scipy.sparse as sparse

from tqdm import tqdm


def compute_normals(vertices, faces):
    """
    Compute normals of a triangular mesh

    Parameters
    -----------------------------
    vertices : (n,3) array of vertices coordinates
    faces    : (m,3) array of vertex indices defining faces

    Output
    -----------------------------
    normals : (m,3) array of normalized per-face normals
    """
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]

    normals = np.cross(v2-v1, v3-v1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    return normals


def compute_faces_areas(vertices, faces):
    """
    Compute per-face areas of a triangular mesh

    Parameters
    -----------------------------
    vertices : (n,3) array of vertices coordinates
    faces    : (m,3) array of vertex indices defining faces

    Output
    -----------------------------
    faces_areas : (m,) array of per-face areas
    """

    v1 = vertices[faces[:,0]]  # (m,3)
    v2 = vertices[faces[:,1]]  # (m,3)
    v3 = vertices[faces[:,2]]  # (m,3)
    faces_areas = 0.5 * np.linalg.norm(np.cross(v2-v1,v3-v1),axis=1)  # (m,)

    return faces_areas


def compute_vertex_areas(vertices, faces, faces_areas=None):
    """
    Compute per-vertex areas of a triangular mesh.
    Area of a vertex, approximated as one third of the sum of the area
    of its adjacent triangles

    Parameters
    -----------------------------
    vertices : (n,3) array of vertices coordinates
    faces    : (m,3) array of vertex indices defining faces

    Output
    -----------------------------
    vert_areas : (n,) array of per-vertex areas
    """
    N = vertices.shape[0]

    if faces_areas is None:
        faces_areas = compute_faces_areas(vertices,faces)  # (m,)


    # This is way faster than using np.add.at for some reason...

    I = np.concatenate([faces[:,0], faces[:,1], faces[:,2]])
    J = np.zeros_like(I)
    V = np.concatenate([faces_areas, faces_areas, faces_areas])/3

    # Get the (n,) array of vertex areas
    vertex_areas = np.array(sparse.coo_matrix((V, (I, J)), shape=(N, 1)).todense()).flatten()

    return vertex_areas


def neigh_faces(faces):
    """
    Return the indices of neighbor faces for each vertex. This supposed all vertices appear in
    the face list.

    Parameters
    --------------------
    faces : (m,3) list of faces

    Output
    --------------------
    neighbors : (n,) list of indices of neighbor faces for each vertex
    """
    n_vertices = 1+faces.max()

    neighbors = [[] for i in range(n_vertices)]

    for face_ind, (i,j,k) in enumerate(faces):
        neighbors[i].append(face_ind)
        neighbors[j].append(face_ind)
        neighbors[k].append(face_ind)

    neighbors = [np.unique(x) for x in neighbors]

    return neighbors


def per_vertex_normal(vertices, faces, face_normals=None):
    """
    Computes per-vertex normals as an average of adjacent face normals.

    Parameters
    --------------------
    vertices : (n,3) coordinates of vertices
    faces : (m,3) faces defined as indices of vertices
    face_normals : (m,3) per-face normals (optional)

    Output
    --------------------
    vert_normals : (n,3) array of per-vertex normals
    """
    if face_normals is None:
        face_normals = compute_normals(vertices, faces)  # (m,3)
    vert2faces = neigh_faces(faces)  # (n, p_i)

    vert_normals = np.array([face_normals[vert2faces[i]].mean(0) for i in range(vertices.shape[0])])
    vert_normals /= np.linalg.norm(vert_normals, axis=1, keepdims=True)

    return vert_normals


def edges_from_faces(faces):
    """
    Compute all edges in the mesh

    Parameters
    --------------------------------
    faces : (m,3) array defining faces with vertex indices

    Output
    --------------------------
    edges : (p,2) array of all edges defined by vertex indices
            with no particular order
    """
    # Number of verties
    N = 1 + np.max(faces)

    # Use a sparse matrix and find non-zero elements
    I = np.concatenate([faces[:,0], faces[:,1], faces[:,2]])
    J = np.concatenate([faces[:,1], faces[:,2], faces[:,0]])
    V = np.ones_like(I)
    M = sparse.coo_matrix((V, (I, J)), shape=(N, N))

    inds1,inds2 = M.nonzero()  # (p,), (p,)
    edges = np.hstack([inds1[:,None], inds2[:,None]])

    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def geodesic_distmat(vertices, faces):
    """
    Compute geodesic distance matrix using Dijkstra algorithm.
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

    geod_dist = sparse.csgraph.dijkstra(graph)

    return geod_dist


def heat_geodesic_from(inds, vertices, faces, normals, A, W=None, t=1e-3, face_areas=None,
                       vert_areas=None, grads=None, solver_heat=None, solver_lap=None):
    """
    Computes geodesic distances between vertices of index inds and all other vertices
    using the Heat Method

    Parameters
    -------------------------
    inds        : int or (p,) array of ints - index of the source vertex (or vertices)
    vertices    : (n,3) vertices coordinates
    faces       : (m,3) triangular faces defined by 3 vertices index
    normals     : (m,3) per-face normals
    A           : (n,n) sparse - area matrix of the mesh so that the laplacian L = A^-1 W
    W           : (n,n) sparse - stiffness matrix so that the laplacian L = A^-1 W.
                  Optional if solvers are given !
    t           : float - time parameter for which to solve the heat equation
    face_area   : (m,) - Optional, array of per-face area, for faster computation
    vert_areas  : (n,) - Optional, array of per-vertex area, for faster computation
    solver_heat : callable -Optional, solver for (A + tW)x = b given b
    solver_lap  : callable -Optional, solver for Wx = b given b

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
    delta[(inds,np.arange(n_inds))] = 1  # works even if inds is an int
    delta = delta.squeeze()  # (n,) if n_inds is 1

    # Solve (I + tL)u = d. Actually (A + tW)u = Ad
    if solver_heat is not None:
        u = solver_heat(delta)
    else:
        u = sparse.linalg.spsolve(A + t*W, delta)  # (n,) or (n,p)

    # Compute and normalize the gradient of the solution
    g = grad_f(u, vertices, faces, normals, face_areas=face_areas, grads=grads)  # (m,3) or (m,p,3)
    h = - g / np.linalg.norm(g, axis=-1, keepdims=True)  # (m,3) or (m,p,3)

    # Solve L*phi = div(h). Actually W*phi = A*div(h)
    div_h = div_f(h, vertices, faces, normals, vert_areas=vert_areas, grads=grads)  # (n,) or (n,p)

    if solver_lap is not None:
        phi = solver_lap(A@div_h)  # (n,) or (n,p)
    else:
        phi = sparse.linalg.spsolve(W, A @ div_h)  # (n,) or (n,p)

    # Phi is defined up to an additive constant. Minimum distance is 0
    phi -= np.min(phi, axis=0, keepdims=True)  # (n,) or (n,p)

    if n_inds > 1:
        phi[(inds, np.arange(n_inds))] = 0
    else:
        phi[inds] = 0

    return phi.squeeze()


def heat_geodmat(vertices, faces, normals, A, W, t=1e-3, face_areas=None, vert_areas=None,
                 batch_size=None, verbose=False):
    """
    Computes geodesic distances between all pairs of vertices using the Heat Method

    Parameters
    -------------------------
    vertices   : (n,3) vertices coordinates
    faces      : (m,3) triangular faces defined by 3 vertices index
    normals    : (m,3) per-face normals
    A          : (n,n) sparse - area matrix of the mesh so that the laplacian L = A^-1 W
    W          : (n,n) sparse - stiffness matrix so that the laplacian L = A^-1 W
    t          : float - time parameter for which to solve the heat equation
    face_areas : (m,) - Optional, array of per-face area, for faster computation
    vert_areas : (n,) - Optional, array of per-vertex area, for faster computation
    batch_size : int - size of batches to use for computation. None means full shape

    """
    n_vertices = vertices.shape[0]

    if face_areas is None:
        face_areas = compute_faces_areas(vertices, faces)
    if vert_areas is None:
        vert_areas = compute_vertex_areas(vertices, faces, face_areas)

    # Prefactor linear systems
    solver_heat = sparse.linalg.factorized(A.tocsc() + t * W)
    solver_lap = sparse.linalg.factorized(W)

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
            batch = np.arange(batch_ind*batch_size, min(n_vertices, (1 + batch_ind) * batch_size))
        else:
            batch = batch_ind
        if batch_ind == n_batches - 1 and n_vertices % batch_size == 1:
            batch = batch[0]
        distmat[:,batch] = heat_geodesic_from(batch, vertices, faces, normals, A, W=None, t=t,
                                              face_areas=face_areas, vert_areas=vert_areas, grads=grads,
                                              solver_heat=solver_heat, solver_lap=solver_lap)

    return distmat


def farthest_point_sampling(d, k, random_init=True, n_points=None):
    """
    Samples points using farthest point sampling using either a complete distance matrix
    or a function giving distances to a given index i

    Parameters
    -------------------------
    d           : (n,n) array or callable - Either a distance matrix between points or
                  a function computing geodesic distance from a given index.
    k           : int - number of points to sample
    random_init : Whether to sample the first point randomly or to take the furthest away
                  from all the other ones. Only used if d is a distance matrix
    n_points    : In the case where d is callable, specifies the size of the output

    Output
    --------------------------
    fps : (k,) array of indices of sampled points
    """

    if callable(d):
        return farthest_point_sampling_call(d, k, n_points=n_points)

    else:
        if d.shape[0] != d.shape[1]:
            raise ValueError(f"D should be a n x n matrix not a {d.shape[0]} x {d.shape[1]}")

        return farthest_point_sampling_distmat(d, k, random_init=random_init)


def farthest_point_sampling_distmat(D, k, random_init=True):
    """
    Samples points using farthest point sampling using a complete distance matrix

    Parameters
    -------------------------
    D           : (n,n) distance matrix between points
    k           : int - number of points to sample
    random_init : Whether to sample the first point randomly or to
                  take the furthest away from all the other ones

    Output
    --------------------------
    fps : (k,) array of indices of sampled points
    """
    if random_init:
        rng = np.random.default_rng()
        inds = [rng.integers(D.shape[0])]
    else:
        inds = [np.argmax(D.sum(1))]

    dists = D[inds[0]]

    for _ in range(k-1):
        newid = np.argmax(dists)
        inds.append(newid)
        dists = np.minimum(dists, D[newid])

    return np.asarray(inds)


def farthest_point_sampling_call(d_func, k, n_points=None, verbose=False):
    """
    Samples points using farthest point sampling, initialized randomly

    Parameters
    -------------------------
    d_func   : callable - for index i, d_func(i) is a (n_points,) array of geodesic distance to
               other points
    k        : int - number of points to sample
    n_points : Number of points. If not specified, checks d_func(0)

    Output
    --------------------------
    fps : (k,) array of indices of sampled points
    """
    rng = np.random.default_rng()

    if n_points is None:
        n_points = d_func(0).shape

    else:
        assert n_points > 0

    inds = [rng.integers(n_points)]
    dists = d_func(inds[0])

    iterable = range(k-1) if not verbose else tqdm(range(k))
    for i in iterable:
        if i == k-1:
            continue
        newid = np.argmax(dists)
        inds.append(newid)
        dists = np.minimum(dists, d_func(newid))

    return np.asarray(inds)
