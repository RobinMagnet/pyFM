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

    I = np.concatenate([faces[:,0], faces[:,1], faces[:,2]])
    J = np.zeros_like(I)
    V = np.concatenate([faces_areas, faces_areas, faces_areas])/3

    # Get the (n,) array of vertex areas
    vertex_areas = np.array(sparse.coo_matrix((V, (I, J)), shape=(N, 1)).todense()).flatten()

    return vertex_areas


def grad_f(f, vertices, faces, normals, face_areas=None, use_sym=False):
    """
    Compute the gradient of a function on a mesh

    Parameters
    --------------------------
    f          : (n,) function value on each vertex
    vertices   : (n,3) coordinates of vertices
    faces      : (m,3) indices of vertices for each face
    normals    : (m,3) normals coordinate for each face
    face_area : (m,) - Optional, array of per-face area, for faster computation
    use_sym    : bool - If true, uses the (slower but) symmetric expression
                 of the gradient

    Output
    --------------------------
    gradient : (m,3) gradient of f on the mesh
    """
    v1 = vertices[faces[:,0]]  # (m,3)
    v2 = vertices[faces[:,1]]  # (m,3)
    v3 = vertices[faces[:,2]]  # (m,3)

    f1 = f[faces[:,0]]  # (m,)
    f2 = f[faces[:,1]]  # (m,)
    f3 = f[faces[:,2]]  # (m,)

    # Compute area for each face
    if face_areas is None:
        face_areas = 0.5 * np.linalg.norm(np.cross(v2-v1,v3-v1),axis=1)  # (m)

    if not use_sym:
        grad2 = np.cross(normals, v1-v3)/(2*face_areas[:,None])  # (m,3)
        grad3 = np.cross(normals, v2-v1)/(2*face_areas[:,None])  # (m,3)

        gradient = (f2-f1)[:,None] * grad2 + (f3-f1)[:,None] * grad3

    else:
        grad1 = np.cross(normals, v3-v2)/(2*face_areas[:,None])  # (m,3)
        grad2 = np.cross(normals, v1-v3)/(2*face_areas[:,None])  # (m,3)
        grad3 = np.cross(normals, v2-v1)/(2*face_areas[:,None])  # (m,3)

        gradient = f1[:,None] * grad1 + f2[:,None] * grad2 + f3[:,None] * grad3

    return gradient


def div_f(f, vertices, faces, normals, vert_areas=None):
    """
    Compute the divergence of a vector field on a mesh

    Parameters
    --------------------------
    f          : (m,3) vector field on each face
    vertices   : (n,3) coordinates of vertices
    faces      : (m,3) indices of vertices for each face
    normals    : (m,3) normals coordinate for each face
    faces_area : (m,) - Optional, array of per-face area, for faster computation

    Output
    --------------------------
    divergence : (n,) divergence of f on the mesh
    """
    n_vertices = vertices.shape[0]

    v1 = vertices[faces[:,0]]  # (m,3)
    v2 = vertices[faces[:,1]]  # (m,3)
    v3 = vertices[faces[:,2]]  # (m,3)

    # Compute area for each face
    if vert_areas is None:
        vert_areas = compute_vertex_areas(vertices, faces, faces_areas=None)  # (n,)

    grad1 = np.einsum('ij,ij->i', np.cross(normals, v3 - v2) / 2, f)  # (m,)
    grad2 = np.einsum('ij,ij->i', np.cross(normals, v1 - v3) / 2, f)  # (m,)
    grad3 = np.einsum('ij,ij->i', np.cross(normals, v2 - v1) / 2, f)  # (m,)

    I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])  # (3*m)
    J = np.zeros_like(I)
    V = np.concatenate([grad1, grad2, grad3])

    div_val = sparse.coo_matrix((V, (I, J)), shape=(n_vertices, 1)).todense()

    return np.asarray(div_val).flatten() / vert_areas


def get_orientation_op(grad_field, vertices, faces, normals, per_vert_area, rotated=False):
    """
    Compute the linear orientation operator associated to a gradient field grad(f).

    This operator computes g -> < grad(f) x grad(g), n> (given at each vertex) for any function g
    In practice, we compute < n x grad(f), grad(g) > for simpler computation.

    Parameters
    --------------------------------
    grad_field    : (n_f,3) gradient field on the mesh
    vertices      : (n_v,3) coordinates of vertices
    faces         : (n_f,3) indices of vertices for each face
    normals       : (n_f,3) normals coordinate for each face
    per_vert_area : (n_v,) voronoi area for each vertex
    rotated       : bool - whether gradient field is already rotated by n x grad(f)

    Output
    --------------------------
    operator : (n_v,n_v) orientation operator.
    """
    n_vertices = per_vert_area.shape[0]
    per_vert_area = np.asarray(per_vert_area)

    v1 = vertices[faces[:,0]]  # (n_f,3)
    v2 = vertices[faces[:,1]]  # (n_f,3)
    v3 = vertices[faces[:,2]]  # (n_f,3)

    # Define (normalized) gradient directions for each barycentric coordinate on each face
    # Remove normalization since it will disappear later on after multiplcation
    Jc1 = np.cross(normals, v3-v2)/2
    Jc2 = np.cross(normals, v1-v3)/2
    Jc3 = np.cross(normals, v2-v1)/2

    # Rotate the gradient field
    if rotated:
        rot_field = grad_field
    else:
        rot_field = np.cross(normals,grad_field)  # (n_f,3)

    I = np.concatenate([faces[:,0], faces[:,1], faces[:,2]])
    J = np.concatenate([faces[:,1], faces[:,2], faces[:,0]])

    # Compute pairwise dot products between the gradient directions
    # and the gradient field
    Sij = 1/3*np.concatenate([np.einsum('ij,ij->i', Jc2, rot_field),
                              np.einsum('ij,ij->i', Jc3, rot_field),
                              np.einsum('ij,ij->i', Jc1, rot_field)])

    Sji = 1/3*np.concatenate([np.einsum('ij,ij->i', Jc1, rot_field),
                              np.einsum('ij,ij->i', Jc2, rot_field),
                              np.einsum('ij,ij->i', Jc3, rot_field)])

    In = np.concatenate([I, J, I, J])
    Jn = np.concatenate([J, I, I, J])
    Sn = np.concatenate([Sij, Sji, -Sij, -Sji])

    W = sparse.coo_matrix((Sn, (In, Jn)), shape=(n_vertices, n_vertices)).tocsc()
    inv_area = sparse.diags(1/per_vert_area, shape=(n_vertices, n_vertices), format='csc')

    return inv_area @ W


def geodesic_distmat(vertices, faces):
    """
    Compute geodesic distance matrix using Dijkstra algorithm.
    """
    N = vertices.shape[0]
    edges = edges_from_faces(faces)

    I = edges[:,0]  # (p,)
    J = edges[:,1]  # (p,)
    V = np.linalg.norm(vertices[J] - vertices[I], axis=1)  # (p,)

    In = np.concatenate([I,J])
    Jn = np.concatenate([J,I])
    Vn = np.concatenate([V,V])

    graph = sparse.coo_matrix((Vn, (In, Jn)), shape=(N, N)).tocsc()

    geod_dist = sparse.csgraph.dijkstra(graph)

    return geod_dist


def heat_geodesic_from(i, vertices, faces, normals, A, W=None, t=1e-3, face_areas=None, vert_areas=None, solver_heat=None, solver_lap=None):
    """
    Computes geodesic distances between vertex i and all other vertices
    using the Heat Method

    Parameters
    -------------------------
    i           : int of array of ints - index of the source vertex (or vertices)
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

    # Define the dirac function  d on the given index. Not that the area normalization
    # will be simplified later on so this is actually A*d with A the area matrix
    delta = np.zeros(n_vertices)
    delta[i] = 1

    # Solve (I + tL)u = d. Actually (A + tW)u = Ad
    if solver_heat is not None:
        u = solver_heat(delta)
    else:
        u = sparse.linalg.spsolve(A + t*W, delta)  # (n,)

    # Compute and normalize the gradient of the solution
    g = grad_f(u, vertices, faces, normals, face_areas=face_areas)  # (m,3)
    h = - g / np.linalg.norm(g, axis=1, keepdims=True)  # (m,3)

    # Solve L*phi = div(h). Actually W*phi = A*div(h)
    div_h = div_f(h, vertices, faces, normals, vert_areas=vert_areas)  # (n,1)

    if solver_lap is not None:
        phi = solver_lap(A@div_h)
    else:
        phi = sparse.linalg.spsolve(W, A @ div_h)  # (n,1)

    phi -= np.min(phi)  # phi is defined up to an additive constant. Minimum distance is 0
    phi[i] = 0

    return phi.flatten()


def heat_geodmat(vertices, faces, normals, A, W, t=1e-3, face_areas=None, vert_areas=None, verbose=False):
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
    face_area  : (m,) - Optional, array of per-face area, for faster computation
    vert_areas : (n,) - Optional, array of per-vertex area, for faster computation

    """
    n_vertices = vertices.shape[0]

    if face_areas is None:
        face_areas = compute_faces_areas(vertices, faces)
    if vert_areas is None:
        vert_areas = compute_vertex_areas(vertices, faces, face_areas)

    solver_heat = sparse.linalg.factorized(A.tocsc() + t * W)
    solver_lap = sparse.linalg.factorized(W)

    distmat = np.zeros((n_vertices,n_vertices))

    if verbose:
        ind_list = tqdm(range(n_vertices))
    else:
        ind_list = range(n_vertices)
    for index in ind_list:
        distmat[index] = heat_geodesic_from(index, vertices, faces, normals, A, W=None, t=t,
                                            face_areas=face_areas, vert_areas=vert_areas,
                                            solver_heat=solver_heat, solver_lap=solver_lap)

    return distmat


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

    return edges


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
        inds = [np.random.randint(D.shape[0])]
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

    if n_points is None:
        n_points = d_func(0).shape

    else:
        assert n_points > 0

    inds = [np.random.randint(n_points)]
    dists = d_func(inds[0])

    iterable = range(k-1) if not verbose else tqdm(range(k-1))
    for _ in iterable:
        newid = np.argmax(dists)
        inds.append(newid)
        dists = np.minimum(dists, d_func(newid))

    return np.asarray(inds)
