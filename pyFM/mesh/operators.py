import numpy as np
import scipy.sparse as sparse

from . import geometry as geom


def _get_grad_dir(vertices, faces, normals, face_areas=None):
    """
    Compute the gradient directions for each face using linear interpolationof the hat
    basis on each face.

    Parameters
    --------------------------
    vertices   : (n,3) coordinates of vertices
    faces      : (m,3) indices of vertices for each face
    normals    : (m,3) normals coordinate for each face
    face_areas : (m,) - Optional, array of per-face area, for faster computation

    Output
    --------------------------
    grads : (3,m,3) array of per-face gradients.
    """

    v1 = vertices[faces[:, 0]]  # (m,3)
    v2 = vertices[faces[:, 1]]  # (m,3)
    v3 = vertices[faces[:, 2]]  # (m,3)

    if face_areas is None:
        face_areas = 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1), axis=1)  # (m,)

    grad1 = np.cross(normals, v3-v2)/(2*face_areas[:, None])  # (m,3)
    grad2 = np.cross(normals, v1-v3)/(2*face_areas[:, None])  # (m,3)
    grad3 = np.cross(normals, v2-v1)/(2*face_areas[:, None])  # (m,3)

    return np.asarray([grad1, grad2, grad3])


def grad_f(f, vertices, faces, normals, face_areas=None, use_sym=False, grads=None):
    """
    Compute the gradient of one or multiple functions on a mesh

    Parameters
    --------------------------
    f          : (n,p) or (n,) functions value on each vertex
    vertices   : (n,3) coordinates of vertices
    faces      : (m,3) indices of vertices for each face
    normals    : (m,3) normals coordinate for each face
    face_areas : (m,) - Optional, array of per-face area, for faster computation
    use_sym    : bool - If true, uses the (slower but) symmetric expression
                 of the gradient
    grads      : iterable of size 3 containing arrays of size (m,3) giving gradient directions
                 for all faces (see function `_get_grad_dir`).
    Output
    --------------------------
    gradient : (m,p,3) or (n,3) gradient of f on the mesh
    """

    if grads is not None:
        grad1, grad2, grad3 = grads[0], grads[1], grads[2]

    v1 = vertices[faces[:,0]]  # (m,3)
    v2 = vertices[faces[:,1]]  # (m,3)
    v3 = vertices[faces[:,2]]  # (m,3)

    f1 = f[faces[:,0]]  # (m,p) or (m,)
    f2 = f[faces[:,1]]  # (m,p) or (m,)
    f3 = f[faces[:,2]]  # (m,p) or (m,)

    # Compute area for each face
    if face_areas is None:
        face_areas = 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1), axis=1)  # (m,)

    # Check whether to use a symmetric computation fo the gradient (slower but more stable)
    if not use_sym:
        # Compute gradient directions
        if grads is None:
            grad2 = np.cross(normals, v1-v3)/(2*face_areas[:,None])  # (m,3)
            grad3 = np.cross(normals, v2-v1)/(2*face_areas[:,None])  # (m,3)

        # One or multiple functions
        if f.ndim == 1:
            gradient = (f2-f1)[:,None] * grad2 + (f3-f1)[:,None] * grad3  # (m,3)
        else:
            gradient = (f2-f1)[:,:,None] * grad2[:,None,:] + (f3-f1)[:,:,None] * grad3[:,None,:]  # (m,3)

    else:
        # Compute gradient directions
        if grads is None:
            grad1 = np.cross(normals, v3-v2)/(2*face_areas[:,None])  # (m,3)
            grad2 = np.cross(normals, v1-v3)/(2*face_areas[:,None])  # (m,3)
            grad3 = np.cross(normals, v2-v1)/(2*face_areas[:,None])  # (m,3)

        # One or multiple functions
        if f.ndim == 1:
            gradient = f1[:,None] * grad1 + f2[:,None] * grad2 + f3[:,None] * grad3  # (m,p,3)
        else:
            gradient = f1[:,:,None] * grad1[:,None,:] + f2[:,:,None] * grad2[:,None,:] + f3[:,:,None] * grad3[:,None,:]  # (m,p,3)

    return gradient


def div_f(f, vertices, faces, normals, vert_areas=None, grads=None, face_areas=None):
    """
    Compute the vertex-wise divergence of a vector field on a mesh

    Parameters
    --------------------------
    f          : (m,3) or (n,m,3) vector field(s) on each face
    vertices   : (n,3) coordinates of vertices
    faces      : (m,3) indices of vertices for each face
    normals    : (m,3) normals coordinate for each face
    vert_areas : (m,) - Optional, array of per-vertex area, for faster computation
    grads      : iterable of size 3 containing arrays of size (m,3) giving gradient directions
                 for all faces
    face_areas : (m,) - Optional, array of per-face area, for faster computation
                  ONLY USED IF grads is given

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
        vert_areas = geom.compute_vertex_areas(vertices, faces, faces_areas=None)  # (n,)

    # Compute gradient direction not normalized by face areas (normalization would disappear later)
    if grads is None:
        grad1_n = np.cross(normals, v3 - v2) / 2
        grad2_n = np.cross(normals, v1 - v3) / 2
        grad3_n = np.cross(normals, v2 - v1) / 2
    else:
        if face_areas is None:
            face_areas = 0.5 * np.linalg.norm(np.cross(v2-v1,v3-v1),axis=1)  # (m,)
        grad1_n = face_areas[:,None] * grads[0]
        grad2_n = face_areas[:,None] * grads[1]
        grad3_n = face_areas[:,None] * grads[2]

    # Check if a single gradient field is given (ndim == 2) or multiple (ndim == 3)
    if f.ndim == 2:
        grad1 = np.einsum('ij,ij->i', grad1_n, f)  # (m,)
        grad2 = np.einsum('ij,ij->i', grad2_n, f)  # (m,)
        grad3 = np.einsum('ij,ij->i', grad3_n, f)  # (m,)
        I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])  # (3*m)
        J = np.zeros_like(I)
        V = np.concatenate([grad1, grad2, grad3])

        div_val = sparse.coo_matrix((V, (I, J)), shape=(n_vertices, 1)).todense()
        div_val = np.asarray(div_val).flatten() / vert_areas  # (n,)

    else:
        grad1 = np.einsum('ij,ipj->ip', grad1_n, f)  # (m,p)
        grad2 = np.einsum('ij,ipj->ip', grad2_n, f)  # (m,p)
        grad3 = np.einsum('ij,ipj->ip', grad3_n, f)  # (m,p)

        div_val = np.zeros((n_vertices,f.shape[1]))  # (n,p)
        np.add.at(div_val, faces[:, 0], grad1)  # (n,p)
        np.add.at(div_val, faces[:, 1], grad2)  # (n,p)
        np.add.at(div_val, faces[:, 2], grad3)  # (n,p)

        div_val /= vert_areas[:, None]  # (n,p)

    return div_val


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
