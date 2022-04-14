import time

import numpy as np
from tqdm.auto import tqdm
import scipy.sparse as sparse

from .nn_utils import knn_query


def project_pc_to_triangles(vert_emb, faces, points_emb, precompute_dmin=True, batch_size=None, n_jobs=1, verbose=False):
    """
    Project a pointcloud on a set of triangles in p-dimension. Projection is defined as
    barycentric coordinates on one of the triangle.
    Line i for the output has 3 non-zero values at indices j,k and l of the vertices of the
    triangle point i zas projected on.

    Parameters
    ----------------------------
    vert_emb        : (n1, p) coordinates of the mesh vertices
    faces           : (m1, 3) faces of the mesh defined as indices of vertices
    points_emb      : (n2, p) coordinates of the pointcloud
    precompute_dmin : Whether to precompute all the values of delta_min.
                      Faster but heavier in memory.
    batch_size      : If precompute_dmin is False, projects batches of points on the surface
    n_jobs          : number of parallel process for nearest neighbor precomputation


    Output
    ----------------------------
    precise_map : (n2,n1) - precise point to point map.
    """
    if batch_size is not None:
        batch_size = None if batch_size < 2 else batch_size
    n_points = points_emb.shape[0]
    n_vertices = vert_emb.shape[0]

    face_match = np.zeros(n_points, dtype=int)
    bary_coord = np.zeros((n_points, 3))

    if verbose:
        print('Precompute edge lengths...')
        start_time = time.time()
    lmax = compute_lmax(vert_emb, faces)  # (n1,)
    if verbose:
        print(f'\tDone in {time.time()-start_time:.2f}s')

    if verbose:
        print('Precompute nearest vertex...')
        start_time = time.time()
    Deltamin = compute_Deltamin(vert_emb, points_emb, n_jobs=n_jobs)  # (n2,)
    if verbose:
        print(f'\tDone in {time.time()-start_time:.2f}s')

    dmin = None
    if precompute_dmin:
        if verbose:
            print('Precompute nearest vertex in each face...')
            start_time = time.time()
        dmin = compute_all_dmin(vert_emb, faces, points_emb)  # (n_f1,n2)
        dmin_params = None
        if verbose:
            print(f'\tDone in {time.time()-start_time:.2f}s')

    else:
        vert_sqnorms = np.linalg.norm(vert_emb, axis=1)**2
        points_sqnorm = np.linalg.norm(points_emb, axis=1)**2
        dmin_params = {
                       'vert_sqnorms': vert_sqnorms,
                       'points_sqnorm': points_sqnorm
                       }

    # Iterate along all points
    if precompute_dmin or batch_size is None:
        iterable = range(n_points) if not verbose else tqdm(range(n_points))
        # for vertind in tqdm(range(n2)):
        for vertind in iterable:
            faceind, bary = project_to_mesh(vert_emb, faces, points_emb, vertind, lmax, Deltamin,
                                            dmin=dmin, dmin_params=dmin_params)
            face_match[vertind] = faceind
            bary_coord[vertind] = bary

    else:
        n_batches = n_points // batch_size + int((n_points % batch_size) > 0)
        iterable = range(n_batches) if not verbose else tqdm(range(n_batches))

        for batchind in iterable:
            batch_minmax = [batch_size*batchind, min(n_points, batch_size*(1+batchind))]
            # print(batch_minmax)
            dmin_batch = compute_all_dmin(vert_emb, faces, points_emb[batch_minmax[0]:batch_minmax[1]],
                                          vert_sqnorm=vert_sqnorms, points_sqnorm=points_sqnorm[batch_minmax[0]:batch_minmax[1]])

            batch_iterable = range(*batch_minmax)  #if not verbose else tqdm(range(*batch_minmax))
            for vertind in batch_iterable:
                batch_vertind = vertind - batch_minmax[0]
                faceind, bary = project_to_mesh(vert_emb, faces, points_emb[batch_minmax[0]:batch_minmax[1]],
                                                batch_vertind, lmax, Deltamin[batch_minmax[0]:batch_minmax[1]],
                                                dmin=dmin_batch, dmin_params=dmin_params)

                face_match[vertind] = faceind
                bary_coord[vertind] = bary

    return barycentric_to_precise(faces, face_match, bary_coord, n_vertices=n_vertices)


def compute_lmax(vert_emb, faces):
    """
    Computes the maximum edge length

    Parameters
    ----------------------------
    vert_emb      : (n1, p) coordinates of the mesh vertices
    faces         : (m1, 3) faces of the mesh defined as indices of vertices

    Output
    ----------------------------
    lmax : (m1,) maximum edge length
    """

    emb0 = vert_emb[faces[:,0]]  # (m1,k1)
    emb1 = vert_emb[faces[:,1]]  # (m1,k1)
    emb2 = vert_emb[faces[:,2]]  # (m1,k1)

    # Compute edge lengths in embedding space
    term1 = np.linalg.norm(emb1 - emb0, axis=1, keepdims=True)  # (m,1)
    term2 = np.linalg.norm(emb2 - emb1, axis=1, keepdims=True)  # (m,1)
    term3 = np.linalg.norm(emb0 - emb2, axis=1, keepdims=True)  # (m,1)

    # return np.max([np.max(term1), np.max(term2), np.max(term3)])
    return np.max(np.hstack([term1, term2, term3]), axis=1)  # (m1,)


def compute_Deltamin(vert_emb, points_emb, n_jobs=1):
    """
    For each point in the pointcloud gives the distance to the nearest vertex
    on the mesh.


    Compute Delta_min for each vertex in the target shape
        min_v2 ||A_{v2,*} - b||_2
    with notations from "Deblurring and Denoising of Maps between Shapes".

    Corresponds to nearest neighbor seach.

    Parameters
    ----------------------------
    vert_emb   : (n1, p) coordinates of the mesh vertices
    points_emb : (n2, p) coordinates of the pointcloud
    n_jobs     : number of paraller processes

    Output
    ----------------------------
    Delta_min : (n2,) Delta_min for each vertex on the target shape
    """

    # tree = KDTree(mesh1.eigenvectors[:,:k1])  # Tree on (n1,k1)
    # dists,_ = tree.query(mesh2.eigenvectors[:,:k2] @ FM, k=1)  # query on (n2,k1)

    dists, _ = knn_query(vert_emb, points_emb, k=1,
                         return_distance=True, n_jobs=n_jobs)

    return dists.flatten()  # (n2,)


def mycdist(X, Y, sqnormX=None, sqnormY=None, squared=False):
    """
    Compute pairwise euclidean distance between two collections of vectors in a k-dimensional space

    Parameters
    --------------
    X       : (n1, k) first collection
    Y       : (n2, k) second collection or (k,) if single point
    squared : bool - whether to compute the squared euclidean distance

    Output
    --------------
    distmat : (n1, n2) or (n2,) distance matrix
    """

    if sqnormX is None:
        sqnormX = np.linalg.norm(X, axis=1)**2

    if sqnormY is None:
        if Y.ndim == 2:
            sqnormY = np.linalg.norm(Y, axis=1)**2
        else:
            sqnormY = np.linalg.norm(Y)

    distmat = X @ Y.T
    distmat *= -2

    if Y.ndim == 2:
        distmat += sqnormX[:, None]
        distmat += sqnormY[None, :]
    else:
        distmat += sqnormX
        distmat += sqnormY

    np.maximum(distmat, 0, out=distmat)

    if not squared:
        np.sqrt(distmat, out=distmat)

    return distmat


def compute_dmin(vert_emb, faces, points_emb, vertind, vert_sqnorms=None, points_sqnorm=None):
    """
    Given a vertex in the pointcloud and each face on the surface, gives the minimum distance
    to between the vertex and each of the 3 points of the triangle.

    For a given face on the source shape and vertex on the target shape:
        delta_min = min_{i=1..3} ||A_{c_i,*} - b||_2
    with notations from "Deblurring and Denoising of Maps between Shapes".

    Parameters
    ----------------------------
    vert_emb      : (n1, p) coordinates of the mesh vertices
    faces         : (m1, 3) faces of the mesh defined as indices of vertices
    points_emb    : (n2, p) coordinates of the pointcloud
    vertind       : index of the vertex for which to compute dmin
    vert_sqnorm   : (n1,) squared norm of each vertex
    points_sqnorm : (n2,) squared norm of each point
    Output

    ----------------------------
    delta_min : (m1,n2) delta_min for each face on the source shape.
    """
    if vert_sqnorms is None:
        vert_sqnorms = np.linalg.norm(vert_emb, axis=1)**2

    emb0 = vert_emb[faces[:,0]]  # (m1,k1)
    emb1 = vert_emb[faces[:,1]]  # (m1,k1)
    emb2 = vert_emb[faces[:,2]]  # (m1,k1)
    b = points_emb[vertind]  # (k1,)

    b_sqnorm = np.linalg.norm(b)**2 if points_sqnorm is None else points_sqnorm[vertind]

    dmin = mycdist(emb0, b, sqnormX=vert_sqnorms[faces[:,0]], sqnormY=b_sqnorm, squared=True)
    np.minimum(dmin, mycdist(emb1, b, sqnormX=vert_sqnorms[faces[:,1]], sqnormY=b_sqnorm, squared=True), out=dmin)
    np.minimum(dmin, mycdist(emb2, b, sqnormX=vert_sqnorms[faces[:,2]], sqnormY=b_sqnorm, squared=True), out=dmin)
    np.sqrt(dmin, out=dmin)

    return dmin


def compute_all_dmin(vert_emb, faces, points_emb, vert_sqnorm=None, points_sqnorm=None):
    """
    For each vertex in the pointcloud and each face on the surface, gives the minimum distance
    to between the vertex and each of the 3 points of the triangle.

    For a given face on the source shape and vertex on the target shape:
        delta_min = min_{i=1..3} ||A_{c_i,*} - b||_2
    with notations from "Deblurring and Denoising of Maps between Shapes".

    Parameters
    ----------------------------
    vert_emb      : (n1, p) coordinates of the mesh vertices
    faces         : (m1, 3) faces of the mesh defined as indices of vertices
    points_emb    : (n2, p) coordinates of the pointcloud
    vert_sqnorm   : (n1,) squared norm of each vertex
    points_sqnorm : (n2,) squared norm of each point
    Output

    ----------------------------
    delta_min : (m1,n2) delta_min for each face on the source shape.
    """
    emb0 = vert_emb[faces[:, 0]]  # (m1,k1)
    emb1 = vert_emb[faces[:, 1]]  # (m1,k1)
    emb2 = vert_emb[faces[:, 2]]  # (m1,k1)

    if points_sqnorm is None:
        points_sqnorm = np.linalg.norm(points_emb, axis=1)**2
    if vert_sqnorm is None:
        vert_sqnorm = np.linalg.norm(vert_emb, axis=1)**2

    distmat = mycdist(emb0, points_emb, sqnormX=vert_sqnorm[faces[:, 0]], sqnormY=points_sqnorm, squared=True)
    np.minimum(distmat, mycdist(emb1, points_emb, sqnormX=vert_sqnorm[faces[:, 1]], sqnormY=points_sqnorm, squared=True), out=distmat)
    np.minimum(distmat, mycdist(emb2, points_emb, sqnormX=vert_sqnorm[faces[:, 2]], sqnormY=points_sqnorm, squared=True), out=distmat)
    np.sqrt(distmat, out=distmat)
    return distmat  # (m1,n2)


def project_to_mesh(vert_emb, faces, points_emb, vertind, lmax, Deltamin, dmin=None, dmin_params=None):
    """
    Project a pointcloud on a p-dimensional triangle mesh

    Parameters
    ----------------------------
    vert_emb    : (n1, p) coordinates of the mesh vertices
    faces       : (m1, 3) faces of the mesh defined as indices of vertices
    points_emb  : (n2, p) coordinates of the pointcloud
    vertind     : int - index of the vertex to project
    lmax        : (m1,) value of lmax (max edge length for each face)
    Deltamin    : (n2,) value of Deltamin (distance to nearest vertex)
    dmin        : (m1,n2) - optional - values of dmin (distance to the nearest vertex of each face
                  for each vertex). Can be computed on the fly
    dmin_params : dict - optional - if dmin is None, stores 'vert_sqnorm' a (n1,) array of squared norms
                  of vertices embeddings, and 'points_sqnorm' a (n2,) array of squared norms
                  of points embeddings. Helps speed up computation of dmin

    Output
    -----------------------------
    min_faceind : int - index of the face on which the vertex is projected
    min_bary    : (3,) - barycentric coordinates on the chosen face
    """
    dmin_params = dict() if dmin_params is None else dmin_params
    # Obtain deltamin
    if dmin is None:
        deltamin = compute_dmin(vert_emb, faces, points_emb, vertind, **dmin_params)  # (m1,)
    else:
        deltamin = dmin[:, vertind]  # (m1,)

    query_faceinds = np.where(deltamin - lmax < Deltamin[vertind])[0]  # (p)

    # Projection can be done on multiple triangles
    query_triangles = vert_emb[faces[query_faceinds]]  # (p, 3, k1)
    query_point = points_emb[vertind]
    # query_triangles = mesh1.eigenvectors[mesh1.facelist[query_faceinds], :k1]  # (p, 3,k1)
    # query_point = FM.T @ mesh2.eigenvectors[vertind,:k2]  # (k1,)

    if len(query_faceinds) == 1:
        min_dist, proj, min_bary = pointTriangleDistance(query_triangles.squeeze(), query_point, return_bary=True)
        return query_faceinds, min_bary

    dists, proj, bary_coords = point_to_triangles_projection(query_triangles, query_point, return_bary=True)

    min_ind = dists.argmin()

    min_faceind = query_faceinds[min_ind]
    min_bary = bary_coords[min_ind]

    return min_faceind, min_bary


def barycentric_to_precise(faces, face_match, bary_coord, n_vertices=None):
    """
    Transforms set of barycentric coordinates into a precise map

    Parameters
    ----------------------------
    faces      : (m,3) - Set of faces defined by index of vertices.
    face_match : (n2,) - indices of the face assigned to each point
    bary_coord : (n2,3) - barycentric coordinates of each point within the face
    n_vertices : int - number of vertices in the target mesh (on which faces are defined)

    Output
    ----------------------------
    precise_map : (n2,n1) - precise point to point map
    """
    if n_vertices is None:
        n_vertices = 1 + faces.max()

    n_points = face_match.shape[0]

    v0 = faces[face_match,0]  # (n2,)
    v1 = faces[face_match,1]  # (n2,)
    v2 = faces[face_match,2]  # (n2,)

    I = np.arange(n_points)  # (n2)

    In = np.concatenate([I, I, I])
    Jn = np.concatenate([v0, v1, v2])
    Sn = np.concatenate([bary_coord[:,0], bary_coord[:,1], bary_coord[:,2]])

    precise_map = sparse.csr_matrix((Sn, (In,Jn)), shape=(n_points, n_vertices))
    return precise_map


def point_to_triangles_projection(triangles, point, return_bary=False):
    r"""

    This functions projects a p-dimensional point on each of the given p-dimensional triangle.

    This is a parallelized version of pointTriangleDistance.

    All operations are parallelized, which makes the code quite hard to read. For an easier take,
    follow the code in the function below (not written by me) for projection on a single triangle.

    The algorithm is based on [1]

    The algorithm first find for each triangle in which of the following region the projected point
    lies, then solves for each region.


           ^t
     \     |
      \reg2|
       \   |
        \  |
         \ |
          \|
           *P2
           |\
           | \
     reg3  |  \ reg1
           |   \
           |reg0\
           |     \
           |      \ P1
    -------*-------*------->s
           |P0      \
     reg4  | reg5    \ reg6

     Most notations come from :
        [1] "David Eberly, 'Distance Between Point and Triangle in 3D',
    Geometric Tools, LLC, (1999)"

    Parameters
    -------------------------------
    triangles   : (m,3,p) set of m p-dimensional triangles
    point       : (p,) coordinates of the point
    return_bary : Whether to return barycentric coordinates inside each triangle

    Output
    -------------------------------
    final_dists : (m,) distance from the point to each of the triangle
    projections : (m,p) coordinates of the projected point
    bary_coords : (m,3) barycentric coordinates of the projection within each triangle
    """

    if point.ndim == 2:
        point = point.squeeze()  # (p,)

    # rewrite triangles in normal form base + axis
    bases = triangles[:, 0]  # (m,p)
    axis1 = triangles[:, 1] - bases  # (m,p)
    axis2 = triangles[:, 2] - bases  # (m,p)

    diff = bases - point[None, :]  # (m,p)

    #  Precompute quantities with notations from [1]

    a = np.einsum('ij,ij->i', axis1, axis1)  # (m,)
    b = np.einsum('ij,ij->i', axis1, axis2)  # (m,)
    c = np.einsum('ij,ij->i', axis2, axis2)  # (m,)
    d = np.einsum('ij,ij->i', axis1, diff)  # (m,)
    e = np.einsum('ij,ij->i', axis2, diff)  # (m,)
    f = np.einsum('ij,ij->i', diff, diff)  # (m,)

    det = a * c - b**2  # (m,)
    s = b * e - c * d  # (m,)
    t = b * d - a * e  # (m,)

    # Array of barycentric coordinates (s,t) and distances
    final_s = np.zeros(s.size)  # (m,)
    final_t = np.zeros(t.size)  # (m,)
    final_dists = np.zeros(t.size)  # (m,)

    # Find for which triangles which zone the point belongs to

    # s + t <= det
    test1 = (s+t <= det)  # (m,) with (m1) True values
    inds_0345 = np.where(test1)[0]  # (m1)
    inds_126 = np.where(~test1)[0]  # (m-m1)

    # s < 0 | s + t <= det
    test11 = s[inds_0345] < 0  # (m1,) with (m11) True values
    inds_34 = inds_0345[test11]  # (m11)
    inds_05 = inds_0345[~test11]  # (m1-m11)

    # t < 0 | (s + t <= det) and (s < 0)
    test111 = t[inds_34] < 0  # (m11) with (m111) True values
    inds_4 = inds_34[test111]  # (m111)
    inds_3 = inds_34[~test111]  # (m11 - m111)

    # t < 0 | s + t <= det and (s >= 0)
    test12 = t[inds_05] < 0  # (m-m11) with (m12) True values
    inds_5 = inds_05[test12]  # (m12;)
    inds_0 = inds_05[~test12]  # (m-m11-m12,)

    # s < 0 | s + t > det
    test21 = s[inds_126] < 0  # (m-m1) with (m21) True values
    inds_2 = inds_126[test21]  # (m21,)
    inds_16 = inds_126[~test21]  # (m-m1-m21)

    # t < 0 | (s + t > det) and (s > 0)
    test22 = t[inds_16] < 0  # (m-m1-m21) with (m22) True values
    inds_6 = inds_16[test22]  # (m22,)
    inds_1 = inds_16[~test22]  # (m-m1-m21-m22)

    # DEAL REGION BY REGION (in parallel within each)

    # REGION 4
    if len(inds_4) > 0:
        # print('Case 4',inds_4)
        test4_1 = d[inds_4] < 0
        inds4_1 = inds_4[test4_1]
        inds4_2 = inds_4[~test4_1]

        # FIRST PART - SUBDIVIDE IN 2
        final_t[inds4_1] = 0  # Useless already done

        test4_11 = (-d[inds4_1] >= a[inds4_1])
        inds4_11 = inds4_1[test4_11]
        inds4_12 = inds4_1[~test4_11]

        final_s[inds4_11] = 1.
        final_dists[inds4_11] = a[inds4_11] + 2.0 * d[inds4_11] + f[inds4_11]

        final_s[inds4_12] = -d[inds4_12] / a[inds4_12]
        final_dists[inds4_12] = d[inds4_12] * s[inds4_12] + f[inds4_12]

        # SECOND PART - SUBDIVIDE IN 2
        final_s[inds4_2] = 0  # Useless already done

        test4_21 = (e[inds4_2] >= 0)
        inds4_21 = inds4_2[test4_21]
        inds4_22 = inds4_2[~test4_21]

        final_t[inds4_21] = 0
        final_dists[inds4_21] = f[inds4_21]

        # SECOND PART OF SECOND PART - SUBDIVIDE IN 2
        test4_221 = (-e[inds4_22] >= c[inds4_22])
        inds4_221 = inds4_22[test4_221]
        inds4_222 = inds4_22[~test4_221]

        final_t[inds4_221] = 1
        final_dists[inds4_221] = c[inds4_221] + 2.0 * e[inds4_221] + f[inds4_221]

        final_t[inds4_222] = -e[inds4_222] / c[inds4_222]
        final_dists[inds4_222] = e[inds4_222] * t[inds4_222] + f[inds4_222]

    if len(inds_3) > 0:
        # print('Case 3', inds_3)
        final_s[inds_3] = 0

        test3_1 = e[inds_3] >= 0
        inds3_1 = inds_3[test3_1]
        inds3_2 = inds_3[~test3_1]

        final_t[inds3_1] = 0
        final_dists[inds3_1] = f[inds3_1]

        # SECOND PART - SUBDIVIDE IN 2

        test3_21 = (-e[inds3_2] >= c[inds3_2])
        inds3_21 = inds3_2[test3_21]
        inds3_22 = inds3_2[~test3_21]

        # print(inds3_21, inds3_22)

        final_t[inds3_21] = 1
        final_dists[inds3_21] = c[inds3_21] + 2.0 * e[inds3_21] + f[inds3_21]

        final_t[inds3_22] = -e[inds3_22] / c[inds3_22]
        final_dists[inds3_22] = e[inds3_22] * final_t[inds3_22] + f[inds3_22]  # -e*t ????

    if len(inds_5) > 0:
        # print('Case 5', inds_5)
        final_t[inds_5] = 0

        test5_1 = d[inds_5] >= 0
        inds5_1 = inds_5[test5_1]
        inds5_2 = inds_5[~test5_1]

        final_s[inds5_1] = 0
        final_dists[inds5_1] = f[inds5_1]

        test5_21 = (-d[inds5_2] >= a[inds5_2])
        inds5_21 = inds5_2[test5_21]
        inds5_22 = inds5_2[~test5_21]

        final_s[inds5_21] = 1
        final_dists[inds5_21] = a[inds5_21] + 2.0 * d[inds5_21] + f[inds5_21]

        final_s[inds5_22] = -d[inds5_22] / a[inds5_22]
        final_dists[inds5_22] = d[inds5_22] * final_s[inds5_22] + f[inds5_22]

    if len(inds_0) > 0:
        # print('Case 0', inds_0)
        invDet = 1.0 / det[inds_0]
        final_s[inds_0] = s[inds_0] * invDet
        final_t[inds_0] = t[inds_0] * invDet
        final_dists[inds_0] = final_s[inds_0] * (a[inds_0] * final_s[inds_0] + b[inds_0] * final_t[inds_0] + 2.0 * d[inds_0]) +\
                              final_t[inds_0] * (b[inds_0] * final_s[inds_0] + c[inds_0] * final_t[inds_0] + 2.0 * e[inds_0]) + f[inds_0]

    if len(inds_2) > 0:
        # print('Case 2', inds_2)

        tmp0 = b[inds_2] + d[inds_2]
        tmp1 = c[inds_2] + e[inds_2]

        test2_1 = tmp1 > tmp0
        inds2_1 = inds_2[test2_1]
        inds2_2 = inds_2[~test2_1]

        numer = tmp1[test2_1] - tmp0[test2_1]
        denom = a[inds2_1] - 2.0 * b[inds2_1] + c[inds2_1]

        test2_11 = (numer >= denom)
        inds2_11 = inds2_1[test2_11]
        inds2_12 = inds2_1[~test2_11]

        final_s[inds2_11] = 1
        final_t[inds2_11] = 0
        final_dists[inds2_11] = a[inds2_11] + 2.0 * d[inds2_11] + f[inds2_11]

        final_s[inds2_12] = numer[~test2_11] / denom[~test2_11]
        final_t[inds2_12] = 1 - final_s[inds2_12]
        final_dists[inds2_12] = final_s[inds2_12] * (a[inds2_12] * final_s[inds2_12] + b[inds2_12] * final_t[inds2_12] + 2 * d[inds2_12]) +\
                                final_t[inds2_12] * (b[inds2_12] * final_s[inds2_12] + c[inds2_12] * final_t[inds2_12] + 2 * e[inds2_12]) + f[inds2_12]


        final_s[inds2_2] = 0.

        test2_21 = (tmp1[~test2_1] <= 0.)
        inds2_21 = inds2_2[test2_21]
        inds2_22 = inds2_2[~test2_21]

        final_t[inds2_21] = 1
        final_dists[inds2_21] = c[inds2_21] + 2.0 * e[inds2_21] + f[inds2_21]

        test2_221 = (e[inds2_22] >= 0.)
        inds2_221 = inds2_22[test2_221]
        inds2_222 = inds2_22[~test2_221]

        final_t[inds2_221] = 0.
        final_dists[inds2_221] = f[inds2_221]

        final_t[inds2_222] = -e[inds2_222] / c[inds2_222]
        final_dists[inds2_222] = e[inds2_222] * final_t[inds2_222] + f[inds2_222]

    if len(inds_6) > 0:
        # print('Case 6', inds_6)
        tmp0 = b[inds_6] + e[inds_6]
        tmp1 = a[inds_6] + d[inds_6]

        test6_1 = tmp1 > tmp0
        inds6_1 = inds_6[test6_1]
        inds6_2 = inds_6[~test6_1]

        numer = tmp1[test6_1] - tmp0[test6_1]
        denom = a[inds6_1] - 2.0 * b[inds6_1] + c[inds6_1]

        test6_11 = (numer >= denom)
        inds6_11 = inds6_1[test6_11]
        inds6_12 = inds6_1[~test6_11]

        final_t[inds6_11] = 1
        final_s[inds6_11] = 0
        final_dists[inds6_11] = c[inds6_11] + 2.0 * e[inds6_11] + f[inds6_11]

        final_t[inds6_12] = numer[~test6_11] / denom[~test6_11]
        final_s[inds6_12] = 1 - final_t[inds6_12]
        final_dists[inds6_12] = final_s[inds6_12] * (a[inds6_12] * final_s[inds6_12] + b[inds6_12] * final_t[inds6_12] + 2.0 * d[inds6_12]) + \
                                final_t[inds6_12] * (b[inds6_12] * final_s[inds6_12] + c[inds6_12] * final_t[inds6_12] + 2.0 * e[inds6_12]) + f[inds6_12]


        final_t[inds6_2] = 0.

        test6_21 = (tmp1[~test6_1] <= 0.)
        inds6_21 = inds6_2[test6_21]
        inds6_22 = inds6_2[~test6_21]

        final_s[inds6_21] = 1
        final_dists[inds6_21] = a[inds6_21] + 2.0 * d[inds6_21] + f[inds6_21]

        test6_221 = (d[inds6_22] >= 0.)
        inds6_221 = inds6_22[test6_221]
        inds6_222 = inds6_22[~test6_221]

        final_s[inds6_221] = 0.
        final_dists[inds6_221] = f[inds6_221]

        final_s[inds6_222] = -d[inds6_222] / a[inds6_222]
        final_dists[inds6_222] = d[inds6_222] * final_s[inds6_222] + f[inds6_222]

    if len(inds_1) > 0:
        # print('Case 1', inds_1)
        numer = c[inds_1] + e[inds_1] - b[inds_1] - d[inds_1]

        test1_1 = numer <= 0
        inds1_1 = inds_1[test1_1]
        inds1_2 = inds_1[~test1_1]

        final_s[inds1_1] = 0
        final_t[inds1_1] = 1
        final_dists[inds1_1] = c[inds1_1] + 2.0 * e[inds1_1] + f[inds1_1]

        denom = a[inds1_2] - 2.0 * b[inds1_2] + c[inds1_2]

        test1_21 = (numer[~test1_1] >= denom)
        # print(denom, numer, numer[~test1_1], test1_21, inds1_2)
        inds1_21 = inds1_2[test1_21]
        inds1_22 = inds1_2[~test1_21]

        final_s[inds1_21] = 1
        final_t[inds1_21] = 0
        final_dists[inds1_21] = a[inds1_21] + 2.0 * d[inds1_21] + f[inds1_21]

        final_s[inds1_22] = numer[~test1_1][~test1_21]/denom[~test1_21]
        final_t[inds1_22] = 1 - final_s[inds1_22]
        final_dists[inds1_22] = final_s[inds1_22] * (a[inds1_22] * final_s[inds1_22] + b[inds1_22] * final_t[inds1_22] + 2.0 * d[inds1_22]) +\
                                final_t[inds1_22] * (b[inds1_22] * final_s[inds1_22] + c[inds1_22] * final_t[inds1_22] + 2.0 * e[inds1_22]) + f[inds1_22]

    final_dists[final_dists < 0] = 0
    final_dists = np.sqrt(final_dists)

    projections = bases + final_s[:,None]*axis1 + final_t[:,None]*axis2
    if return_bary:
        bary_coords = np.concatenate([1-final_s[:,None]-final_t[:,None], final_s[:,None], final_t[:,None]],axis=1)
        return final_dists, projections, bary_coords

    return final_dists, projections


def pointTriangleDistance(TRI, P, return_bary=False):
    r"""
    Computes distance between a point and a triangle in a p-dimensional space

    Based on the implementation in (modified to return barycentric coordinates of the projection):
    https://gist.github.com/joshuashaffer/99d58e4ccbd37ca5d96e

    DESCRIPTION
      Calculate the distance of a given point P from a triangle TRI.
      Point P is a row vector of the form 1x3. The triangle is a matrix
      formed by three rows of points TRI = [P1;P2;P3] each of size 1x3.
      dist = pointTriangleDistance(TRI,P) returns the distance of the point P
      to the triangle TRI.
      [dist,PP0] = pointTriangleDistance(TRI,P) additionally returns the
      closest point PP0 to P on the triangle TRI.

    The algorithm is based on
    "David Eberly, 'Distance Between Point and Triangle in 3D',
    Geometric Tools, LLC, (1999)"
    http:\\www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf

           ^t
     \     |
      \reg2|
       \   |
        \  |
         \ |
          \|
           *P2
           |\
           | \
     reg3  |  \ reg1
           |   \
           |reg0\
           |     \
           |      \ P1
    -------*-------*------->s
           |P0      \
     reg4  | reg5    \ reg6



    Parameters
    -------------------------------
    TRI         : (3,p) a p-dimensional triangle
    P           : (p,) coordinates of the point
    return_bary : Whether to return barycentric coordinates inside each triangle

    Output
    -------------------------------
    dist        : float - distance from the point to each of the triangle
    projection  : (p,) coordinates of the projected point
    bary_coords : (3,) barycentric coordinates of the projection within each triangle
    """
    # rewrite triangle in normal form
    B = TRI[0, :]
    E0 = TRI[1, :] - B
    # E0 = E0/sqrt(sum(E0.^2)); %normalize vector
    E1 = TRI[2, :] - B
    # E1 = E1/sqrt(sum(E1.^2)); %normalize vector
    D = B - P
    a = np.dot(E0, E0)
    b = np.dot(E0, E1)
    c = np.dot(E1, E1)
    d = np.dot(E0, D)
    e = np.dot(E1, D)
    f = np.dot(D, D)

    # print "{0} {1} {2} ".format(B,E1,E0)
    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    # Terible tree of conditionals to determine in which region of the diagram
    # shown above the projection of the point into the triangle-plane lies.
    if (s + t) <= det:
        if s < 0.0:
            if t < 0.0:
                # region4
                if d < 0:
                    t = 0.0
                    if -d >= a:
                        s = 1.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
                else:
                    s = 0.0
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        if -e >= c:
                            t = 1.0
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = -e / c
                            sqrdistance = e * t + f

                            # of region 4
            else:
                # region 3
                s = 0
                if e >= 0:
                    t = 0
                    sqrdistance = f
                else:
                    if -e >= c:
                        t = 1
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 3
        else:
            if t < 0:
                # region 5
                t = 0
                if d >= 0:
                    s = 0
                    sqrdistance = f
                else:
                    if -d >= a:
                        s = 1
                        sqrdistance = a + 2.0 * d + f;  # GF 20101013 fixed typo d*s ->2*d
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
            else:
                # region 0
                invDet = 1.0 / det
                s = s * invDet
                t = t * invDet
                sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
    else:
        if s < 0.0:
            # region 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:  # minimum on edge s+t=1
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                if numer >= denom:
                    s = 1.0
                    t = 0.0
                    sqrdistance = a + 2.0 * d + f;  # GF 20101014 fixed typo 2*b -> 2*d
                else:
                    s = numer / denom
                    t = 1 - s
                    sqrdistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 * e) + f

            else:  # minimum on edge s=0
                s = 0.0
                if tmp1 <= 0.0:
                    t = 1
                    sqrdistance = c + 2.0 * e + f
                else:
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 2
        else:
            if t < 0.0:
                # region6
                tmp0 = b + e
                tmp1 = a + d
                if tmp1 > tmp0:
                    numer = tmp1 - tmp0
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        t = 1.0
                        s = 0
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = numer / denom
                        s = 1 - t
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

                else:
                    t = 0.0
                    if tmp1 <= 0.0:
                        s = 1
                        sqrdistance = a + 2.0 * d + f
                    else:
                        if d >= 0.0:
                            s = 0.0
                            sqrdistance = f
                        else:
                            s = -d / a
                            sqrdistance = d * s + f
            else:
                # region 1
                numer = c + e - b - d
                if numer <= 0:
                    s = 0.0
                    t = 1.0
                    sqrdistance = c + 2.0 * e + f
                else:
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        s = 1.0
                        t = 0.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = numer / denom
                        t = 1 - s
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

    # account for numerical round-off error
    if sqrdistance < 0:
        sqrdistance = 0

    dist = np.sqrt(sqrdistance)

    PP0 = B + s * E0 + t * E1

    if not return_bary:
        return dist, PP0

    else:
        return dist,PP0,np.array([1-s-t, s, t])
