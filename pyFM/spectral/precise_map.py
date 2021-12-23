"""
This code implements the procedure described in
"Deblurring and Denoising of Maps between Shapes", by Danielle Ezuz and Mirela Ben-Chen.

Notations of variables will follow those from the paper, except that our functional
maps go from mesh1 to mesh2 while their directions are reversed in the paper.
"""
import numpy as np
import scipy.spatial.distance
from tqdm import tqdm
import scipy.sparse as sparse
# from sklearn.neighbors import KDTree
from .nn_utils import knn_query


def precise_map(mesh1, mesh2, FM, precompute_dmin=True, n_jobs=1):
    """
    Parameters
    ----------------------------
    mesh1           : Source mesh (for the functional map)
    mesh2           : Target mesh (for the functional map)
    FM              : (k2,k1) Functional map
    precompute_dmin : Whether to precompute all the values of delta_min.
                      Faster but heavier in memory

    Output
    ----------------------------
    precise_map : (n2,n1) - precise point to point map
    """
    n2 = mesh2.n_vertices
    k2,k1 = FM.shape

    face_match = np.zeros(n2, dtype=int)
    bary_coord = np.zeros((n2, 3))

    print('Compute lmax')
    lmax = compute_lmax(mesh1, k1)  # (n1,)

    print('Compute Deltamin')
    Deltamin = compute_Deltamin(mesh1, mesh2, FM, n_jobs=n_jobs)  # (n2,)

    dmin = None
    if precompute_dmin:
        print('Precompute dmin')
        dmin = compute_dmin(mesh1, mesh2, FM, vertind=None)  # (n_f1,n2)

    # Iterate along all points
    for vertind in tqdm(range(n2)):
        faceind, bary = project_to_mesh(mesh1,mesh2,FM,vertind,lmax,Deltamin,dmin=dmin)
        face_match[vertind] = faceind
        bary_coord[vertind] = bary

    return barycentric_to_precise(mesh1, mesh2, face_match, bary_coord)


def compute_lmax(mesh1, k1):
    """
    Return for each face on the source shape,
        max_{i,j=1..3} ||A_{c_i,*} - A_{c_j,*}||_2

    with notations from the map deblurring paper

    Parameters
    -------------------------
    mesh1 : Source mesh (for the functional map)
    k1    : Dimension to use

    Output
    --------------------------
    lmax  : (m1,) maximum embedding distance of edges for each face.
    """
    ev0 = mesh1.eigenvectors[mesh1.facelist[:,0], :k1]  # (m1,k1)
    ev1 = mesh1.eigenvectors[mesh1.facelist[:,1], :k1]  # (m1,k1)
    ev2 = mesh1.eigenvectors[mesh1.facelist[:,2], :k1]  # (m1,k1)

    # Compute edge lengths in embedding space
    term1 = np.linalg.norm(ev1 - ev0, axis=1, keepdims=True)  # (m,1)
    term2 = np.linalg.norm(ev2 - ev1, axis=1, keepdims=True)  # (m,1)
    term3 = np.linalg.norm(ev0 - ev2, axis=1, keepdims=True)  # (m,1)

    return np.max(np.hstack([term1, term2, term3]), axis=1)  # (m1,)


def compute_Deltamin(mesh1, mesh2, FM, n_jobs=1):
    """
    Compute Delta_min for each vertex in the target shape
        min_v2 ||A_{v2,*} - b||_2
    with notations from the map deblurring paper

    Corresponds to nearest neighbor seach.

    Parameters
    ----------------------------
    mesh1 : Source mesh (for the functional map)
    mesh2 : Target mesh (for the functional map)
    FM    : (k2,k1) Functional map

    Output
    ----------------------------
    Delta_min : (n2,) Delta_min for each vertex on the target shape
    """
    k2, k1 = FM.shape

    # tree = KDTree(mesh1.eigenvectors[:,:k1])  # Tree on (n1,k1)
    # dists,_ = tree.query(mesh2.eigenvectors[:,:k2] @ FM, k=1)  # query on (n2,k1)

    dists, _ = knn_query(mesh1.eigenvectors[:,:k1], mesh2.eigenvectors[:,:k2]@FM, k=1,
                         return_distance=True, n_jobs=n_jobs)

    return dists.flatten()  # (n2,)


def compute_dmin(mesh1, mesh2, FM, vertind=None):
    """
    Compute for each face on the source shape, either delta_min for a given vertex on the target
    shape or the values for all the vertices.
    For a given face on the source shape and vertex on the target shape:
        delta_min = min_{i=1..3} ||A_{c_i,*} - b||_2
    with notations from the map deblurring paper.

    Parameters
    ----------------------------
    mesh1   : Source mesh (for the functional map)
    mesh2   : Target mesh (for the functional map)
    FM      : (k2,k1) Functional map
    vertind : int - vertex index for which to compute dmin for all faces.
              If not specified, values for all possible vertices are
              computed

    Output
    ----------------------------
    delta_min : (m1,) or (m1,n2) delta_min for each face on the source shape.
    """
    k2,k1 = FM.shape
    ev0 = mesh1.eigenvectors[mesh1.facelist[:,0], :k1]  # (m1,k1)
    ev1 = mesh1.eigenvectors[mesh1.facelist[:,1], :k1]  # (m1,k1)
    ev2 = mesh1.eigenvectors[mesh1.facelist[:,2], :k1]  # (m1,k1)

    if vertind is None:
        distmat = scipy.spatial.distance.cdist(ev0, mesh2.eigenvectors[:,:k2] @ FM)  # (m1, n2)
        distmat = np.minimum(distmat, scipy.spatial.distance.cdist(ev1,mesh2.eigenvectors[:,:k2]@FM))
        distmat = np.minimum(distmat, scipy.spatial.distance.cdist(ev2,mesh2.eigenvectors[:,:k2]@FM))
        return distmat  # (m1,n2)

    else:
        b = mesh2.eigenvectors[vertind,None,:k2]@FM
        term1 = np.linalg.norm(ev1 - b, axis=1, keepdims=True)  # (m1,1)
        term2 = np.linalg.norm(ev2 - b, axis=1, keepdims=True)  # (m1,1)
        term3 = np.linalg.norm(ev0 - b, axis=1, keepdims=True)  # (m1,1)

        return np.min(np.hstack([term1,term2,term3]),axis=1)  # (m1,)


def project_to_mesh(mesh1, mesh2, FM, vertind, lmax, Deltamin, dmin=None):
    """
    Project a vertex of the target mesh to a face on the first mesh using its embedding.

    Parameters
    ----------------------------
    mesh1    : Source mesh (for the functional map)
    mesh2    : Target mesh (for the functional map)
    FM       : (k2,k1) Functional map
    vertind  : int - index of the vertex to project
    lmax     : (m1,) value of lmax
    Deltamin : (n2,) value of Deltamin
    dmin     : (m1,n2) values of dmin. If not specified, the required value
                value of deltamin is computed.

    Output
    -----------------------------
    min_faceind : int - index of the face on which the vertex is projected
    min_bary    : (3,) - barycentric coordinates on the chosen face
    """
    k2,k1 = FM.shape

    # Obtain deltamin
    if dmin is None:
        deltamin = compute_dmin(mesh1,mesh2,FM,vertind=vertind)  # (m1,)
    else:
        deltamin = dmin[:,vertind]  # (m1,)

    query_faceinds = np.where(deltamin - lmax < Deltamin[vertind])[0]  # (p)

    # Projection can be done on multiple triangles
    query_triangles = mesh1.eigenvectors[mesh1.facelist[query_faceinds], :k1]  # (p, 3,k1)
    query_point = FM.T @ mesh2.eigenvectors[vertind,:k2]  # (k1,)

    if len(query_faceinds) == 1:
        min_dist, proj, min_bary = pointTriangleDistance(query_triangles.squeeze(), query_point, return_bary=True)
        return query_faceinds, min_bary

    dists, proj, bary_coords = point_to_triangles_projection(query_triangles, query_point, return_bary=True)

    min_ind = dists.argmin()

    min_faceind = query_faceinds[min_ind]
    min_bary = bary_coords[min_ind]

    return min_faceind, min_bary


def barycentric_to_precise(mesh1, mesh2, face_match, bary_coord):
    """
    Transforms set of barycentric coordinates into a precise map

    Parameters
    ----------------------------
    mesh1      : Source mesh (for the functional map)
    mesh2      : Target mesh (for the functional map)
    face_match : (n2,) - indices of the face assigned to each vertex
    bary_coord : (n2,3) - barycentric coordinates for each face

    Output
    ----------------------------
    precise_map : (n2,n1) - precise point to point map
    """

    v0 = mesh1.facelist[face_match,0]  # (n2,)
    v1 = mesh1.facelist[face_match,1]  # (n2,)
    v2 = mesh1.facelist[face_match,2]  # (n2,)

    I = np.arange(mesh2.n_vertices)  # (n2)

    In = np.concatenate([I, I, I])
    Jn = np.concatenate([v0, v1, v2])
    Sn = np.concatenate([bary_coord[:,0], bary_coord[:,1], bary_coord[:,2]])

    precise_map = sparse.coo_matrix((Sn, (In,Jn)), shape=(mesh2.n_vertices, mesh1.n_vertices)).tocsc()
    return precise_map


def point_to_triangles_projection(triangles, point, return_bary=False):
    r"""

    This functions projects a p-dimensional point on each of the given p-dimensional triangle.
    All operations are parallelized, which makes the code quite hard to read. For an easier take,
    follow the code in the function below (not written by me) for projection on a single triangle.

    The first estimates for each triangle in which of the following region the point lies, then
    solves for each region.


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
        point = point.squeeze()  # (1,p)

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
    function [dist,PP0] = pointTriangleDistance(TRI,P)
    calculate distance between a point and a triangle in 3D
    SYNTAX
      dist = pointTriangleDistance(TRI,P)
      [dist,PP0] = pointTriangleDistance(TRI,P)

    DESCRIPTION
      Calculate the distance of a given point P from a triangle TRI.
      Point P is a row vector of the form 1x3. The triangle is a matrix
      formed by three rows of points TRI = [P1;P2;P3] each of size 1x3.
      dist = pointTriangleDistance(TRI,P) returns the distance of the point P
      to the triangle TRI.
      [dist,PP0] = pointTriangleDistance(TRI,P) additionally returns the
      closest point PP0 to P on the triangle TRI.

    Author: Gwolyn Fischer
    Release: 1.0
    Release date: 09/02/02
    Release: 1.1 Fixed Bug because of normalization
    Release: 1.2 Fixed Bug because of typo in region 5 20101013
    Release: 1.3 Fixed Bug because of typo in region 2 20101014

    Possible extention could be a version tailored not to return the distance
    and additionally the closest point, but instead return only the closest
    point. Could lead to a small speed gain.

    Example:
    %% The Problem
    P0 = [0.5 -0.3 0.5]

    P1 = [0 -1 0]
    P2 = [1  0 0]
    P3 = [0  0 0]

    vertices = [P1; P2; P3]
    faces = [1 2 3]

    %% The Engine
    [dist,PP0] = pointTriangleDistance([P1;P2;P3],P0)

    %% Visualization
    [x,y,z] = sphere(20)
    x = dist*x+P0(1)
    y = dist*y+P0(2)
    z = dist*z+P0(3)

    figure
    hold all
    patch('Vertices',vertices,'Faces',faces,'FaceColor','r','FaceAlpha',0.8)
    plot3(P0(1),P0(2),P0(3),'b*')
    plot3(PP0(1),PP0(2),PP0(3),'*g')
    surf(x,y,z,'FaceColor','b','FaceAlpha',0.3)
    view(3)

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
