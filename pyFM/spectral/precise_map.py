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
from sklearn.neighbors import KDTree


def precise_map(mesh1, mesh2, FM, precompute_dmin=True):
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
    Deltamin = compute_Deltamin(mesh1, mesh2, FM)  # (n2,)

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


def compute_Deltamin(mesh1, mesh2, FM):
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

    tree = KDTree(mesh1.eigenvectors[:,:k1])  # Tree on (n1,k1)
    dists,_ = tree.query(mesh2.eigenvectors[:,:k2] @ FM, k=1)  # query on (n2,k1)

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

    # Initialize values
    min_dist = np.inf
    min_faceind = 0
    min_bary = np.zeros(3)
    # min_point = np.zeros(k1)

    # Iterate along client faces
    for faceind in np.where(deltamin - lmax < Deltamin[vertind])[0]:
        Af = mesh1.eigenvectors[mesh1.facelist[faceind], :k1]  # (3,k1)
        b = mesh2.eigenvectors[vertind,None,:k2]@FM  # (1,k1)

        # Project to triangle
        dist, nearest_p, bary_coord = pointTriangleDistance(Af, b.flatten(), return_bary=True)

        if dist < min_dist:
            min_faceind = faceind
            min_bary = bary_coord.copy()  # (3,)
            min_dist = dist
            # min_point = nearest_p.copy()

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
