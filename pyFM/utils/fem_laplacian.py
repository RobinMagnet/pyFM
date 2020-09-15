"""
CODE WAS TAKEN FROM MINDBOGGLE GITHUB, MINIMALLY ADAPTED
I think there is a lot of speedup to be done with their computeAB function
since it doesn't use any broadcast or einstein summation.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, lobpcg

def area_of_faces(points, faces):
    """
    Compute the areas of all triangles on the mesh.
    Parameters
    ----------
    points : list of lists of 3 floats
        x,y,z coordinates for each vertex of the structure
    faces : list of lists of 3 integers
        3 indices to vertices that form a triangle on the mesh
    Returns
    -------
    area: 1-D numpy array
        area[i] is the area of the i-th triangle
    """

    area = np.zeros(len(faces))

    points = np.array(points)

    for i, triangle in enumerate(faces):

        a = np.linalg.norm(points[triangle[0]] - points[triangle[1]])
        b = np.linalg.norm(points[triangle[1]] - points[triangle[2]])
        c = np.linalg.norm(points[triangle[2]] - points[triangle[0]])
        s = (a+b+c) / 2.0

        area[i] = np.sqrt(s*(s-a)*(s-b)*(s-c))

    return area


def computeAB(points, faces):
    """
    Compute matrices for the Laplace-Beltrami operator.
    The matrices correspond to A and B from Reuter's 2009 article.
    Note ::
        All points must be on faces. Otherwise, a singular matrix error
        is generated when inverting D.
    Parameters
    ----------
    points : list of lists of 3 floats
        x,y,z coordinates for each vertex
    faces : list of lists of 3 integers
        each list contains indices to vertices that form a triangle on a mesh
    Returns
    -------
    A : csr_matrix
    B : csr_matrix
    """

    points = np.array(points)
    faces = np.array(faces)
    nfaces = faces.shape[0]

    # Linear local matrices on unit triangle:
    tB = (np.ones((3,3)) + np.eye(3)) / 24.0

    tA00 = np.array([[ 0.5,-0.5, 0.0],
                     [-0.5, 0.5, 0.0],
                     [ 0.0, 0.0, 0.0]])

    tA11 = np.array([[ 0.5, 0.0,-0.5],
                     [ 0.0, 0.0, 0.0],
                     [-0.5, 0.0, 0.5]])

    tA0110 = np.array([[ 1.0,-0.5,-0.5],
                       [-0.5, 0.0, 0.5],
                       [-0.5, 0.5, 0.0]])

    # Replicate into third dimension for each triangle
    # (for tB, 1st index is the 3rd index in MATLAB):
    tB = np.array([np.tile(tB, (1, 1)) for i in range(nfaces)])
    tA00 = np.array([np.tile(tA00, (1, 1)) for i in range(nfaces)])
    tA11 = np.array([np.tile(tA11, (1, 1)) for i in range(nfaces)])
    tA0110 = np.array([np.tile(tA0110,(1, 1)) for i in range(nfaces)])

    # Compute vertex coordinates and a difference vector for each triangle:
    v1 = points[faces[:, 0], :]
    v2 = points[faces[:, 1], :]
    v3 = points[faces[:, 2], :]
    v2mv1 = v2 - v1
    v3mv1 = v3 - v1

    def reshape_and_repeat(A):
        """
        For a given 1-D array A, run the MATLAB code below.
            M = reshape(M,1,1,nfaces);
            M = repmat(M,3,3);
        Please note that a0 is a 3-D matrix, but the 3rd index in NumPy
        is the 1st index in MATLAB.  Fortunately, nfaces is the size of A.
        """
        return np.array([np.ones((3,3))*x for x in A])

    # Compute length^2 of v3mv1 for each triangle:
    a0 = np.sum(v3mv1 * v3mv1, axis=1)
    a0 = reshape_and_repeat(a0)

    # Compute length^2 of v2mv1 for each triangle:
    a1 = np.sum(v2mv1 * v2mv1, axis=1)
    a1 = reshape_and_repeat(a1)

    # Compute dot product (v2mv1*v3mv1) for each triangle:
    a0110 = np.sum(v2mv1 * v3mv1, axis=1)
    a0110 = reshape_and_repeat(a0110)

    # Compute cross product and 2*vol for each triangle:
    cr  = np.cross(v2mv1,v3mv1)
    vol = np.sqrt(np.sum(cr*cr, axis=1))
    # zero vol will cause division by zero below, so set to small value:
    vol_mean = 0.001*np.mean(vol)
    vol = [vol_mean if x == 0 else x for x in vol]
    vol = reshape_and_repeat(vol)

    # Construct all local A and B matrices (guess: for each triangle):
    localB = vol * tB
    localA = (1.0/vol) * (a0*tA00 + a1*tA11 - a0110*tA0110)

    # Construct row and col indices.
    # (Note: J in numpy is I in MATLAB after flattening,
    #  because numpy is row-major while MATLAB is column-major.)
    J = np.array([np.tile(x, (3,1)) for x in faces])
    I = np.array([np.transpose(np.tile(x, (3,1))) for x in faces])

    # Flatten arrays and swap I and J:
    J_new = I.flatten()
    I_new = J.flatten()
    localA = localA.flatten()
    localB = localB.flatten()

    # Construct sparse matrix:
    A = sparse.csr_matrix((localA, (I_new, J_new)))
    B = sparse.csr_matrix((localB, (I_new, J_new)))

    return A, B



def area_normalize(points, faces, spectrum):
    """
    Normalize a spectrum using areas as suggested in Reuter et al. (2006)
    Parameters
    ----------
    points : list of lists of 3 floats
        x,y,z coordinates for each vertex of the structure
    faces : list of lists of 3 integers
        3 indices to vertices that form a triangle on the mesh
    spectrum : list of floats
        LB spectrum of a given shape defined by _points_ and _faces_
    Returns
    -------
    new_spectrum : list of floats
        LB spectrum normalized by area
    Examples

    """

    area = area_of_faces(points, faces)
    total_area = sum(area)

    new_spectrum = [x*total_area for x in spectrum]

    return new_spectrum


def fem_laplacian(points, faces, spectrum_size=10, normalization="areaindex", verbose=False):
    """
    Compute linear finite-element method Laplace-Beltrami spectrum
    after Martin Reuter's MATLAB code.
    Comparison of fem_laplacian() with Martin Reuter's Matlab eigenvalues:
    fem_laplacian() results for Twins-2-1 left hemisphere (6 values):
    [4.829758648026221e-18,
    0.0001284173002467199,
    0.0002715181572272745,
    0.0003205150847159417,
    0.0004701628070486448,
    0.0005768904023010318]
    Martin Reuter's shapeDNA-tria Matlab code:
    {-4.7207711983791511358e-18 ;
    0.00012841730024672144738 ;
    0.00027151815722727096853 ;
    0.00032051508471592313632 ;
    0.0004701628070486902353  ;
    0.00057689040230097490998 }
    fem_laplacian() results for Twins-2-1 left postcentral (1022):
    [6.3469513010430304e-18,
    0.0005178862383467463,
    0.0017434911095630772,
    0.003667561767487686,
    0.005429017880363784,
    0.006309346984678924]
    Martin Reuter's Matlab code:
    {-2.1954862991027e-18 ;
    0.0005178862383468 ;
    0.0017434911095628 ;
    0.0036675617674875 ;
    0.0054290178803611 ;
    0.006309346984678 }
    Julien Lefevre, regarding comparison with Spongy results:
    "I have done some comparisons between my Matlab codes and yours
    on python and everything sounds perfect:
    The evaluation has been done only for one mesh (about 10000 vertices).
    - L2 error between your A and B matrices and mine are about 1e-16.
    - I have also compared eigenvalues of the generalized problem;
    even if the error is slightly increasing, it remains on the order
    of machine precision.
    - computation time for 1000 eigenvalues was 67s with python
    versus 63s in matlab. And it is quite the same behavior for lower orders.
    - Since the eigenvalues are increasing with order,
    it is also interesting to look at the relative error...
    high frequencies are not so much perturbed."
    Parameters
    ----------
    points : list of lists of 3 floats
        x,y,z coordinates for each vertex of the structure
    faces : list of lists of 3 integers
        3 indices to vertices that form a triangle on the mesh
    spectrum_size : integer
        number of eigenvalues to be computed (the length of the spectrum)
    normalization : string
        the method used to normalize eigenvalues
        if None, no normalization is used
        if "area", use area of the 2D structure as in Reuter et al. 2006
        if "index", divide eigenvalue by index to account for linear trend
        if "areaindex", do both (default)
    verbose : bool
        print statements?
    Returns
    -------
    spectrum : list
        first spectrum_size eigenvalues for Laplace-Beltrami spectrum
    """

    # ----------------------------------------------------------------
    # Compute A and B matrices (from Reuter et al., 2009):
    # ----------------------------------------------------------------
    A, B = computeAB(points, faces)
    if A.shape[0] <= spectrum_size:
        if verbose:
            print("The 3D shape has too few vertices ({0} <= {1}). Skip.".
                  format(A.shape[0], spectrum_size))
        return None

    # ----------------------------------------------------------------
    # Use the eigsh eigensolver:
    # ----------------------------------------------------------------
    try :

        # eigs is for nonsymmetric matrices while
        # eigsh is for real-symmetric or complex-Hermitian matrices:
        # Martin Reuter: "small sigma shift helps prevent numerical
        #   instabilities with zero eigenvalue"
        #B/=np.sum(B)
        eigenvalues, eigenvectors = eigsh(A, k=spectrum_size, M=B,
                                          sigma=-0.01)
        
        spectrum = eigenvalues.tolist()

    # ----------------------------------------------------------------
    # Use the lobpcg eigensolver:
    # ----------------------------------------------------------------
    except RuntimeError:     
           
        if verbose:
            print("eigsh() failed. Now try lobpcg.")
            print("Warning: lobpcg can produce different results from "
                  "Reuter (2006) shapeDNA-tria software.")
        # Initial eigenvector values:
        init_eigenvecs = np.random.random((A.shape[0], spectrum_size))

        # maxiter = 40 forces lobpcg to use 20 iterations.
        # Strangely, largest=false finds largest eigenvalues
        # and largest=True gives the smallest eigenvalues:
        eigenvalues, eigenvectors = lobpcg(A, init_eigenvecs, B=B,
                                           largest=True, maxiter=40)
        # Extract the real parts:
        spectrum = [value.real for value in eigenvalues]

        # For some reason, the eigenvalues from lobpcg are not sorted:
        sorting_arr = np.argsort(spectrum)
        spectrum.sort()
        eigenvectors = eigenvectors[sorting_arr]

    # ----------------------------------------------------------------
    # Normalize by area:
    # ----------------------------------------------------------------
    if normalization == "area":
        spectrum = area_normalize(points, faces, spectrum)
        if verbose:
            print("Compute area-normalized linear FEM Laplace-Beltrami "
                  "spectrum")
    elif normalization == "index":
        spectrum = index_normalize(spectrum)
        if verbose:
            print("Compute index-normalized linear FEM Laplace-Beltrami"
                  " spectrum")
    elif normalization == "areaindex":
        spectrum = index_normalize(spectrum)
        spectrum = area_normalize(points,faces,spectrum)
        if verbose:
            print("Compute area and index-normalized linear FEM "
                  "Laplace-Beltrami spectrum")
    else:
        if verbose:
            print("Compute linear FEM Laplace-Beltrami spectrum")

    return np.asarray(spectrum), eigenvectors, A,B


def index_normalize(spectrum):
    """
    Normalize a spectrum by division of index to account for linear increase of
    Eigenvalue magnitude (Weyl's law in 2D) as suggested in Reuter et al. (2006)
    and used in BrainPrint (Wachinger et al. 2015)
    Parameters
    ----------
    spectrum : list of floats
        LB spectrum of a given shape
    Returns
    -------
    new_spectrum : list of floats
        Linearly re-weighted LB spectrum
    """

    # define index list of floats
    idx = [float(i) for i in range(1,len(spectrum) + 1)]
    # if first entry is zero, shift index
    if (abs(spectrum[0]<1e-09)):
        idx = [i-1 for i in idx]
        idx[0] = 1.0
    # divide each element by its index
    new_spectrum = [x/i for x, i in zip(spectrum, idx)]

    return new_spectrum