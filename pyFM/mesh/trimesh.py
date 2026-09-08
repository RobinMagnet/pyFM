import copy
import os
import time
import warnings

import numpy as np
import potpourri3d as pp3d
import robust_laplacian
import scipy.sparse as sparse

from . import file_utils, laplacian
from . import geometry as geom

GEODESIC_METHODS = ("heat", "heat_pure", "dijkstra", "fast_marching")
READ_MESH_EXTENSIONS = (".off", ".obj", ".ply")
WRITE_MESH_EXTENSIONS = (".off", ".obj")


def _read_mesh_file(path):
    """
    Read vertices and faces from a mesh file.

    Parameters
    ----------
    path : str
        Path to a ``.off`` or ``.obj`` file.

    Returns
    -------
    vertices : (n, 3) np.ndarray
        Coordinates of the mesh vertices.
    faces : (m, 3) np.ndarray or None
        Vertex indices defining the faces.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".off":
        return file_utils.read_off(path)
    if ext == ".obj":
        return file_utils.read_obj(path)
    if ext == ".ply":
        return pp3d.read_mesh(path)
    raise ValueError(
        f"Cannot read '{ext}' files. Supported formats: {', '.join(READ_MESH_EXTENSIONS)}"
    )


def _alias(target, doc):
    """Read/write property forwarding to another attribute."""
    return property(
        lambda self: getattr(self, target),
        lambda self, value: setattr(self, target, value),
        doc=doc,
    )


def _deprecated_alias(target, name):
    """Read/write property forwarding to ``target``, warning when used."""
    message = f"`{name}` is deprecated, use `{target}` instead."

    def fget(self):
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return getattr(self, target)

    def fset(self, value):
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        setattr(self, target, value)

    return property(fget, fset, doc=message)


def _deprecated_method(target, name):
    """Method forwarding to ``target``, warning when used."""
    message = f"`{name}()` is deprecated, use `{target}()` instead."

    def wrapper(self, *args, **kwargs):
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return getattr(self, target)(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__doc__ = message
    return wrapper


class TriMesh:
    """
    Triangle mesh, or point cloud when no faces are given.

    Derived quantities (edges, normals, areas, Laplace-Beltrami operators and
    their spectrum) are computed on demand and stored.

    Parameters
    ------------------
    vertices : np.ndarray
        (n,3) coordinates of the vertices. A file path is also accepted, but
        this is deprecated: use :meth:`TriMesh.load` instead.
    faces : np.ndarray, optional
        (m,3) indices of the vertices of each triangle. Leave empty for a point cloud.
    area_normalize : bool, optional
        If True, scale the mesh to unit area
    center : bool, optional
        If True, move the center of mass to the origin
    normalize : bool, optional
        Shorthand for ``area_normalize=True, center=True``
    rotation : np.ndarray, optional
        (3,3) rotation matrix, applied first
    translation : np.ndarray, optional
        (3,) translation vector, applied after the rotation
    name : str, optional
        Name of the mesh. Defaults to the file stem when loaded from a file.

    Attributes
    ------------------
    path : str
        Path the mesh was loaded from, or None if the geometry has been modified.
    name : str
        Name of the mesh. Preserved even when the geometry is modified.
    stiffness : scipy.sparse
        (n,n) cotangent weight matrix, also available as ``W``.
    mass : scipy.sparse
        (n,n) area matrix, either diagonal or built with finite elements,
        also available as ``A``.
    eigenvalues : np.ndarray
        (k,) eigenvalues of the Laplace-Beltrami operator
    eigenvectors : np.ndarray
        (n,k) eigenvectors of the Laplace-Beltrami operator
    """

    #: Discretizations available to build the Laplace-Beltrami operators.
    #: Registering a new scheme (for instance one based on intrinsic
    #: triangulations with a signpost data structure) means adding an entry
    #: here; nothing else in the class needs to change.
    _LAPLACIAN_BUILDERS = {
        "cotan": "_build_cotan_operators",
        "robust": "_build_robust_operators",
        "intrinsic": "_build_intrinsic_operators",
    }

    #: Prefactorized solvers. They wrap native objects that can be neither
    #: copied nor pickled, so they are dropped rather than duplicated and
    #: rebuilt on demand.
    _SOLVER_ATTRIBUTES = (
        "_solver_heat",
        "_solver_lap",
        "_solver_geod_heat",
        "_solver_geod_fmarch",
    )

    def __init__(
        self,
        vertices,
        faces=None,
        *,
        area_normalize=False,
        center=False,
        normalize=False,
        rotation=None,
        translation=None,
        name=None,
    ):
        self._init_all_attributes()
        self.name = name

        path = None
        if isinstance(vertices, (str, os.PathLike)):
            if faces is not None:
                raise TypeError("Cannot give both a file path and a list of faces")
            warnings.warn(
                "Building a TriMesh from a file path is deprecated, "
                "use TriMesh.load(path) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            path = os.fspath(vertices)
            vertices, faces = _read_mesh_file(path)

        if normalize:
            area_normalize = True
            center = True

        self.vertices = vertices
        self.faces = faces

        if path is not None:
            self.path = path
            if self.name is None:
                self.name = os.path.splitext(os.path.basename(path))[0]

        if rotation is not None:
            self.rotate(rotation)
        if translation is not None:
            self.translate(translation)

        if area_normalize:
            self.area_normalize()

        if center:
            self.translate(-self.center_mass)

    @classmethod
    def load(cls, path, **kwargs):
        """
        Read a mesh from a ``.off`` or ``.obj`` file.

        Parameters
        ------------------
        path : str or os.PathLike
            path to the file to read
        **kwargs
            Any keyword argument accepted by :class:`TriMesh`.

        Returns
        ------------------
        mesh : TriMesh
            the loaded mesh
        """
        path = os.fspath(path)
        vertices, faces = _read_mesh_file(path)

        mesh = cls(vertices, faces, **kwargs)
        mesh.path = path
        if mesh.name is None:
            mesh.name = os.path.splitext(os.path.basename(path))[0]
        return mesh

    #: Alias for :meth:`load`.
    from_file = load

    def __repr__(self):
        name = "" if self.name is None else f" {self.name!r}"
        if self.is_point_cloud:
            return f"<{type(self).__name__}{name}: {self.n_vertices} vertices (point cloud)>"
        return f"<{type(self).__name__}{name}: {self.n_vertices} vertices, {self.n_faces} faces>"

    # ------------------------------------------------------------------
    # Core data
    # ------------------------------------------------------------------

    @property
    def vertices(self):
        """
        Get or set the vertices. Checks the format when setting.

        Returns
        -----------------
        vertices : np.ndarray
            (n,3) array of vertices
        """
        return self._vertices

    @vertices.setter
    def vertices(self, vertices):
        vertices = np.asarray(vertices, dtype=float)
        if vertices.ndim != 2:
            raise ValueError("Vertex list has to be 2D")
        elif vertices.shape[1] != 3:
            raise ValueError("Vertex list requires 3D coordinates")

        if self._vertices is not None:
            self._modified = True
            self._normalized = False

        self._reset_vertex_attributes()
        self.path = None
        self._vertices = vertices.copy()

    @property
    def faces(self):
        """
        Get or set the faces. Checks the format when setting.

        Returns
        -----------------
        faces : np.ndarray
            (m,3) array of faces, or None for a point cloud
        """
        return self._faces

    @faces.setter
    def faces(self, faces):
        if faces is None:
            self._faces = None
        else:
            faces = np.asarray(faces)
            if faces.ndim != 2:
                raise ValueError("Faces list has to be 2D")
            elif faces.shape[1] != 3:
                raise ValueError("Each face is made of 3 points")
            self._faces = faces.copy()

        self._reset_face_attributes()
        self.path = None

    vertlist = _deprecated_alias("vertices", "vertlist")
    facelist = _deprecated_alias("faces", "facelist")

    @property
    def is_point_cloud(self):
        """
        Whether the mesh has no faces.

        Returns
        -----------------
        is_point_cloud : bool
            True if no faces are defined
        """
        return self._faces is None

    def _require_faces(self, operation):
        """Raise a clear error when a face-based quantity is asked of a point cloud."""
        if self.is_point_cloud:
            raise ValueError(f"Cannot compute {operation}: this mesh has no faces.")

    @property
    def n_vertices(self):
        """
        Number of vertices in the mesh.

        Returns
        -----------------
        n_vertices : int
            number of vertices in the mesh
        """
        return self.vertices.shape[0]

    @property
    def n_faces(self):
        """
        Number of faces in the mesh, 0 for a point cloud.

        Returns
        -----------------
        n_faces : int
            number of faces in the mesh
        """
        if self.is_point_cloud:
            return 0
        return self.faces.shape[0]

    # ------------------------------------------------------------------
    # Derived quantities, computed on first access
    # ------------------------------------------------------------------

    @property
    def edges(self):
        """
        (p,2) array of edges, defined by vertex indices.

        Returns
        -----------------
        edges : np.ndarray
            (p,2) array of edges
        """
        if self._edges is None:
            self.compute_edges()
        return self._edges

    @property
    def edge_lengths(self):
        """
        (p,) array of edge lengths.

        Returns
        -----------------
        edge_lengths : np.ndarray
            (p,) array of edge lengths
        """
        if self._edge_lengths is None:
            edges = self.edges
            self._edge_lengths = np.linalg.norm(
                self.vertices[edges[:, 1]] - self.vertices[edges[:, 0]], axis=1
            )
        return self._edge_lengths

    @property
    def face_normals(self):
        """
        (m,3) array of face normals.

        Returns
        -----------------
        face_normals : np.ndarray
            (m,3) array of face normals
        """
        if self._face_normals is None:
            self.compute_normals()
        return self._face_normals

    @face_normals.setter
    def face_normals(self, face_normals):
        self._face_normals = face_normals

    @property
    def vertex_normals(self):
        """
        (n,3) array of vertex normals.

        Returns
        -----------------
        vertex_normals : np.ndarray
            (n,3) array of vertex normals
        """
        if self._vertex_normals is None:
            self.compute_vertex_normals()
        return self._vertex_normals

    @vertex_normals.setter
    def vertex_normals(self, vertex_normals):
        self._vertex_normals = vertex_normals

    @property
    def face_areas(self):
        """
        (m,) array of face areas.

        Returns
        -----------------
        face_areas : np.ndarray
            (m,) array of face areas
        """
        if self._face_areas is None:
            self._require_faces("face areas")
            self._face_areas = geom.compute_faces_areas(self.vertices, self.faces)
        return self._face_areas

    @face_areas.setter
    def face_areas(self, face_areas):
        self._face_areas = face_areas

    # Short forms, kept because they read well in formulas such as ``f @ mesh.W @ g``.
    W = _alias("stiffness", "Alias for :attr:`stiffness`.")
    A = _alias("mass", "Alias for :attr:`mass`.")

    normals = _deprecated_alias("face_normals", "normals")
    faces_areas = _deprecated_alias("face_areas", "faces_areas")
    edges_lengths = _deprecated_alias("edge_lengths", "edges_lengths")
    meshname = _deprecated_alias("name", "meshname")

    @property
    def is_intrinsic(self):
        """
        Whether the operators were built on an intrinsic triangulation.

        Returns
        -----------------
        is_intrinsic : bool
            True if an intrinsic triangulation was used
        """
        return self._intrinsic

    @property
    def vertex_areas(self):
        """
        Per-vertex area.

        Returns
        -----------------
        vertex_areas : np.ndarray
            (n,) array of vertex areas
        """
        if self.mass is None:
            return geom.compute_vertex_areas(self.vertices, self.faces)

        return np.asarray(self.mass.sum(1)).squeeze()

    @property
    def area(self):
        """
        Area of the mesh, None for an unprocessed point cloud.

        Returns
        -----------------
        area : float
            area of the mesh
        """
        if self.mass is None:
            if self.is_point_cloud:
                return None
            return self.face_areas.sum()

        return self.mass.sum()

    @property
    def sqrt_area(self):
        """
        Square root of the area.

        Returns
        -----------------
        sqrt_area : float
            square root of the area
        """
        return np.sqrt(self.area)

    sqrtarea = _deprecated_alias("sqrt_area", "sqrtarea")

    @property
    def center_mass(self):
        """
        Center of mass.

        Returns
        -----------------
        center_mass : np.ndarray
            (3,) array of the center of mass
        """
        return np.average(self.vertices, axis=0, weights=self.vertex_areas)

    @property
    def is_normalized(self):
        """
        Whether the mesh has been area normalized with :meth:`area_normalize`.

        Returns
        -----------------
        is_normalized : bool
            whether the mesh has been area normalized
        """
        return self._normalized

    @property
    def is_modified(self):
        """
        Whether the mesh has been modified from the file it was read from,
        with non-isometric deformations.

        Returns
        -----------------
        is_modified : bool
            whether the mesh was modified with respect to the original input
        """
        return self._modified

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def _init_all_attributes(self):
        """Set every attribute to its empty value."""
        self.path = None
        self.name = None

        self._vertices = None
        self._faces = None

        self._modified = False
        self._normalized = False

        self._vertex_normals_weighting = "area"

        self._edges = None
        self._reset_vertex_attributes()

    def _reset_vertex_attributes(self):
        """
        Reset everything that depends on the vertex positions.

        Called whenever the vertices move in a way that is not a rigid motion.
        """
        self._edge_lengths = None
        self._face_areas = None

        self._face_normals = None
        self._vertex_normals = None

        self._intrinsic = False
        self._laplacian_method = None

        self.stiffness = None
        self.mass = None

        self.eigenvalues = None
        self.eigenvectors = None

        self._reset_solvers()

    def _reset_face_attributes(self):
        """
        Reset everything that depends on the faces.

        Changing the faces changes everything changing the vertices does, and
        the connectivity on top of it.
        """
        self._edges = None
        self._reset_vertex_attributes()

    def _reset_solvers(self):
        """Drop the prefactorized solvers, which are rebuilt on demand."""
        for attribute in self._SOLVER_ATTRIBUTES:
            setattr(self, attribute, None)

    # ------------------------------------------------------------------
    # Copying (some tricks regarding solvers)
    # ------------------------------------------------------------------

    def copy(self, deep=True):
        """
        Return a copy of the mesh.

        Cached solvers are never copied. They wrap native objects that cannot be
        duplicated, and are rebuilt on demand on the copy.

        Parameters
        -----------------
        deep : bool, optional
            If True, copy the underlying arrays as well. Defaults to True.

        Returns
        -----------------
        mesh : TriMesh
            a copy of the mesh
        """
        return copy.deepcopy(self) if deep else copy.copy(self)

    def __copy__(self):
        other = type(self).__new__(type(self))
        other.__dict__.update(self.__dict__)
        other._reset_solvers()
        return other

    def __deepcopy__(self, memo):
        other = type(self).__new__(type(self))
        memo[id(self)] = other
        for key, value in self.__dict__.items():
            if key in self._SOLVER_ATTRIBUTES:
                other.__dict__[key] = None
            else:
                other.__dict__[key] = copy.deepcopy(value, memo)
        return other

    def __getstate__(self):
        state = self.__dict__.copy()
        for attribute in self._SOLVER_ATTRIBUTES:
            state[attribute] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def area_normalize(self):
        """
        Normalize the mesh by its area, keeping the center of mass fixed.

        Returns
        -----------------
        self : TriMesh
            the mesh itself
        """
        center_mass = self.center_mass
        self.translate(-center_mass)
        self.scale(1 / self.sqrt_area)
        self.translate(center_mass)
        self._normalized = True
        return self

    def rotate(self, R):
        """
        Rotate the mesh and its normals.

        Parameters
        -----------------
        R : np.ndarray
            (3,3) rotation matrix

        Returns
        -----------------
        self : TriMesh
            the mesh itself
        """
        R = np.asarray(R)
        if R.shape != (3, 3) or not np.isclose(np.linalg.det(R), 1):
            raise ValueError("Rotation should be a 3x3 matrix with unit determinant")

        self._vertices = self.vertices @ R.T

        if self._face_normals is not None:
            self._face_normals = self._face_normals @ R.T

        if self._vertex_normals is not None:
            self._vertex_normals = self._vertex_normals @ R.T

        return self

    def translate(self, t):
        """
        Translate the mesh.

        Parameters
        -----------------
        t : np.ndarray
            (3,) translation vector

        Returns
        -----------------
        self : TriMesh
            the mesh itself
        """
        self._vertices += np.asarray(t).squeeze()[None, :]
        return self

    def scale(self, alpha):
        """
        Multiply the mesh by alpha, updating areas, spectrum and geodesic distances.

        Parameters
        -----------------
        alpha : float
            scaling factor

        Returns
        -----------------
        self : TriMesh
            the mesh itself
        """
        self._vertices *= alpha

        if self.mass is not None:
            self.mass = alpha**2 * self.mass

        if self._face_areas is not None:
            self._face_areas = alpha**2 * self._face_areas

        if self._edge_lengths is not None:
            self._edge_lengths = alpha * self._edge_lengths

        if self.eigenvalues is not None:
            self.eigenvalues = 1 / alpha**2 * self.eigenvalues

        if self.eigenvectors is not None:
            self.eigenvectors = 1 / alpha * self.eigenvectors

        # Solvers were factorized for the previous scale.
        self._reset_solvers()

        self._modified = True
        self._normalized = False
        return self

    def center(self):
        """
        Center the mesh on its center of mass.

        Returns
        -----------------
        self : TriMesh
            the mesh itself
        """
        self.translate(-self.center_mass)
        return self

    # ------------------------------------------------------------------
    # Laplace-Beltrami operators and spectrum
    # ------------------------------------------------------------------

    def _build_cotan_operators(self):
        """Cotangent weights with a lumped diagonal area matrix."""
        return (
            laplacian.cotangent_weights(self.vertices, self.faces),
            laplacian.dia_area_mat(self.vertices, self.faces),
        )

    def _build_robust_operators(self, mollify_factor=1e-5):
        """Tufted Laplacian, which also handles point clouds."""
        if self.is_point_cloud:
            return robust_laplacian.point_cloud_laplacian(
                self.vertices, mollify_factor=mollify_factor
            )
        return robust_laplacian.mesh_laplacian(
            self.vertices, self.faces, mollify_factor=mollify_factor
        )

    def _build_intrinsic_operators(self):
        """Laplacian on an intrinsic triangulation, without mollification."""
        return self._build_robust_operators(mollify_factor=0.0)

    def _build_operators(self, method):
        """
        Build the stiffness and mass matrices with the given discretization.

        Parameters
        -----------------
        method : str
            name of a discretization registered in ``_LAPLACIAN_BUILDERS``

        Returns
        -----------------
        stiffness : scipy.sparse
            (n,n) stiffness matrix
        mass : scipy.sparse
            (n,n) mass matrix
        """
        try:
            builder = self._LAPLACIAN_BUILDERS[method]
        except KeyError:
            raise ValueError(
                f"Unknown Laplacian discretization '{method}', "
                f"expected one of {tuple(self._LAPLACIAN_BUILDERS)}"
            ) from None

        return getattr(self, builder)()

    def compute_operators(self, intrinsic=False, robust=False):
        """
        Build the Laplace-Beltrami operators, without computing the spectrum.

        Parameters
        -------------------------
        intrinsic : bool, optional
            Use an intrinsic triangulation. Defaults to False
        robust : bool, optional
            Use the tufted Laplacian, forced for point clouds. Defaults to False

        Returns
        -------------------------
        self : TriMesh
            the mesh itself
        """
        if self.is_point_cloud:
            robust = True

        if robust:
            method = "robust"
        elif intrinsic:
            method = "intrinsic"
        else:
            method = "cotan"

        self.stiffness, self.mass = self._build_operators(method)
        self._laplacian_method = method
        self._intrinsic = bool(intrinsic) and method != "cotan"

        return self

    def compute_spectrum(
        self, k, intrinsic=False, return_spectrum=True, robust=False, verbose=False
    ):
        """
        Compute the Laplace-Beltrami operators and their spectrum.

        Consider using :meth:`process` for easier use.

        Parameters
        -------------------------
        k : int
            number of eigenvalues to compute
        intrinsic : bool, optional
            Use an intrinsic triangulation. Defaults to False
        return_spectrum : bool, optional
            Whether to return the computed spectrum, defaults to True
        robust : bool, optional
            use the tufted Laplacian, defaults to False
        verbose : bool, optional
            print progress. Defaults to False

        Returns
        -------------------------
        eigenvalues : np.ndarray, optional
            (k,) - Only if return_spectrum is True.
        eigenvectors : np.ndarray, optional
            (n,k) - Only if return_spectrum is True.
        """
        self.compute_operators(intrinsic=intrinsic, robust=robust)

        # If k is 0, stop here
        if k > 0:
            if verbose:
                print(f"Computing {k} eigenvectors")
                start_time = time.time()

            self.eigenvalues, self.eigenvectors = laplacian.laplacian_spectrum(
                self.stiffness, self.mass, spectrum_size=k
            )

            if verbose:
                print(f"\tDone in {time.time() - start_time:.2f} s")

            if return_spectrum:
                return self.eigenvalues, self.eigenvectors

    laplacian_spectrum = _deprecated_method("compute_spectrum", "laplacian_spectrum")

    def process(self, k=200, skip_normals=True, intrinsic=False, robust=False, verbose=False):
        """
        Compute the LB spectrum and store it.

        Parameters
        -----------------------
        k : int
            (default = 200) Number of eigenvalues to compute
        skip_normals : bool, optional
            If set to True, skip normals computation. Defaults to True
        intrinsic : bool, optional
            Use an intrinsic triangulation. Defaults to False
        robust : bool, optional
            use the tufted Laplacian
        verbose : bool, optional
            print progress

        Returns
        -----------------------
        self : TriMesh
            the mesh itself
        """
        if not skip_normals:
            _ = self.face_normals

        if (
            (self.eigenvectors is not None)
            and (self.eigenvalues is not None)
            and (len(self.eigenvalues) >= k)
        ):
            self.eigenvectors = self.eigenvectors[:, :k]
            self.eigenvalues = self.eigenvalues[:k]

        else:
            self.compute_spectrum(
                k,
                return_spectrum=False,
                intrinsic=intrinsic,
                robust=robust,
                verbose=verbose,
            )

        return self

    # ------------------------------------------------------------------
    # Spectral projection
    # ------------------------------------------------------------------

    def project(self, func, k=None):
        """
        Project one or multiple functions on the spectral basis.

        Parameters
        -----------------------
        func : np.ndarray
            (n,p) or (n,) functions on the shape
        k : int, optional
            dimension of the LB basis on which to project. If None use all the computed basis

        Returns
        -----------------------
        projected_func : np.ndarray
            (k,p) or (k,) projected function
        """
        if k is None:
            return self.eigenvectors.T @ (self.mass @ func)

        elif k <= self.eigenvectors.shape[1]:
            return self.eigenvectors[:, :k].T @ (self.mass @ func)

        else:
            raise ValueError(f"At least {k} eigenvectors should be computed before projecting")

    def unproject(self, projection):
        """
        Build a function from its coefficients in the spectral basis.

        Parameters
        -----------------------
        projection : np.ndarray
            (k,p) or (k,) functions on the reduced basis of the shape

        Returns
        -----------------------
        func : np.ndarray
            (n,p) or (n,) reconstructed function on the vertices
        """
        k = projection.shape[0]
        if k <= self.eigenvectors.shape[1]:
            return self.eigenvectors[:, :k] @ projection

        else:
            raise ValueError(f"At least {k} eigenvectors should be computed before decoding")

    decode = _deprecated_method("unproject", "decode")

    def reconstruct(self, func, k=None):
        """
        Reconstruct a function with the LB eigenbasis, ie project on the spectral
        basis and rebuild values on all vertices.

        Parameters
        -----------------------
        func : np.ndarray
            (n,p) or (n,) - functions on the shape
        k : int, optional
            Number of eigenfunctions to use. If None, uses the complete computed basis.

        Returns
        -----------------------
        func : np.ndarray
            (n,p) or (n,) projected function
        """
        return self.unproject(self.project(func, k=k))

    # ------------------------------------------------------------------
    # Norms and inner products
    # ------------------------------------------------------------------

    def l2_sqnorm(self, func):
        """
        Return the squared L2 norm of one or multiple functions on the mesh.

        For a single function f, this returns f.T @ A @ f with A the area matrix.

        Parameters
        -----------------
        func : np.ndarray
            (n,p) or (n,) functions on the mesh

        Returns
        -----------------
        sqnorm : np.ndarray
            (p,) array of squared l2 norms or a float only one function was provided.
        """
        return self.l2_inner(func, func)

    def l2_inner(self, func1, func2):
        """
        Return the L2 inner product of two functions, or pairwise inner products
        if lists of functions are given.

        For two functions f1 and f2, this returns f1.T @ A @ f2 with A the area matrix.

        Parameters
        -----------------
        func1 : np.ndarray
            (n,p) or (n,) functions on the mesh
        func2 : np.ndarray
            (n,p) or (n,) functions on the mesh

        Returns
        -----------------
        sqnorm : np.ndarray
            (p,) array of L2 inner products or a float if only one function per
            argument was provided.
        """
        assert func1.shape == func2.shape, "Shapes must be equal"

        if func1.ndim == 1:
            return func1 @ self.mass @ func2

        return np.einsum("np,np->p", func1, self.mass @ func2)

    def h1_sqnorm(self, func):
        """
        Return the squared H^1_0 norm (L2 norm of the gradient) of one or multiple
        functions on the mesh.

        For a single function f, this returns f.T @ W @ f with W the stiffness matrix.

        Parameters
        -----------------
        func : np.ndarray
            (n,p) or (n,) functions on the mesh

        Returns
        -----------------
        sqnorm : np.ndarray
            (p,) array of squared H1 norms or a float only one function was provided.
        """
        return self.h1_inner(func, func)

    def h1_inner(self, func1, func2):
        """
        Return the H1 inner product of two functions, or pairwise inner products
        if lists of functions are given.

        For two functions f1 and f2, this returns f1.T @ W @ f2 with W the stiffness matrix.

        Parameters
        -----------------
        func1 : np.ndarray
            (n,p) or (n,) functions on the mesh
        func2 : np.ndarray
            (n,p) or (n,) functions on the mesh

        Returns
        -----------------
        sqnorm : np.ndarray
            (p,) array of H1 inner products or a float if only one function per
            argument was provided.
        """
        assert func1.shape == func2.shape, "Shapes must be equal"

        if func1.ndim == 1:
            return func1 @ self.stiffness @ func2

        return np.einsum("np,np->p", func1, self.stiffness @ func2)

    def integrate(self, func):
        """
        Integrate a function or a set of functions on the mesh.

        Parameters
        -----------------
        func : np.ndarray
            (n,p) or (n,) functions on the mesh

        Returns
        -----------------
        integral : np.ndarray
            (p,) array of integrals or a float only one function was provided.
        """
        if func.ndim == 1:
            return np.sum(self.mass @ func)
        return np.sum(self.mass @ func, axis=0)

    # ------------------------------------------------------------------
    # Geodesics
    # ------------------------------------------------------------------

    def _heat_time(self):
        """Squared mean edge length, used as the diffusion time of the heat method."""
        return self.edge_lengths.mean() ** 2

    def geodesic_matrix(
        self,
        method="heat",
        sym=False,
        batch_size=500,
        verbose=False,
    ):
        """
        Compute the full matrix of pairwise geodesic distances.

        This is an (n,n) dense matrix, so it gets expensive quickly. Save it
        yourself with ``np.save`` if you need it more than once, and hand it back
        to :meth:`farthest_point_sampling` through its ``distances`` argument.

        Parameters
        -----------------
        method : str, optional
            Method to use to compute geodesic distances. One of:

            - "heat" : potpourri3d robust heat method (default)
            - "heat_pure" : pure-python heat method (falls back to "heat" if the mesh
              uses an intrinsic triangulation, since the pure-python path
              needs faces/normals that intrinsic meshes may lack)
            - "dijkstra" : graph-based Dijkstra algorithm
            - "fast_marching" : potpourri3d fast marching method

            Defaults to "heat".
        sym : bool, optional
            Symmetrize the matrix if computed with the heat or fast marching method.
            Defaults to False
        batch_size : int, optional
            If method is "heat_pure", compute distances by batch
        verbose : bool, optional
            Print progress

        Returns
        -----------------
        distances : np.ndarray
            (n,n) matrix of geodesic distances
        """
        if method not in GEODESIC_METHODS:
            raise ValueError(f"method must be one of {GEODESIC_METHODS}, got '{method}'")

        self._require_faces("geodesic distances")

        if method == "dijkstra":
            geod_dist = geom.geodesic_distmat_dijkstra(self.vertices, self.faces)

        elif method == "fast_marching":
            geod_dist = geom.geodesic_distmat_fast_marching(self.vertices, self.faces)

        elif method == "heat" or (method == "heat_pure" and self.is_intrinsic):
            geod_dist = geom.heat_geodmat_robust(self.vertices, self.faces, verbose=verbose)

        else:
            # Ensure LB matrices are processed.
            if self.mass is None or self.stiffness is None:
                self.process(k=0)

            geod_dist = geom.heat_geodmat(
                self.vertices,
                self.faces,
                self.face_normals,
                self.mass,
                self.stiffness,
                t=self._heat_time(),
                batch_size=batch_size,
                verbose=verbose,
            )

        if sym and method != "dijkstra":
            geod_dist *= 0.5
            geod_dist += geod_dist.T

        return geod_dist

    get_geodesic = _deprecated_method("geodesic_matrix", "get_geodesic")

    def geodesic_from(self, i, method="heat"):
        """
        Compute geodesic distances from vertex (or vertices) i using the given method.

        Parameters
        ----------------------
        i : int or (p,) array of ints
            index (or indices) of the source vertex/vertices
        method : str, optional
            Method to use to compute geodesic distances. One of:

            - "heat" : potpourri3d robust heat method (default)
            - "heat_pure" : pure-python heat method (falls back to "heat" if the mesh
              uses an intrinsic triangulation, since the pure-python path
              needs faces/normals that intrinsic meshes may lack)
            - "dijkstra" : graph-based Dijkstra algorithm
            - "fast_marching" : potpourri3d fast marching method

            Defaults to "heat".

        Returns
        ----------------------
        dist : np.ndarray
            (n,) distances to vertex i, or (n,p) if i is a sequence of length p
        """
        if method not in GEODESIC_METHODS:
            raise ValueError(f"method must be one of {GEODESIC_METHODS}, got '{method}'")

        self._require_faces("geodesic distances")

        if method == "fast_marching":
            if self._solver_geod_fmarch is None:
                self._solver_geod_fmarch = pp3d.MeshFastMarchingDistanceSolver(
                    self.vertices, self.faces
                )

            if np.issubdtype(type(i), np.integer):
                return self._solver_geod_fmarch.compute_distance([[(i, [])]])
            else:
                return np.array(
                    [self._solver_geod_fmarch.compute_distance([[(x, [])]]) for x in i]
                ).T

        elif method == "dijkstra":
            graph = geom.build_dijkstra_graph(self.vertices, self.faces)
            return geom.dijkstra_from(i, graph)

        elif method == "heat" or (method == "heat_pure" and self.is_intrinsic):
            if self._solver_geod_heat is None:
                self._solver_geod_heat = pp3d.MeshHeatMethodDistanceSolver(
                    self.vertices, self.faces
                )

            if np.issubdtype(type(i), np.integer):
                return self._solver_geod_heat.compute_distance(i)
            else:
                return np.array([self._solver_geod_heat.compute_distance(x) for x in i]).T

        # method == "heat_pure" and not self.is_intrinsic
        if self.mass is None or self.stiffness is None:
            self.process(k=0)

        t = self._heat_time()

        if self._solver_heat is None:
            self._solver_heat = sparse.linalg.factorized(self.mass.tocsc() + t * self.stiffness)
            self._solver_lap = sparse.linalg.factorized(self.stiffness)

        # Compute distance with cached solvers
        return geom.heat_geodesic_from(
            i,
            self.vertices,
            self.faces,
            self.face_normals,
            self.mass,
            W=None,
            t=t,
            solver_heat=self._solver_heat,
            solver_lap=self._solver_lap,
        )

    geod_from = _deprecated_method("geodesic_from", "geod_from")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def farthest_point_sampling(
        self, size, random_init=True, geodesic=True, distances=None, verbose=False
    ):
        """
        Sample points using farthest point sampling.

        Distances to each new sample are computed on the fly, unless a full
        distance matrix is given as ``distances``.

        Parameters
        -------------------------
        size : int
            number of points to sample
        random_init : bool, optional
            Whether to sample the first point randomly or to take the furthest away
            from all the other ones. The latter needs the full distance matrix, so
            this is only read when ``distances`` is given. Defaults to True
        geodesic : bool, optional
            If True perform geodesic fps, else euclidean. Defaults to True
        distances : np.ndarray, optional
            (n,n) matrix of precomputed pairwise distances, as returned by
            :meth:`geodesic_matrix`. Saves recomputing distances at each step.
        verbose : bool, optional
            Print progress. Defaults to False

        Returns
        --------------------------
        fps : np.ndarray
            (size,) array of indices of sampled points (given on the complete mesh)

        Notes
        --------------------------
        Without ``distances``, the first point is always drawn at random, so the
        result varies between calls. Pass ``distances`` with ``random_init=False``
        for a reproducible sample.
        """
        if distances is not None:
            return geom.farthest_point_sampling(
                distances, size, random_init=random_init, verbose=verbose
            )

        if geodesic:
            return geom.farthest_point_sampling_call(
                self.geodesic_from, size, n_points=self.n_vertices, verbose=verbose
            )

        def dist_func(i):
            return np.linalg.norm(self.vertices - self.vertices[i, None, :], axis=1)

        return geom.farthest_point_sampling_call(
            dist_func, size, n_points=self.n_vertices, verbose=verbose
        )

    extract_fps = _deprecated_method("farthest_point_sampling", "extract_fps")

    def farthest_point_sampling_sub(
        self,
        size,
        sub_points,
        return_sub_inds=False,
        random_init=True,
        geodesic=True,
        distances=None,
        verbose=False,
    ):
        """
        Sample points using farthest point sampling, restricted to a subset of vertices.

        Distances to each new sample are computed on the fly, unless a full
        distance matrix is given as ``distances``.

        Parameters
        -------------------------
        size : int
            number of points to sample
        sub_points : np.ndarray
            (size,) array of indices of the sub points
        return_sub_inds : bool, optional
            Whether to also return the indices in the sub mesh. Defaults to False
        random_init : bool, optional
            Whether to sample the first point randomly or to take the furthest away
            from all the other ones. Defaults to True
        geodesic : bool, optional
            If True perform geodesic fps, else euclidean. Defaults to True
        distances : np.ndarray, optional
            (n,n) matrix of precomputed pairwise distances, as returned by
            :meth:`geodesic_matrix`. Saves recomputing distances at each step.
        verbose : bool, optional
            Print progress. Defaults to False

        Returns
        --------------------------
        fps : np.ndarray
            (size,) array of indices of sampled points (given on the complete mesh)
        fps_sub : np.ndarray
            (size,) array of indices of sampled points (given on the sub mesh)
        """
        if distances is not None:
            fps_sub = geom.farthest_point_sampling(
                distances[np.ix_(sub_points, sub_points)],
                size,
                random_init=random_init,
                verbose=verbose,
            )
            if return_sub_inds:
                return sub_points[fps_sub], fps_sub
            return sub_points[fps_sub]

        if geodesic:
            dist_func = self.geodesic_from
        else:

            def dist_func(i):
                return np.linalg.norm(self.vertices - self.vertices[i, None, :], axis=1)

        return geom.farthest_point_sampling_call_sub(
            dist_func,
            size,
            sub_points,
            return_sub_inds=return_sub_inds,
            random_init=random_init,
            verbose=verbose,
        )

    extract_fps_sub = _deprecated_method("farthest_point_sampling_sub", "extract_fps_sub")

    # ------------------------------------------------------------------
    # Differential operators
    # ------------------------------------------------------------------

    def gradient(self, f, normalize=False):
        """
        Compute the gradient of a function using linear interpolation between vertices.

        Parameters
        --------------------------
        f : np.ndarray
            (n_v,) function value on each vertex
        normalize : bool, optional
            Whether the gradient should be normalized on each face

        Returns
        --------------------------
        gradient : np.ndarray
            (n_f,3) gradient of f on each face
        """
        self._require_faces("gradients")

        grad = geom.grad_f(f, self.vertices, self.faces, self.face_normals)  # (n_f,3)

        if normalize:
            grad /= np.linalg.norm(grad, axis=1, keepdims=True)

        return grad

    def divergence(self, f):
        """
        Compute the divergence of a vector field on the mesh.

        Parameters
        --------------------------
        f : np.ndarray
            (n_f, 3) vector value on each face

        Returns
        --------------------------
        divergence : np.ndarray
            (n_v,) divergence of f on each vertex
        """
        self._require_faces("divergence")

        return geom.div_f(f, self.vertices, self.faces, self.face_normals)

    def orientation_op(self, gradf, normalize=False):
        """
        Compute the orientation operator associated to a gradient field gradf.

        For a given function g on the vertices, this operator linearly computes
        < grad(f) x grad(g), n> for each vertex by averaging along the adjacent faces.
        In practice, we compute < n x grad(f), grad(g) > for simpler computation.

        Parameters
        --------------------------
        gradf : np.ndarray
            (n_f,3) gradient field on the mesh
        normalize : bool, optional
            Whether to normalize the gradient on each face

        Returns
        --------------------------
        operator : scipy.sparse.csc_matrix
            (n_v,n_v) orientation operator.
        """
        self._require_faces("the orientation operator")

        if normalize:
            gradf /= np.linalg.norm(gradf, axis=1, keepdims=True)

        return geom.get_orientation_op(
            gradf, self.vertices, self.faces, self.face_normals, self.vertex_areas
        )

    # ------------------------------------------------------------------
    # Normals and edges
    # ------------------------------------------------------------------

    def compute_normals(self):
        """
        Compute the per-face normals.

        Returns
        -----------------
        face_normals : np.ndarray
            (m,3) array of face normals
        """
        self._require_faces("face normals")
        self._face_normals = geom.compute_normals(self.vertices, self.faces)
        return self._face_normals

    def compute_vertex_normals(self):
        """
        Compute the per-vertex normals.

        Returns
        -----------------
        vertex_normals : np.ndarray
            (n,3) array of vertex normals
        """
        self._require_faces("vertex normals")
        self._vertex_normals = geom.per_vertex_normal(
            self.vertices, self.faces, weighting=self._vertex_normals_weighting
        )
        return self._vertex_normals

    def compute_edges(self):
        """
        Compute the edges.

        Returns
        -----------------
        edges : np.ndarray
            (p,2) array of edges
        """
        self._require_faces("edges")
        self._edges = geom.edges_from_faces(self.faces)
        return self._edges

    def set_vertex_normal_weighting(self, weight_type):
        """
        Set the weighting scheme for vertex normals, between 'area' and 'uniform'.

        Parameters
        -----------------
        weight_type : str
            weighting scheme for vertex normals, either 'area' or 'uniform'
        """
        weight_type = weight_type.lower()
        assert weight_type in [
            "uniform",
            "area",
        ], "Only implemented uniform and area weighting"

        if weight_type != self._vertex_normals_weighting:
            self._vertex_normals_weighting = weight_type
            self._vertex_normals = None

    # ------------------------------------------------------------------
    # Writing to file
    # ------------------------------------------------------------------

    def save(
        self,
        path,
        precision=None,
        face_colors=None,
        vertex_normals=None,
        uv=None,
        texture=None,
        mtl_file="material.mtl",
        verbose=False,
    ):
        """
        Write the mesh to a ``.off`` or ``.obj`` file.

        The format is chosen from the extension, defaulting to ``.off`` when the
        path has none.

        Parameters
        -----------------------------
        path : str or os.PathLike
            path of the file to write
        precision : int, optional
            number of significant digits to write for each float
        face_colors : np.ndarray, optional
            (m,3) color of each face. Only supported by the ``.off`` format.
        vertex_normals : np.ndarray or bool, optional
            (n,3) normal at each vertex, or True to use the normals of the mesh.
            Only supported by the ``.obj`` format.
        uv : np.ndarray, optional
            (n,2) uv coordinates of each vertex. Only supported by the ``.obj`` format.
        texture : str, optional
            name or path of the image defining the texture. Requires ``uv``, and
            writes the accompanying ``.mtl`` file.
        mtl_file : str, optional
            name or path of the ``.mtl`` file to write alongside the texture
        verbose : bool, optional
            whether to print information

        Returns
        -----------------------------
        self : TriMesh
            the mesh itself
        """
        path = os.fspath(path)
        ext = os.path.splitext(path)[1].lower()
        if ext == "":
            path += ".off"
            ext = ".off"

        if ext not in WRITE_MESH_EXTENSIONS:
            raise ValueError(
                f"Cannot write '{ext}' files. Supported formats: {', '.join(WRITE_MESH_EXTENSIONS)}"
            )

        if vertex_normals is True:
            vertex_normals = self.vertex_normals
        elif vertex_normals is False:
            vertex_normals = None

        if ext == ".off":
            unsupported = [
                name
                for name, value in (("vertex normals", vertex_normals), ("uv coordinates", uv))
                if value is not None
            ]
            if unsupported:
                raise ValueError(
                    f"The .off format cannot store {' or '.join(unsupported)}, use .obj instead"
                )

            file_utils.write_off(
                path,
                self.vertices,
                self.faces,
                precision=precision,
                face_colors=face_colors,
            )

        else:
            if face_colors is not None:
                raise ValueError("The .obj format cannot store face colors, use .off instead")

            if texture is not None:
                if uv is None:
                    raise ValueError("uv coordinates are required to write a texture")
                file_utils.write_obj_texture(
                    path,
                    self.vertices,
                    self.faces,
                    uv=uv,
                    mtl_file=mtl_file,
                    texture_im=texture,
                    precision=precision,
                    vertex_normals=vertex_normals,
                    verbose=verbose,
                )
            else:
                file_utils.write_obj(
                    path,
                    self.vertices,
                    faces=self.faces,
                    uv=uv,
                    vertex_normals=vertex_normals,
                    precision=precision,
                )

        return self

    export = _deprecated_method("save", "export")

    def get_uv(self, ind1, ind2, mult_const, rotation=None):
        """
        Extract UV coordinates for each vertex.

        Extracted by orthogonal projection on 2 of the x,y,z axes.

        Parameters
        -----------------------------
        ind1 : int
            column index to use as first coordinate
        ind2 : int
            column index to use as second coordinate
        mult_const : float
            number of times to repeat the pattern
        rotation : np.ndarray, optional
            (3,3) rotation matrix applied before the projection

        Returns
        ------------------------------
        uv : np.ndarray
            (n,2) UV coordinates of each vertex
        """
        vert = self.vertices if rotation is None else self.vertices @ rotation.T
        return file_utils.get_uv(vert, ind1, ind2, mult_const=mult_const)

    def export_texture(
        self,
        filename,
        uv,
        mtl_file="material.mtl",
        texture_im="texture_1.jpg",
        precision=None,
        verbose=False,
    ):
        """
        Write a .obj file with texture, using uv coordinates.

        Deprecated, use ``save(path, uv=uv, texture=texture_im)`` instead.

        Parameters
        ------------------------------
        filename : str
            path to the .obj file to write
        uv : np.ndarray
            (n,2) uv coordinates of each vertex
        mtl_file : str, optional
            name of the .mtl file
        texture_im : str, optional
            name of the .jpg file defining the texture
        precision : int, optional
            number of significant digits to write for each float
        verbose : bool, optional
            whether to print information

        Returns
        ------------------------------
        self : TriMesh
            the mesh itself
        """
        warnings.warn(
            "`export_texture()` is deprecated, use `save(path, uv=uv, texture=...)` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        filename = os.fspath(filename)
        if os.path.splitext(filename)[1] != ".obj":
            filename += ".obj"

        return self.save(
            filename,
            uv=uv,
            texture=texture_im,
            mtl_file=mtl_file,
            precision=precision,
            verbose=verbose,
        )
