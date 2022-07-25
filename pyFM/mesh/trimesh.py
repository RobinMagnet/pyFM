import os
import time

import numpy as np

from . import file_utils
from . import geometry as geom
from . import laplacian
import scipy.linalg
import scipy.sparse as sparse

import potpourri3d as pp3d
import robust_laplacian


class TriMesh:
    """
    Mesh Class (can also represent point clouds)
    ________

    Attributes
    ------------------
    # FILE INFO
    path         : str - path the the loaded .off file. Set to None if the geometry is modified.
    meshname     : str - name of the .off file. Remains even when geometry is modified. '_n' is
                   added at the end if the mesh was normalized.

    # GEOMETRY
    vertlist       : (n,3) array of n vertices coordinates
    facelist       : (m,3) array of m triangle indices
    normals        : (m,3) array of normals
    vertex_normals : (n,3) array of vertex normals
                     (change weighting type with self.set_vertex_normal_weighting)

    # SPECTRAL INFORMATION
    W            : (n,n) sparse cotangent weight matrix
    A            : (n,n) sparse area matrix (either diagonal or computed with finite elements)
    eigenvalues  : (K,) eigenvalues of the Laplace Beltrami Operator
    eigenvectors : (n,K) eigenvectors of the Laplace Beltrami Operator

    Properties
    ------------------
    area         : float - area of the mesh
    face_areas   : (m,) per face area
    vertex_areas : (n,) per vertex area
    center_mass  : (3,) center of mass

    n_vertices   : int - number of vertices
    n_faces      : int - number of faces
    edges        : (p,2) edges defined by vertex indices
    """
    def __init__(self, *args, **kwargs):
        # area_normalize=False, center=False, rotation=None, translation=None):
        """
        Read the mesh. Give either the path to a .off file or a list of vertices
        and corrresponding triangles

        Parameters
        ----------------------
        path           : path to a .off file
        vertices       : (n,3) vertices coordinates
        faces          : (m,3) list of indices of triangles
        area_normalize : If True, normalize the mesh
        """
        self._init_all_attributes()
        assert 0 < len(args) < 3, "Provide a path or vertices / faces"

        rotation, translation, area_normalize, center = self._read_init_kwargs(kwargs)

        # Differnetiate between [path] or [vertex] or [vertex, faces]
        if len(args) == 1 and type(args[0]) is str:
            self._load_mesh(args[0])
        elif len(args) == 1:
            self.vertlist = args[0]
            self.facelist = None
        else:
            self.vertlist = args[0]
            self.facelist = args[1]

        if rotation is not None:
            self.rotate(rotation)
        if translation is not None:
            self.translate(translation)

        if area_normalize:
            self.area_normalize()

        if center:
            self.translate(-self.center_mass)

    @property
    def vertlist(self):
        """
        Get or set the vertices.
        Checks the format when setting
        """
        return self._vertlist

    @vertlist.setter
    def vertlist(self, vertlist):
        vertlist = np.asarray(vertlist, dtype=float)
        if vertlist.ndim != 2:
            raise ValueError('Vertex list has to be 2D')
        elif vertlist.shape[1] != 3:
            raise ValueError('Vertex list requires 3D coordinates')

        self._reset_vertex_attributes()
        if hasattr(self, "_vertlist") and self._vertlist is not None:
            self._modified = True
            self._normalized = False
        self.path = None
        self._vertlist = vertlist

    @property
    def facelist(self):
        """
        Get or set the faces.
        Checks the format when setting
        """
        return self._facelist

    @facelist.setter
    def facelist(self, facelist):
        facelist = np.asarray(facelist) if facelist is not None else None
        if facelist is not None:
            if facelist.ndim != 2:
                raise ValueError('Faces list has to be 2D')
            elif facelist.shape[1] != 3:
                raise ValueError('Each face is made of 3 points')
            self._facelist = np.asarray(facelist)
        else:
            self._facelist = None
        self.path = None

    @property
    def vertices(self):
        "alias for vertlist"
        return self.vertlist

    @property
    def faces(self):
        "alias for facelist"
        return self.facelist

    @property
    def n_vertices(self):
        """
        return the number of vertices in the mesh
        """
        return self.vertlist.shape[0]

    @property
    def n_faces(self):
        """
        return the number of faces in the mesh
        """
        if self.facelist is None:
            return 0
        return self.facelist.shape[0]

    @property
    def area(self):
        """
        Returns the area of the mesh
        """
        if self.A is None:
            if self.facelist is None:
                return None
            faces_areas = geom.compute_faces_areas(self.vertlist, self.facelist)
            return faces_areas.sum()

        return self.A.sum()

    @property
    def sqrtarea(self):
        """
        square root of the area
        """
        return np.sqrt(self.area)

    @property
    def edges(self):
        """
        return a (p,2) array of edges defined by vertex indices.
        """
        if self._edges is None:
            self.compute_edges()
        return self._edges

    @property
    def normals(self):
        """
        face normals
        """
        if self._normals is None:
            self.compute_normals()
        return self._normals

    @normals.setter
    def normals(self, normals):
        self._normals = normals

    @property
    def vertex_normals(self):
        """
        per vertex_normal
        """
        if self._vertex_normals is None:
            self.compute_vertex_normals()
        return self._vertex_normals

    @vertex_normals.setter
    def vertex_normals(self, vertex_normals):
        self._vertex_normals = vertex_normals

    @property
    def vertex_areas(self):
        """
        per vertex area
        """
        if self.A is None:
            return geom.compute_vertex_areas(self.vertlist, self.facelist)

        return np.asarray(self.A.sum(1)).squeeze()

    @property
    def faces_areas(self):
        """
        per face area
        """
        if self._faces_areas is None:
            self._faces_areas = geom.compute_faces_areas(self.vertlist, self.facelist)
        return self._faces_areas

    @faces_areas.setter
    def face_areas(self, face_areas):
        self._faces_areas = face_areas

    @property
    def center_mass(self):
        """
        center of mass
        """
        return np.average(self.vertlist, axis=0, weights=self.vertex_areas)

    @property
    def is_normalized(self):
        """
        Whether the mash has been manually normalized using the self.area_normalize method
        """
        if not hasattr(self, "_normalized"):
            self._normalized = False
        return self._normalized

    @property
    def is_modified(self):
        """
        Whether the mash has been modified from path with
        non-isometric deformations
        """
        if not hasattr(self, "_modified"):
            self._modified = False
        return self._modified

    def area_normalize(self):
        self.scale(1/self.sqrtarea)
        self._normalized = True
        return self

    def rotate(self, R):
        """
        Rotate mesh and normals
        """
        if R.shape != (3, 3) or not np.isclose(scipy.linalg.det(R), 1):
            raise ValueError("Rotation should be a 3x3 matrix with unit determinant")

        self._vertlist = self.vertlist @ R.T
        if self._normals is not None:
            self.normals = self.normals @ R.T

        if self._per_vertex_normals is not None:
            self.per_vertex_normals = self.per_vertex_normals @ R.T

        return self

    def translate(self, t):
        """
        translate mesh
        """
        self._vertlist += np.asarray(t).squeeze()[None, :]
        return self

    def scale(self, alpha):
        """
        Multiply mesh by alpha.
        modify vertices, area, spectrum, geodesic distances
        """
        self._vertlist *= alpha

        if self.A is not None:
            self.A = alpha**2 * self.A

        if self._faces_areas is not None:
            self._faces_area *= alpha

        if self.eigenvalues is not None:
            self.eigenvalues = 1 / alpha**2 * self.eigenvalues

        if self.eigenvectors is not None:
            self.eigenvectors = 1 / alpha * self.eigenvectors

        self._solver_heat = None
        self._solver_lap = None
        self._solver_geod = None

        self._modified = True
        self._normalized = False
        return self

    def center(self):
        """
        center the mesh
        """
        self.translate(-self.center_mass)
        return self

    def laplacian_spectrum(self, k, intrinsic=False, return_spectrum=True, robust=False, verbose=False):
        """
        Compute the Laplace Beltrami Operator and its spectrum.
        Consider using the .process() function for easier use !

        Parameters
        -------------------------
        K               : int - number of eigenvalues to compute
        intrinsic       : bool - Use intrinsic triangulation
        robust          : bool - use tufted laplacian
        return_spectrum : bool - Whether to return the computed spectrum

        Output
        -------------------------
        eigenvalues, eigenvectors : (k,), (n,k) - Only if return_spectrum is True.
        """
        if self.facelist is None:
            robust = True

        if robust:
            mollify_factor = 1e-5
        elif intrinsic:
            mollify_factor = 0

        if robust or intrinsic:
            self._intrinsic = intrinsic
            if self.facelist is not None:
                self.W, self.A = robust_laplacian.mesh_laplacian(self.vertlist, self.facelist, mollify_factor=mollify_factor)
            else:
                self.W, self.A = robust_laplacian.point_cloud_laplacian(self.vertlist, mollify_factor=mollify_factor)

        else:
            self.W = laplacian.cotangent_weights(self.vertlist, self.facelist)
            self.A = laplacian.dia_area_mat(self.vertlist, self.facelist)

        # If k is 0, stop here
        if k > 0:
            if verbose:
                print(f"Computing {k} eigenvectors")
                start_time = time.time()
            self.eigenvalues, self.eigenvectors = laplacian.laplacian_spectrum(self.W, self.A,
                                                                               spectrum_size=k)

            if verbose:
                print(f"\tDone in {time.time()-start_time:.2f} s")

            if return_spectrum:
                return self.eigenvalues, self.eigenvectors

    def process(self, k=200, skip_normals=True, intrinsic=False, robust=False, verbose=False):
        """
        Process the LB spectrum and saves it.
        Additionnaly computes per-face normals

        Parameters:
        -----------------------
        k            : int - (default = 200) Number of eigenvalues to compute
        skip_normals : bool - If set to True, skip normals computation
        intrinsic    : bool - Use intrinsic triangulation
        robust       : bool - use tufted laplacian
        """
        if not skip_normals and self._normals is None:
            self.compute_normals()

        if (self.eigenvectors is not None) and (self.eigenvalues is not None)\
           and (len(self.eigenvalues) >= k):
            self.eigenvectors = self.eigenvectors[:,:k]
            self.eigenvalues = self.eigenvalues[:k]

        else:
            if self.facelist is None:
                robust = True
            self.laplacian_spectrum(k, return_spectrum=False, intrinsic=intrinsic, robust=robust,
                                    verbose=verbose)

        return self

    def project(self, func, k=None):
        """
        Project one or multiple functions on the spectral basis

        Parameters
        -----------------------
        func : (n,p) or (n,) functions on the shape
        k    : dimension of the LB basis on which to project. If None use all the computed basis

        Output
        -----------------------
        projected_func : (k,p) or (k,) projected function
        """
        if k is None:
            return self.eigenvectors.T @ (self.A @ func)

        elif k <= self.eigenvectors.shape[1]:
            return self.eigenvectors[:,:k].T @ (self.A @ func)

        else:
            raise ValueError(f'At least {k} eigenvectors should be computed before projecting')

    def decode(self, projection):
        """
        Build a function from its coefficient in the spectral basis

        Parameters
        -----------------------
        projection : (k,p) or (k,) functions on the reduced basis of the shape

        Output
        -----------------------
        func : (n,p) or (n,) projected function
        """
        k = projection.shape[0]
        if k <= self.eigenvectors.shape[1]:
            return self.eigenvectors[:,:k]@projection

        else:
            raise ValueError(f'At least {k} eigenvectors should be computed before decoding')

    def unproject(self, projection):
        """
        Alias for decode
        """
        return self.decode(projection)

    def reconstruct(self, func, k=None):
        """
        Reconstruct function with the LB eigenbasis, ie project on the spectral basis
        and rebuild values on all vertices.

        Parameters
        -----------------------
        func : (n,p) or (n,) - functions on the shape
        k    : int - Number of eigenfunctions to use. If None, uses the complete computed basis.

        Output
        -----------------------
        func : (n,p) or (n,) projected function
        """
        return self.unproject(self.project(func, k=k))

    def get_geodesic(self, dijkstra=False, robust=True, save=False,
                     force_compute=False, sym=False, batch_size=500, verbose=False):
        """
        Compute the geodesic distance matrix using either the Dijkstra algorithm or the Heat Method.
        Loads from cache if possible.

        Parameters
        -----------------
        dijkstra      : bool - If True, use Dijkstra algorithm instead of the
                        heat method
        robust        : Robust heat method
        save          : bool - If True, save the resulting distance matrix at
                        '{path}/geod_cache/{meshname}.npy' with 'path/meshname.{ext}' path of the
                        current mesh.
        force_compute : bool - If True, doesn't look for a cached distance matrix.
        sym           : Symmetrize the matrix if computed with heat method
        batch_size    : If robust is False, compute distances by batch

        Output
        -----------------
        distances : (n,n) matrix of geodesic distances
        """
        # Load cache if possible and not explicitly forbidden
        if not force_compute:
            geod_dist = self._get_geod_cache(verbose=verbose)
            if geod_dist is not None:
                return geod_dist

        # Else compute the complete matrix
        if dijkstra:
            geod_dist = geom.geodesic_distmat_dijkstra(self.vertlist, self.facelist)

        elif robust or self._intrinsic:
            geod_dist = geom.heat_geodmat_robust(self.vertlist, self.facelist, verbose=verbose)

        else:
            # Ensure LB matrices are processed.
            if self.A is None or self.W is None:
                self.process(k=0)
            if self._normals is None:
                self.compute_normals()

            # Set the time parameter as the squared mean edge length
            edges = self.edges
            v1 = self.vertlist[edges[:, 0]]
            v2 = self.vertlist[edges[:, 1]]
            t = np.linalg.norm(v2-v1, axis=1).mean()**2

            geod_dist = geom.heat_geodmat(self.vertlist, self.facelist, self.normals,
                                          self.A, self.W, t=t, batch_size=batch_size,
                                          verbose=verbose)

        if sym and not dijkstra:
            geod_dist *= .5
            geod_dist += geod_dist.T

        # Save the geodesic distance matrix if required
        if save:
            if self.path is None:
                raise ValueError('No path specified')

            root_dir = os.path.dirname(self.path)

            if self.is_normalized:
                geod_filename = os.path.join(root_dir, 'geod_cache', f'{self.meshname}_n.npy')
            elif self.is_modified:
                geod_filename = os.path.join(root_dir, 'geod_cache', f'{self.meshname}_mod.npy')
            else:
                geod_filename = os.path.join(root_dir, 'geod_cache', f'{self.meshname}.npy')

            os.makedirs(os.path.dirname(geod_filename), exist_ok=True)
            np.save(geod_filename, geod_dist)

        return geod_dist

    def geod_from(self, i, robust=True):
        """
        Compute geodesic distances from vertex i sing the Heat Method

        Parameters
        ----------------------
        i      : int - index from source
        robust : Robust heat method

        Output
        ----------------------
        dist : (n,) distances to vertex i
        """

        if robust or self._intrinsic:
            if self._solver_geod is None:
                self._solver_geod = pp3d.MeshHeatMethodDistanceSolver(self.vertlist, self.facelist)

            return self._solver_geod.compute_distance(i)

        if self.A is None or self.W is None:
            self.process(k=0)
        if self._normals is None:
            self.compute_normals()

        edges = self.edges
        v1 = self.vertlist[edges[:,0]]
        v2 = self.vertlist[edges[:,1]]
        t = np.linalg.norm(v2-v1, axis=1).mean()**2

        if self._solver_heat is None:
            solver_heat = sparse.linalg.factorized(self.A.tocsc() + t * self.W)
            solver_lap = sparse.linalg.factorized(self.W)
            self._solver_heat = solver_heat
            self._solver_lap = solver_lap

        # Compute distance with cached solvers
        dists = geom.heat_geodesic_from(i, self.vertlist, self.facelist, self.normals,
                                        self.A, W=None, t=t,
                                        solver_heat=solver_heat, solver_lap=solver_lap)

        return dists

    def l2_sqnorm(self, func):
        """
        Return the squared L2 norm of one or multiple functions on the mesh.
        For a single function f, this returns f.T @ A @ f with A the area matrix.

        Parameters
        -----------------
        func : (n,p) or (n,) functions on the mesh

        Returns
        -----------------
        sqnorm : (p,) array of squared l2 norms or a float only one function was provided.
        """
        return self.l2_inner(func, func)

    def l2_inner(self, func1, func2):
        """
        Return the L2 inner product of two functions, or pairwise inner products if lists
        of function is given.

        For two functions f1 and f2, this returns f1.T @ A @ f2 with A the area matrix.

        Parameters
        -----------------
        func1 : (n,p) or (n,) functions on the mesh
        func2 : (n,p) or (n,) functions on the mesh

        Returns
        -----------------
        sqnorm : (p,) array of L2 inner product or a float only one function per argument
                  was provided.
        """
        assert func1.shape == func2.shape, "Shapes must be equal"

        if func1.ndim == 1:
            return func1 @ self.A @ func2

        return np.einsum('np,np->p', func1, self.A@func2)

    def h1_sqnorm(self, func):
        """
        Return the squared H^1_0 norm (L2 norm of the gradient) of one or multiple functions
        on the mesh.
        For a single function f, this returns f.T @ W @ f with W the stiffness matrix.

        Parameters
        -----------------
        func : (n,p) or (n,) functions on the mesh

        Returns
        -----------------
        sqnorm : (p,) array of squared H1 norms or a float only one function was provided.
        """
        return self.h1_inner(func, func)

    def h1_inner(self, func1, func2):
        """
        Return the H1 inner product of two functions, or pairwise inner products if lists
        of function is given.

        For two functions f1 and f2, this returns f1.T @ W @ f2 with W the stiffness matrix.

        Parameters
        -----------------
        func1 : (n,p) or (n,) functions on the mesh
        func2 : (n,p) or (n,) functions on the mesh

        Returns
        -----------------
        sqnorm : (p,) array of H1 inner product or a float only one function per argument
                  was provided.
        """
        assert func1.shape == func2.shape, "Shapes must be equal"

        if func1.ndim == 1:
            return func1 @ self.W @ func2

        return np.einsum('np,np->p', func1, self.W@func2)

    def integrate(self, func):
        """
        Integrate a function or a set of function on the mesh

        Parameters
        -----------------
        func : (n,p) or (n,) functions on the mesh

        Returns
        -----------------
        integral : (p,) array of integrals or a float only one function was provided.
        """
        if func.ndim == 1:
            return np.sum(self.A @ func)
        return np.sum(self.A @ func, axis=0)

    def extract_fps(self, size, random_init=True, geodesic=True, no_load=False, verbose=False):
        """
        Samples points using farthest point sampling with geodesic distances. If the geodesic matrix
        is precomputed (in the cache folder) uses it, else computes geodesic distance in real time

        Parameters
        -------------------------
        size        : int - number of points to sample
        random_init : Whether to sample the first point randomly or to take the furthest away from
                      all the other ones. This is only done if the geodesic matrix is accessible
                      from cache
        geodesic    : bool - If True perform geodesic fps, else eucliden.
        no_load     : bool - if True never loads cache

        Output
        --------------------------
        fps : (size,) array of indices of sampled points
        """
        if not geodesic:
            def dist_func(i):
                return np.linalg.norm(self.vertlist - self.vertlist[i,None,:], axis=1)

            fps = geom.farthest_point_sampling_call(dist_func, size, n_points=self.n_vertices, verbose=verbose)

            return fps

        # Check if the geodesic matrix is accessible from cache
        A_geod = self._get_geod_cache() if not no_load else None

        if A_geod is None:
            # Set the time parameter as the squared mean edge length
            def geod_func(i):
                return self.geod_from(i)

            # Use the self.geod_from function as callable
            fps = geom.farthest_point_sampling_call(geod_func, size, n_points=self.n_vertices, verbose=verbose)

        else:
            fps = geom.farthest_point_sampling(A_geod, size, random_init=random_init, verbose=verbose)

        return fps

    def gradient(self, f, normalize=False):
        """
        computes the gradient of a function on f using linear
        interpolation between vertices.

        Parameters
        --------------------------
        f         : (n_v,) function value on each vertex
        normalize : bool - Whether the gradient should be normalized on each face

        Output
        --------------------------
        gradient : (n_f,3) gradient of f on each face
        """

        grad = geom.grad_f(f, self.vertlist, self.facelist, self.normals)  # (n_f,3)

        if normalize:
            grad /= np.linalg.norm(grad,axis=1,keepdims=True)

        return grad

    def divergence(self, f):
        """
        Computes the divergence of a vector field on the mesh

        Parameters
        --------------------------
        f         : (n_f, 3) vector value on each face

        Output
        --------------------------
        divergence : (n_v,) divergence of f on each vertex
        """
        div = geom.div_f(f, self.vertlist, self.facelist, self.normals)

        return div

    def orientation_op(self, gradf, normalize=False):
        """
        Compute the orientation operator associated to a gradient field gradf.

        For a given function g on the vertices, this operator linearly computes
        < grad(f) x grad(g), n> for each vertex by averaging along the adjacent faces.
        In practice, we compute < n x grad(f), grad(g) > for simpler computation.

        Parameters
        --------------------------
        gradf     : (n_f,3) gradient field on the mesh
        normalize : Whether to normalize the gradient on each face

        Output
        --------------------------
        operator : (n_v,n_v) orientation operator.
        """
        if normalize:
            gradf /= np.linalg.norm(gradf, axis=1, keepdims=True)

        operator = geom.get_orientation_op(gradf, self.vertlist, self.facelist, self.normals,
                                           self.vertex_areas)

        return operator

    def export(self, filename, precision=None):
        """
        Write the mesh in a .off file

        Parameters
        -----------------------------
        filename  : path to the file to write
        precision : floating point precision
        """
        # assert os.path.splitext(filename)[1] in ['.off',''], "Can only export .off files"
        file_ext = os.path.splitext(filename)[1]
        if file_ext == '':
            filename += '.off'
            file_ext = '.off'

        if file_ext == '.off':
            file_utils.write_off(filename, self.vertlist, self.facelist, precision=precision)

        elif file_ext == '.obj':
            file_utils.write_obj(filename, self.vertlist, self.facelist, precision=precision)

        return self

    def get_uv(self, ind1, ind2, mult_const, rotation=None):
        """
        Extracts UV coordinates for each vertices

        Parameters
        -----------------------------
        ind1       : int - column index to use as first coordinate
        ind2       : int - column index to use as second coordinate
        mult_const : float - number of time to repeat the pattern

        Output
        ------------------------------
        uv : (n,2) UV coordinates of each vertex
        """
        vert = self.vertlist if rotation is None else self.vertlist @ rotation.T
        return file_utils.get_uv(vert, ind1, ind2, mult_const=mult_const)

    def export_texture(self, filename, uv, mtl_file='material.mtl', texture_im='texture_1.jpg',
                       precision=None, verbose=False):
        """
        Write a .obj file with texture using uv coordinates

        Parameters
        ------------------------------
        filename   : str - path to the .obj file to write
        uv         : (n,2) uv coordinates of each vertex
        mtl_file   : str - name of the .mtl file
        texture_im : str - name of the .jpg file definig texture
        """
        if os.path.splitext(filename)[1] != '.obj':
            filename += '.obj'

        file_utils.write_obj(filename, self.vertlist, self.facelist, uv=uv,
                             mtl_file=mtl_file, texture_im=texture_im, verbose=verbose)

        return self

    def compute_normals(self):
        """
        Compute normal vectors for each face
        """
        self.normals = geom.compute_normals(self.vertlist, self.facelist)

    def set_vertex_normal_weighting(self, weight_type):
        """
        Set weighting type for vertex normals between 'area' and 'uniform'
        """
        weight_type = weight_type.lower()
        assert weight_type in ['uniform', 'area'], "Only implemented uniform and area weighting"

        if weight_type != self._vertex_normals_weighting:
            self._vertex_normals_weighting = weight_type
            self._vertex_normals = None

    def compute_vertex_normals(self):
        """
        computes vertex normals in self.vertex_normals
        """
        self.vertex_normals = geom.per_vertex_normal(self.vertlist, self.facelist,
                                                     weighting=self._vertex_normals_weighting)

    def compute_edges(self):
        """
        computes edges in self.edges
        """
        self._edges = geom.edges_from_faces(self.facelist)

    def _reset_vertex_attributes(self):
        """
        Resets attributes which depend on the vertex positions
        in the case of nonisometric deformation
        """
        self._face_areas = None

        self._normals = None
        self._vertex_normals = None

        self._intrinsic = False

        self.W = None
        self.A = None

        self.eigenvalues = None
        self.eigenvectors = None

        self._solver_heat = None
        self._solver_lap = None
        self._solver_geod = None

    def _get_geod_cache(self, verbose=False):
        # Check if the mesh has a stored path
        if self.path is None:
            return None

        root_dir = os.path.dirname(self.path)
        if self.is_normalized:
            geod_filename = os.path.join(root_dir, 'geod_cache', f'{self.meshname}_n.npy')

        elif self.is_modified:
            return None

        else:
            geod_filename = os.path.join(root_dir, 'geod_cache', f'{self.meshname}.npy')

        # Check if the geodesic matrix exists
        if os.path.isfile(geod_filename):
            if verbose:
                print('Loading Geodesic Distances from cache')
            return np.load(geod_filename)

        return None

    def _load_mesh(self, meshpath):
        """
        Load a mesh from a file

        Parameters:
        --------------------------
        meshpath : path to file
        """

        if os.path.splitext(meshpath)[1] == '.off':
            self.vertlist, self.facelist = file_utils.read_off(meshpath)
        elif os.path.splitext(meshpath)[1] == '.obj':
            self.vertlist, self.facelist = file_utils.read_obj(meshpath)

        else:
            raise ValueError('Provide file in .off or .obj format')

        self.path = meshpath
        self.meshname = os.path.splitext(os.path.basename(meshpath))[0]

    def _read_init_kwargs(self, kwargs):
        rotation = kwargs['rotation'] if 'rotation' in kwargs.keys() else None
        translation = kwargs['translation'] if 'translation' in kwargs.keys() else None
        area_normalize = kwargs['area_normalize'] if 'area_normalize' in kwargs.keys() else False
        center = kwargs['center'] if 'center' in kwargs.keys() else False
        return rotation, translation, area_normalize, center

    def _init_all_attributes(self):

        self.path = None
        self.meshname = None

        self._vertlist = None
        self._facelist = None

        self._modified = False
        self._normalized = False

        self._edges = None
        self._normals = None

        self._vertex_normals_weighting = 'area'
        self._vertex_normals = None

        self.W = None
        self.A = None
        self._intrinsic = False

        self._faces_areas = None

        self.eigenvalues = None
        self.eigenvectors = None

        self._solver_geod = None
        self._solver_heat = None
        self._solver_lap = None
