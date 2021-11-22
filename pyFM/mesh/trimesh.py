import os
import time

import numpy as np

from . import file_utils
from . import geometry as geom
from . import laplacian
import scipy.linalg
import scipy.sparse as sparse


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
    vertlist     : (n,3) array of n vertices coordinates
    facelist     : (m,3) array of m triangle indices
    normals      : (m,3) array of normals

    # SPECTRAL INFORMATION
    W            : (n,n) sparse cotangent weight matrix
    A            : (n,n) sparse area matrix (either diagonal or computed with finite elements)
    eigenvalues  : (K,) eigenvalues of the Laplace Beltrami Operator
    eigenvectors : (n,K) eigenvectors of the Laplace Beltrami Operator

    # GEODESIC INFORMATION
    t            : float - temperature parameter for geodesic computation with heat method
    solver_heat  : callable - given b, solves for x in (A + tW)x = b
    solver_lap   : callable - given b, solver for x in Wx = b

    Properties
    ------------------
    area         : float - area of the mesh
    n_vertices   : int - number of vertices
    n_faces      : int - number of faces
    edges        : (p,2) edges defined by vertex indices
    """
    def __init__(self, path=None, vertices=None, faces=None, area_normalize=False,
                 rotation=None, translation=None):
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

        self._vertlist = None
        self._facelist = None

        self._normals = None

        self.path = None
        self.meshname = None

        if vertices is None and faces is None:
            if path is None:
                raise ValueError("You should provide either a path to an .off file or \
                                  a list of vertices (and faces)")
            if os.path.splitext(path)[1] == '.off':
                self.vertlist, self.facelist = file_utils.read_off(path)
            elif os.path.splitext(path)[1] == '.obj':
                self.vertlist, self.facelist = file_utils.read_obj(path)

            self.path = path
            self.meshname = os.path.splitext(os.path.basename(path))[0]

        else:
            self.vertlist = np.asarray(vertices, dtype=float)
            self.facelist = np.asarray(faces, dtype=int) if faces is not None else None

        if rotation is not None:
            self.rotate(rotation)
        if translation is not None:
            self.translate(translation)

        self.normals = None
        self.W = None
        self.A = None

        self.eigenvalues = None
        self.eigenvectors = None

        self.t = None
        self.solver_heat = None
        self.solver_lap = None

        if area_normalize:
            new_meshname = None
            if self.meshname is not None:
                new_meshname = self.meshname + '_n'
                new_path = self.path
            tau = np.sqrt(self.area)
            self.vertlist /= tau

            if new_meshname is not None:
                self.path, self.meshname = new_path, new_meshname

    @property
    def vertlist(self):
        """
        Get or set the vertices.
        Checks the format when checking
        """
        return self._vertlist

    @vertlist.setter
    def vertlist(self, vertlist):
        if vertlist.ndim != 2:
            raise ValueError('Vertex list has to be 2D')
        elif vertlist.shape[1] != 3:
            raise ValueError('Vertex list requires 3D coordinates')

        self._reset_vertex_attributes()
        self.path = None
        self._vertlist = np.asarray(vertlist, dtype=float)

    @property
    def facelist(self):
        """
        Get or set the faces.
        Checks the format when checking
        """
        return self._facelist

    @facelist.setter
    def facelist(self, facelist):
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
            faces_areas = geom.compute_faces_areas(self.vertlist, self.facelist)
            return faces_areas.sum()

        return self.A.sum()

    @property
    def edges(self):
        """
        return a (p,2) array of edges defined by vertex indices.
        """
        return geom.edges_from_faces(self.facelist)

    @property
    def normals(self):
        if self._normals is None:
            self.compute_normals()
        return self._normals

    @normals.setter
    def normals(self, normals):
        self._normals = normals

    @property
    def vertex_normals(self):
        return geom.per_vertex_normal(self.vertlist, self.facelist, self.normals)

    @property
    def vertex_areas(self):
        if self.A is None:
            return geom.compute_vertex_areas(self.vertlist, self.facelist)

        return np.array(self.A.sum(1)).squeeze()

    @property
    def center_mass(self):
        return np.average(self.vertlist, axis=0, weights=self.vertex_areas)

    def rotate(self, R):
        if R.shape != (3, 3) or scipy.linalg.det(R) != 1:
            raise ValueError("Rotation should be a 3x3 matrix with unit determinant")

        self._vertlist = self.vertlist @ R.T
        if self._normals is not None:
            self.normals = self.normals @ R.T
        return self

    def translate(self, t):
        self._vertlist += t[None, :]
        return self

    def center(self):
        self.translate(self, -self.center_mass)
        return self

    def _reset_vertex_attributes(self):
        """
        Resets attributes which depend on the vertex positions
        in the case of nonisometric deformation
        """
        self._normals = None
        self.W = None
        self.A = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.t = None
        self.solver_heat = None
        self.solver_lap = None

    def _get_geod_cache(self, verbose=False):
        # Check if the mesh has a stored path
        if self.path is None:
            return None

        root_dir = os.path.dirname(self.path)
        geod_filename = os.path.join(root_dir, 'geod_cache', f'{self.meshname}.npy')

        # Check if the geodesic matrix exists
        if os.path.isfile(geod_filename):
            if verbose:
                print('Loading Geodesic Distances from cache')
            return np.load(geod_filename)

        return None

    def compute_normals(self):
        """
        Compute normal vectors for each face
        """
        self.normals = geom.compute_normals(self.vertlist, self.facelist)

    def laplacian_spectrum(self, k, fem_area=False, return_spectrum=True, verbose=False):
        """
        Compute the LB operator and its spectrum.
        Consider using the .process() function for easier use !

        Parameters
        -------------------------
        K               : int - number of eigenvalues to compute
        fem_area        : bool - Whether to compute the area matrix using finite element method
                          instead of the diagonal matrix.
        return_spectrum : bool - Whether to return the computed spectrum

        Output
        -------------------------
        eigenvalues, eigenvectors : Only if return_spectrum is True.
        """
        self.W = laplacian.cotangent_weights(self.vertlist, self.facelist)
        if fem_area:
            self.A = laplacian.fem_area_mat(self.vertlist, self.facelist)
        else:
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

    def process(self, k=200, fem_area=False, skip_normals=False, verbose=False):
        """
        Process the LB spectrum and saves it.
        Additionnaly computes per-face normals

        Parameters:
        -----------------------
        k            : int - (default = 200) Number of eigenvalues to compute
        fem_area     : bool - Whether to compute the area matrix using finite element method instead
                       of the diagonal matrix.
        skip_normals : bool - If set to True, skip normals computation
        """
        if not skip_normals and self.normals is None:
            self.compute_normals()

        if (self.eigenvectors is not None) and (self.eigenvalues is not None)\
           and (len(self.eigenvalues) >= k):
            self.eigenvectors = self.eigenvectors[:,:k]
            self.eigenvalues = self.eigenvalues[:k]

        else:
            self.laplacian_spectrum(k, return_spectrum=False, fem_area=fem_area, verbose=verbose)

        return self

    def project(self, func, k=None):
        """
        Project one or multiple functions on the LB basis

        Parameters
        -----------------------
        func : (n,p) or (n,) functions on the shape
        k    : dimension of the LB basis on which to project. If None use all the computed basis

        Output
        -----------------------
        projected_func : (k,p) or (k,) projected function
        """
        if k is None:
            return self.eigenvectors.T@self.A@func

        elif k <= self.eigenvectors.shape[1]:
            return self.eigenvectors[:,:k].T@self.A@func

        else:
            raise ValueError(f'At least {k} eigenvectors should be computed before projecting')

    def decode(self, projection):
        """
        Decode one or multiple functions from their LB basis projection

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

    def reconstruct(self, func, k=None):
        """
        Reconstruct function with the LB eigenbasis

        Parameters
        -----------------------
        func : (n,p) or (n,) - functions on the shape
        k    : int - Number of eigenfunctions to use. If None, uses the complete computed basis.

        Output
        -----------------------
        func : (n,p) or (n,) projected function
        """
        return self.decode(self.project(func, k=k))

    def get_geodesic(self, dijkstra=False, save=False, force_compute=False, batch_size=500,
                     verbose=False):
        """
        Compute the geodesic distance matrix using either the Dijkstra algorithm or the Heat Method.
        Loads from cache if possible.

        Parameters
        -----------------
        dijkstra      : bool - If True, use Dijkstra algorithm instead of the
                        heat method
        save          : bool - If True, save the resulting distance matrix at
                        '{path}/geod_cache/{meshname}.npy' with 'path/meshname.off' path of the
                        current mesh.
        force_compute : bool - If True, doesn't look for a cached distance matrix.

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
            geod_dist = geom.geodesic_distmat(self.vertlist, self.facelist)
        else:
            # Ensure LB matrices are processed.
            if self.A is None or self.W is None:
                self.process(k=0)
            if self.normals is None:
                self.compute_normals()

            # Set the time parameter as the squared mean edge length
            edges = self.edges
            v1 = self.vertlist[edges[:, 0]]
            v2 = self.vertlist[edges[:, 1]]
            t = np.linalg.norm(v2-v1, axis=1).mean()**2

            geod_dist = geom.heat_geodmat(self.vertlist, self.facelist, self.normals,
                                          self.A, self.W, t=t, batch_size=batch_size, verbose=verbose)

        # Save the geodesic distance matrix if required
        if save:
            if self.path is None:
                raise ValueError('No path specified')

            root_dir = os.path.dirname(self.path)
            geod_filename = os.path.join(root_dir, 'geod_cache', f'{self.meshname}.npy')

            os.makedirs(os.path.dirname(geod_filename), exist_ok=True)
            np.save(geod_filename, geod_dist)

        return geod_dist

    def geod_from(self, i, t=None, save=True):
        """
        Compute geodesic distances from vertex i sing the Heat Method

        Parameters
        ----------------------
        i    : int - index from source
        t    : float - time parameter. If not specified, uses the squared mean edge length
        save : bool - optional, if True, saves precfactorized linear systems for geodesic
               distance computation with heat method

        Output
        ----------------------
        dist : (n,) distances to vertex i
        """
        if self.A is None or self.W is None:
            self.process(k=0)
        if self.normals is None:
            self.compute_normals()

        # Compute temperature parameter
        if t is None:
            # Set the time parameter as the squared mean edge length
            edges = self.edges
            v1 = self.vertlist[edges[:,0]]
            v2 = self.vertlist[edges[:,1]]
            t = np.linalg.norm(v2-v1, axis=1).mean()**2

        # Useless to precompute or fetch cached solver in this case
        if (self.t is None or self.t != t) and not save:
            dists = geom.heat_geodesic_from(i, self.vertlist, self.facelist, self.normals,
                                            self.A, W=self.W, t=t)
            return dists

        # Load the cached solver if similar t
        elif self.t is not None and np.isclose(self.t, t):
            solver_heat = self.solver_heat
            solver_lap = self.solver_lap

        # Else compute and solver the solvers
        else:
            solver_heat = sparse.linalg.factorized(self.A.tocsc() + t * self.W)
            solver_lap = sparse.linalg.factorized(self.W)
            self.t = t
            self.solver_heat = solver_heat
            self.solver_lap = solver_lap

        # Compute distance with cached solvers
        dists = geom.heat_geodesic_from(i, self.vertlist, self.facelist, self.normals,
                                        self.A, W=None, t=t,
                                        solver_heat=solver_heat, solver_lap=solver_lap)

        return dists

    def l2_sqnorm(self, func):
        """
        Return the squared L2 norm of one or multiple functions on the mesh (area weighted).
        For a single function f, this returns f.T @ A @ f with A the area matrix.

        Parameters
        -----------------
        func : (n,p) or (n,) functions on the mesh

        Returns
        -----------------
        sqnorm : (p,) array of squared l2 norms or a float only one function was provided.
        """
        if len(func.shape) == 1:
            func = func[:,None]
            return np.einsum('np,np->p', func, self.A@func).flatten().item()

        return np.einsum('np,np->p', func, self.A@func).flatten()

    def extract_fps(self, size, random_init=True, geodesic=True, verbose=False):
        """
        Samples points using iterative farthest point sampling with geodesic or euclidean distances.
        If the geodesic matrix is precomputed (in the cache folder) uses it, else computes geodesic
        distances in real time

        Parameters
        -------------------------
        size        : int - number of points to sample
        random_init : Whether to sample the first point randomly or to take the furthest away from
                      all the other ones. This is only done if the geodesic matrix is accessible
                      from cache
        geodesic    : bool - whether to use geodesic distance or euclidean one.

        Output
        --------------------------
        fps : (size,) array of indices of sampled points
        """
        if not geodesic:
            def dist_func(i):
                return np.linalg.norm(self.vertlist - self.vertlist[i,None,:], axis=1)

            fps = geom.farthest_point_sampling_call(dist_func, size, n_points=self.n_vertices, verbose=verbose)

            return fps

        # Else
        # Check if the geodesic matrix is accessible from cache
        A_geod = self._get_geod_cache()
        # A_geod = self.get_geodesic()

        if A_geod is None:
            # Set the time parameter as the squared mean edge length
            edges = self.edges
            v1 = self.vertlist[edges[:,0]]
            v2 = self.vertlist[edges[:,1]]
            t = np.linalg.norm(v2-v1, axis=1).mean()**2

            def geod_func(i):
                return self.geod_from(i, t=t)

            # Use the self.geod_from function as callable
            fps = geom.farthest_point_sampling_call(geod_func, size, n_points=self.n_vertices, verbose=verbose)

        else:
            fps = geom.farthest_point_sampling(A_geod, size, random_init=random_init)

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

        per_vert_area = np.asarray(self.A.sum(1)).flatten()
        operator = geom.get_orientation_op(gradf, self.vertlist, self.facelist, self.normals,
                                           per_vert_area)

        return operator

    def export(self, filename, precision=6):
        """
        Write the mesh in a .off file

        Parameters
        -----------------------------
        filename : path to the file to write
        precision : floating point precision
        """
        if os.path.splitext(filename)[1] != '.off':
            filename += '.off'
        file_utils.write_off(filename, self.vertlist, self.facelist, precision=precision)
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

    def export_obj(self, filename, uv, mtl_file='material.mtl', texture_im='texture_1.jpg',
                   precision=6, verbose=False):
        """
        Write a .obj file with texture using uv coordinates

        Parameters
        ------------------------------
        filename   : str - path to the .obj file to write
        uv         : (n,2) uv coordinates of each vertex
        mtl_file   : str - name of the .mtl file
        texture_im : str - name of the .jpg file definig texture
        precision : floating point precision
        """
        if os.path.splitext(filename)[1] != '.obj':
            filename += '.obj'

        file_utils.write_obj(filename, self.vertlist, self.facelist, uv,
                             mtl_file=mtl_file, texture_im=texture_im,
                             precision=precision, verbose=verbose)

        return self
