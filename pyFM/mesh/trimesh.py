import os

import numpy as np

from . import file_utils
from . import geometry as geom
from . import laplacian


class TriMesh:
    """
    Mesh Class
    ________

    Attributes
    ------------------
    vertlist     : (n,3) array of n vertices coordinates
    facelist     : (m,3) array of m triangle indices
    normals      : (m,3) array of normals
    W            : (n,n) sparse cotangent weight matrix
    A            : (n,n) sparse area matrix (either diagonal or computed with finite elements)
    eigenvalues  : (K,) eigenvalues of the Laplace Beltrami Operator
    eigenvectors : (n,K) eigenvectors of the Laplace Beltrami Operator

    Properties
    ------------------
    area         : float - area of the mesh
    n_vertices   : int - number of vertices
    n_faces      : int - number of faces
    edges        : (p,2) edges defined by vertex indices
    """
    def __init__(self, path=None, vertices=None, faces=None, area_normalize=False):
        """
        Read the mesh. Give either the path to a .off file
        or a list of vertices and corrresponding triangles

        Parameters
        ----------------------
        path           : path to a .off file
        vertices       : (n,3) vertices coordinates
        faces          : (m,3) list of indices of triangles
        area_normalize :
        """
        self.path = path

        self._vertlist = None
        self._facelist = None

        if vertices is None and faces is None:
            assert path is not None, "You should provide either a path to an .off file or a list of vertices and faces"
            self.vertlist,self.facelist = file_utils.read_off(path)

        else:
            self.vertlist = np.asarray(vertices)
            self.facelist = np.asarray(faces)

        self.normals = None
        self.W = None
        self.A = None

        self.eigenvalues = None
        self.eigenvectors = None

        if area_normalize:
            tau = np.sqrt(self.area)
            self.vertlist /= tau

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
        self._vertlist = np.asarray(vertlist)

    @property
    def facelist(self):
        """
        Get or set the faces.
        Checks the format when checking
        """
        return self._facelist

    @facelist.setter
    def facelist(self, facelist):
        if facelist.ndim != 2:
            raise ValueError('Faces list has to be 2D')
        elif facelist.shape[1] != 3:
            raise ValueError('Each face is made of 3 points')

        self._facelist = np.asarray(facelist)

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
        return self.facelist.shape[0]

    @property
    def area(self):
        """
        Returns the area of the mesh
        """
        if self.A is None:
            faces_areas = geom.compute_faces_areas(self.vertlist,self.facelist)
            return faces_areas.sum()

        return self.A.sum()

    @property
    def edges(self):
        """
        return a (p,2) array of edges defined by vertex indices.
        """
        return geom.edges_from_faces(self.facelist)

    def compute_normals(self):
        """
        Compute normal vectors for each face
        """
        self.normals = geom.compute_normals(self.vertlist,self.facelist)

    def laplacian_spectrum(self, k, fem_area=True, return_spectrum=True, verbose=False):
        """
        Compute the LB operator and its spectrum.
        Consider using the .process() function for easier use !

        Parameters
        -------------------------
        K               : int - number of eigenvalues to compute
        fem_area        : bool - Whether to compute the area matrix using finite element method instead
                          of the diagonal matrix.
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
            self.eigenvalues, self.eigenvectors = laplacian.laplacian_spectrum(self.W, self.A,
                                                                               spectrum_size=max(k, 1))

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
           and (len(self.eigenvalues) > k):
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

    def get_geodesic(self, dijkstra=False, save=False, force_compute=False):
        """
        Compute the geodesic distances using either the Dijkstra algorithm
        or the Heat Method.
        Loads from cache if possible

        Parameters
        -----------------
        dijkstra      : bool - If True, use Dijkstra algorithm instead of the
                        heat method
        save          : bool - If True, save the resulting distance matrix at
                        '{path}/geod_cache/{meshname}.npy' with 'path/meshname.off'
                        being the current mesh.
        force_compute : bool - If True, doesn't look for a cached distance matrix.

        Output
        -----------------
        distances : (n,n) matrix of geodesic distances
        """
        # Load cache if possible
        if not force_compute and self.path is not None:
            root_dir,filename = os.path.split(self.path)
            meshname = os.path.splitext(filename)[0]
            geod_filename = os.path.join(root_dir,'geod_cache',f'{meshname}.npy')
            if os.path.isfile(geod_filename):
                return np.load(geod_filename)

        if dijkstra:
            geod_dist = geom.geodesic_distmat(self.vertlist, self.facelist)
        else:
            if self.A is None or self.W is None:
                self.process(k=0)
            if self.normals is None:
                self.compute_normals()

            # Set the time parameter as the squared mean edge length
            edges = self.edges
            v1 = self.vertlist[edges[:,0]]
            v2 = self.vertlist[edges[:,1]]
            t = np.linalg.norm(v2-v1).mean()**2

            geod_dist = geom.heat_geodmat(self.vertlist, self.facelist, self.normals,
                                          self.A, self.W, t=t)

        if save:
            if self.path is None:
                raise ValueError('No path specified')
            root_dir, filename = os.path.split(self.path)
            meshname = os.path.splitext(filename)[0]
            geod_filename = os.path.join(root_dir, 'geod_cache', f'{meshname}.npy')

            os.makedirs(os.path.dirname(geod_filename), exist_ok=True)
            np.save(geod_filename,geod_dist)

        return geod_dist

    def geod_from(self, i, t=None):
        """
        Compute geodesic distances from vertex i sing the Heat Method

        Parameters
        ----------------------
        i : int - index from source
        t : float - time parameter. If not specified, uses the squared
            mean edge length

        Output
        ----------------------
        dist : (n,) distances to vertex i
        """
        if self.A is None or self.W is None:
            self.process(k=0)
        if self.normals is None:
            self.compute_normals()

        # Set the time parameter as the squared mean edge length
        edges = self.edges
        v1 = self.vertlist[edges[:,0]]
        v2 = self.vertlist[edges[:,1]]
        t = np.linalg.norm(v2-v1).mean()**2

        dists = geom.heat_geodesic_from(i, self.vertlist, self.facelist, self.normals,
                                        self.A, self.W, t=t)

        return dists

    def l2_sqnorm(self, func):
        """
        Return the squared L2 norm of a function on the mesh (area weighted).

        Parameters
        -----------------
        func : (n,p) or (n,) functions on the mesh

        Returns
        -----------------
        sqnorm : (p,) or float
        """
        if len(func.shape) == 1:
            func = func[:,None]
            return np.einsum('np,np->p', func, self.A@func).flatten().item()

        return np.einsum('np,np->p', func, self.A@func).flatten()

    def extract_fps(self, size, random_init=True):
        """
        Samples points using farthest point sampling using geodesic distances

        Parameters
        -------------------------
        size        : int - number of points to sample
        random_init : Whether to sample the first point randomly or to
                      take the furthest away from all the other ones

        Output
        --------------------------
        fps : (size,) array of indices of sampled points
        """
        A_geod = self.get_geodesic()

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
        f         : (n_f, 3) vector0 value on each face

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
            gradf /= np.linalg.norm(gradf,axis=1,keepdims=True)

        per_vert_area = np.asarray(self.A.sum(1)).flatten()
        operator = geom.get_orientation_op(gradf, self.vertlist, self.facelist, self.normals,
                                           per_vert_area)

        return operator

    def export(self, filename):
        """
        Write the mesh in a .off file

        Parameters
        -----------------------------
        filename : path to the file to write

        """
        assert os.path.splitext(filename)[1] == '.off', "Can only export .off files"
        file_utils.write_off(filename, self.vertlist, self.facelist)
        return self

    def get_uv(self, ind1, ind2, mult_const):
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
        return file_utils.get_uv(self.vertlist, ind1, ind2, mult_const=mult_const)

    def export_obj(self,filename, uv, mtl_file='material.mtl', texture_im='texture_1.jpg', verbose=False):
        """
        Write a .obj file with texture using uv coordinates

        Parameters
        ------------------------------
        filename   : str - path to the .obj file to write
        uv         : (n,2) uv coordinates of each vertex
        mtl_file   : str - name of the .mtl file
        texture_im : str - name of the .jpg file definig texture
        """

        file_utils.write_obj(filename, self.vertlist, self.facelist, uv,
                             mtl_file=mtl_file, texture_im=texture_im, verbose=verbose)

        return self
