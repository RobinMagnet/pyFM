import os

import numpy as np
from scipy import sparse

import pyFM.file_utils as file_utils
from pyFM.utils.fem_laplacian import fem_laplacian
import pyFM.utils.tools as tools

class TriMesh:
    """
    Mesh Class
    ________

    Attributes
    ------------------
    vertlist     : (n,3) array of n vertices coordinates
    facelist     : (m,3) array of m triangle indices
    W            : (n,n) sparse cotangent weight matrix (computed with finite elements)
    A            : (n,n) sparse area matrix (computed with finite elements)
    eigenvalues  : (K,) eigenvalues of the Laplace Beltrami Operator
    eigenvectors : (n,K) eigenvectors of the Laplace Beltrami Operator

    Properties
    ------------------
    area         : float - area of the mesh
    n_vertices   : int - number of vertices
    n_faces      : int - number of faces
    """
    def __init__(self,path = None,vertices = None, faces =None):
        """
        Read the mesh. Give either the path to a .off file
        or a list of vertices and corrresponding triangles

        Parameters
        ----------------------
        path     : path to a .off file
        vertices : (n,3) vertices coordinates
        faces    : (m,3) list of indices of triangles
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

        self.W = None
        self.A = None

        self.eigenvalues = None
        self.eigenvectors = None



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
        returns the area of the mesh.
        Compute matrix A if necessary !
        """
        if self.A is None:
            self.process(K=0,verbose=False)
    
        return self.A.sum()
            
    def laplacian_spectrum(self,K,return_spectrum=True,saving=True, verbose=False):
        """
        Compute the LB operator and its spectrum. Consider using the process function for usual use !

        Parameters
        -------------------------
        K : int - number of eigenvalues to compute
        return_spectrum : bool - Whether to return the computed spectrum
        saving          : bool - Whether to save the computed spectrum in the object (for memory usage)

        Output
        -------------------------
        eigenvalues, eigenvectors : Only if return_spectrum is True.
        """

        eigenvalues, eigenvectors,W,A = fem_laplacian(self.vertlist, self.facelist, spectrum_size=max(K,1), normalization=None,verbose=verbose)

        if saving:
            self.eigenvalues, self.eigenvectors,self.W,self.A = eigenvalues, eigenvectors, W, A
        
        if return_spectrum:
            return eigenvalues,eigenvectors

        return self

    def process(self,k=200,verbose=False):
        """
        Process the LB spectrum and saves it.

        Parameters:
        -----------------------
        k : int - (default = 200) Number of eigenvalues to compute
        """
        if (self.eigenvectors is not None) and (self.eigenvalues is not None)\
           and (len(self.eigenvectors) > k) and (len(self.eigenvectors) == len(self.eigenvalues)):
            self.eigenvectors = self.eigenvectors[:,:k]
            self.eigenvalues = self.eigenvalues[:k]

        else:
            self.laplacian_spectrum(k,return_spectrum=False,saving=True,verbose=verbose)

        return self

    def project(self,func,k=None):
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

    def decode(self,projection):
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

    def reconstruct(self,func,k=None):
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
        return self.decode(self.project(func,k=k))

    def get_geodesic(self, save=False):
        """
        Compute the geodesic distances using the Dijkstra algorithm.
        Loads from cache if possible

        Output
        -----------------
        distances : (n,n) matrix of geodesic distances
        """

        if self.path is not None:
            root_dir,filename = os.path.split(self.path)
            meshname = os.path.splitext(filename)[0]
            geod_filename = os.path.join(root_dir,'geod_cache',f'{meshname}.npy')
            if os.path.isfile(geod_filename):
                return np.load(geod_filename)

        edges = self.get_edges()
        
        I = edges[:,0]
        J = edges[:,1]
        V = np.linalg.norm(self.vertlist[J]-self.vertlist[I],axis=1)
        distmat= sparse.coo_matrix((V,(I,J)),shape=(self.n_vertices,self.n_vertices))
        
        graph = (distmat + distmat.T).tocsr()

        geod_dist = sparse.csgraph.dijkstra(graph)

        if save:
            if self.path is None:
                raise ValueError('No path specified')
            root_dir,filename = os.path.split(self.path)
            meshname = os.path.splitext(filename)[0]
            geod_filename = os.path.join(root_dir,'geod_cache',f'{meshname}.npy')

            os.makedirs(os.path.dirname(geod_filename),exist_ok=True)
            np.save(geod_filename,geod_dist)

        return  geod_dist

    def get_edges(self):
        """
        Compute a list of all edges

        Output
        --------------------------
        edges : (p,) array of all edges
        """
        edge_list = set()
        for (i,j,k) in self.facelist:

            edge_list.add((min(i,j),max(i,j)))
            edge_list.add((min(j,k),max(j,k)))
            edge_list.add((min(k,i),max(k,i)))

        edges = np.array(list(edge_list))
        return edges


    def l2_sqnorm(self,func):
        """
        Return the squared L2 norm of a function on the mesh (area weighted).

        Parameters
        -----------------
        func : (n,p) or (n,) functions on the mesh

        Returns
        -----------------
        sqnorm : (p,) or float
        """
        if len(func.shape)==1:
            func = func[:,None]
            return np.einsum('np,np->p', func, self.A@func).flatten().item()

        return np.einsum('np,np->p', func, self.A@func).flatten()

    def extract_fps(self,size):
        A_geod = self.get_geodesic()

        fps = tools.farthest_point(A_geod,size,init='farthest')
        return fps




    def export(self,filename):
        """
        Write the mesh in a .off file

        Parameters
        -----------------------------
        filename : path to the file to write

        """
        assert os.path.splitext(filename)[1] == '.off', "Can only export .off files"
        file_utils.write_off(filename, self.vertlist, self.facelist)
        return self
