import copy
from collections import defaultdict
import time

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
import scipy.linalg
from scipy.optimize import linprog

import pyFM.spectral as spectral

from tqdm import tqdm
from sklearn.neighbors import KDTree, NearestNeighbors

try:
    import pynndescent
    index = pynndescent.NNDescent(np.random.random((100, 3)), n_jobs=2)
    del index
    ANN = True
except ImportError:
    ANN = False


class FMN:
    def __init__(self, meshlist, maps_dict=None):
        # Mesh of each Node
        self.meshlist = copy.deepcopy(meshlist)  # List of n TriMesh

        # Edges are determined by (i,j) pair of indices
        # A map is associated to each edge (via dictionnary)
        # Weights of edges are stored in a sparse (n,n) matrix
        # For computation, an arbitraty ordering of edges is stored.

        # Network attribute
        self.edges = None  # List of couples (i,j)
        self.maps = None  # Dictionnary of maps
        self.weights = None  # (n,n) sparse matrix of weights
        self.edge2ind = None  # Ordering of edges

        # (n,K) array of indices of K vertices per shape in the network.
        self.subsample = None

        # ISCM weights attributes
        self.cycles = None  # List of 3-cycles (i,j,k)
        self.A = None  # (n_cycle, n_edges) binary matrix (1 if edge j in cycle i)
        self.A_sub = None  # (n_edge_in_cycle,) indices of edges contained in a 3-cycle

        self.use_iscm = False  # Whether ISCM or adjacency weights are used.
        self.cycle_weight = None  # Weights of each 3-cycle (map-dependant)
        self.edge_weights = None  # Weight of each edge (map-dependant)

        # CLB and CCLB attributes
        self.W = None  # (n*M, n*M) sparse matrix. Quadratic form for CLB computation.
        self.CLB = None  # (n,M,M) array of Consistent Latent Basis for each mesh.
        self.CCLB = None  # (n,M,m) array of Canonical Consistent Latent Basis for each mesh
        self.cclb_eigenvalues = None  # (m,) eigenvalues of the CCLB

        # Extra information
        self.p2p = None  # Dictionnary of pointwise maps associated to each edge
        self._M = None

        if maps_dict is not None:
            self.set_maps(maps_dict=maps_dict, verbose=True)

    @property
    def n_meshes(self):
        return len(self.meshlist)

    @property
    def M(self):
        """
        Return the current shared dimension for functional maps
        (which are square matrices).

        If not specified, returns the sized of the first found map.
        """
        if self._M is not None:
            return self._M
        else:
            return self.maps[self.edges[0]].shape[0]

    @M.setter
    def M(self, M):
        self._M = M

    @property
    def m_cclb(self):
        """
        Return the dimension of the Canonical Consistent Latent Basis
        """
        return self.CCLB.shape[2]

    def _reset_map_attributes(self):
        """
        Resets all attributes depending on the Functional Maps
        """
        # Resets ISCM weights variables
        if self.use_iscm:
            self.use_iscm = False  # Whether ISCM or adjacency weights are used.
            self.cycle_weight = None  # Weights of each 3-cycle (map-dependant)
            self.edge_weights = None  # Weight of each edge (map-dependant)
            self.weights = None  # (n,n) sparse matrix of weights

        # Reset map-dependant attributes
        self.W = None  # (n*M, n*M) sparse matrix. Quadratic form for CLB computation.
        self.CLB = None  # (n,M,M) array containing the Consistent Latent Basis for each mesh.
        self.CCLB = None  # (n,M,m) array of Canonical Consistent Latent Basis for each mesh
        self.cclb_eigenvalues = None  # (m,) eigenvalues of the CCLB
        self.p2p = None  # Dictionnary of pointwise

    def set_maps(self, maps_dict, verbose=False):
        """
        Set the edges of the graph with maps.
        Saves extra information about the edges.

        Parameters
        --------------------------
        maps_dict : dict - dictionnary, key (i,j) gives functional map FM
                    between mesh i and j.
                    FM can be of different size depending on the edge
        """
        self.maps = copy.deepcopy(maps_dict)

        # Sort edges for later faster optimization
        self.edges = sorted(list(maps_dict.keys()))

        self.edge2ind = dict()
        for edge_ind, edge in enumerate(self.edges):
            self.edge2ind[edge] = edge_ind

        if verbose:
            print(f'Setting {len(self.edges)} edges on {self.n_meshes} nodes.')

        return self

    def set_subsample(self, subsample):
        """
        Set subsamples an all shapes in the network

        Parameters
        -----------------------------------
        subsample : (n, size)
        """
        self.subsample = subsample

        return self

    def compute_subsample(self, size=1000, geodesic=False, verbose=False):
        """
        Subsample vertices on each shape using farthest point sampling.
        Store in an (n,size) array of indices

        Parameters
        ---------------------------------
        size : int - number of vertices to subsample on each shape
        """
        if verbose:
            print(f'Computing a {size}-sized subsample for each mesh')
        self.subsample = np.zeros((self.n_meshes, size), dtype=int)
        for i in range(self.n_meshes):
            self.subsample[i] = self.meshlist[i].extract_fps(size, geodesic=geodesic, random_init=False)

    def set_weights(self, weights=None, weight_type='iscm', verbose=False):
        """
        Set weights for each edge in the graph

        Parameters
        -------------------------
        weights     : sparse (n,n) matrix.
                      If not specified, sets weights according to 'weight_type' argument
        weight_type : 'iscm' | 'adjacency' : if 'weights' is not specified, computes weights
                      according to the Consistent Zoomout adaptation of iscm or using the adjacency
                      matrix of the graph.
        """
        if weights is not None:
            self.use_iscm = False
            self.weights = copy.deepcopy(weights)

        elif weight_type == 'iscm':
            self.use_iscm = True

            # Process cycles if necessary
            if self.cycles is None:
                if verbose:
                    print("Computing cycle information")
                self.extract_3_cycles()
                self.compute_Amat()

            # Compute original ISCM weights d_ij for each edge (i,j)
            # Final weight is set to exp(-d_ij^2/(2*sigma^2))
            # With sigma = median(d_ij)
            weight_arr = self.optimize_iscm(verbose=verbose)  # (n_edges,)
            median_val = np.median(weight_arr[self.A_sub])
            if np.isclose(median_val, 0, atol=1e-4):
                weight_arr /= np.mean(weight_arr[self.A_sub])
            else:
                weight_arr /= median_val
            new_w = np.exp(-np.square(weight_arr)/2)  # (n_edges,)

            I = [x[0] for x in self.edges]
            J = [x[1] for x in self.edges]
            self.weights = sparse.csr_matrix((new_w, (I, J)), shape=(self.n_meshes, self.n_meshes))

        elif weight_type == 'adjacency':
            self.use_iscm = False
            I = [x[0] for x in self.edges]
            J = [x[1] for x in self.edges]
            V = [1 for x in range(len(self.edges))]
            self.weights = sparse.csr_matrix((V, (I, J)),shape=(self.n_meshes, self.n_meshes))

        else:
            raise ValueError(f'"weight_type" should be "iscm" or "adjacency, not {weight_type}')

        return self

    def set_isometries(self, M=None):
        """
        For each edge (i,j), if (j,i) is also an edge then,
        the corresponding functional maps are set as transpose of each other
        chosing the closest to orthogonal of both.

        Since this modifies the maps, ISCM weights are deleted

        Parameters
        -----------------------
        M : int - dimension with wich to compare the functional maps.
            If None, uses the current self.M
        """
        # Dictionnary with False as a default value for any key
        visited = defaultdict(bool)

        if M is None:
            M = self.M

        for (i, j) in self.edges:
            if not visited[(i, j)] and (j, i) in self.edges:
                FM1 = self.maps[(i, j)][:M, :M]
                FM2 = self.maps[(j, i)][:M, :M]

                dist1 = np.linalg.norm(FM1.T @ FM1 - np.eye(FM1.shape[1]))
                dist2 = np.linalg.norm(FM2.T @ FM2 - np.eye(FM2.shape[1]))

                if dist1 <= dist2:
                    self.maps[(j, i)] = np.transpose(self.maps[(i, j)])
                else:
                    self.maps[(i, j)] = np.transpose(self.maps[(j, i)])

                visited[(j,i)] = True

        # Reset all map-dependant attributes
        self._reset_map_attributes()

    def compute_W(self, M=None, verbose=False):
        """
        Computes the quadratic form for Consistent Latent Basis (CLB) computation.

        Parameters
        ---------------------------
        M : int - (optional) size of the functional maps to use,
            uses projection of FM on this dimension.
            If not specified, used the size of the first found functional map
        """
        if self.maps is None:
            raise ValueError('Functional maps should be set')

        if self.weights is None:
            self.set_weights(verbose=verbose)

        if M is not None:
            self.M = M

        self.W = CLB_quad_form(self.maps, self.weights, M=self.M)

    def compute_CLB(self, equals_id=False, verbose=False):
        """
        Computes the Consistent Latent Basis CLB using the quadratic form
        associated to the problem.
        The first M vectors for each basis are computed in order.

        Parameters
        --------------------------
        equals_id : If False, the sum of Y.T@Y are expected to give n*Id.
                    If True,  the sum of Y.T@Y are expected to give Id.
        """
        if self.W is None:
            self.compute_W(verbose=verbose)

        # W is a real symmetric matrix !
        # There is a bug in sparse eigenvalues computation where 'LM' returns the smallest
        # eigenvalues whereas 'SM' does not.
        if verbose:
            print(f'Computing {self.M} CLB eigenvectors...')
            start_time = time.time()
        if equals_id:
            # Returns (n*M,), (n*M,M) array

            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(self.W, k=self.M,
                                                                  which='LM', sigma=-1e-6)
        else:
            # Returns (n*M,), (n*M,M) array
            M_mat = 1/self.n_meshes * scipy.sparse.eye(self.W.shape[0])
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(self.W, M=M_mat, k=self.M,
                                                                  which='LM', sigma=-1e-6)

        if verbose:
            print(f'\tDone in {time.time() - start_time:.1f}s')
        # In any case, make sure they are real and sorted.
        # eigenvalues = np.real(eigenvalues)
        # sorting = np.argsort(eigenvalues)
        # eigenvalues = eigenvalues[sorting]
        # eigenvectors = np.real(eigenvectors)[:,sorting] # NM,M
        eigenvalues[0] = 0

        self.CLB = eigenvectors.reshape((self.n_meshes, self.M, self.M))  # (n,M,M)

    def compute_CCLB(self, m, verbose=True):
        """
        Compute the Canonical Consistent Latent Basis CCLB from the CLB.

        Parameters:
        ------------------------------
        m : int - size of the CCLB to compute.
        """
        if self.CLB is None:
            self.compute_CLB(verbose=verbose)

        # Matrix E from Algorithm 1 in the Limit Shape paper
        E_mat = np.zeros((m, m))

        for i in range(self.n_meshes):
            Y = self.CLB[i, :, :m]  # (M,m)
            evals = self.meshlist[i].eigenvalues[:self.M]  # (M,)
            E_mat += Y.T @ (evals[:, None] * Y)  # (m,m)

        # Compute the eigendecomposition of E
        b = self.n_meshes * np.eye(E_mat.shape[0])
        eigenvalues, eigenvectors = scipy.linalg.eig(E_mat, b=b)  # (m,), (m,m)

        eigenvalues = np.real(eigenvalues)  # (m,)
        sorting = np.argsort(eigenvalues)  # (m,)
        eigenvalues = eigenvalues[sorting]  # (m,)
        eigenvectors = np.real(eigenvectors)[:, sorting]  # (m,m)

        # CCLB is stored as an (n,M,m) array
        self.cclb_eigenvalues = eigenvalues  # (m,)
        self.CCLB = np.array([self.CLB[i, :, :m] @ eigenvectors for i in range(self.n_meshes)])

        return self

    def get_CSD(self, i):
        """
        Returns the Characterisic Shape Difference operators CSD for mesh i

        Parameters
        --------------------------
        i : int - index of the mesh on which to returns the two CSD

        Output
        --------------------------
        CSD_a,CSD_c - (m,m), (m,m) array of area and conformal CSD expressed in the Latent Space
        """
        # Functional map from the Limit Shape to shape i
        FM = self.CCLB[i]

        CSD_a = FM.T@FM
        CSD_c = np.linalg.pinv(np.diag(self.cclb_eigenvalues)) @ FM.T @ (self.meshlist[i].eigenvalues[:self.M,None]*FM)

        return CSD_a, CSD_c

    def get_LB(self, i, complete=True):
        """
        Returns the latent basis LB for mesh i

        Parameters
        --------------------------
        i        : int - index of the mesh on which to returns the LB
        complete : bool - If False, only computes values on the self.subsample[i] vertices

        Output
        --------------------------
        latent_basis - (n_i,m) latent basis on mesh i
        """
        cclb = self.CCLB[i]  # / np.linalg.norm(self.CCLB[i],axis=0,keepdims=True)  # (M,m)
        if not complete and self.subsample is not None:
            latent_basis = self.meshlist[i].eigenvectors[self.subsample[i], :self.M] @ cclb
            return latent_basis  # (n_i',m)

        latent_basis = self.meshlist[i].eigenvectors[:, :self.M] @ cclb  # (N_i,m)
        return latent_basis

    def compute_p2p(self, complete=True, use_ANN=False, n_jobs=1):
        """
        Computes vertex to vertex maps for each (directed) edge using the factorization of
        functional maps CCLB. Only maps related to existing edges are computed.
        Vertex to vertex maps are saved in a dictionnary the same way as functional maps,
        although their direction are reversed.

        Parameters
        --------------------------
        complete : If False, uses self.subsample to obtain pointwise maps between
                   subsamples of vertices for each shape
        use_ANN  : If True, uses pynndescent to compute approximate nearest neighbors between shapes
        """
        if use_ANN and not ANN:
            raise ValueError('Please install pydescent to achieve Approximate Nearest Neighbor')

        self.p2p = dict()
        curr_vind = -1
        for (i, j) in self.edges:

            if i != curr_vind:
                curr_v = i
                LB_1 = self.get_LB(curr_v, complete=False)  # (n_1',m)

                if use_ANN:
                    index = pynndescent.NNDescent(LB_1, n_jobs=n_jobs)  # (n_2',m)
                else:
                    tree = NearestNeighbors(n_neighbors=1, leaf_size=40, algorithm="kd_tree", n_jobs=n_jobs)
                    _ = tree.fit(LB_1)

            # LB_1 = self.get_LB(i, complete=complete)  # (n_1',m)
            LB_2 = self.get_LB(j, complete=complete)  # (n_2',m)

            if use_ANN:
                p2p,_ = index.query(LB_2, k=1)
            else:
                t_, p2p = tree.kneighbors(LB_2)

            p2p = p2p.flatten()

            self.p2p[(i, j)] = p2p  # (n_2',)

    def compute_maps(self, M, complete=True):
        """
        Convert pointwise maps into Functional Maps of size M.

        Parameters
        ------------------------
        M : int - size of the functional map to compute
        """
        self.M = M
        for (i, j) in self.edges:
            if not complete and self.subsample is not None:
                sub = (self.subsample[i], self.subsample[j])
            else:
                sub = None

            FM = spectral.mesh_p2p_to_FM(self.p2p[(i, j)], self.meshlist[i], self.meshlist[j],
                                         dims=M, subsample=sub)
            self.maps[(i, j)] = FM

        # Reset map-dependant variables
        self._reset_map_attributes()

    def extract_3_cycles(self):
        """
        Extract all 3-cycles from the graph in a list of 3-uple (i,j,k)
        """
        self.cycles = []

        # Ugly triple for loop, but only has to be run once
        # Saves cycles (i,j,k) with either i<j<k or i>j>k
        for i in range(self.n_meshes):
            for j in range(i):
                for k in range(j):
                    if (i, j) in self.edges and (j, k) in self.edges and (k, i) in self.edges:
                        self.cycles.append((i, j, k))

            for j in range(i+1,self.n_meshes):
                for k in range(j+1,self.n_meshes):
                    if (i, j) in self.edges and (j, k) in self.edges and (k, i) in self.edges:
                        self.cycles.append(tuple((i, j, k)))

    def compute_Amat(self):
        """
        Compute matrix A for iscm weights optimization.  Binary matrix telling which edge
        belongs to which cycle.
        Uses the arbitraty edge ordering createede in the self.set_maps method
        """
        self.A = np.zeros((len(self.cycles), len(self.edges)))  # (n_cycles, n_edges)

        for cycle_ind, (i, j, k) in enumerate(self.cycles):
            self.A[cycle_ind, self.edge2ind[(i, j)]] = 1
            self.A[cycle_ind, self.edge2ind[(j, k)]] = 1
            self.A[cycle_ind, self.edge2ind[(k, i)]] = 1

        self.A_sub = np.where(self.A.sum(0) > 0)[0]  # (n_edges_in_cycle)

    def compute_3cycle_weights(self, M=None):
        """
        Compute per-cycle costs and per-edge costs for ISCM optimization.
        Cycle weights are given by the self.get_cycle_weight method (deviation from Id map)
        Edge weight is the inverse of the sum of all weights of the cycles the edge belongs to.

        Parameters :
        -----------------------
        M : Dimension of functional maps to use. If None, uses self.M
        """
        if M is None:
            M = self.M

        self.cycle_weight = np.zeros(len(self.cycles))
        for cycle_ind, cycle in enumerate(self.cycles):
            self.cycle_weight[cycle_ind] = self.get_cycle_weight(cycle,M=M)  # n_cycles

        self.edge_weights = np.zeros(len(self.edges))
        self.edge_weights[self.A_sub] = 1/(self.A[:, self.A_sub]*self.cycle_weight[:, None]).sum(0)

    def optimize_iscm(self, verbose=False):
        """
        Solves the linear problem for ISCM weights computation
        min w.T @ x
        s.t.  A@x >= C_gamma and x >= 0

        Edges which are not part of a cycle are given 0-weigths

        Output
        ------------------------
        opt_weights : (n_edges,) (positive) weights for each edge.
        """
        self.compute_3cycle_weights(M=self.M)

        if verbose:
            print('Optimizing Cycle Weights...')
            start_time = time.time()
        # Solve Linear Program
        res = linprog(self.edge_weights, A_ub=-self.A, b_ub=-self.cycle_weight, bounds=(0, float("inf")), method='highs-ds')

        if verbose:
            print(f'\tDone in {time.time() - start_time:.5f}s')
        opt_weights = np.zeros(len(self.edges))  # (n_edges,)
        opt_weights[self.A_sub] = res.x[self.A_sub]

        return opt_weights

    def get_cycle_weight(self, cycle, M=None):
        """
        Given a cycle (i,j,k), compute its cost using the functional maps.
        Cost is given as the maximum deviation to the identity map when
        going through the complete cycle (3 possibilities)

        Parameters
        -----------------------
        cycle : 3-uple with node indices creating a cycle
        M     : Dimension of functional maps to use. If None use self.M

        Output
        -----------------------
        cost : cost of the cycle
        """
        if M is None:
            M = self.M

        (i, j, k) = cycle

        Cij = self.maps[(i, j)][:M, :M]
        Cjk = self.maps[(j, k)][:M, :M]
        Cki = self.maps[(k, i)][:M, :M]

        Cii = Cij@Cjk@Cki
        Cjj = Cjk@Cki@Cij
        Ckk = Cki@Cij@Cjk

        costi = np.linalg.norm(Cii - np.eye(M))
        costj = np.linalg.norm(Cjj - np.eye(M))
        costk = np.linalg.norm(Ckk - np.eye(M))

        return max(max(costi, costj), costk)

    def zoomout_iteration(self, cclb_size, M_init, M_final, isometric=True, weight_type='iscm',
                          n_jobs=1, equals_id=False, use_ANN=True, complete=False):
        """
        Performs an iteration of Consistent Zoomout refinement

        Parameters
        -----------------------------
        cclb_size   : size of the CCLB to compute
        M_init      : initial dimension of maps
        M_final     : dimension at the end of the iteration
        isometric   : whether to use the reduced space strategy of ConsistentZoomout-iso
        weight_type : 'iscm' or 'adjacency', type of weights to use
        equals_id   : Whether the CLB optimization uses Id or n*Id as a constraint
        use_ANN     : Whether to use Approximate Nearest Neighbor.
        complete    : If vertex-to-vertex and functional maps should be computed with all vertices 
                      instead of the subsampling.
        """
        if isometric:
            self.set_isometries(M=M_init)

        if weight_type == 'iscm':
            self.set_weights(weight_type=weight_type)
        elif self.weights is None:
            # Only computed at first iteration
            self.set_weights(weight_type='adjacency')

        self.compute_W(M=M_init)
        self.compute_CLB(equals_id=equals_id)
        self.compute_CCLB(cclb_size)
        self.compute_p2p(complete=complete, use_ANN=use_ANN, n_jobs=n_jobs)
        self.compute_maps(M_final, complete=complete)

    def zoomout_refine(self, nit=10, step=1, subsample=1000, isometric=True, weight_type='iscm',
                       M_init=None, cclb_ratio=.9, n_jobs=1, equals_id=False, use_ANN=True,
                       verbose=False):
        """
        Refines the functional maps using Consistent Zoomout refinement

        Parameters
        -----------------------------
        nit         : number of zoomout iterations
        step        : dimension increase at each iteration
        subsample   : size of vertices subsample. If set to 0 or None, all vertices are used.
        isometric   : whether to use the reduced space strategy of ConsistentZoomout-iso
        weight_type : 'iscm' or 'adjacency', type of weights to use
        M_init      : original size of functional maps. If None, uses self.M
        cclb_ratio  : size of CCLB as a ratio of the current dimension M
        equals_id   : Whether the CLB optimization uses Id or n*Id as a constraint
        use_ANN     : Whether to use Approximate Nearest Neighbor. This will only be activate once
                      the dimension hits 80 since KDTree are faster before.
        """
        if (np.issubdtype(type(subsample), np.integer) and subsample == 0) or subsample is None:
            use_sub = False
            self.subsample = None
        else:
            use_sub = True
            if np.issubdtype(type(subsample), np.integer):
                self.compute_subsample(size=subsample, verbose=verbose)
            else:
                self.set_subsample(subsample)

        if M_init is not None:
            self.M = M_init
        else:
            M_init = self.M

        ANN_faster = False  # Whether it helps using ANN instead of NN.
        for i in tqdm(range(nit-1)):
            new_M = self.M + step
            m_cclb = int(cclb_ratio * self.M)
            # If not the last iteration
            if i < nit - 1:
                if use_ANN and m_cclb > 80:
                    ANN_faster = True
                self.zoomout_iteration(m_cclb, self.M, new_M, weight_type=weight_type,
                                       equals_id=equals_id, use_ANN=ANN_faster,
                                       n_jobs=n_jobs, complete=not use_sub)

            # Last iteration
            else:
                self.zoomout_iteration(m_cclb, self.M, new_M, weight_type=weight_type,
                                       equals_id=equals_id, use_ANN=False,
                                       n_jobs=n_jobs, complete=True)


def CLB_quad_form(maps, weights, M=None):
    """
    Computes the quadratic form associated to a Functional Maps Network, for Consistent Latent Basis
    computation.

    Parameters:
    -----------------------------
    maps    : dict - dictionnary of functional maps associated to key (i,j) representing an edge
    weights : (n,n) sparse matrix of weights. Entry (i,j) represent the weight of edge (i,j)
    M       : Dimension of Functional maps to consider

    Output
    -----------------------------
    W : (N*M,N*M) sparse matrix representing the quadratic form for CLB computation.
    """
    edges = list(maps.keys())
    N = 1 + np.max(edges)

    if M is None:
        M = maps[edges[0]].shape[0]

    # Prepare a block-sparse matrix
    grid = [[None for _ in range(N)] for _ in range(N)]
    for i in range(N):
        grid[i][i] = sparse.csr_matrix(np.zeros((M, M)))

    for (i,j) in edges:
        FM = maps[(i, j)][:M, :M]

        grid[i][i] += sparse.csr_matrix(weights[i, j] * (FM.T @ FM))
        grid[j][j] += sparse.csr_matrix(weights[i, j] * np.eye(M))

        if grid[i][j] is None:
            grid[i][j] = sparse.csr_matrix(np.zeros((M, M)))

        grid[i][j] -= sparse.csr_matrix(weights[i, j] * FM.T)

        if grid[j][i] is None:
            grid[j][i] = sparse.csr_matrix(np.zeros((M, M)))

        grid[j][i] -= sparse.csr_matrix(weights[i, j] * FM)

    # Build block sparse matrix
    W = sparse.bmat(grid,format='csr')
    return W
