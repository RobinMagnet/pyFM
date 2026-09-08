import copy
import time
from collections import defaultdict

import numpy as np
import scipy.linalg
import scipy.sparse as sparse
import scipy.sparse.linalg
from scipy.optimize import linprog
from tqdm.auto import tqdm

from .. import spectral
from ..spectral.nearest_neighbor import knn_query


class FMN:
    """
    Functional Map Network.

    Parameters
    ----------
    meshlist : list
        List of TriMesh objects.
    maps_dict : dict, optional
        Dictionary of functional maps between each pair of meshes.
        Keys are (i, j) with i, j indices of the meshes in the list.
    """

    def __init__(self, meshlist, maps_dict=None):
        # Mesh of each Node
        self.meshlist = copy.deepcopy(meshlist)  # List of n TriMesh

        # Edges are determined by (i,j) pair of indices
        # A map is associated to each edge (via dictionary)
        # Weights of edges are stored in a sparse (n,n) matrix
        # For computation, an arbitrary ordering of edges is stored.

        # Network attribute
        self.edges = None  # List of couples (i,j)
        self.maps = None  # Dictionary of maps
        self.weights = None  # (n,n) sparse matrix of weights
        self.edge2ind = None  # Ordering of edges

        # (n,K) array of indices of K vertices per shape in the network.
        self.subsample = None

        # icsm weights attributes
        self.cycles = None  # List of 3-cycles (i,j,k)
        self.A = None  # (n_cycle, n_edges) binary matrix (1 if edge j in cycle i)
        self.A_sub = None  # (n_edge_in_cycle,) indices of edges contained in a 3-cycle

        self.use_icsm = False  # Whether icsm or adjacency weights are used.
        self.cycle_weight = None  # Weights of each 3-cycle (map-dependent)
        self.edge_weights = None  # Weight of each edge (map-dependent)

        # CLB and CCLB attributes
        self.W = None  # (n*M, n*M) sparse matrix. Quadratic form for CLB computation.
        self.CLB = None  # (n,M,M) array of Consistent Latent Basis for each mesh.
        self.CCLB = None  # (n,M,m) array of Canonical Consistent Latent Basis for each mesh
        self.cclb_eigenvalues = None  # (m,) eigenvalues of the CCLB

        # Extra information
        self.p2p = None  # Dictionary of pointwise maps associated to each edge
        self._M = None

        if maps_dict is not None:
            self.set_maps(maps_dict=maps_dict, verbose=True)

    @property
    def n_meshes(self):
        """
        Return the number of meshes (nodes) in the network.

        Returns
        -------
        n_meshes : int
            Number of meshes in the network.
        """
        return len(self.meshlist)

    @property
    def M(self):
        """
        Return the current shared dimension for functional maps
        (which are square matrices).

        If not specified, returns the size of the first found map.

        Returns
        -------
        M : int
            Size of the functional maps.
        """
        if self._M is not None:
            return self._M
        else:
            return self.maps[self.edges[0]].shape[0]

    @M.setter
    def M(self, M):
        size, ind = self._spectrum_size()
        if size is not None and M > size:
            raise ValueError(
                f"Functional maps of size {M} need {M} eigenvectors on each mesh, but "
                f"mesh {ind} only has {size}. Process the meshes with `process(k={M})`, "
                "or use smaller maps."
            )
        self._M = M

    def _spectrum_size(self):
        """
        Return the smallest number of eigenvectors available, with the mesh it belongs to.

        Returns
        -------
        size : int or None
            Smallest spectrum size, None if no mesh has been processed.
        ind : int or None
            Index of the mesh achieving it.
        """
        sizes = [
            (evects.shape[1], i)
            for i, evects in enumerate(
                getattr(mesh, "eigenvectors", None) for mesh in self.meshlist
            )
            if evects is not None
        ]
        return min(sizes) if sizes else (None, None)

    @property
    def m_cclb(self):
        """
        Return the dimension of the Canonical Consistent Latent Basis.

        Returns
        -------
        m : int
            Size of the CCLB.
        """
        return self.CCLB.shape[2]

    def _reset_map_attributes(self):
        """
        Reset all attributes depending on the functional maps.
        """
        # Resets icsm weights variables
        if self.use_icsm:
            self.use_icsm = False  # Whether icsm or adjacency weights are used.
            self.cycle_weight = None  # Weights of each 3-cycle (map-dependent)
            self.edge_weights = None  # Weight of each edge (map-dependent)
            self.weights = None  # (n,n) sparse matrix of weights

        # Reset map-dependent attributes
        self.W = None  # (n*M, n*M) sparse matrix. Quadratic form for CLB computation.
        self.CLB = None  # (n,M,M) array containing the Consistent Latent Basis for each mesh.
        self.CCLB = None  # (n,M,m) array of Canonical Consistent Latent Basis for each mesh
        self.cclb_eigenvalues = None  # (m,) eigenvalues of the CCLB
        self.p2p = None  # Dictionary of pointwise

    def set_maps(self, maps_dict, verbose=False):
        """
        Set the edges of the graph with maps.

        Saves extra information about the edges.

        Parameters
        ----------
        maps_dict : dict
            Dictionary where key (i, j) gives the functional map FM between mesh
            i and j. FM can be of different size depending on the edge.
        verbose : bool, optional
            Whether to print information about the edges being set.

        Returns
        -------
        self : FMN
            The current object, with edges set.
        """
        self.maps = copy.deepcopy(maps_dict)

        # Sort edges for later faster optimization
        self.edges = sorted(list(maps_dict.keys()))

        self.edge2ind = dict()
        for edge_ind, edge in enumerate(self.edges):
            self.edge2ind[edge] = edge_ind

        if verbose:
            print(f"Setting {len(self.edges)} edges on {self.n_meshes} nodes.")

        return self

    def set_subsample(self, subsample):
        """
        Set the subsample of vertices on all shapes in the network.

        Parameters
        ----------
        subsample : (n, size) np.ndarray
            Array of indices of vertices to subsample on each shape.

        Returns
        -------
        self : FMN
            The current object, with the subsample set.
        """
        self.subsample = subsample

        return self

    def compute_subsample(self, size=1000, geodesic=False, verbose=False):
        """
        Subsample vertices on each shape using farthest point sampling.

        Store the result in an (n, size) array of indices.

        Parameters
        ----------
        size : int
            Number of vertices to subsample on each shape.
        geodesic : bool, optional
            Whether to use geodesic distances for farthest point sampling.
        verbose : bool, optional
            Whether to print information during computation.

        Returns
        -------
        None
            The subsample is stored in ``self.subsample`` in place.
        """
        if verbose:
            print(f"Computing a {size}-sized subsample for each mesh")
        self.subsample = np.zeros((self.n_meshes, size), dtype=int)
        for i in range(self.n_meshes):
            self.subsample[i] = self.meshlist[i].farthest_point_sampling(size, geodesic=geodesic)

    def set_weights(self, weights=None, weight_type="icsm", verbose=False):
        """
        Set weights for each edge in the graph.

        Parameters
        ----------
        weights : (n, n) sparse matrix, optional
            Matrix of edge weights. If not specified, sets weights according to
            the ``weight_type`` argument.
        weight_type : str, optional
            'icsm' | 'adjacency'. If ``weights`` is not specified, computes
            weights according to the Consistent Zoomout adaptation of icsm or
            using the adjacency matrix of the graph.
        verbose : bool, optional
            Whether to print information during computation.

        Returns
        -------
        self : FMN
            The current object, with weights set.
        """
        if weights is not None:
            self.use_icsm = False
            self.weights = copy.deepcopy(weights)

        elif weight_type == "icsm":
            self.use_icsm = True

            # Process cycles if necessary
            if self.cycles is None:
                if verbose:
                    print("Computing cycle information")
                self.extract_3_cycles()
                self.compute_Amat()

            # Compute original icsm weights d_ij for each edge (i,j)
            # Final weight is set to exp(-d_ij^2/(2*sigma^2))
            # With sigma = median(d_ij)
            weight_arr = self.optimize_icsm(verbose=verbose)  # (n_edges,)
            median_val = np.median(weight_arr[self.A_sub])
            if np.isclose(median_val, 0, atol=1e-4):
                weight_arr /= np.mean(weight_arr[self.A_sub])
            else:
                weight_arr /= median_val
            new_w = np.exp(-np.square(weight_arr) / 2)  # (n_edges,)

            I = [x[0] for x in self.edges]
            J = [x[1] for x in self.edges]
            self.weights = sparse.csr_matrix((new_w, (I, J)), shape=(self.n_meshes, self.n_meshes))

        elif weight_type == "adjacency":
            self.use_icsm = False
            I = [x[0] for x in self.edges]
            J = [x[1] for x in self.edges]
            V = [1 for x in range(len(self.edges))]
            self.weights = sparse.csr_matrix((V, (I, J)), shape=(self.n_meshes, self.n_meshes))

        else:
            raise ValueError(f'"weight_type" should be "icsm" or "adjacency, not {weight_type}')

        return self

    def set_isometries(self, M=None):
        """
        Symmetrize functional maps of reciprocal edges.

        For each edge (i, j), if (j, i) is also an edge, the corresponding
        functional maps are set as the transpose of each other, choosing the
        closest to orthogonal of both.

        Since this modifies the maps, icsm weights are deleted.

        Parameters
        ----------
        M : int, optional
            Dimension with which to compare the functional maps.
            If None, uses the current ``self.M``.

        Returns
        -------
        None
            The maps are modified in place and map-dependent attributes reset.
        """
        # Dictionary with False as a default value for any key
        visited = defaultdict(bool)

        if M is None:
            M = self.M

        for i, j in self.edges:
            if not visited[(i, j)] and (j, i) in self.edges:
                FM1 = self.maps[(i, j)][:M, :M]
                FM2 = self.maps[(j, i)][:M, :M]

                dist1 = np.linalg.norm(FM1.T @ FM1 - np.eye(FM1.shape[1]))
                dist2 = np.linalg.norm(FM2.T @ FM2 - np.eye(FM2.shape[1]))

                if dist1 <= dist2:
                    self.maps[(j, i)] = np.transpose(self.maps[(i, j)])
                else:
                    self.maps[(i, j)] = np.transpose(self.maps[(j, i)])

                visited[(j, i)] = True

        # Reset all map-dependent attributes
        self._reset_map_attributes()

    def compute_W(self, M=None, verbose=False):
        """
        Compute the quadratic form for Consistent Latent Basis (CLB) computation.

        Parameters
        ----------
        M : int, optional
            Size of the functional maps to use; uses the projection of each FM
            on this dimension. If not specified, uses the size of the first
            found functional map.
        verbose : bool, optional
            Whether to print information during computation.

        Returns
        -------
        None
            The quadratic form is stored in ``self.W`` in place.
        """
        if self.maps is None:
            raise ValueError("Functional maps should be set")

        if self.weights is None:
            self.set_weights(verbose=verbose)

        if M is not None:
            self.M = M

        self.W = CLB_quad_form(self.maps, self.weights, M=self.M, n_meshes=self.n_meshes)

    def compute_CLB(self, equals_id=False, verbose=False):
        """
        Compute the Consistent Latent Basis (CLB) using the quadratic form.

        The first M vectors for each basis are computed in order.

        Parameters
        ----------
        equals_id : bool, optional
            If False, the sum of Y.T @ Y is expected to give n * Id.
            If True, the sum of Y.T @ Y is expected to give Id.
        verbose : bool, optional
            Whether to print information during computation.

        Returns
        -------
        None
            The CLB is stored in ``self.CLB`` in place.
        """
        if self.W is None:
            self.compute_W(verbose=verbose)

        # W is a real symmetric matrix !
        # There is a bug in sparse eigenvalues computation where 'LM' returns the smallest
        # eigenvalues whereas 'SM' does not.
        if verbose:
            print(f"Computing {self.M} CLB eigenvectors...")
            start_time = time.time()
        if equals_id:
            # Returns (n*M,), (n*M,M) array

            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                self.W, k=self.M, which="LM", sigma=-1e-6
            )
        else:
            # Returns (n*M,), (n*M,M) array
            M_mat = 1 / self.n_meshes * scipy.sparse.eye(self.W.shape[0])
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                self.W, M=M_mat, k=self.M, which="LM", sigma=-1e-6
            )

        if verbose:
            print(f"\tDone in {time.time() - start_time:.1f}s")
        # In any case, make sure they are real and sorted.
        # eigenvalues = np.real(eigenvalues)
        # sorting = np.argsort(eigenvalues)
        # eigenvalues = eigenvalues[sorting]
        # eigenvectors = np.real(eigenvectors)[:,sorting] # NM,M
        eigenvalues[0] = 0

        self.CLB = eigenvectors.reshape((self.n_meshes, self.M, self.M))  # (n,M,M)

    def compute_CCLB(self, m, verbose=True):
        """
        Compute the Canonical Consistent Latent Basis (CCLB) from the CLB.

        Parameters
        ----------
        m : int
            Size of the CCLB to compute.
        verbose : bool, optional
            Whether to print information during computation.

        Returns
        -------
        self : FMN
            The current object, with the CCLB computed.
        """
        if self.CLB is None:
            self.compute_CLB(verbose=verbose)

        # Matrix E from Algorithm 1 in the Limit Shape paper
        E_mat = np.zeros((m, m))

        for i in range(self.n_meshes):
            Y = self.CLB[i, :, :m]  # (M,m)
            evals = self.meshlist[i].eigenvalues[: self.M]  # (M,)
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
        Return the Characteristic Shape Difference (CSD) operators for mesh i.

        Parameters
        ----------
        i : int
            Index of the mesh on which to return the two CSD.

        Returns
        -------
        CSD_a : (m, m) np.ndarray
            Area-based CSD expressed in the latent space.
        CSD_c : (m, m) np.ndarray
            Conformal CSD expressed in the latent space.
        """
        # Functional map from the Limit Shape to shape i
        FM = self.CCLB[i]

        CSD_a = FM.T @ FM
        CSD_c = (
            np.linalg.pinv(np.diag(self.cclb_eigenvalues))
            @ FM.T
            @ (self.meshlist[i].eigenvalues[: self.M, None] * FM)
        )

        return CSD_a, CSD_c

    def get_LB(self, i, complete=True):
        """
        Return the latent basis (LB) for mesh i.

        Parameters
        ----------
        i : int
            Index of the mesh on which to return the LB.
        complete : bool, optional
            If False, only computes values on the ``self.subsample[i]`` vertices.

        Returns
        -------
        latent_basis : (n_i, m) np.ndarray
            Latent basis on mesh i.
        """
        cclb = self.CCLB[i]  # / np.linalg.norm(self.CCLB[i],axis=0,keepdims=True)  # (M,m)
        if not complete and self.subsample is not None:
            latent_basis = self.meshlist[i].eigenvectors[self.subsample[i], : self.M] @ cclb
            return latent_basis  # (n_i',m)

        latent_basis = self.meshlist[i].eigenvectors[:, : self.M] @ cclb  # (N_i,m)
        return latent_basis

    def compute_p2p(self, complete=True, n_jobs=None):
        """
        Compute vertex-to-vertex maps for each (directed) edge from the CCLB.

        Uses the factorization of functional maps through the CCLB. Only maps
        related to existing edges are computed. Vertex-to-vertex maps are saved
        in a dictionary the same way as functional maps, although their
        direction is reversed.

        Parameters
        ----------
        complete : bool, optional
            If False, uses ``self.subsample`` to obtain pointwise maps between
            subsamples of vertices for each shape.
        n_jobs : int, optional
            Number of parallel jobs. None (default) decides automatically.

        Returns
        -------
        None
            The pointwise maps are stored in ``self.p2p`` in place.
        """

        # Embeddings only depend on the node, whereas each node appears in multiple edges.
        LB_sub = [self.get_LB(i, complete=False) for i in range(self.n_meshes)]
        LB_target = (
            [self.get_LB(i, complete=True) for i in range(self.n_meshes)] if complete else LB_sub
        )

        self.p2p = dict()
        for i, j in self.edges:
            LB_1 = LB_sub[i]  # (n_1',m)
            LB_2 = LB_target[j]  # (n_2',m)

            self.p2p[(i, j)] = knn_query(LB_1, LB_2, k=1, n_jobs=n_jobs)  # (n_2',)

    def compute_maps(self, M, complete=True):
        """
        Convert pointwise maps into functional maps of size M.

        Parameters
        ----------
        M : int
            Size of the functional map to compute.
        complete : bool, optional
            If False and a subsample is set, uses the subsample of vertices to
            convert the pointwise maps.

        Returns
        -------
        None
            The functional maps are stored in ``self.maps`` in place and
            map-dependent attributes are reset.
        """
        self.M = M
        for i, j in self.edges:
            if not complete and self.subsample is not None:
                sub = (self.subsample[i], self.subsample[j])
            else:
                sub = None

            FM = spectral.mesh_p2p_to_FM(
                self.p2p[(i, j)],
                self.meshlist[i],
                self.meshlist[j],
                dims=M,
                subsample=sub,
            )
            self.maps[(i, j)] = FM

        # Reset map-dependent variables
        self._reset_map_attributes()

    def extract_3_cycles(self):
        """
        Extract all 3-cycles from the graph as a list of 3-tuples (i, j, k).

        Returns
        -------
        None
            The cycles are stored in ``self.cycles`` in place.
        """
        self.cycles = []

        # Ugly triple for loop, but only has to be run once.
        # Membership is tested on the edge2ind dict, not on the edges list, to avoid
        # a linear scan over all edges at each test.
        edgeset = self.edge2ind
        # Saves cycles (i,j,k) with either i<j<k or i>j>k
        for i in range(self.n_meshes):
            for j in range(i):
                for k in range(j):
                    if (i, j) in edgeset and (j, k) in edgeset and (k, i) in edgeset:
                        self.cycles.append((i, j, k))

            for j in range(i + 1, self.n_meshes):
                for k in range(j + 1, self.n_meshes):
                    if (i, j) in edgeset and (j, k) in edgeset and (k, i) in edgeset:
                        self.cycles.append(tuple((i, j, k)))

    def compute_Amat(self):
        """
        Compute matrix A for icsm weights optimization.

        Binary matrix telling which edge belongs to which cycle. Uses the
        arbitrary edge ordering created in the ``self.set_maps`` method.

        Returns
        -------
        None
            The matrix is stored in ``self.A`` (as a sparse matrix) and the
            indices of edges in a cycle in ``self.A_sub``, in place.
        """
        n_cycles, n_edges = len(self.cycles), len(self.edges)

        # Each cycle uses exactly 3 edges, so A only has 3 non-zeros per row.
        cols = np.array(
            [
                (self.edge2ind[(i, j)], self.edge2ind[(j, k)], self.edge2ind[(k, i)])
                for (i, j, k) in self.cycles
            ],
            dtype=int,
        ).reshape(n_cycles, 3)
        rows = np.repeat(np.arange(n_cycles), 3)

        self.A = sparse.csr_matrix(
            (np.ones(cols.size), (rows, cols.ravel())), shape=(n_cycles, n_edges)
        )  # (n_cycles, n_edges)

        self.A_sub = np.unique(cols)  # (n_edges_in_cycle)

    def compute_3cycle_weights(self, M=None):
        """
        Compute per-cycle costs and per-edge costs for icsm optimization.

        Cycle weights are given by the ``self.get_cycle_weight`` method
        (deviation from the identity map). Each edge weight is the inverse of
        the sum of all weights of the cycles the edge belongs to.

        Parameters
        ----------
        M : int, optional
            Dimension of functional maps to use. If None, uses ``self.M``.

        Returns
        -------
        None
            Cycle weights are stored in ``self.cycle_weight`` and edge weights
            in ``self.edge_weights``, in place.
        """
        if M is None:
            M = self.M

        self.cycle_weight = np.zeros(len(self.cycles))
        for cycle_ind, cycle in enumerate(self.cycles):
            self.cycle_weight[cycle_ind] = self.get_cycle_weight(cycle, M=M)  # n_cycles

        # Sum of the weights of all cycles each edge belongs to. A being binary,
        # this is simply A.T @ cycle_weight.
        col_sums = self.A.T @ self.cycle_weight  # (n_edges,)

        self.edge_weights = np.zeros(len(self.edges))
        self.edge_weights[self.A_sub] = 1 / col_sums[self.A_sub]

    def optimize_icsm(self, verbose=False):
        r"""
        Solve the linear problem for icsm weights computation.

        Solves $\min w^{\top}  x$ subject to $A x \geq C_{\gamma}$ and
        $x \geq 0$. Edges which are not part of a cycle are given zero weights.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print information during optimization.

        Returns
        -------
        opt_weights : (n_edges,) np.ndarray
            (positive) weights for each edge.
        """
        self.compute_3cycle_weights(M=self.M)

        if verbose:
            print("Optimizing Cycle Weights...")
            start_time = time.time()
        # Solve Linear Program
        res = linprog(
            self.edge_weights,
            A_ub=-self.A,
            b_ub=-self.cycle_weight,
            bounds=(0, float("inf")),
            method="highs-ds",
        )

        if verbose:
            print(f"\tDone in {time.time() - start_time:.5f}s")
        opt_weights = np.zeros(len(self.edges))  # (n_edges,)
        opt_weights[self.A_sub] = res.x[self.A_sub]

        return opt_weights

    def get_cycle_weight(self, cycle, M=None):
        """
        Compute the cost of a cycle (i, j, k) using the functional maps.

        Cost is given as the maximum deviation to the identity map when going
        through the complete cycle (3 possibilities).

        Parameters
        ----------
        cycle : tuple
            3-tuple with node indices creating a cycle.
        M : int, optional
            Dimension of functional maps to use. If None, uses ``self.M``.

        Returns
        -------
        cost : float
            Cost of the cycle.
        """
        if M is None:
            M = self.M

        i, j, k = cycle

        Cij = self.maps[(i, j)][:M, :M]
        Cjk = self.maps[(j, k)][:M, :M]
        Cki = self.maps[(k, i)][:M, :M]

        Cii = Cij @ Cjk @ Cki
        Cjj = Cjk @ Cki @ Cij
        Ckk = Cki @ Cij @ Cjk

        costi = np.linalg.norm(Cii - np.eye(M))
        costj = np.linalg.norm(Cjj - np.eye(M))
        costk = np.linalg.norm(Ckk - np.eye(M))

        return max(max(costi, costj), costk)

    def zoomout_iteration(
        self,
        cclb_size,
        M_init,
        M_final,
        isometric=True,
        weight_type="icsm",
        n_jobs=None,
        equals_id=False,
        complete=False,
    ):
        """
        Perform an iteration of Consistent Zoomout refinement.

        Parameters
        ----------
        cclb_size : int
            Size of the CCLB to compute.
        M_init : int
            Initial dimension of maps.
        M_final : int
            Dimension at the end of the iteration.
        isometric : bool, optional
            Whether to use the reduced space strategy of ConsistentZoomout-iso.
        weight_type : str, optional
            'icsm' or 'adjacency', type of weights to use.
        n_jobs : int, optional
            Number of parallel jobs. None (default) decides automatically.
        equals_id : bool, optional
            Whether the CLB optimization uses Id or n * Id as a constraint.
        complete : bool, optional
            Whether vertex-to-vertex and functional maps should be computed with
            all vertices instead of the subsampling.

        Returns
        -------
        None
            The maps are refined in place.
        """
        if isometric:
            self.set_isometries(M=M_init)

        if weight_type == "icsm":
            self.set_weights(weight_type=weight_type)
        elif self.weights is None:
            # Only computed at first iteration
            self.set_weights(weight_type="adjacency")

        self.compute_W(M=M_init)
        self.compute_CLB(equals_id=equals_id)
        self.compute_CCLB(cclb_size)
        self.compute_p2p(complete=complete, n_jobs=n_jobs)
        self.compute_maps(M_final, complete=complete)

    def zoomout_refine(
        self,
        nit=10,
        step=1,
        subsample=1000,
        isometric=True,
        weight_type="icsm",
        M_init=None,
        cclb_ratio=0.9,
        n_jobs=None,
        equals_id=False,
        verbose=False,
    ):
        """
        Refine the functional maps using Consistent Zoomout refinement.

        Parameters
        ----------
        nit : int, optional
            Number of zoomout iterations.
        step : int, optional
            Dimension increase at each iteration.
        subsample : int or np.ndarray, optional
            Size of vertices subsample. If set to 0 or None, all vertices are used.
        isometric : bool, optional
            Whether to use the reduced space strategy of ConsistentZoomout-iso.
        weight_type : str, optional
            'icsm' or 'adjacency', type of weights to use.
        M_init : int, optional
            Original size of functional maps. If None, uses ``self.M``.
        cclb_ratio : float, optional
            Size of CCLB as a ratio of the current dimension M.
        n_jobs : int, optional
            Number of parallel jobs. None (default) decides automatically.
        equals_id : bool, optional
            Whether the CLB optimization uses Id or n * Id as a constraint.
        verbose : bool, optional
            Whether to print information during refinement.

        Returns
        -------
        None
            The maps are refined in place.
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

        # Fail now rather than in the middle of the refinement.
        size, ind = self._spectrum_size()
        M_final = M_init + nit * step
        if size is not None and M_final > size:
            raise ValueError(
                f"Refining {nit} times by {step} takes the maps from {M_init} to "
                f"{M_final}, but mesh {ind} only has {size} eigenvectors. Process the "
                f"meshes with `process(k={M_final})`, or reduce `nit` or `step`."
            )

        for i in tqdm(range(nit), disable=not verbose):
            new_M = self.M + step
            m_cclb = int(cclb_ratio * self.M)
            # On the last iteration, always recompute the maps on the full mesh
            # (not just the subsample).
            is_last = i == nit - 1
            self.zoomout_iteration(
                m_cclb,
                self.M,
                new_M,
                isometric=isometric,
                weight_type=weight_type,
                equals_id=equals_id,
                n_jobs=n_jobs,
                complete=is_last or not use_sub,
            )


def CLB_quad_form(maps, weights, M=None, n_meshes=None):
    """
    Compute the quadratic form of a Functional Maps Network for CLB computation.

    Parameters
    ----------
    maps : dict
        Dictionary of functional maps, keyed by (i, j) representing an edge.
    weights : (n, n) sparse matrix
        Matrix of weights. Entry (i, j) represents the weight of edge (i, j).
    M : int, optional
        Dimension of functional maps to consider.
    n_meshes : int, optional
        Number of meshes. If not specified, inferred from the highest index in ``maps``.

    Returns
    -------
    W : scipy.sparse.csr_matrix
        (N*M, N*M) sparse matrix representing the quadratic form for CLB
        computation.
    """
    edges = list(maps.keys())
    N = 1 + np.max(edges) if n_meshes is None else n_meshes

    if M is None:
        M = maps[edges[0]].shape[0]

    # Scalar indexing in a sparse matrix is slow, and each weight is read 4 times.
    weights = weights.toarray() if sparse.issparse(weights) else np.asarray(weights)

    # Prepare a block-sparse matrix
    grid = [[None for _ in range(N)] for _ in range(N)]
    for i in range(N):
        grid[i][i] = sparse.csr_matrix(np.zeros((M, M)))

    for i, j in edges:
        FM = maps[(i, j)][:M, :M]
        w_ij = weights[i, j]

        grid[i][i] += sparse.csr_matrix(w_ij * (FM.T @ FM))
        grid[j][j] += sparse.csr_matrix(w_ij * np.eye(M))

        if grid[i][j] is None:
            grid[i][j] = sparse.csr_matrix(np.zeros((M, M)))

        grid[i][j] -= sparse.csr_matrix(w_ij * FM.T)

        if grid[j][i] is None:
            grid[j][i] = sparse.csr_matrix(np.zeros((M, M)))

        grid[j][i] -= sparse.csr_matrix(w_ij * FM)

    # Build block sparse matrix
    W = sparse.bmat(grid, format="csr")
    return W
