import copy
import time

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

import pyFM.signatures as sg
import pyFM.optimize as opt_func
import pyFM.refine
import pyFM.spectral as spectral


class FunctionalMapping:
    """
    Compute a functional map between two meshes.

    Typical workflow::

        model = FunctionalMapping(mesh1, mesh2)
        model.preprocess(n_ev=(50, 50), descr_type='WKS')
        model.fit()                          # sets model.FM

        p2p = model.get_p2p()               # from model.FM
        FM_icp = model.icp_refine()
        p2p_icp = model.get_p2p(FM_icp)

    Parameters
    ----------
    mesh1, mesh2 : TriMesh

    Attributes
    ----------
    FM : (k2, k1) ndarray or None
        Functional map set by fit(). Refinement methods return a new FM
        rather than overwriting this one.
    descr1, descr2 : (n, p) ndarray or None
        Descriptors set by preprocess().
    """

    def __init__(self, mesh1, mesh2):
        self.mesh1 = copy.deepcopy(mesh1)
        self.mesh2 = copy.deepcopy(mesh2)

        self.descr1 = None
        self.descr2 = None

        self.FM = None

        self._k1 = None
        self._k2 = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def k1(self):
        if self.FM is not None:
            return self.FM.shape[1]
        return self._k1

    @property
    def k2(self):
        if self.FM is not None:
            return self.FM.shape[0]
        return self._k2

    # ------------------------------------------------------------------
    # Status properties
    # ------------------------------------------------------------------

    @property
    def preprocessed(self):
        return (
            self.mesh1.eigenvalues is not None
            and self.mesh2.eigenvalues is not None
            and self.mesh1.eigenvectors is not None
            and self.mesh2.eigenvectors is not None
        )

    @property
    def fitted(self):
        return self.FM is not None

    # ------------------------------------------------------------------
    # Pointwise map extraction
    # ------------------------------------------------------------------

    def get_p2p(self, FM=None, use_adj=False, n_jobs=1):
        """
        Compute a pointwise map from mesh2 to mesh1.

        Parameters
        ----------
        FM      : (k2, k1) ndarray, optional
            Functional map to convert. Defaults to self.FM.
        use_adj : bool
            Whether to use the adjoint map.
        n_jobs  : int
            Number of parallel jobs for nearest-neighbour search.

        Returns
        -------
        p2p_21 : (n2,) ndarray
            p2p_21[i] is the index on mesh1 corresponding to vertex i on mesh2.
        """
        if FM is None:
            if not self.fitted:
                raise ValueError("No FM available — run fit() first or pass an FM.")
            FM = self.FM

        return spectral.mesh_FM_to_p2p(
            FM, self.mesh1, self.mesh2, use_adj=use_adj, n_jobs=n_jobs
        )

    def get_precise_map(
        self,
        FM=None,
        precompute_dmin=True,
        use_adj=True,
        batch_size=None,
        n_jobs=1,
        verbose=False,
    ):
        """
        Compute a precise (barycentric) map from mesh2 to mesh1.

        See "Deblurring and Denoising of Maps between Shapes" (Ezuz & Ben-Chen).

        Parameters
        ----------
        FM             : (k2, k1) ndarray, optional
            Functional map to convert. Defaults to self.FM.
        precompute_dmin : bool
            Precompute all delta_min values. Faster but heavier in memory.
        use_adj        : bool
        batch_size     : int, optional
        n_jobs         : int
        verbose        : bool

        Returns
        -------
        P21 : (n2, n1) sparse matrix
        """
        if FM is None:
            if not self.fitted:
                raise ValueError("No FM available — run fit() first or pass an FM.")
            FM = self.FM

        return spectral.mesh_FM_to_p2p_precise(
            FM,
            self.mesh1,
            self.mesh2,
            precompute_dmin=precompute_dmin,
            use_adj=use_adj,
            batch_size=batch_size,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(
        self,
        K=50,
        n_descr=100,
        descr_type="WKS",
        landmarks=None,
        subsample_step=1,
        k_process=None,
        verbose=False,
    ):
        """
        Compute the LBO spectrum and descriptors needed for fit().

        Parameters
        ----------
        K             : int or (int, int)
            Number of LBO eigenvectors to keep for each mesh.
        n_descr       : int
            Number of descriptor values per mesh.
        descr_type    : "WKS" | "HKS" | None
            Built-in descriptor type. Pass None to skip descriptor computation
            and supply descriptors manually via add_descriptors().
        landmarks     : (p,) or (p, 2) ndarray, optional
            Landmark indices. Shape (p,) uses the same indices on both meshes;
            shape (p, 2) uses column 0 for mesh1 and column 1 for mesh2.
        subsample_step : int
            Keep every nth descriptor column.
        k_process     : int, optional
            Number of eigenvalues to compute (default 200).
        verbose       : bool
        """
        if np.issubdtype(type(K), np.integer):
            K = (K, K)
        k1, k2 = K

        self._k1, self._k2 = k1, k2

        if k_process is None:
            k_process = 200

        if verbose:
            print("\nComputing Laplacian spectrum")
        self.mesh1.process(max(k1, k_process), verbose=verbose)
        self.mesh2.process(max(k2, k_process), verbose=verbose)

        if descr_type is None:
            if verbose:
                print("\nSkipping descriptor computation (descr_type=None)")
            return self

        if verbose:
            print("\nComputing descriptors")

        use_lm = landmarks is not None and len(landmarks) > 0
        lmks1, lmks2 = self._parse_landmarks(landmarks) if use_lm else (None, None)

        if descr_type == "HKS":
            self.descr1 = sg.mesh_HKS(self.mesh1, n_descr, k=k1)
            self.descr2 = sg.mesh_HKS(self.mesh2, n_descr, k=k2)
            if use_lm:
                self.descr1 = np.hstack(
                    [
                        self.descr1,
                        sg.mesh_HKS(self.mesh1, n_descr, landmarks=lmks1, k=k1),
                    ]
                )
                self.descr2 = np.hstack(
                    [
                        self.descr2,
                        sg.mesh_HKS(self.mesh2, n_descr, landmarks=lmks2, k=k2),
                    ]
                )

        elif descr_type == "WKS":
            self.descr1 = sg.mesh_WKS(self.mesh1, n_descr, k=k1)
            self.descr2 = sg.mesh_WKS(self.mesh2, n_descr, k=k2)
            if use_lm:
                self.descr1 = np.hstack(
                    [
                        self.descr1,
                        sg.mesh_WKS(self.mesh1, n_descr, landmarks=lmks1, k=k1),
                    ]
                )
                self.descr2 = np.hstack(
                    [
                        self.descr2,
                        sg.mesh_WKS(self.mesh2, n_descr, landmarks=lmks2, k=k2),
                    ]
                )

        else:
            raise ValueError(
                f'descr_type must be "HKS", "WKS", or None, got "{descr_type}"'
            )

        self.descr1 = self.descr1[:, ::subsample_step]  # (n1, p//s)
        self.descr2 = self.descr2[:, ::subsample_step]  # (n2, p//s)

        # L2-normalise each descriptor column
        no1 = np.sqrt(self.mesh1.l2_sqnorm(self.descr1))
        no2 = np.sqrt(self.mesh2.l2_sqnorm(self.descr2))
        self.descr1 /= no1[None, :]
        self.descr2 /= no2[None, :]

        if verbose:
            n_lmks = np.asarray(landmarks).shape[0] if use_lm else 0
            print(
                f"\n\t{self.descr1.shape[1]} / {n_descr * (1 + n_lmks)} descriptors kept"
            )

        return self

    def add_descriptors(self, descr1, descr2, normalize=True):
        """
        Append custom descriptors for both meshes.

        Can be called after preprocess() to mix custom descriptors with
        built-in ones, or after preprocess(descr_type=None) for a fully
        custom descriptor workflow.

        Parameters
        ----------
        descr1    : (n1, p) or (n1,) ndarray
        descr2    : (n2, p) or (n2,) ndarray
        normalize : bool
            L2-normalize each descriptor column using the mesh area metric.

        Returns
        -------
        self
        """
        descr1 = np.asarray(descr1, dtype=float)
        descr2 = np.asarray(descr2, dtype=float)

        if descr1.ndim == 1:
            descr1 = descr1[:, None]
        if descr2.ndim == 1:
            descr2 = descr2[:, None]

        if descr1.shape[0] != self.mesh1.n_vertices:
            raise ValueError(
                f"descr1 must have {self.mesh1.n_vertices} rows, got {descr1.shape[0]}"
            )
        if descr2.shape[0] != self.mesh2.n_vertices:
            raise ValueError(
                f"descr2 must have {self.mesh2.n_vertices} rows, got {descr2.shape[0]}"
            )
        if descr1.shape[1] != descr2.shape[1]:
            raise ValueError(
                f"descr1 and descr2 must have the same number of columns, "
                f"got {descr1.shape[1]} and {descr2.shape[1]}"
            )

        if normalize:
            no1 = np.sqrt(self.mesh1.l2_sqnorm(descr1))
            no2 = np.sqrt(self.mesh2.l2_sqnorm(descr2))
            descr1 = descr1 / no1[None, :]
            descr2 = descr2 / no2[None, :]

        if self.descr1 is None:
            self.descr1 = descr1
            self.descr2 = descr2
        else:
            self.descr1 = np.hstack([self.descr1, descr1])
            self.descr2 = np.hstack([self.descr2, descr2])

        return self

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def fit(
        self,
        w_descr=1e-1,
        w_lap=1e-3,
        w_dcomm=1,
        w_orient=0,
        orient_reversing=False,
        optinit="zeros",
        verbose=False,
    ):
        """
        Solve the functional map optimization and store the result in self.FM.

        Minimises::

            w_descr  * ||C A - B||²
          + w_lap    * ||C L1 - L2 C||²        (LBO commutativity)
          + w_dcomm  * Σ_i ||C D_Ai - D_Bi C||²  (descriptor commutativity)
          + w_orient * Σ_i ||C G_Ai - G_Bi C||²  (orientation term)

        Calls preprocess() automatically if it has not been done yet.

        Parameters
        ----------
        w_descr          : float
        w_lap            : float
        w_dcomm          : float
        w_orient         : float
            Set to 0 to disable the orientation term.
        orient_reversing : bool
            Use orientation-reversing instead of orientation-preserving operators.
        optinit          : "zeros" | "identity" | "random"
        verbose          : bool
        """
        if optinit not in ("zeros", "identity", "random"):
            raise ValueError(
                f'optinit must be "zeros", "identity" or "random", got "{optinit}"'
            )

        if not self.preprocessed:
            if verbose:
                print(
                    "Preprocessing not done — running preprocess() with default parameters."
                )
            self.preprocess(verbose=verbose)

        if self.descr1 is None:
            raise ValueError(
                "No descriptors set — call add_descriptors() or preprocess() "
                "with a descr_type before fitting."
            )

        k1, k2 = self._k1, self._k2

        descr1_red = self.project(self.descr1, mesh_ind=1)  # (k1, p)
        descr2_red = self.project(self.descr2, mesh_ind=2)  # (k2, p)

        list_descr = []
        if w_dcomm > 0:
            list_descr = self._compute_descr_op()

        orient_op = []
        if w_orient > 0:
            orient_op = self._compute_orientation_op(reversing=orient_reversing)

        ev_sqdiff = np.square(
            self.mesh1.eigenvalues[None, :k1] - self.mesh2.eigenvalues[:k2, None]
        )  # (k2, k1)
        ev_sqdiff_sum = ev_sqdiff.sum()
        ev_sqdiff /= ev_sqdiff_sum
        if verbose:
            print(f"\tLBO commutativity weight scaled by {1 / ev_sqdiff_sum:.2e}")

        # Rescale orientation weight relative to the other terms
        if w_orient > 0:
            C_eye = np.eye(k2, k1)
            eval_native = opt_func.energy_func_std(
                C_eye,
                w_descr,
                w_lap,
                w_dcomm,
                0,
                descr1_red,
                descr2_red,
                list_descr,
                orient_op,
                ev_sqdiff,
            )
            eval_orient = opt_func.oplist_commutation(C_eye, orient_op)
            if eval_orient > 0:
                scale = eval_native / eval_orient
                w_orient *= scale
                if verbose:
                    print(f"\tOrientation weight scaled by {scale:.2e}")
            elif verbose:
                print(
                    "\tOrientation operator has zero energy; "
                    "skipping orientation weight rescaling"
                )

        args = (
            w_descr,
            w_lap,
            w_dcomm,
            w_orient,
            descr1_red,
            descr2_red,
            list_descr,
            orient_op,
            ev_sqdiff,
        )

        x0 = self._get_x0(optinit)

        if verbose:
            print(
                f"\nOptimization:\n"
                f"\t{k1} eigenvectors on source, {k2} on target\n"
                f"\t{self.descr1.shape[1]} descriptors\n"
                f"\tw_descr={w_descr:.2e}  w_dcomm={w_dcomm:.2e}  "
                f"w_lap={w_lap:.2e}  w_orient={w_orient:.2e}"
            )

        t0 = time.time()
        res = fmin_l_bfgs_b(
            opt_func.energy_and_grad_std,
            x0.ravel(),
            args=args,
        )
        if verbose:
            info = res[2]
            print(
                f"\ttask={info['task']}  funcalls={info['funcalls']}  "
                f"nit={info['nit']}  warnflag={info['warnflag']}"
            )
            print(f"\tDone in {time.time() - t0:.2f}s")

        self.FM = res[0].reshape((k2, k1))

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def icp_refine(
        self, FM=None, nit=10, tol=None, use_adj=False, n_jobs=1, verbose=False
    ):
        """
        Refine a functional map with ICP.

        Parameters
        ----------
        FM      : (k2, k1) ndarray, optional
            FM to refine. Defaults to self.FM.
        nit     : int
        tol     : float, optional
        use_adj : bool
        n_jobs  : int
        verbose : bool

        Returns
        -------
        FM_icp : (k2, k1) ndarray
        """
        if FM is None:
            if not self.fitted:
                raise ValueError("No FM available — run fit() first or pass an FM.")
            FM = self.FM

        return pyFM.refine.mesh_icp_refine(
            FM,
            self.mesh1,
            self.mesh2,
            nit=nit,
            tol=tol,
            use_adj=use_adj,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def zoomout_refine(self, FM=None, nit=10, step=1, subsample=None, verbose=False):
        """
        Refine a functional map with ZoomOut.

        Parameters
        ----------
        FM        : (k2, k1) ndarray, optional
            FM to refine. Defaults to self.FM.
        nit       : int
        step      : int
            Dimension increase per iteration.
        subsample : int, optional
            Number of vertices to subsample via FPS. None means no subsampling.
        verbose   : bool

        Returns
        -------
        FM_zo : (k2 + nit*step, k1 + nit*step) ndarray
        """
        if FM is None:
            if not self.fitted:
                raise ValueError("No FM available — run fit() first or pass an FM.")
            FM = self.FM

        sub = None
        if subsample:
            sub = (self.mesh1.extract_fps(subsample), self.mesh2.extract_fps(subsample))

        return pyFM.refine.mesh_zoomout_refine(
            FM,
            self.mesh1,
            self.mesh2,
            nit,
            step=step,
            subsample=sub,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Shape difference operators
    # ------------------------------------------------------------------

    def compute_SD(self):
        """
        Compute area- and conformal-based shape difference operators.
        Stores results in self.SD_a and self.SD_c.
        """
        if not self.fitted:
            raise ValueError(
                "Fit the model before computing shape difference operators."
            )

        self.SD_a = spectral.area_SD(self.FM)
        self.SD_c = spectral.conformal_SD(
            self.FM, self.mesh1.eigenvalues, self.mesh2.eigenvalues
        )

    # ------------------------------------------------------------------
    # Transfer Functions
    # ------------------------------------------------------------------

    def project(self, func, k=None, mesh_ind=1):
        """
        Project a function onto the LBO basis.

        Parameters
        ----------
        func     : (n, p) ndarray
        k        : int, optional — number of coefficients (default: k1 or k2)
        mesh_ind : 1 | 2

        Returns
        -------
        coeffs : (k, p) ndarray
        """
        if mesh_ind == 1:
            return self.mesh1.project(func, k=k if k is not None else self._k1)
        elif mesh_ind == 2:
            return self.mesh2.project(func, k=k if k is not None else self._k2)
        raise ValueError(f"mesh_ind must be 1 or 2, got {mesh_ind}")

    def decode(self, coeffs, mesh_ind=2):
        """
        Reconstruct a function from LBO coefficients.

        Parameters
        ----------
        coeffs   : (k, p) ndarray
        mesh_ind : 1 | 2

        Returns
        -------
        func : (n, p) ndarray
        """
        if mesh_ind == 1:
            return self.mesh1.decode(coeffs)
        elif mesh_ind == 2:
            return self.mesh2.decode(coeffs)
        raise ValueError(f"mesh_ind must be 1 or 2, got {mesh_ind}")

    def transport(self, coeffs, reverse=False):
        """
        Apply the functional map to spectral coefficients.

        Parameters
        ----------
        coeffs  : (k1, p) ndarray  (or (k2, p) if reverse=True)
        reverse : bool — use FM.T to go from basis 2 → basis 1

        Returns
        -------
        transported : (k2, p) or (k1, p) ndarray
        """
        if not self.fitted:
            raise ValueError("Fit the model before transporting functions.")
        return self.FM.T @ coeffs if reverse else self.FM @ coeffs

    def transfer(self, func, reverse=False):
        """
        Transfer a function between meshes (project → transport → decode).

        Parameters
        ----------
        func    : (n1, p) ndarray (or (n2, p) if reverse=True)
        reverse : bool — transfer from mesh2 to mesh1 via FM.T

        Returns
        -------
        transferred : (n2, p) or (n1, p) ndarray
        """
        if not reverse:
            return self.decode(
                self.transport(self.project(func, mesh_ind=1)), mesh_ind=2
            )
        return self.decode(
            self.transport(self.project(func, mesh_ind=2), reverse=True), mesh_ind=1
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_landmarks(self, landmarks):
        lm = np.asarray(landmarks).squeeze()
        if lm.ndim == 1:
            return lm, lm.copy()
        return lm[:, 0], lm[:, 1]

    def _get_x0(self, optinit):
        k1, k2 = self._k1, self._k2
        if optinit == "random":
            x0 = np.random.random((k2, k1))
        elif optinit == "identity":
            x0 = np.eye(k2, k1)
        else:
            x0 = np.zeros((k2, k1))

        # Fix the constant-function coefficient analytically
        ev_sign = np.sign(self.mesh1.eigenvectors[0, 0] * self.mesh2.eigenvectors[0, 0])
        area_ratio = np.sqrt(self.mesh2.area / self.mesh1.area)
        x0[:, 0] = 0.0
        x0[0, 0] = ev_sign * area_ratio
        return x0

    def _compute_descr_op(self):
        """Return per-descriptor multiplicative operators in the reduced basis."""

        evecs1 = self.mesh1.eigenvectors[:, : self._k1]  # (n1, k1)
        evecs2 = self.mesh2.eigenvectors[:, : self._k2]  # (n2, k2)

        pinv1 = evecs1.T @ self.mesh1.A  # (k1, n1)
        pinv2 = evecs2.T @ self.mesh2.A  # (k2, n2)

        ops1 = [
            pinv1 @ (self.descr1[:, i, None] * evecs1)
            for i in range(self.descr1.shape[1])
        ]  # (p, k1, k1)
        ops2 = [
            pinv2 @ (self.descr2[:, i, None] * evecs2)
            for i in range(self.descr2.shape[1])
        ]  # (p, k2, k2)

        return np.stack(ops1, axis=0), np.stack(ops2, axis=0)

    def _compute_orientation_op(self, reversing=False, normalize=False):
        """Return per-descriptor orientation operators in the reduced basis."""
        evecs1 = self.mesh1.eigenvectors[:, : self._k1]  # (n1, k1)
        evecs2 = self.mesh2.eigenvectors[:, : self._k2]  # (n2, k2)

        pinv1 = evecs1.T @ self.mesh1.A  # (k1, n1)
        pinv2 = evecs2.T @ self.mesh2.A  # (k2, n2)

        grads1 = [
            self.mesh1.gradient(self.descr1[:, i], normalize=normalize)
            for i in range(self.descr1.shape[1])
        ]
        grads2 = [
            self.mesh2.gradient(self.descr2[:, i], normalize=normalize)
            for i in range(self.descr2.shape[1])
        ]

        ops1 = [pinv1 @ self.mesh1.orientation_op(g) @ evecs1 for g in grads1]
        sign = -1 if reversing else 1
        ops2 = [sign * pinv2 @ self.mesh2.orientation_op(g) @ evecs2 for g in grads2]

        return np.stack(ops1, axis=0), np.stack(ops2, axis=0)
