import copy
import time
import warnings

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

import pyFM.optimize as opt_func
import pyFM.refine
import pyFM.signatures as sg
import pyFM.spectral as spectral
from pyFM.mesh import TriMesh
from pyFM.optimize.weights import canonical_taus, legacy_taus, resolvent_mask


class FunctionalMapping:
    """
    Compute a functional map between two meshes.

    Typical workflow::

        model = FunctionalMapping(mesh1, mesh2)
        model.preprocess(descr_type='WKS')
        model.fit(K=(50, 50))                # sets model.FM_12

        p2p = model.get_p2p()                # from model.FM_12
        FM_icp = model.icp_refine()
        p2p_icp = model.get_p2p(FM_icp)

    Parameters
    ----------
    mesh1, mesh2 : TriMesh

    Attributes
    ----------
    FM_12 : (k2, k1) ndarray or None
        Functional map set by fit(). Refinement methods return a new FM
        rather than overwriting this one.
    descr1, descr2 : (n, p) ndarray or None
        Descriptors set by preprocess().
    """

    def __init__(self, mesh1: TriMesh, mesh2: TriMesh):
        self.mesh1 = copy.deepcopy(mesh1)
        self.mesh2 = copy.deepcopy(mesh2)

        self.descr1 = None
        self.descr2 = None

        self.FM_12 = None

        self._k1 = None
        self._k2 = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def k1(self):
        if self.FM_12 is not None:
            return self.FM_12.shape[1]
        return self._k1

    @property
    def k2(self):
        if self.FM_12 is not None:
            return self.FM_12.shape[0]
        return self._k2

    def _set_k(self, k):
        if np.issubdtype(type(k), np.integer):
            self._k1 = k
            self._k2 = k
        else:
            self._k1, self._k2 = k

        self.FM_12 = None  # invalidate any previous FM

    # ------------------------------------------------------------------
    # Status properties
    # ------------------------------------------------------------------

    @property
    def has_spectral(self):
        return (
            self.mesh1.eigenvalues is not None
            and self.mesh2.eigenvalues is not None
            and self.mesh1.eigenvectors is not None
            and self.mesh2.eigenvectors is not None
        )

    @property
    def fitted(self):
        return self.FM_12 is not None

    # ------------------------------------------------------------------
    # Deprecated aliases (pre-1.3 API)
    # ------------------------------------------------------------------

    @property
    def FM(self):
        """Deprecated alias for :attr:`FM_12`."""
        warnings.warn("`FM` is deprecated, use `FM_12` instead.", DeprecationWarning, stacklevel=2)
        return self.FM_12

    @FM.setter
    def FM(self, value):
        warnings.warn("`FM` is deprecated, use `FM_12` instead.", DeprecationWarning, stacklevel=2)
        self.FM_12 = value

    @property
    def preprocessed(self):
        """Deprecated alias for :attr:`has_spectral`."""
        warnings.warn(
            "`preprocessed` is deprecated, use `has_spectral` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.has_spectral

    def project(self, func, k=None, mesh_ind=1):
        """Deprecated alias for ``model.mesh1.project`` / ``model.mesh2.project``."""
        warnings.warn(
            "`FunctionalMapping.project()` is deprecated, "
            "use `model.mesh1.project()` or `model.mesh2.project()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if mesh_ind == 1:
            return self.mesh1.project(func, k=k if k is not None else self._k1)
        elif mesh_ind == 2:
            return self.mesh2.project(func, k=k if k is not None else self._k2)
        raise ValueError(f"mesh_ind must be 1 or 2, got {mesh_ind}")

    def decode(self, coeffs, mesh_ind=2):
        """Deprecated alias for ``model.mesh1.unproject`` / ``model.mesh2.unproject``."""
        warnings.warn(
            "`FunctionalMapping.decode()` is deprecated, "
            "use `model.mesh1.unproject()` or `model.mesh2.unproject()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if mesh_ind == 1:
            return self.mesh1.unproject(coeffs)
        elif mesh_ind == 2:
            return self.mesh2.unproject(coeffs)
        raise ValueError(f"mesh_ind must be 1 or 2, got {mesh_ind}")

    def transport(self, coeffs, reverse=False):
        """Deprecated, apply :attr:`FM_12` directly."""
        warnings.warn(
            "`transport()` is deprecated, apply `FM_12` to the coefficients directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self.fitted:
            raise ValueError("Fit the model before transporting functions.")
        return self.FM_12.T @ coeffs if reverse else self.FM_12 @ coeffs

    # ------------------------------------------------------------------
    # Pointwise map extraction
    # ------------------------------------------------------------------

    def get_p2p(self, FM=None, use_adj=False, n_jobs=None):
        """
        Compute a pointwise map from mesh2 to mesh1.

        Parameters
        ----------
        FM      : (k2, k1) ndarray, optional
            Functional map to convert. Defaults to self.FM_12.
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
            FM = self.FM_12

        return spectral.mesh_FM_to_p2p(FM, self.mesh1, self.mesh2, use_adj=use_adj, n_jobs=n_jobs)

    def get_precise_map(
        self,
        FM=None,
        precompute_dmin=True,
        use_adj=True,
        batch_size=None,
        n_jobs=None,
        verbose=False,
    ):
        """
        Compute a precise (barycentric) map from mesh2 to mesh1.

        See "Deblurring and Denoising of Maps between Shapes" (Ezuz & Ben-Chen).

        Parameters
        ----------
        FM             : (k2, k1) ndarray, optional
            Functional map to convert. Defaults to self.FM_12.
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
            FM = self.FM_12

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
    def compute_spectral_descriptors(
        self,
        n_descr=100,
        descr_type="WKS",
        k_descr=128,
        landmarks=None,
        landmarks_only=False,
        subsample_step=1,
        normalization="l2",
    ):
        """
        Compute the LBO spectrum and descriptors needed for fit().

        Parameters
        ----------
        n_descr       : int
            Number of descriptor values per mesh.
        descr_type    : "WKS" | "HKS" | None
            Built-in descriptor type. Pass None to skip descriptor computation
            and supply descriptors manually via add_descriptors().
        landmarks     : (p,) or (p, 2) ndarray, optional
            Landmark indices. Shape (p,) uses the same indices on both meshes;
            shape (p, 2) uses column 0 for mesh1 and column 1 for mesh2.
        landmarks_only : bool
            If True, compute descriptors only at the landmark vertices.
        subsample_step : int
            Keep every nth descriptor column.
        k_descr       : int, optional
            Number of eigenvalues to compute for descriptor computation (default 128).
        verbose       : bool
        """
        use_lm = landmarks is not None and len(landmarks) > 0
        lmks1, lmks2 = self._parse_landmarks(landmarks) if use_lm else (None, None)

        descr1 = np.empty((self.mesh1.n_vertices, 0))
        descr2 = np.empty((self.mesh2.n_vertices, 0))
        if descr_type == "HKS":
            if not use_lm or not landmarks_only:
                descr1_raw = sg.mesh_HKS(self.mesh1, n_descr, k=k_descr)
                descr2_raw = sg.mesh_HKS(self.mesh2, n_descr, k=k_descr)
                descr1 = np.hstack([descr1, descr1_raw])
                descr2 = np.hstack([descr2, descr2_raw])
            if use_lm:
                descr1_lm = sg.mesh_HKS(self.mesh1, n_descr, landmarks=lmks1, k=k_descr)
                descr2_lm = sg.mesh_HKS(self.mesh2, n_descr, landmarks=lmks2, k=k_descr)
                descr1 = np.hstack([descr1, descr1_lm])
                descr2 = np.hstack([descr2, descr2_lm])

        elif descr_type == "WKS":
            if not use_lm or not landmarks_only:
                descr1_raw = sg.mesh_WKS(self.mesh1, n_descr, k=k_descr)
                descr2_raw = sg.mesh_WKS(self.mesh2, n_descr, k=k_descr)
                descr1 = np.hstack([descr1, descr1_raw])
                descr2 = np.hstack([descr2, descr2_raw])
            if use_lm:
                descr1_lm = sg.mesh_WKS(self.mesh1, n_descr, landmarks=lmks1, k=k_descr)
                descr2_lm = sg.mesh_WKS(self.mesh2, n_descr, landmarks=lmks2, k=k_descr)
                descr1 = np.hstack([descr1, descr1_lm])
                descr2 = np.hstack([descr2, descr2_lm])
        else:
            raise ValueError(f'descr_type must be "HKS", "WKS", or None, got "{descr_type}"')

        descr1 = descr1[:, ::subsample_step]  # (n1, p//s)
        descr2 = descr2[:, ::subsample_step]  # (n2, p//s)

        if normalization == "l2":
            # L2-normalise each descriptor column
            no1 = np.sqrt(self.mesh1.l2_sqnorm(descr1))
            no2 = np.sqrt(self.mesh2.l2_sqnorm(descr2))
            descr1 /= no1[None, :]
            descr2 /= no2[None, :]

        elif normalization in [None, "none"]:
            pass

        else:
            raise ValueError(f"Unknown normalization: {normalization}")

        self.descr1 = descr1
        self.descr2 = descr2

    def preprocess(
        self,
        n_descr=100,
        descr_type="WKS",
        landmarks=None,
        landmarks_only=False,
        subsample_step=1,
        k_process=200,
        k_descr=128,
        verbose=False,
        K=None,
    ):
        """
        Compute the LBO spectrum and descriptors needed for fit().

        Parameters
        ----------
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
        K             : int or (int, int), optional
            Deprecated. The functional map size is now set on :meth:`fit`.
        """
        if K is not None:
            warnings.warn(
                "`K` has moved from `preprocess()` to `fit()`; pass it there instead. "
                "Note that `preprocess()`'s first positional argument is now `n_descr`.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._set_k(K)

        if k_process is None:
            k_process = 200
        k_process = max(k_process, k_descr)

        if verbose:
            print("\nComputing Laplacian spectrum")
        self.mesh1.process(k_process, verbose=verbose)
        self.mesh2.process(k_process, verbose=verbose)

        if descr_type is None:
            if verbose:
                print("\nSkipping descriptor computation (descr_type=None)")
            return self

        if verbose:
            print("\nComputing descriptors")

        if descr_type not in ["WKS", "HKS"]:
            raise ValueError(f'descr_type must be "WKS", "HKS", or None, got "{descr_type}"')

        self.compute_spectral_descriptors(
            n_descr=n_descr,
            descr_type=descr_type,
            k_descr=k_descr,
            landmarks=landmarks,
            landmarks_only=landmarks_only,
            subsample_step=subsample_step,
            normalization="l2",
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
        K=(50, 50),
        w_descr=1e-1,
        w_lap=1e-3,
        w_dcomm=1,
        w_orient=0,
        orient_reversing=False,
        use_resolvent_laplacian=True,
        resolvent_gamma=0.5,
        optinit="zeros",
        factr=1e7,
        pgtol=1e-5,
        normalize_weights=True,
        verbose=False,
    ):
        """
        Solve the functional map optimization and store the result in self.FM_12.

        Minimises::

            w_descr  * ||C A - B||²
          + w_lap    * ||C L1 - L2 C||²        (LBO commutativity)
          + w_dcomm  * Σ_i ||C D_Ai - D_Bi C||²  (descriptor commutativity)
          + w_orient * Σ_i ||C G_Ai - G_Bi C||²  (orientation term)

        Only the *ratios* of the weights matter: they are normalized internally,
        so multiplying all of them by a common factor leaves the result unchanged.

        Calls preprocess() automatically if it has not been done yet.

        Parameters
        ----------
        w_descr          : float
        w_lap            : float
        w_dcomm          : float
        w_orient         : float
            Relative weight of the orientation term, rescaled internally so that
            w_orient=1 makes it comparable to the sum of the other terms at C=I.
            Set to 0 to disable it.
        orient_reversing : bool
            Use orientation-reversing instead of orientation-preserving operators.
        optinit          : "zeros" | "identity" | "random"
        factr            : float
            L-BFGS-B relative tolerance on the energy decrease. Lower it (e.g. 1e2)
            for a tighter solve.
        pgtol            : float
            L-BFGS-B tolerance on the max-norm of the projected gradient. This test
            is absolute, so lower it if the energy is small in absolute terms.
        verbose          : bool
        """
        if optinit not in ("zeros", "identity", "random"):
            raise ValueError(f'optinit must be "zeros", "identity" or "random", got "{optinit}"')

        if not self.has_spectral:
            if verbose:
                print("Preprocessing not done — running preprocess() with default parameters.")
            self.preprocess(verbose=verbose)

        if self.descr1 is None:
            raise ValueError(
                "No descriptors set — call add_descriptors() or preprocess() "
                "with a descr_type before fitting."
            )

        self._set_k(K)

        for name, k, mesh in [("k1", self.k1, self.mesh1), ("k2", self.k2, self.mesh2)]:
            n_computed = len(mesh.eigenvalues)
            if k > n_computed:
                raise ValueError(
                    f"Requested {name}={k} exceeds the {n_computed} eigenvalues computed "
                    f"on {name.replace('k', 'mesh')}. Call preprocess() with a larger k_process."
                )

        w_total = w_descr + w_lap + w_dcomm + w_orient
        if w_total <= 0:
            raise ValueError("At least one of the weights must be positive")
        w_descr, w_lap, w_dcomm, w_orient = (
            w / w_total for w in (w_descr, w_lap, w_dcomm, w_orient)
        )

        descr1_red = self.mesh1.project(self.descr1, k=self.k1)  # (k1, p)
        descr2_red = self.mesh2.project(self.descr2, k=self.k2)  # (k2, p)

        descr_op = []
        if w_dcomm > 0:
            descr_op = self._compute_descr_op()

        orient_op = []
        if w_orient > 0:
            orient_op = self._compute_orientation_op(reversing=orient_reversing)

        if use_resolvent_laplacian:
            lap_mask = resolvent_mask(
                self.mesh1.eigenvalues[: self.k1],
                self.mesh2.eigenvalues[: self.k2],
                gamma=resolvent_gamma,
            )
        else:
            lap_mask = np.square(
                self.mesh1.eigenvalues[None, : self.k1] - self.mesh2.eigenvalues[: self.k2, None]
            )

        if normalize_weights:
            tau_descr, tau_lap, tau_descr_op, tau_orient_op = canonical_taus(
                descr1_red,
                descr2_red,
                descr_op,
                orient_op,
                lap_mask,
            )

        else:
            if verbose:
                print("\tUsing legacy weighting")
            tau_descr, tau_lap, tau_descr_op, tau_orient_op = legacy_taus(
                K,
                w_descr,
                w_lap,
                w_dcomm,
                w_orient,
                descr1_red,
                descr2_red,
                descr_op,
                orient_op,
                lap_mask,
                verbose=verbose,
            )

        args = (
            w_descr / tau_descr,
            w_lap / tau_lap,
            w_dcomm / tau_descr_op,
            w_orient / tau_orient_op,
            descr1_red,
            descr2_red,
            descr_op,
            orient_op,
            lap_mask,
        )

        x0 = self._get_x0(optinit)

        if verbose:
            print(
                f"\nOptimization:\n"
                f"\t{self.k1} eigenvectors on source, {self.k2} on target\n"
                f"\t{self.descr1.shape[1]} descriptors\n"
                f"\tw_descr={w_descr:.2e}  w_dcomm={w_dcomm:.2e}  "
                f"w_lap={w_lap:.2e}  w_orient={w_orient:.2e}"
            )

        t0 = time.time()
        res = fmin_l_bfgs_b(
            opt_func.energy_and_grad_std,
            x0.ravel(),
            args=args,
            factr=factr,
            pgtol=pgtol,
        )
        if verbose:
            info = res[2]
            print(
                f"\ttask={info['task']}  funcalls={info['funcalls']}  "
                f"nit={info['nit']}  warnflag={info['warnflag']}"
            )
            print(f"\tDone in {time.time() - t0:.2f}s")

        self.FM_12 = res[0].reshape((self.k2, self.k1))

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def icp_refine(self, FM=None, nit=10, tol=1e-10, use_adj=False, n_jobs=None, verbose=False):
        """
        Refine a functional map with ICP.

        Parameters
        ----------
        FM      : (k2, k1) ndarray, optional
            FM to refine. Defaults to self.FM_12.
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
            FM = self.FM_12

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
            FM to refine. Defaults to self.FM_12.
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
            FM = self.FM_12

        sub = None
        if subsample:
            sub = (
                self.mesh1.farthest_point_sampling(subsample),
                self.mesh2.farthest_point_sampling(subsample),
            )

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
            raise ValueError("Fit the model before computing shape difference operators.")

        self.SD_a = spectral.area_SD(self.FM_12)
        self.SD_c = spectral.conformal_SD(
            self.FM_12, self.mesh1.eigenvalues, self.mesh2.eigenvalues
        )

    # ------------------------------------------------------------------
    # Transfer Functions
    # ------------------------------------------------------------------

    def transfer(self, func):
        """
        Transfer a function between meshes (project → transport → decode).

        Parameters
        ----------
        func    : (n1, p) ndarray

        Returns
        -------
        transferred : (n2, p)
        """

        return self.mesh2.unproject(self.FM_12 @ self.mesh1.project(func, k=self.k1))

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
            x0 = np.random.default_rng().random((k2, k1))
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
            pinv1 @ (self.descr1[:, i, None] * evecs1) for i in range(self.descr1.shape[1])
        ]  # (p, k1, k1)
        ops2 = [
            pinv2 @ (self.descr2[:, i, None] * evecs2) for i in range(self.descr2.shape[1])
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
