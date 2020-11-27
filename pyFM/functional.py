import copy
import time

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

import pyFM.signatures as sg
import pyFM.optimize as opt_func
import pyFM.refine.zoomout as zoomout
import pyFM.refine.icp as icp
import pyFM.spectral as spectral


class FunctionalMapping:
    """
    A class to compute functional maps between two meshes

    Attributes
    ----------------------
    mesh1  : first mesh
    mesh2  : second mesh
    descr1 : (n1,p) descriptors on the first mesh
    descr2 : (n2,p) descriptors on the second mesh
    D_a    : (k1,k1) area-based shape differnence operator
    D_c    : (k1,k1) conformal-based shape differnence operator

    Properties
    ----------------------
    FM_type : 'classic' | 'icp' | 'zoomout' which FM is currently used
    k1      : dimension of the first eigenspace (varies depending on the type of FM)
    k2      : dimension of the seconde eigenspace (varies depending on the type of FM)
    FM      : (k2,k1) current FM
    p2p     : (n2,) point to point map associated to the current functional map
    """
    def __init__(self, mesh1, mesh2):

        self.mesh1 = copy.deepcopy(mesh1)
        self.mesh2 = copy.deepcopy(mesh2)

        # Complete descriptors
        self.descr1 = None
        self.descr2 = None

        self.FM_type = 'classic'
        self.FM_base = None
        self.FM_icp = None
        self.FM_zo = None

        self._k1, self._k2 = None, None

    # DIMENSION PROPERTIES
    @property
    def k1(self):
        if self._k1 is None and not self.preprocessed and not self.fitted:
            raise ValueError('No information known about dimensions')
        if self.fitted:
            return self.FM.shape[1]
        else:
            return self._k1

    @k1.setter
    def k1(self, k1):
        self._k1 = k1

    @property
    def k2(self):
        if self._k2 is None and not self.preprocessed and not self.fitted:
            raise ValueError('No information known about dimensions')
        if self.fitted:
            return self.FM.shape[0]
        else:
            return self._k2

    @k2.setter
    def k2(self, k2):
        self._k2 = k2

    # FUNCTIONAL MAP SWITCHER (REFINED OR NOT)
    @property
    def FM_type(self):
        return self._FM_type

    @FM_type.setter
    def FM_type(self, FM_type):
        if FM_type.lower() not in ['classic', 'icp', 'zoomout']:
            raise ValueError(f'FM_type can only be set to "classic", "icp" or "zoomout", not {FM_type}')
        self._FM_type = FM_type

    def change_FM_type(self, FM_type):
        self.FM_type = FM_type

    @property
    def FM(self):
        """
        Returns the current functional map depending on the value of FM_type

        Output
        ----------------
        FM : (k2,k1) current FM
        """
        if self.FM_type.lower() == 'classic':
            return self.FM_base
        elif self.FM_type.lower() == 'icp':
            return self.FM_icp
        elif self.FM_type.lower() == 'zoomout':
            return self.FM_zo

    @FM.setter
    def FM(self, FM):
        self.FM_base = FM

    # BOOLEAN PROPERTIES
    @property
    def preprocessed(self):
        """
        check if enough information is provided to fit the model
        """
        test_descr = (self.descr1 is not None) and (self.descr2 is not None)
        test_evals = (self.mesh1.eigenvalues is not None) and (self.mesh2.eigenvalues is not None)
        test_evects = (self.mesh1.eigenvectors is not None) and (self.mesh2.eigenvectors is not None)

        return test_descr and test_evals and test_evects

    @property
    def fitted(self):
        return self.FM is not None

    @property
    def p2p(self):
        """
        Computes the current point to point map

        Output
        --------------------
        p2p : (n2,) point to point map associated to the current functional map
        """
        if not self.fitted:
            raise ValueError('Model should be fit and fit to obtain p2p map')

        return spectral.FM_to_p2p(self.FM, self.mesh1.eigenvectors, self.mesh2.eigenvectors)

    def precise_map(self,precompute=True):
        """
        Returns a precise map between the two meshes using the map deblurring paper

        Paramaters
        -------------------
        precompute : Set to precompute values for faster computation but heavier
                     memory use.

        Output
        -------------------
        precise_map : (n2,n1) sparse - precise map
        """
        if not self.fitted:
            raise ValueError('Model should be fit and fit to obtain p2p map')

        return spectral.precise_map.precise_map(self.mesh1,self.mesh2, self.FM, precompute_dmin=precompute)

    def preprocess(self, n_ev=(50,50), n_descr=100, descr_type='WKS', landmarks=None, subsample_step=1, fem_area=False, verbose=False):
        """
        Saves the information about the Laplacian mesh for opt

        Parameters
        -----------------------------
        n_ev           : (k1, k2) tuple - with the number of Laplacian eigenvalues to consider.
        n_descr        : int - number of descriptors to consider
        descr_type     : str - "HKS" | "WKS"
        landmarks      : (p,1|2) array of indices of landmarks to match.
                         If (p,1) uses the same indices for both.
        subsample_step : int - step with which to subsample the descriptors.
        fem_area       : bool - Whether to compute the area matrix using finite element method
                         instead of the diagonal matrix.
        """
        self.k1,self.k2 = n_ev

        use_lm = landmarks is not None and len(landmarks) > 0

        # Compute the Laplacian spectrum
        if verbose:
            print('\nComputing Laplacian spectrum')
        if self.mesh1.eigenvalues is None or len(self.mesh1.eigenvalues) < self.k1:
            self.mesh1.process(max(self.k1,200), fem_area=fem_area, verbose=verbose)
        if self.mesh2.eigenvalues is None or len(self.mesh2.eigenvalues) < self.k2:
            self.mesh2.process(max(self.k2,200), fem_area=fem_area, verbose=verbose)

        if verbose:
            print('\nComputing descriptors')

        # Extract landmarks indices
        if use_lm:
            if len(landmarks.shape) == 1 or landmarks.shape[1] == 1:
                if verbose:
                    print('\tUsing same landmarks indices for both meshes')
                lm1 = landmarks.flatten()
                lm2 = lm1
            else:
                lm1,lm2 = landmarks[:,0],landmarks[:,1]

        # Compute descriptors
        if descr_type == 'HKS':
            self.descr1 = sg.mesh_HKS(self.mesh1,n_descr,k=self.k1)  # (N1, n_descr)
            self.descr2 = sg.mesh_HKS(self.mesh2,n_descr,k=self.k2)  # (N2, n_descr)

            if use_lm:
                lm_descr1 = sg.mesh_HKS(self.mesh1,n_descr,landmarks=lm1,k=self.k1)  # (N1, p*n_descr)
                lm_descr2 = sg.mesh_HKS(self.mesh2,n_descr,landmarks=lm2,k=self.k2)  # (N2, p*n_descr)

                self.descr1 = np.hstack([self.descr1,lm_descr1])  # (N1, (p+1)*n_descr)
                self.descr2 = np.hstack([self.descr2,lm_descr2])  # (N2, (p+1)*n_descr)

        elif descr_type == 'WKS':
            self.descr1 = sg.mesh_WKS(self.mesh1,n_descr,k=self.k1)  # (N1, n_descr)
            self.descr2 = sg.mesh_WKS(self.mesh2,n_descr,k=self.k2)  # (N2, n_descr)

            if use_lm:
                lm_descr1 = sg.mesh_WKS(self.mesh1,n_descr,landmarks=lm1,k=self.k1)  # (N1, p*n_descr)
                lm_descr2 = sg.mesh_WKS(self.mesh2,n_descr,landmarks=lm2,k=self.k2)  # (N2, p*n_descr)

                self.descr1 = np.hstack([self.descr1,lm_descr1])  # (N1, (p+1)*n_descr)
                self.descr2 = np.hstack([self.descr2,lm_descr2])  # (N2, (p+1)*n_descr)

        else:
            raise ValueError(f'Descriptor type "{descr_type}" not implemented')

        # Subsample descriptors
        self.descr1 = self.descr1[:, np.arange(0, self.descr1.shape[1], subsample_step)]
        self.descr2 = self.descr2[:, np.arange(0, self.descr2.shape[1], subsample_step)]

        # Normalize descriptors
        if verbose:
            print('\tNormalizing descriptors')

        no1 = np.sqrt(self.mesh1.l2_sqnorm(self.descr1))  # (p,)
        no2 = np.sqrt(self.mesh2.l2_sqnorm(self.descr2))  # (p,)

        self.descr1 /= no1[None, :]
        self.descr2 /= no2[None, :]

        if verbose:
            n_lmks = np.asarray(landmarks).shape[0] if use_lm else 0
            print(f'\n\t{self.descr1.shape[1]} out of {n_descr*(1+n_lmks)} possible descriptors kept')

        return self

    def fit(self, descr_mu=1e-1, lap_mu=1e-3, descr_comm_mu=1, orient_mu=0, orient_reversing=False, optinit='zeros', verbose=False):
        """
        Solves the functional map optimization problem :

        min_C descr_mu * ||C@A - B||^2 + descr_comm_mu * (sum_i ||C@D_Ai - D_Bi@C||^2)
              + lap_mu * ||C@L1 - L2@C||^2 + orient_mu * (sum_i ||C@G_Ai - G_Bi@C||^2)

        with A and B descriptors, D_Ai and D_Bi multiplicative operators extracted
        from the i-th descriptors, L1 and L2 laplacian on each shape, G_Ai and G_Bi
        orientation preserving (or reversing) operators association to the i-th descriptors.

        Parameters
        -------------------------------
        descr_mu         : scaling for the descriptor preservation term
        lap_mu           : scaling of the laplacian commutativity term
        descr_comm_mu    : scaling of the multiplicative operator commutativity
        orient_mu        : scaling of the orientation preservation term
                           (in addition to relative scaling with the other terms as in the original
                           code)
        orient_reversing : Whether to use the orientation reversing term instead of the orientation
                           preservation one
        optinit          : 'random' | 'identity' | 'zeros' initialization.
                           In any case, the first column of the functional map is computed by hand
                           and not modified during optimization
        """
        if optinit not in ['random','identity', 'zeros']:
            raise ValueError(f"optinit arg should be 'random', 'identity' or 'zeros', not {optinit}")

        if not self.preprocessed:
            self.preprocess()

        # Project the descriptors on the LB basis
        descr1_red = self.project(self.descr1, mesh_ind=1)  # (n_ev1, n_descr)
        descr2_red = self.project(self.descr2, mesh_ind=2)  # (n_ev2, n_descr)

        # Compute multiplicative operators associated to each descriptor
        list_descr = []
        if descr_comm_mu > 0:
            if verbose:
                print('Computing commutativity operators')
            list_descr = self.compute_new_descr()  # (n_descr, ((k1,k1), (k2,k2)) )

        # Compute orientation operators associated to each descriptor
        orient_op = []
        if orient_mu > 0:
            if verbose:
                print('Computing orientation operators')
            orient_op = self.compute_orientation_op(reversing=orient_reversing)  # (n_descr,)

        # Compute the squared differences between eigenvalues for LB commutativity
        ev_sqdiff = np.square(self.mesh1.eigenvalues[None, :self.k1] - self.mesh2.eigenvalues[:self.k2, None])  # (n_ev2,n_ev1)
        ev_sqdiff /= np.linalg.norm(ev_sqdiff)**2

        # rescale orientation term
        if orient_mu > 0:
            args_native = (np.eye(self.k2,self.k1),
                           descr_mu, lap_mu, descr_comm_mu, 0,
                           descr1_red, descr2_red,list_descr, orient_op, ev_sqdiff)

            eval_native = opt_func.energy_func_std(*args_native)
            eval_orient = opt_func.oplist_commutation(np.eye(self.k2,self.k1), orient_op)
            orient_mu *= eval_native / eval_orient
            if verbose:
                print(f'\tScaling orientation preservation weight by {eval_native / eval_orient:.1e}')

        # Arguments for the optimization problem
        args = (descr_mu, lap_mu, descr_comm_mu, orient_mu,
                descr1_red, descr2_red, list_descr, orient_op, ev_sqdiff)

        # Initialization
        x0 = self.get_x0(optinit=optinit)

        if verbose:
            print(f'\nOptimization :\n'
                  f'\t{self.k1} Ev on source - {self.k2} Ev on Target\n'
                  f'\tUsing {self.descr1.shape[1]} Descriptors\n'
                  f'\tHyperparameters :\n'
                  f'\t\tDescriptors preservation :{descr_mu:.1e}\n'
                  f'\t\tDescriptors commutativity :{descr_comm_mu:.1e}\n'
                  f'\t\tLaplacian commutativity :{lap_mu:.1e}\n'
                  f'\t\tOrientation preservation :{orient_mu:.1e}\n'
                  )

        # Optimization
        start_time = time.time()
        res = fmin_l_bfgs_b(opt_func.energy_func_std, x0.ravel(), fprime=opt_func.grad_energy_std, args=args)
        opt_time = time.time() - start_time
        self.FM = res[0].reshape((self.k2,self.k1))

        if verbose:
            print("\tTask : {task}, funcall : {funcalls}, nit : {nit}, warnflag : {warnflag}".format(**res[2]))
            print(f'\tDone in {opt_time:.2f} seconds')

    def icp_refine(self, nit=5, tol=None, overwrite=True, verbose=False):
        """
        Refines the functional map using ICP and saves the result

        Parameters
        -------------------
        nit       : int - number of iterations of icp to apply
        tol       : float - threshold of change in functional map in order to stop refinement
                    (only applies if nit is None)
        overwrite : bool - If True changes FM type to 'icp' so that next call of self.FM
                    will be the icp refined FM
        """
        if not self.fitted:
            raise ValueError("The Functional map must be fit before refining it")

        self.FM_icp = icp.mesh_icp_refine(self.mesh1, self.mesh2, self.FM,
                                          nit=nit, tol=tol, verbose=verbose)
        if overwrite:
            self.FM_type = 'icp'
        return self

    def zoomout_refine(self, nit=10, step=1, subsample=None, use_ANN=False, overwrite=True, verbose=False):
        """
        Refines the functional map using ZoomOut and saves the result

        Parameters
        -------------------
        nit       : int - number of iterations to do
        step      : increase in dimension at each Zoomout Iteration
        subsample : int - number of points to subsample for ZoomOut. If None or 0, no subsampling is done.
        use_ANN   : bool - If True, use approximate nearest neighbor
        overwrite : bool - If True changes FM type to 'zoomout' so that next call of self.FM
                    will be the zoomout refined FM (larger than the other 2)
        """
        if not self.fitted:
            raise ValueError("The Functional map must be fit before refining it")

        if subsample is None or subsample == 0:
            sub = None
        else:
            sub1 = self.mesh1.extract_fps(subsample)
            sub2 = self.mesh2.extract_fps(subsample)
            sub = (sub1,sub2)

        self.FM_zo = zoomout.mesh_zoomout_refine(self.mesh1, self.mesh2, self.FM, nit,
                                                 step=step, subsample=sub, use_ANN=use_ANN, verbose=verbose)
        if overwrite:
            self.FM_type = 'zoomout'
        return self

    def compute_SD(self):
        """
        Compute the shape difference operators associated to the functional map
        """
        if not self.fitted:
            raise ValueError("The Functional map must be fit before computing the shape difference")

        self.D_a = spectral.area_SD(self.FM)
        self.D_c = spectral.conformal_SD(self.FM, self.mesh1.eigenvalues, self.mesh2.eigenvalues)

    def get_x0(self, optinit="zeros"):
        """
        Returns the initial functional map for optimization.

        Parameters
        ------------------------
        optinit : 'random' | 'identity' | 'zeros' initialization.
                  In any case, the first column of the functional map is computed by hand
                  and not modified during optimization

        Output
        ------------------------
        x0 : corresponding initial vector
        """
        if optinit == 'random':
            x0 = np.random.random((self.k2, self.k1))
        elif optinit == 'identity':
            x0 = np.eye(self.k2, self.k1)
        else:
            x0 = np.zeros((self.k2, self.k1))

        # Sets the equivalence between the constant functions
        ev_sign = np.sign(self.mesh1.eigenvectors[0,0]*self.mesh2.eigenvectors[0,0])
        area_ratio = np.sqrt(self.mesh2.area/self.mesh1.area)

        x0[:,0] = np.zeros(self.k2)
        x0[0,0] = ev_sign * area_ratio

        return x0

    def compute_new_descr(self):
        """
        Compute the multiplication operators associated with the descriptors

        Output
        ---------------------------
        operators : n_descr long list of ((k1,k1),(k2,k2)) operators.
        """
        if not self.preprocessed:
            raise ValueError("Preprocessing must be done before computing the new descriptors")

        pinv1 = self.mesh1.eigenvectors[:,:self.k1].T @ self.mesh1.A  # (k1,n)
        pinv2 = self.mesh2.eigenvectors[:,:self.k2].T @ self.mesh2.A  # (k2,n)

        list_descr = [
                      (pinv1@(self.descr1[:,i,None]*self.mesh1.eigenvectors[:,:self.k1]),
                       pinv2@(self.descr2[:,i,None]*self.mesh2.eigenvectors[:,:self.k2])
                       )
                      for i in range(self.descr1.shape[1])
                      ]

        return list_descr

    def compute_orientation_op(self, reversing=False, normalize=False):
        """
        Compute orientation preserving or reversing operators associated to each descriptor.

        Parameters
        ---------------------------------
        reversing : whether to return operators associated to orientation inversion instead
                    of orientation preservation (return the opposite of the second operator)
        normalize : whether to normalize the gradient on each face. Might improve results
                    according to the authors

        Output
        ---------------------------------
        list_op : (n_descr,) where term i contains (D1,D2) respectively of size (k1,k1) and
                  (k2,k2) which represent operators supposed to commute.
        """
        n_descr = self.descr1.shape[1]

        # Precompute the inverse of the eigenvectors matrix
        pinv1 = self.mesh1.eigenvectors[:,:self.k1].T @ self.mesh1.A  # (k1,n)
        pinv2 = self.mesh2.eigenvectors[:,:self.k2].T @ self.mesh2.A  # (k2,n)

        # Compute the gradient of each descriptor
        grads1 = [self.mesh1.gradient(self.descr1[:,i], normalize=normalize) for i in range(n_descr)]
        grads2 = [self.mesh2.gradient(self.descr2[:,i], normalize=normalize) for i in range(n_descr)]

        # Compute the operators in reduced basis
        can_op1 = [pinv1 @ self.mesh1.orientation_op(gradf) @ self.mesh1.eigenvectors[:, :self.k1]
                   for gradf in grads1]

        if reversing:
            can_op2 = [- pinv2 @ self.mesh2.orientation_op(gradf) @ self.mesh2.eigenvectors[:, :self.k2]
                       for gradf in grads2]
        else:
            can_op2 = [pinv2 @ self.mesh2.orientation_op(gradf) @ self.mesh2.eigenvectors[:, :self.k2]
                       for gradf in grads2]

        list_op = list(zip(can_op1,can_op2))

        return list_op

    def project(self, func, k=None, mesh_ind=1):
        """
        Projects a function on the LB basis

        Parameters
        -----------------------
        func    : array - (n1|n2,p) evaluation of the function
        mesh_in : int  1 | 2 index of the mesh on which to encode

        Output
        -----------------------
        encoded_func : (n1|n2,p) array of decoded f
        """
        if k is None:
            k = self.k1 if mesh_ind == 1 else self.k2

        if mesh_ind == 1:
            return self.mesh1.project(func,k=k)
        elif mesh_ind == 2:
            return self.mesh2.project(func,k=k)
        else:
            raise ValueError(f'Only indices 1 or 2 are accepted, not {mesh_ind}')

    def decode(self, encoded_func, mesh_ind=2):
        """
        Decode a function from the LB basis

        Parameters
        -----------------------
        encoded_func : array - (k1|k2,p) encoding of the functions
        mesh_ind     : int  1 | 2 index of the mesh on which to decode

        Output
        -----------------------
        func : (n1|n2,p) array of decoded f
        """

        if mesh_ind == 1:
            return self.mesh1.decode(encoded_func)
        elif mesh_ind == 2:
            return self.mesh2.decode(encoded_func)
        else:
            raise ValueError(f'Only indices 1 or 2 are accepted, not {mesh_ind}')

    def transport(self, encoded_func, reverse=False):
        """
        transport a function from LB basis 1 to LB basis 2. 
        If reverse is True, then the functions are transposed the other way
        using the transpose of the functional map matrix

        Parameters
        -----------------------
        encoded_func : array - (k1|k2,p) encoding of the functions
        reverse      : bool If true, transpose from 2 to 1 using the transpose of the FM

        Output
        -----------------------
        transp_func : (n2|n1,p) array of new encoding of the functions
        """
        if not self.preprocessed:
            raise ValueError("The Functional map must be fit before transporting a function")

        if not reverse:
            return self.FM @ encoded_func
        else:
            return self.FM.T @ encoded_func

    def transfer(self, func, reverse=False):
        """
        Transfer a function from mesh1 to mesh2.
        If 'reverse' is set to true, then the transfer goes
        the other way using the transpose of the functional
        map as approximate inverser transfer.

        Parameters
        ----------------------
        func : (n1|n2,p) evaluation of the functons

        Output
        -----------------------
        transp_func : (n2|n1,p) transfered function

        """
        if not reverse:
            return self.decode(self.transport(self.project(func)))

        else:
            encoding = self.project(func, mesh_ind=2)
            return self.decode(self.transport(encoding,reverse=True),
                               mesh_ind=1
                               )
