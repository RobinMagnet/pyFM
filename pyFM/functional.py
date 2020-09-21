import copy
import time

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

import pyFM.signatures as sg
import pyFM.optimize as opt_func
import pyFM.refine
import pyFM.utils.spectral as spectral
import pyFM.utils.tools as tools


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
    def __init__(self,mesh1,mesh2):


        self.mesh1 = copy.deepcopy(mesh1)
        self.mesh2 = copy.deepcopy(mesh2)

        # Complete descriptors
        self.descr1 = None
        self.descr2 = None

        self.FM_type = 'classic'
        self.FM_base = None
        self.FM_icp = None
        self.FM_zo = None

        self._k1, self._k2 = None,None

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
    def k1(self,k1):
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
    def k2(self,k2):
        self._k2 = k2

    # FUNCTIONAL MAP SWITCHER (REFINED OR NOT)
    @property
    def FM_type(self):
        return self._FM_type

    @FM_type.setter
    def FM_type(self,FM_type):
        if FM_type.lower() not in ['classic', 'icp', 'zoomout']:
            raise ValueError(f'FM_type can only be set to "classic", "icp" or "zoomout", not {FM_type}')
        self._FM_type = FM_type

    def change_FM_type(self,FM_type):
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

    @property
    def p2p(self):
        """
        Computes the current point to point map

        Output
        --------------------
        p2p : (n2,) point to point map associated to the current functional map
        """
        if not self.fitted or not self.preprocessed:
            raise ValueError('Model should be processed and fit to obtain p2p map')

        return spectral.get_P2P(self.FM,self.mesh1.eigenvectors,self.mesh2.eigenvectors)

    
    # BOOLEAN PROPERTIES
    @property
    def preprocessed(self):
        test_descr = (self.descr1 is not None) and (self.descr2 is not None)
        # test_evals = (self.mesh1.eigenvalues is not None) and (self.mesh2.eigenvalues is not None)
        # test_evects = (self.mesh1.eigenvectors is not None) and (self.mesh2.eigenvectors is not None)

        return test_descr #and test_evals and test_evects
    
    @property
    def fitted(self):
        return self.FM is not None
    

    def preprocess(self, n_ev = (50,50), n_descr=100, descr_type='WKS', landmarks=None, subsample_step = 1, verbose = False):
        """
        Saves the information about the Laplacian mesh for opt

        Parameters
        -----------------------------
        n_ev        : (k1, k2) tuple - with the number of Laplacian eigenvalues to consider. Sets these parameters for the rest !!
        n_descr     : int - number of descriptors to consider
        descr_type  : str - "HKS" | "WKS"
        landmarks   : (p,1|2) array of indices of landmarks to match. If (p,1) uses the same indices for both
        subsample_step : int - step at which to subsample the descriptors (important to use w/ landmarks !)
        """
        self.k1,self.k2 = n_ev

        use_lm = landmarks is not None and len(landmarks) > 0

        # Compute the Laplacian spectrum
        if verbose:
            print('\nComputing Laplacian spectrum')
        if self.mesh1.eigenvalues is None or len(self.mesh1.eigenvalues) < self.k1:
            self.mesh1.process(max(self.k1,200),verbose=verbose)
        if self.mesh2.eigenvalues is None or len(self.mesh2.eigenvalues) < self.k2:
            self.mesh2.process(max(self.k2,200),verbose=verbose)

        
        if verbose:
            print('\nComputing descriptors')

        if use_lm:
            if len(landmarks.shape) == 1 or landmarks.shape[1]==1:
                if verbose:
                    print('\tUsing same landmarks indices for both meshes')
                lm1 = landmarks.flatten()
                lm2 = lm1
            else:
                lm1,lm2 = landmarks[:,0],landmarks[:,1]

        # Compute descriptors
        if descr_type == 'HKS':
            self.descr1 = sg.mesh_HKS(self.mesh1,n_descr,k=self.k1) # (N1, n_descr)
            self.descr2 = sg.mesh_HKS(self.mesh2,n_descr,k=self.k2) # (N2, n_descr)

            if use_lm:
                lm_descr1 = sg.mesh_HKS(self.mesh1,n_descr,landmarks=lm1,k=self.k1) # (N1, p*n_descr)
                lm_descr2 = sg.mesh_HKS(self.mesh2,n_descr,landmarks=lm2,k=self.k2) # (N2, p*n_descr)

                self.descr1 = np.hstack([self.descr1,lm_descr1]) # (N1, (p+1)*n_descr)
                self.descr2 = np.hstack([self.descr2,lm_descr2]) # (N2, (p+1)*n_descr)

        elif descr_type == 'WKS':
            self.descr1 = sg.mesh_WKS(self.mesh1,n_descr,k=self.k1) # (N1, n_descr)
            self.descr2 = sg.mesh_WKS(self.mesh2,n_descr,k=self.k2) # (N2, n_descr)

            if use_lm:
                lm_descr1 = sg.mesh_WKS(self.mesh1,n_descr,landmarks=lm1,k=self.k1) # (N1, p*n_descr)
                lm_descr2 = sg.mesh_WKS(self.mesh2,n_descr,landmarks=lm2,k=self.k2) # (N2, p*n_descr)

                self.descr1 = np.hstack([self.descr1,lm_descr1]) # (N1, (p+1)*n_descr)
                self.descr2 = np.hstack([self.descr2,lm_descr2]) # (N2, (p+1)*n_descr)

        else:
            raise ValueError(f'Descriptor type "{descr_type}" not implemented')
        
        # Subsample descriptors
        self.descr1 = self.descr1[:,np.arange(0,self.descr1.shape[1],subsample_step)]
        self.descr2 = self.descr2[:,np.arange(0,self.descr2.shape[1],subsample_step)]

        # Normalize descriptors
        if verbose:
            print('\tNormalizing descriptors')

        no1 = np.sqrt(self.mesh1.l2_sqnorm(self.descr1))
        no2 = np.sqrt(self.mesh2.l2_sqnorm(self.descr2))

        self.descr1 /= no1[None,:]
        self.descr2 /= no2[None,:]

        if verbose:
            n_lmks = np.asarray(landmarks).shape[0] if use_lm else 0
            print(f'\n\t{self.descr1.shape[1]} out of {n_descr*(1+n_lmks)} possible descriptors kept')
            print('\tDone')

        return self


    
    def fit(self,descr_mu = 1e-1,lap_mu = 1e-3, descr_comm_mu=1, optinit='sign', verbose=False):
        """
        Solves the functional mapping problem and saves the computed Functional Map.

        Parameters
        -------------------------------
        descr_mu  : scaling of the descriptor loss
        lap_mu    : scaling of the laplacian commutativity loss
        descr_comm_mu   : scaling of the descriptor commutativity loss
        optinit : 'random' | 'identity' | 'sign' initialization. 'sign' sets all parameters to 0 and compute the top left component.
        """
        assert optinit in ['random','identity', 'sign'], f"optinit arg should be 'random', 'identity' or 'sign', not {optinit}"

        if not self.preprocessed:
            self.preprocess()

        # Project the descriptors on the LB basis
        descr1_red = self.project(self.descr1,mesh_ind=1) # (n_ev1, n_descr)
        descr2_red = self.project(self.descr2,mesh_ind=2) # (n_ev2, n_descr)

        # Compute the operators associated with each descriptor
        list_descr = []
        if descr_comm_mu > 0:
            if verbose:
                print('Computing new descriptors')
            list_descr = self.compute_new_descr() # (n_descr,)
            if verbose:
                print('\tDone')

        # Compute the squared differences between eigenvalues for LB commutativity
        ev_sqdiff = np.square(self.mesh1.eigenvalues[None,:self.k1] - self.mesh2.eigenvalues[:self.k2,None]) # (n_ev2,n_ev1)
        ev_sqdiff /= np.linalg.norm(ev_sqdiff)**2

        # Defines current optimization functions
        def energy_func(C,descr_mu,lap_mu,descr_comm_mu,descr1_red,descr2_red,list_descr,ev_sqdiff):
            """
            Evaluation of the energy
            """
            k1 = descr1_red.shape[0]
            k2 = descr2_red.shape[0]
            C = C.reshape((k2,k1))

            energy = 0
            
            if descr_mu > 0:
                energy += descr_mu * opt_func.descr_preservation(C,descr1_red,descr2_red)

            if lap_mu > 0:
                energy += lap_mu * opt_func.LB_commutation(C,ev_sqdiff)
            
            if descr_comm_mu > 0:
                energy += descr_comm_mu * opt_func.oplist_commutation(C,list_descr)

            return  energy

        def grad_energy(C,descr_mu,lap_mu,descr_comm_mu,descr1_red,descr2_red,list_descr,ev_sqdiff):
            """
            Gradient of the energy
            """
            k1 = descr1_red.shape[0]
            k2 = descr2_red.shape[0]
            C = C.reshape((k2,k1))

            gradient = np.zeros_like(C)
            
            if descr_mu > 0:
                gradient += descr_mu * opt_func.descr_preservation_grad(C,descr1_red,descr2_red)

            if lap_mu > 0:
                gradient += lap_mu * opt_func.LB_commutation_grad(C,ev_sqdiff)
            
            if descr_comm_mu > 0:
                gradient += descr_comm_mu * opt_func.oplist_commutation_grad(C,list_descr)

            return gradient.reshape(-1)

        # Constants
        args = (descr_mu,lap_mu,descr_comm_mu,descr1_red,descr2_red,list_descr,ev_sqdiff)

        # Initialization
        if optinit == 'random':
            x0 = np.random.random((self.k2,self.k1))
        elif optinit == 'identity':
            x0 = np.eye(self.k2, self.k1)
        elif optinit == 'sign':
            x0 = np.zeros((self.k2, self.k1))
            x0[0,0] = np.sign(self.mesh1.eigenvectors[0,0]*self.mesh2.eigenvectors[0,0])* np.sqrt(self.mesh2.A.sum()/self.mesh1.A.sum())

        if verbose:
            print(f'\nOptimization :\n'
                f'\t{self.k1} Ev on source - {self.k2} Ev on Target\n'
                f'\tUsing {self.descr1.shape[1]} Descriptors\n'
                f'\tHyperparameters :\n'
                f'\t\tDescriptors preservation :{descr_mu:.1e}\n'
                f'\t\tDescriptors commutativity :{descr_comm_mu:.1e}\n'
                f'\t\tLaplacian commutativity :{lap_mu:.1e}\n'
                )

        # Optimization
        start_time = time.time()
        res = fmin_l_bfgs_b(energy_func, x0.reshape(-1),fprime=grad_energy,args=args)
        opt_time = time.time() - start_time
        self.FM = res[0].reshape((self.k2,self.k1))

        if verbose:
            print("\tTask : {task}, funcall : {funcalls}, nit : {nit}, warnflag : {warnflag}".format(**res[2]))
            print(f'\tDone in {opt_time:.2f} seconds')
        
    
    def icp_refine(self,nit=5, overwrite=True):
        """
        Refines the functional map using ICP and saves the result

        Parameters
        -------------------
        nit       : int - number of iterations to do
        overwrite : bool - If True changes FM type to 'icp' so that next call of self.FM
                    will be the icp refined FM
        """
        if not self.fitted:
            raise ValueError("The Functional map must be fit before refining it")

        self.FM_icp = pyFM.refine.icp_refine(self.mesh1.eigenvectors[:,:self.k1],self.mesh2.eigenvectors[:,:self.k2],self.FM,nit)
        if overwrite:
            self.FM_type = 'icp'
        return self


    def zoomout_refine(self,nit=10,step=1, subsample=None, use_ANN=False, overwrite=True):
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


        self.FM_zo = pyFM.refine.zoomout_refine(self.mesh1.eigenvectors,self.mesh2.eigenvectors,self.mesh2.A,self.FM,nit,
                                                step=step, subsample=sub, use_ANN=use_ANN, return_p2p=False)
        if overwrite:
            self.FM_type = 'zoomout'
        return self

    def display_C(self):
        """
        Display the Functional Map
        """
        tools.display_C(self.FM)

    def compute_SD(self):
        """
        Compute the shape difference operators associated to the functional map

        """
        if not self.fitted:
            raise ValueError("The Functional map must be fit before computing the shape difference")

        self.D_a = self.FM.T @ self.FM
        self.D_c = np.linalg.pinv(np.diag(self.mesh1.eigenvalues[:self.k1])) @ self.FM.T @ (self.mesh2.eigenvalues[:self.k2,None] *self.FM)

    def compute_new_descr(self):
        """
        Compute the multiplication operators associated with the descriptors

        Output
        ---------------------------
        operators : n_descr long list of ((k1,k1),(k2,k2)) operators.
        """
        if not self.preprocessed:
            raise ValueError("Preprocessing must be done before computing the new descriptors")

        pinv1 = self.mesh1.eigenvectors[:,:self.k1].T @ self.mesh1.A # (k1,n)
        pinv2 = self.mesh2.eigenvectors[:,:self.k2].T @ self.mesh2.A # (k2,n)

        list_descr = [
                      (pinv1@(self.descr1[:,i,None]*self.mesh1.eigenvectors[:,:self.k1]),
                       pinv2@(self.descr2[:,i,None]*self.mesh2.eigenvectors[:,:self.k2])
                       )
                      for i in range(self.descr1.shape[1])
                      ]

        return list_descr

    def project(self,func,k=None,mesh_ind=1):
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
            k = self.k1 if mesh_ind==1 else self.k2

        if mesh_ind == 1:
            return self.mesh1.project(func,k=k) #self.mesh1.eigenvectors.T @ (self.mesh1.A @ func)
        elif mesh_ind == 2:
            return self.mesh2.project(func,k=k) #self.mesh2.eigenvectors.T @ (self.mesh2.A @ func)
        else:
            raise ValueError(f'Only indices 1 or 2 are accepted, not {mesh_ind}')

    def decode(self,encoded_func,mesh_ind=2):
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

        if mesh_ind==1:
            return self.mesh1.decode(encoded_func) #self.mesh1.eigenvectors @ encoded_func
        elif mesh_ind==2:
            return self.mesh2.decode(encoded_func) #self.mesh2.eigenvectors @ encoded_func
        else:
            raise ValueError(f'Only indices 1 or 2 are accepted, not {mesh_ind}')
        
    def transport(self,encoded_func, reverse=False):
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
        transp_func : (n2|n1,p) array of new encoding og the functions
        """
        if not self.preprocessed:
            raise ValueError("The Functional map must be fit before transporting a function")

        if not reverse:
            return self.FM @ encoded_func
        else:
            return self.FM.T @ encoded_func


    def transfer(self,func, reverse=False):
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
