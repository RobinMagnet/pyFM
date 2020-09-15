import numpy as np
from sklearn.neighbors import KDTree
import scipy.linalg

def icp_refine(L1,L2,C,nit):
    """
    Refine a functional map with ICP

    Parameters
    --------------------
    L1 : (n1,k1') basis on the source shape. k1'>= k1
    L2 : (n2,k2') basis on the target shape. k2'>= k2
    C  : (k2,k1)  functional map between the two reduced basis
    nit : int - number of iterations

    Output
    ---------------------
    C_icp  : icp-refined functional map C
    """

    K2,K1 = C.shape
    
    C_icp = C.copy()
    L1_icp = L1[:,:K1].copy()
    L2_icp = L2[:,:K2].copy()
    
    for _ in range(nit):
        tree = KDTree((C_icp@L1_icp.T).T,leaf_size=20) # Tree on (n1,K2)
        matches = tree.query(L2_icp,k=1,return_distance=False).flatten() # (n2,)        
        W,_,_,_ = scipy.linalg.lstsq(L2_icp,L1_icp[matches]) # (K2,K1)
        U,_,VT = scipy.linalg.svd(W)
        C_icp = U@np.eye(K2,K1)@VT
    
    return C_icp
