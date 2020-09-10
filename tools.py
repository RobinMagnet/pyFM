import numpy as np
from sklearn.neighbors import KDTree
import scipy.sparse

import matplotlib.pyplot as plt

def cosine_list(v1,v2):
    """
    Computes the pairwise cosine between vectors

    Parameters
    ------------------------
        v1 : (n,3) array (works with n,d)
        v2 : (n,3) array (works with n,d)

    Output
    ------------------------
        output : (n,) array with the i-th entry being the cosine between
                 the i-th entry of v1 and the i-th entry of v2
        
    """

    # Normalize each vector
    v1_n = v1/np.linalg.norm(v1,axis=1,keepdims=True)
    v2_n = v2/np.linalg.norm(v2,axis=1,keepdims=True)

    # Compute pairwise dot product efficiently
    return np.einsum('ij,ij->i',v1_n,v2_n)

def get_angle(vec1,vec2):
    """
    returns the angle between two vectors
    """
    return np.arccos(np.dot(vec1, vec2) / \
                        (np.linalg.norm(vec1) * \
                        np.linalg.norm(vec2))
                    )

def cotangent(alpha):
    """
    The cotangent function
    """
    return 1/np.tan(alpha)

def cotangent_list(v1,v2):
    """
    Computes pairwise cotangent between vectors

    Parameters
    ------------------------
        v1 : (n,3) array (works with n,d)
        v2 : (n,3) array (works with n,d)

    Output
    ------------------------
        cotan_list : (n,) array where the i-th entry is the cotan 
                    of the angle between v1[i] and v2[i]
    """

    cosines = cosine_list(v1,v2) # (n,)

    return  cosines/np.sqrt(1-cosines**2)


def read_off(file):
    with open(file,'r') as f:
        if f.readline().strip() != 'OFF':
            raise TypeError('Not a valid OFF header')
        n_verts, n_faces, _ = [int(x) for x in f.readline().strip().split(' ')]
        vertices = [ [float(x) for x in f.readline().strip().split()] for _ in range(n_verts)]
        faces = [[int(x) for x in f.readline().strip().split()][1:] for _ in range(n_faces)]
    
    return np.asarray(vertices), np.asarray(faces)    


def write_off(filename, vertices, faces):
    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0]

    with open(filename,'w') as f:
        f.write('OFF\n')
        f.write(f'{n_vertices} {n_faces} 0\n')
        for i in range(n_vertices):
            f.write(f'{" ".join([str(coord) for coord in vertices[i]])}\n')
        
        for j in range(n_faces):
            f.write(f'3 {" ".join([str(tri) for tri in faces[j]])}\n')


def read_vert(file):
    vertices = [ [float(x) for x in line.strip().split()] for line in open(file,'r')]
    return np.asarray(vertices)

def read_tri(file, from_matlab = True):
    faces = [[int(x) for x in line.strip().split()] for line in open(file,'r')]
    return np.asarray(faces) - int(from_matlab)

def display_C(FM):
    _,axs=plt.subplots(1,3,figsize=(18,4))
    plt.sca(axs[0])
    plt.title(r'$C$')
    plt.imshow(FM);

    plt.sca(axs[1])
    plt.title(r'$C\cdot C^\top$')
    plt.imshow(FM@FM.T)

    plt.sca(axs[2])
    plt.title(r'$C^\top\cdot C$')
    plt.imshow(FM.T@FM);
    plt.show()


def icp_refine(L1,L2,C,nit):
    """
    L1 : (n1,K1)
    L2 : (n2,K2)
    C  : (K2,K1)
    nit : int
    """
    
    # K1 = L1.shape[1]
    # K2 = L2.shape[1]
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


def zoomout_refine(L1,L2,A2,C,nit):
    """
    L1 : (n1,K1) with K1 > K + nit
    L2 : (n2,K2) with K2 > K + nit
    A2 : (n2,n2) sparse area matrix
    C  : (K,K) ; FM from L1 to L2
    nit : int
    """
    
    assert C.shape[0] == C.shape[1], f"Zoomout input should be square not {C.shape}"
    
    
    n1 = L1.shape[0]
    n2 = L2.shape[0]
    
    kinit = C.shape[0]
    
    C_zo = C.copy()
    
    assert L1.shape[1] >= kinit+nit, f"Enough eigenvectors should be provided, here {kinit+nit} are needed when {L1.shape[1]} are provided"
    assert L2.shape[1] >= kinit+nit, f"Enough eigenvectors should be provided, here {kinit+nit} are needed when {L2.shape[1]} are provided"


    for k in range(kinit,kinit+nit):
        
        tree = KDTree((C_zo@L1[:,:k].T).T, leaf_size=20) # Tree on (n1,K2)
        matches = tree.query(L2[:,:k],k=1,return_distance=False).flatten() # (n2,)
        
        P = scipy.sparse.coo_matrix( (np.ones(n2),(np.arange(n2),matches)), shape=(n2,n1)).tocsc() # (N2,N1) point2point
        
        C_zo = L2[:,:k+1].T @ A2 @ P @ L1[:,:k+1] # (k+1,k+1)
        
    return C_zo


def get_P2P(FM,eigvects1,eigvects2):
    k2,k1 = FM.shape

    tree = KDTree((FM@eigvects1[:,:k1].T).T) # Tree on (n1,K2)
    matches = tree.query(eigvects2[:,:k2],k=1,return_distance=False).flatten() # (n2,)
    
    return matches # (n2,)

def farthest_point(D,k):
    inds = [np.random.randint(D.shape[0])]
    
    dists = D[inds]
    
    for _ in range(k-1):
        newid = np.argmax(dists)
        inds.append(newid)
        dists = np.minimum(dists,D[newid])
    
    return np.asarray(inds)


def get_SD(mesh1,mesh2,k1=None,k2=None,mapping=None,SD_type='spectral'):
    """
    SD_type : 'spectral' | 'semican' | 'can'
    
    mapping : (n2,n1)
    """
    assert SD_type in ['spectral', 'semican'], f"Problem with type of SD type : {SD_type}"
    
    if k1 is None:
        k1 = len(mesh1.eigenvalues)

    if k2 is None:
        k2 = 3*k1
        
    if mapping is None:
        mapping = np.eye(mesh1.n_vertices)
    
    # print(mesh1.eigenvectors.shape)
    # print(mesh2.eigenvectors.shape)


    if SD_type == 'spectral':
        FM = mesh2.eigenvectors[:,:k2].T @ mesh2.A @ mapping@ mesh1.eigenvectors[:,:k1] # (K2,K1)
        SD_a = FM.T @ FM # (K1,K1)
        SD_c = np.linalg.pinv(np.diag(mesh1.eigenvalues[:k1])) @ FM.T @ np.diag(mesh2.eigenvalues[:k2]) @FM # (K1,K1)
    
    elif SD_type == 'semican':
        FM = mapping@mesh1.eigenvectors[:,:k1] # (n2,K1)
        SD_a = FM.T @ mesh2.A @ FM # (K1,K1)
        SD_c = np.linalg.pinv(np.diag(mesh1.eigenvalues[:k1])) @ FM.T @ mesh2.W @FM # (K1,K1)
        
    return SD_a, SD_c

