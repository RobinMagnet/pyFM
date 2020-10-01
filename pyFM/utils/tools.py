import numpy as np
import matplotlib.pyplot as plt


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


def farthest_point(D,k,init='random'):
    if init=='random':
        inds = [np.random.randint(D.shape[0])]
    else:
        inds = [np.argmax(D.sum(1))]
    
    dists = D[inds]
    
    for _ in range(k-1):
        newid = np.argmax(dists)
        inds.append(newid)
        dists = np.minimum(dists,D[newid])
    
    return np.asarray(inds)