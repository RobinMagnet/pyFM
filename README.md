# pyFM
pyFM is a pure python implementation of multiple tools used for Functional Maps computations. Namely, it implements shape signatures, functional map optimization, refinement techniques, and above all an easy-to-use interface for computing maps.

## Features

* A TriMesh class allowing to compute the area, geodesic distances (usign the [heat method](https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/)), normals, projection on the LBO basis, export .off or .obj files with textures.
* A pure Python (fast) implementation of Laplace-Beltrami computation using FEM or a diagonal area matrix.
* Implementation of [HKS](http://www.lix.polytechnique.fr/~maks/papers/hks.pdf) and [WKS](http://imagine.enpc.fr/~aubrym/projects/wks/index.html) (and their version for landmarks) with multiple level of automation for parameters selection (from full automatic to total control)
* Implementation of icp and [ZoomOut](https://arxiv.org/abs/1904.07865) on Python
* Conversion from Functional Map to vertex to vertex map or [precise map](https://www.cs.technion.ac.il/~mirela/publications/p2p_recovery.pdf) with barycentric coordinates.
* A FunctionalMapping class allowing straightforward computation of Functional Maps mixing all the previous features
* Functions for evaluating results
* Support for Functional Map Networks : Consistent Latent Basis, Canonical Consistent Latent Basis, consistency weights, Consistent ZoomOut


This code contains implementations of features described in the following papers :
 * [The Heat Method for Distance Computation](https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/)
 * [A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion](http://www.lix.polytechnique.fr/~maks/papers/hks.pdf)
 * [The Wave Kernel Signature: A Quantum Mechanical Approach To Shape Analysis](http://imagine.enpc.fr/~aubrym/projects/wks/index.html)
 * [ZoomOut: Spectral Upsampling for Efficient Shape Correspondence](https://arxiv.org/abs/1904.07865), with MatLab implementation [here](https://github.com/llorz/SGA19_zoomOut)
 * [Deblurring and Denoising of Maps between Shapes](https://www.cs.technion.ac.il/~mirela/publications/p2p_recovery.pdf), with Matlab implementation [here](https://mirela.net.technion.ac.il/publications/)
 * [Functional Maps: A Flexible Representation of Maps Between Shapes](http://www.lix.polytechnique.fr/~maks/papers/obsbg_fmaps.pdf)
 * [Informative Descriptor Preservation via Commutativity for Shape Matching](http://www.lix.polytechnique.fr/~maks/papers/fundescEG17.pdf)
 * [Continuous and Orientation-preserving Correspondences via Functional Maps](https://arxiv.org/abs/1806.04455), only the orientation preserving / reversing term, matlab implementation can be found [here](https://github.com/llorz/SGA18_orientation_BCICP_code)
 * [Map-Based Exploration of Intrinsic Shape Differences and Variability](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.642.4287&rep=rep1&type=pdf)
 * [An Optimization Approach to Improving Collections of Shape Maps](http://fodava.gatech.edu/files/reports/FODAVA-11-22.pdf)
 * [Limit Shapes – A Tool for Understanding Shape Differences and Variability in 3D Model Collections](http://www.lix.polytechnique.fr/~maks/papers/limit_shapes_SGP19.pdf)
 * [CONSISTENT ZOOMOUT: Efficient Spectral Map Synchronization](http://www.lix.polytechnique.fr/~maks/papers/ConsistentZoomOut_SGP2020.pdf), with Matlab implementation [here](https://github.com/llorz/SGA19_zoomOut)

## Dependencies

There are few dependencies, namely `numpy`, `scipy`, `tqdm` and `scikit-learn` for its KDTree implementation.

`pynndescent` (see [here](https://github.com/lmcinnes/pynndescent)) is an optional package which is only required if one wish to use Approximate Nearest Neighbor. Else it is not required.

I did not build on the [trimesh](https://github.com/mikedh/trimesh) package which has some strange behaviour with vertex reordering.

## Example Code

Running the [example notebook](https://github.com/RobinMagnet/pyFM/blob/master/example_notebook.ipynb) gives you an overview of the package functions.
Note that this notebook requires the [`meshplot` package](https://skoch9.github.io/meshplot/), which is an easy to use interface for `pythreejs`, which allows to display mesh in an easy fashion on notebooks.

All functions in the package are documented, with a descriptions of parameters and output.

## Example Code for shape matching

```python
import numpy as np
from pyFM.mesh import TriMesh
from pyFM.functional import FunctionalMapping

mesh1 = TriMesh('data/cat-00.off')
mesh2 = TriMesh('data/lion-00.off')
# One can also specify a mesh using vertices coordinates and indexes for faces

print(f'Mesh 1 : {mesh1.n_vertices:4d} vertices, {mesh1.n_faces:5d} faces\n'
      f'Mesh 2 : {mesh2.n_vertices:4d} vertices, {mesh2.n_faces:5d} faces')


# Initialize a FunctionalMapping object in order to compute maps
process_params = {
    'n_ev': (35,35),  # Number of eigenvalues on source and Target
    'landmarks': np.loadtxt('data/landmarks.txt',dtype=int)[:5],  # loading 5 landmarks
    'subsample_step': 5,  # In order not to use too many descriptors
    'descr_type': 'WKS',  # WKS or HKS
}

model = FunctionalMapping(mesh1,mesh2)
model.preprocess(**process_params,verbose=True)


# Define parameters for optimization and fit the Functional Map
fit_params = {
    'descr_mu': 1e0,
    'lap_mu': 1e-3,
    'descr_comm_mu': 1e-1,
    'orient_mu': 0
}

model.fit(**fit_params, verbose=True)


# One can access the functional map FM and vertex to vertex mapping p2p
FM = model.FM
p2p = model.p2p

# Refining is possible
model.icp_refine()
FM = model.FM
p2p = model.p2p

# Previous information is not lost, one just need to tell which kind of functional map should be time
model.change_FM_type('classic') # Chose between 'classic', 'icp' or 'zoomout'
FM = model.FM # This is now the original FM

model.zoomout_refine(nit=10) # This refines the current model.FM, be careful which FM type is used

# EVALUATION
import pyFM.eval
# Compute geodesic distance matrix on the cat mesh
A_geod = mesh1.get_geodesic(verbose=True)

# Load an approximate ground truth map
gt_p2p = np.loadtxt('data/lion2cat',dtype=int)

# Compute accuracies
model.change_FM_type('classic')
acc_base = pyFM.eval.accuracy(model.p2p, gt_p2p, A_geod, sqrt_area=np.sqrt(mesh1.area))
model.change_FM_type('icp')
acc_icp = pyFM.eval.accuracy(model.p2p, gt_p2p, A_geod, sqrt_area=np.sqrt(mesh1.area))
model.change_FM_type('zoomout')
acc_zo = pyFM.eval.accuracy(model.p2p, gt_p2p, A_geod, sqrt_area=np.sqrt(mesh1.area))

print(f'Accuracy results\n'
      f'\tBasic FM : {1e3*acc_base:.2f}\n'
      f'\tICP refined : {1e3*acc_icp:.2f}\n'
      f'\tZoomOut refined : {1e3*acc_zo:.2f}\n')




# Functions can now be transfered from one shape to another
random_func = np.random.random(model.k1) # Define a random function on the LB basis
random_func/= np.linalg.norm(random_func) # normalize the function
new_func = model.transport(random_func) # (k2,) encoding on the target shape

# If one wish to obtain vertex-wise functions to display them
random_func_canonical = model.decode(random_func,mesh_ind=1) # (n1,) chose on which mesh to decode
new_func_canonical = model.decode(new_func,mesh_ind=2) # (n2,) chose on which mesh to decode

```

## Example code for TriMesh class
```python
import numpy as np
from pyFM.mesh import TriMesh

mesh = TriMesh('data/cat-00.off')

mesh.process() # Computes naturally the first 150 eigenvalues and eigenvectors of the LB spectrum
eigenvalues = mesh.eigenvalues # (K,)
eigenvectors = mesh.eigenvectors # (N,K)
print(f'Total Area : {mesh.area}')

# Compute Geodesic matrix
M = mesh.get_geodesic() # not saved in the mesh for memory efficiency

# Check the eigenvectors are normalized on the mesh
sqnorms = mesh.l2_sqnorm(mesh.eigenvectors) # (K,)
```
