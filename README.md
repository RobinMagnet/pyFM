# pyFM
pyFM is a pure python implementation of multiple tools used for Functional Maps computations. Namely, it implements shape signatures, functional map optimization, refinement techniques, and above all an easy-to-use interface for computing maps.

## Features

* A TriMesh class allowing to compute the area, geodesic distances (using dijkstra not fast marching yet), to use the LB basis to project or decode functions, and compute squared norm of a function on the mesh
* A pure Python (fast) implementation of Laplace-Beltrami computation using FEM (taken from [mindboggles github](https://github.com/nipy/mindboggle))
* Implementation of HKS and WKS (and their version for landmarks) with multiple level of automation for parameters selection (from full automatic to total control)
* Implementation of icp and zoomout on Python
* Small functions like farthest point sampling, shape difference computations, conversion from Functional Map to vertex to vertex map 
* A FunctionalMapping class allowing straightforward computation of Functional Maps mixing all the previous features

Incoming :
* Support for Functional Map Networks : Consistent Latent Basis, Canonical Consistent Latent Basis, ISCM weights, Consistent ZoomOut

## Dependencies

There are few dependencies, namely `numpy`, `scipy`, `scikit-learn` for its KDTree implementation, and `matplotlib` for a single optional function.

`pynndescent' (see [here](https://github.com/lmcinnes/pynndescent)) is an optional package which is only required if one wish to use Approximate Nearest Neighbor. Else it is not required.

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
model = FunctionalMapping(mesh1,mesh2)

# Define parameters for descriptors computations and compute it
descr_type = 'WKS' # WKS or HKS
k1,k2 = 60,60 # Number of eigenvalues on source and Target
landmarks = np.loadtxt('data/landmarks.txt',dtype=int)[:5] # loading 5 landmarks
subsample_step = 5 # In order not to use too many descriptors*

model.preprocess(n_ev=(k1,k2), subsample_step=subsample_step, landmarks=landmarks, descr_type=descr_type, verbose=True)


# Define parameters for optimization and fit the Functional Map
descr_mu = 1e0
lap_mu = 1e-3
descr_comm_mu=1e-1

model.fit(descr_mu=descr_mu, lap_mu=lap_mu, descr_comm_mu=descr_comm_mu, verbose=True)

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
