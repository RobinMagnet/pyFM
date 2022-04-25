# pyFM - Python bindings for functional maps
pyFM is a pure python implementation of multiple tools used for Functional Maps computations. Namely, it implements shape signatures, functional map optimization and refinement algorithm, and above all an easy-to-use interface for using functional maps.

## Features

* A TriMesh class incuding many standard geometric measures with python code, including geodesic distances (using the [heat method](https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/)), normals, projection on the LBO basis, export of .off or .obj files with textures.
* A pure Python (fast) implementation of Laplace-Beltrami Operator.
* Implementation of [HKS](http://www.lix.polytechnique.fr/~maks/papers/hks.pdf) and [WKS](http://imagine.enpc.fr/~aubrym/projects/wks/index.html) (and their version for landmarks) with multiple level of automation for parameters selection (from full automatic to total control)
* Implementation of icp and [ZoomOut](https://arxiv.org/abs/1904.07865) on Python
* *Fast* conversion from Functional Map to vertex to vertex map or [precise map](https://www.cs.technion.ac.il/~mirela/publications/p2p_recovery.pdf) using p-dimensional [vertex to mesh projection](https://github.com/RobinMagnet/pyFM/blob/master/pyFM/spectral/projection_utils.py)
* A FunctionalMapping class for straightforward computation of Functional Maps mixing all the previous features
* Functions for evaluating functional maps.
* Support for Functional Map Networks : Consistent Latent Basis, Canonical Consistent Latent Basis, consistency weights, Consistent ZoomOut


In particular this codebade contains python implementations of the following papers :
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

## Incoming features
Python code for [Discrete Optimization](http://www.lix.polytechnique.fr/~maks/papers/SGP21_DiscMapOpt.pdf) and [Reversible Harmonic Map](https://www.cs.technion.ac.il/~mirela/publications/rhm.pdf) should be released soon.
Don't hesitate to reach out at <rmagnet@> <lix.polytechnique.fr> for requests.

## Dependencies

Hard dependencies are `numpy`, `scipy`, `tqdm`, `scikit-learn` for its KDTree implementation.

The main non-standard (optional) dependencies are [`potpourri3d`](https://github.com/nmwsharp/potpourri3d) for its robust geodesic distance computation and [`robust_laplacian`](https://github.com/nmwsharp/robust-laplacians-py) which provide an implementation of both intrinsic delaunay and tufted Laplacian. If these functionalities are not needed one can remove the imports [here](https://github.com/RobinMagnet/pyFM/blob/master/pyFM/mesh/trimesh.py) and [here](https://github.com/RobinMagnet/pyFM/blob/master/pyFM/mesh/geometry.py).


`pynndescent` (see [here](https://github.com/lmcinnes/pynndescent)) is an optional package which is only required if one wish to use Approximate Nearest Neighbor. Else it is not required.

I did not build on the [trimesh](https://github.com/mikedh/trimesh) package which has some strange behaviour with vertex reordering.

## Remark on Code notations

In the whole codebase, we consider pairs of meshes `mesh1` and `mesh2`. Functional maps always go **from** `mesh1` **to** `mesh2` (denoted `FM_12`) and pointwise maps always **from** `mesh2` **to** `mesh1` (denoted `p2p_21`)

## Example Code

Running the [example notebook](https://github.com/RobinMagnet/pyFM/blob/master/example_notebook.ipynb) gives you an overview of the package functions.
Note that this notebook requires the [`meshplot` package](https://skoch9.github.io/meshplot/), which is an easy to use interface for `pythreejs`, which allows to display mesh in an easy fashion on notebooks.

All functions in the package are documented, with a descriptions of parameters and output.

## Example Code for shape matching

See the [Example Notebook](https://github.com/RobinMagnet/pyFM/blob/master/example_notebook.ipynb) for example of code.
