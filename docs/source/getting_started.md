Getting Started
===============

This short tutorial computes a functional map between two meshes and converts it
to a point-to-point correspondence. See the
[example notebooks](https://github.com/RobinMagnet/pyFM/tree/master/examples)
for complete, runnable versions.

Loading meshes
--------------

A `TriMesh` can be built from a file path or from raw vertex/face arrays:

```python
from pyFM.mesh import TriMesh

mesh1 = TriMesh("cat-00.off", area_normalize=True, center=False)
mesh2 = TriMesh("lion-00.off", area_normalize=True, center=False)
```

Computing a functional map
--------------------------

`FunctionalMapping` drives the standard workflow: compute the Laplace-Beltrami
spectrum and descriptors with `preprocess`, then optimize the map with `fit`.

```python
import numpy as np
from pyFM.functional import FunctionalMapping

model = FunctionalMapping(mesh1, mesh2)

# Spectrum + descriptors
model.preprocess(
    K=(35, 35),          # eigenvectors kept on (source, target)
    descr_type="WKS",    # "WKS" or "HKS"
    subsample_step=5,    # keep every 5th descriptor
    verbose=True,
)

# Optimize the functional map
model.fit(
    w_descr=1e0,         # descriptor preservation
    w_lap=1e-2,          # Laplacian commutativity
    w_dcomm=1e-1,        # descriptor commutativity
    w_orient=0,          # orientation term
    verbose=True,
)
```

Getting a point-to-point map
----------------------------

`get_p2p` converts the functional map ``model.FM`` into a vertex-to-vertex map
`p2p_21` (for each vertex of mesh2, the index of its match on mesh1):

```python
p2p_21 = model.get_p2p()
```

Refinement
----------

The map can be sharpened with ICP or ZoomOut. Both return a new functional map
rather than overwriting `model.FM`:

```python
FM_icp = model.icp_refine(verbose=True)
p2p_21_icp = model.get_p2p(FM_icp)

FM_zo = model.zoomout_refine(nit=10, step=5, verbose=True)
p2p_21_zo = model.get_p2p(FM_zo)
```

Evaluating accuracy
-------------------

Given an (approximate) ground-truth map and a geodesic distance matrix on the
source mesh, `pyFM.eval.accuracy` reports the mean geodesic error:

```python
import pyFM.eval

A_geod = mesh1.get_geodesic()
gt_p2p = np.loadtxt("lion2cat", dtype=int)

acc = pyFM.eval.accuracy(p2p_21, gt_p2p, A_geod, sqrt_area=mesh1.sqrtarea)
print(f"Mean geodesic error: {1e2 * acc:.2f}")
```
