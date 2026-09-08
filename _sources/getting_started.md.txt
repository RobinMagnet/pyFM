Getting Started
===============

This short tutorial computes a functional map between two meshes and converts it
to a point-to-point correspondence. It is the condensed version of
[Computing a map without a correspondence](auto_examples/plot_05_maps_without_correspondence),
which explains every step and shows what each one buys.

Loading meshes
--------------

A `TriMesh` is read from a file with `TriMesh.load`, or built directly from raw vertex/face arrays:

```python
from pyFM.mesh import TriMesh

mesh1 = TriMesh.load("cat-00.off", area_normalize=True, center=True)
mesh2 = TriMesh.load("lion-00.off", area_normalize=True, center=True)
```

Computing a functional map
--------------------------

`FunctionalMapping` drives the standard workflow: compute the Laplace-Beltrami
spectrum and descriptors with `preprocess`, then optimize the map with `fit`.

```python
from pyFM.functional import FunctionalMapping

model = FunctionalMapping(mesh1, mesh2)

# Spectrum + descriptors
model.preprocess(
    descr_type="WKS",    # "WKS" or "HKS"
    subsample_step=2,    # keep every 2nd descriptor
    verbose=True,
)

# Optimize the functional map
model.fit(
    K=30,           # eigenvectors kept on each shape
    w_descr=1e0,    # descriptor preservation
    w_lap=1e-1,     # Laplacian commutativity
    w_dcomm=1e0,    # descriptor commutativity
    w_orient=1e0,   # orientation term
    verbose=True,
)
```

Keep `w_orient` above zero. Descriptors alone cannot tell the left side of a shape
from its right, so with `w_orient=0` the optimization typically returns a
symmetry-flipped map that refinement will not fix.

Getting a point-to-point map
----------------------------

`get_p2p` converts the functional map `model.FM_12` into a vertex-to-vertex map
`p2p_21` (for each vertex of mesh2, the index of its match on mesh1):

```python
p2p_21 = model.get_p2p()
```

Refinement
----------

The output of `fit` is a starting point, not an answer — refinement is where most
of the accuracy comes from. ICP and ZoomOut both return a new functional map
rather than overwriting `model.FM_12`:

```python
FM_icp = model.icp_refine(verbose=True)
p2p_21_icp = model.get_p2p(FM_icp)

FM_zo = model.zoomout_refine(nit=16, step=5, verbose=True)
p2p_21_zo = model.get_p2p(FM_zo)
```

ZoomOut grows the map by `step` at each of its `nit` iterations, so `K = 30` ends
at `30 + 16 * 5 = 110`.

Evaluating accuracy
-------------------

Given an (approximate) ground-truth map and a geodesic distance matrix on the
source mesh, `pyFM.eval.accuracy` reports the mean geodesic error:

```python
import numpy as np
import pyFM.eval

A_geod = mesh1.geodesic_matrix()
gt_p2p = np.loadtxt("lion2cat", dtype=int)

for name, p2p in [("fit", p2p_21), ("ICP", p2p_21_icp), ("ZoomOut", p2p_21_zo)]:
    acc = pyFM.eval.accuracy(p2p, gt_p2p, A_geod, sqrt_area=mesh1.sqrt_area)
    print(f"{name:8s}: {1e2 * acc:.2f}")
```

On this pair that prints roughly `23.85`, `7.39` and `8.53`, in percent of
$\sqrt{\text{area}}$: the raw fit is coarse, and refinement is what makes it usable.

A few matched landmarks, passed to `preprocess` as `landmarks=`, cut the error
further still — see
[Computing a map without a correspondence](auto_examples/plot_05_maps_without_correspondence).
