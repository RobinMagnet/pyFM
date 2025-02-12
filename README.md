# pyFM - Python Functional Maps Library

![](doc/zoomout.gif)

[![](https://github.com/RobinMagnet/pyFM/actions/workflows/documentation.yml/badge.svg)](https://robinmagnet.github.io/pyFM/)

**NEW** API Documentation is available [here](https://robinmagnet.github.io/pyFM/)

This package contains a comprehensive Python implementation for shape correspondence using functional maps, featuring code from [multiple papers](#implemented-papers)

## Core Features
- Complete TriMesh class with geometric measures (geodesics, normals, LBO projections, differential operators)
- Fast Laplace-Beltrami Operator implementation
- Shape descriptors (HKS, WKS) with flexible parameter control
- Efficient correspondence refinement (ICP, ZoomOut)
- Fast functional-to-pointwise map conversion
- Functional Map Network utilities

## Installation
```bash
pip install pyfmaps
```

## Key Dependencies
- Required: numpy, scipy, tqdm, scikit-learn
- Optional: [`potpourri3d`](https://github.com/nmwsharp/potpourri3d) (geodesics), [`robust_laplacian`](https://github.com/nmwsharp/robust-laplacians-py) (Delaunay/tufted Laplacian)

## Notation Convention
- Functional maps (FM_12): mesh1 → mesh2
- Pointwise maps (p2p_21): mesh2 → mesh1

## Documentation & Examples
- [API Documentation](https://robinmagnet.github.io/pyFM/)
- [Example Notebook](https://github.com/RobinMagnet/pyFM/blob/master/example_notebook.ipynb)

## Implemented Papers
This library implements methods from several key papers in shape correspondence, including:

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

## Coming Soon
- [Discrete Optimization for Shape Matching](https://www.lix.polytechnique.fr/~maks/papers/SGP21_DiscMapOpt.pdf) and [Smooth Non-Rigid Shape Matching via Effective Dirichlet Energy Optimization](https://www.lix.polytechnique.fr/Labo/Robin.Magnet/3DV2022_smooth_corres/smooth_corres_main.pdf), already implemented [here](https://github.com/RobinMagnet/SmoothFunctionalMaps)
- [Reversible Harmonic Maps](https://dl.acm.org/doi/10.1145/3202660), already implemented [here](https://github.com/RobinMagnet/ReversibleHarmonicMaps)

# Contact and Citation

robin.magnet@inria.fr