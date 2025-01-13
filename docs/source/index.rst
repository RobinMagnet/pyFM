.. pyfmaps documentation master file, created by
   sphinx-quickstart on Thu Apr 25 23:18:17 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Python Bindings for Functional Map Computations !
=================================================

pyFM (or pyfmaps) is a Python library designed for computing and using functional maps, a powerful framework in shape analysis and geometry processing.
Functional maps provide a compact and robust way to represent correspondences between shapes by transforming point-to-point mappings into small matrices.

This package implements shape signatures, functional map optimization and refinement algorithms, and above all an easy-to-use interface for using functional maps.

The package is now in **v1.0.0** as it has been stable for quite a long time. It had been released on PyPI.

Key Features
------------

- **Spectral Analysis**: Automatically compute Laplace-Beltrami eigenfunctions for shapes, enabling efficient computations in the spectral domain.

- **Differential Geometry Tools**: Implements a variety of differential geometry tools directly in Python for advanced shape analysis workflows.

- **Functional Map Computation**: Straightforward tools to calculate or refine functional maps using descriptors, landmarks, or initial blurry correspondences.

- **Pointwise Correspondences**: Functions for navigating between point-to-point maps and functional maps.


Why pyFM?
---------

pyFM has been originally designed as a way to incorporate existing Matlab code into Python workflows.
It has now grown beyond that, with a variety of tools and utilities for shape analysis and geometry processing.
In short, `pyFM` is designed to be

- **User-Friendly:** With clear APIs and detailed documentation, `pyFM` is accessible to both beginners and experts in shape analysis.
- **Efficient:** Built with performance in mind, avoiding slow python loops.
- **Extensible:** Highly modular. Most functions can be easily extracted from the package and used in other projects, as they usually only require `numpy` arrays as input.
- **Research-Oriented:** Inspired by state-of-the-art research in geometry processing, making it a great choice for prototyping and academic projects.

Whether you are an academic researcher exploring functional map theory or an industry professional working on advanced shape analysis tasks, I hope `pyFM` can be a valuable tool in your experiments.


What’s Next?
------------

To get started with `pyFM`, check out the example notebook for a quick overview of the package’s capabilities.

.. To get started with `pyFM`, check out the [Installation Guide](#installation) and the [Quickstart Tutorial](#getting-started) for a hands-on introduction. For a deeper dive, explore the [API Reference](#detailed-api-reference) and [Tutorials](#tutorials) to learn more about advanced features and workflows.

.. Let’s unlock the potential of functional maps together with `pyFM`!

Table of Contents
-----------------
.. toctree::
   install.md
   getting_started.md
   examples.md

.. toctree::
   api