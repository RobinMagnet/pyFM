.. _api_documentation:

API Documentation
=================

Complete reference for the ``pyFM`` package, generated from the source
docstrings. Every function, class, and method has its own page — use the tables
below or the sidebar to navigate.

Core classes
------------

The high-level objects most users interact with.

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~pyFM.mesh.TriMesh
   ~pyFM.functional.FunctionalMapping

Functional Map Network
----------------------

Consistent map synchronization across a collection of shapes (the ``FMN``
class and its helpers).

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~pyFM.FMN.FMN.FMN
   ~pyFM.FMN.FMN.CLB_quad_form

Shape signatures
----------------

Heat and Wave Kernel Signatures.

.. autosummary::
   :toctree: generated
   :recursive:

   pyFM.signatures

Spectral operations
-------------------

Conversions between pointwise and functional maps, shape difference operators,
and nearest-neighbour utilities.

.. autosummary::
   :toctree: generated
   :recursive:

   pyFM.spectral

Map refinement
--------------

ICP and ZoomOut refinement of functional maps.

.. autosummary::
   :toctree: generated
   :recursive:

   pyFM.refine

Evaluation
----------

Geodesic accuracy, continuity, and coverage metrics for correspondences.

.. autosummary::
   :toctree: generated
   :recursive:

   pyFM.eval

Mesh geometry & I/O
-------------------

Standalone geometric operators (areas, normals, gradients, geodesics),
Laplacians, and mesh file readers/writers.

.. autosummary::
   :toctree: generated
   :recursive:

   pyFM.mesh.geometry
   pyFM.mesh.laplacian
   pyFM.mesh.file_utils

Optimization internals
----------------------

Energy terms and gradients used by :class:`~pyFM.functional.FunctionalMapping`.

.. autosummary::
   :toctree: generated
   :recursive:

   pyFM.optimize
