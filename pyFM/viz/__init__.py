"""
PyVista-based visualization helpers.

This subpackage is optional: it is only importable with the ``viz`` extra installed::

    pip install 'pyfmaps[viz]'

Nothing here imports PyVista at module level, so ``import pyFM`` never requires it.

The plot functions are rather flexible and work with any Object that contains ``.vertices`` and ``.faces`` attributes.
A pointcloud object must have a ``.vertices``, and either no faces attributes, an empty list or None.

IN the functions, ``scalars`` is the data to color the mesh with (scalar or RGB), ``cmap`` is the name of a matplotlib
colormap used to transfer scalars to RGB.
"""

from ._convert import to_polydata
from .plot import plot_arrows, plot_mesh, vertices_to_rgb

__all__ = ["plot_mesh", "plot_arrows", "to_polydata", "vertices_to_rgb"]
