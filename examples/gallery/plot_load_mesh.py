"""
Loading and visualizing a mesh
==============================

The shortest possible pyFM session: read a triangle mesh from disk, inspect what
was loaded, and look at it.

Most of the 3D views below are live — drag to rotate, scroll to zoom. The last
one is rendered as a static image, to show that either is available per figure.
"""

# %%
# Locating the example data
# -------------------------
#
# The example meshes live in the pyFM repository under ``examples/data`` and are
# not shipped with the installed package, so we locate them by walking up from
# the current directory until we find them.

# sphinx_gallery_thumbnail_number = 2

from pathlib import Path

import pyvista as pv

from pyFM.mesh import TriMesh


def example_data(name):
    """Return the path to a mesh in the repository's ``examples/data`` folder."""
    for folder in [Path.cwd(), *Path.cwd().parents]:
        candidate = folder / "examples" / "data" / name
        if candidate.exists():
            return str(candidate)  # TriMesh expects a str, not a Path
    raise FileNotFoundError(
        f"{name!r} not found. These examples read data from the pyFM repository; "
        "clone it and run them from inside the checkout."
    )


# %%
# Reading a mesh
# --------------
#
# :class:`~pyFM.mesh.trimesh.TriMesh` accepts a path to an ``.off`` or ``.obj``
# file. ``area_normalize`` rescales the mesh to unit area and ``center`` moves
# its center of mass to the origin, which makes quantities computed on different
# shapes directly comparable.

mesh = TriMesh(example_data("cat-00.off"), area_normalize=True, center=True)

print(f"{mesh.n_vertices} vertices, {mesh.n_faces} faces")
print(f"total area: {mesh.area:.6f}")

# %%
# Vertices and faces are plain numpy arrays, available as ``vertlist``/``facelist``
# (or the ``vertices``/``faces`` aliases). Most other geometric quantities —
# normals, face areas, edges, the Laplacian — are computed lazily the first time
# you ask for them, and cached afterwards.

print("vertices:", mesh.vertlist.shape, mesh.vertlist.dtype)
print("faces   :", mesh.facelist.shape, mesh.facelist.dtype)
print("face areas:", mesh.faces_areas.shape)

# %%
# A first look
# ------------
#
# PyVista expects a flat connectivity array in which every face is prefixed by
# its vertex count. :meth:`pyvista.PolyData.from_regular_faces` does that for us,
# since all our faces are triangles.

poly = pv.PolyData.from_regular_faces(mesh.vertlist, mesh.facelist)

pl = pv.Plotter()
pl.add_mesh(poly, color="#b0bec5", smooth_shading=True)
pl.show()

# %%
# Coloring by a scalar field
# --------------------------
#
# Any per-vertex array can be used as a scalar field. Here we simply take the
# height of each vertex; later examples use eigenfunctions of the Laplacian,
# descriptors and geodesic distances in exactly the same way.

height = mesh.vertlist[:, 1]

pl = pv.Plotter()
pl.add_mesh(poly, scalars=height, cmap="viridis", smooth_shading=True)
pl.show()

# %%
# Coloring by position
# --------------------
#
# Mapping the ``(x, y, z)`` coordinates onto ``(r, g, b)`` gives each vertex a
# color that varies smoothly across the surface. This is the standard way to
# display a correspondence between two shapes: color the source with these
# values, then transfer them through the map and color the target with the
# result. Matching colors then mean matching points.
#
# This one is rendered as a static image rather than an interactive scene — a
# per-figure choice that keeps the page light.

# sphinx_gallery_start_ignore
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

lo = mesh.vertlist.min(axis=0, keepdims=True)
hi = mesh.vertlist.max(axis=0, keepdims=True)
rgb = (mesh.vertlist - lo) / (hi - lo)

pl = pv.Plotter()
pl.add_mesh(poly, scalars=rgb, rgb=True, smooth_shading=True)
pl.show()

# %%
# .. note::
#
#    ``rgb`` here is an ``(n, 3)`` float array in ``[0, 1]``, which PyVista
#    renders directly when ``rgb=True``. A one-dimensional ``(n,)`` array is
#    instead mapped through the ``cmap`` colormap, as in the previous figure.

print("rgb array:", rgb.shape, f"range [{rgb.min():.2f}, {rgb.max():.2f}]")
