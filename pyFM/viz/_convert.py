"""Conversion from pyFM meshes (or raw arrays) to PyVista objects."""

import numpy as np


def _require_pyvista():
    """Import PyVista, with a message if the optional extra is missing."""
    try:
        import pyvista as pv
    except ImportError as err:  # pragma: no cover - depends on the install
        raise ImportError(
            "pyFM.viz requires PyVista, which is an optional dependency.\n"
            "    pip install 'pyfmaps[viz]'"
        ) from err
    return pv


def to_polydata(mesh=None, vertices=None, faces=None, scalars=None):
    """
    Convert a mesh to a ``pyvista.PolyData``.

    The mesh is given either as an object exposing ``.vertices`` (n,3) and ``.faces`` (m,3), such as
    a :class:`~pyFM.mesh.trimesh.TriMesh`.

    Parameters
    ------------------------------
    mesh     : object exposing ``.vertices`` and ``.faces``. Takes precedence over the
               ``vertices`` / ``faces`` arguments.
    vertices : np.ndarray - (n,3) vertex coordinates, used when ``mesh`` is None
    faces    : np.ndarray - (m,3) triangle indices, used when ``mesh`` is None. If None, a
               point cloud is built.
    scalars  : np.ndarray - (n|m,) scalar values or (n|m,3) RGB values in [0,1].
               Stored under the name ``"scalars"``.

    Output
    ------------------------------
    pv_mesh : pyvista.PolyData
    """
    pv = _require_pyvista()

    if mesh is not None:
        vertices = mesh.vertices
        try:
            faces = mesh.faces
            if faces is None or len(faces) == 0:
                faces = None
        except AttributeError:
            faces = None
    elif vertices is None:
        raise ValueError("Either `mesh` or `vertices` must be provided")

    vertices = np.asarray(vertices)
    n_vertices = len(vertices)
    n_faces = 0 if faces is None else len(faces)

    if n_faces == 0:
        pv_mesh = pv.PolyData(vertices)
    else:
        pv_mesh = pv.PolyData.from_regular_faces(vertices, faces)

    if scalars is not None:
        scalars = np.asarray(scalars)
        if scalars.shape[0] == n_vertices:
            pv_mesh.point_data["scalars"] = scalars
        elif scalars.shape[0] == n_faces:
            pv_mesh.cell_data["scalars"] = scalars
        else:
            raise ValueError(
                f"`scalars` has length {scalars.shape[0]}, expected {n_vertices} (per-vertex)"
                f" or {n_faces} (per-face)"
            )

    return pv_mesh
