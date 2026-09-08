"""Plotting functions for triangle meshes, point clouds and vector fields."""

import numpy as np

from ._convert import _require_pyvista, to_polydata


def _as_scalar_kind(scalars):
    """Return (array, is_rgb) for a scalar argument, or (None, False)."""
    if scalars is None:
        return None, False
    scalars = np.asarray(scalars)
    if scalars.ndim == 1:
        return scalars, False
    if scalars.ndim == 2 and scalars.shape[1] == 3:
        return scalars, True
    raise ValueError(f"`scalars` must be (n,) or (n,3), got shape {scalars.shape}")


def _finish(pl, show_plot, cpos=None):
    """Show the plotter, or hand it back to the caller."""
    if show_plot:
        pl.show(cpos=cpos)
        return None
    return pl


def plot_mesh(
    mesh,
    scalars=None,
    *,
    cmap="viridis",
    clim=None,
    points=None,
    points_scalars=None,
    points_cmap="viridis",
    points_clim=None,
    points_color="red",
    point_size=12.0,
    vfield=None,
    vfield_color="black",
    vfield_cmap="viridis",
    vfield_rescale=1.0,
    vfield_scale_by_norm=True,
    vfield_tolerance=None,
    wireframe=False,
    line_width=None,
    smooth=True,
    opacity=1.0,
    color="white",
    show_colorbar=False,
    interpolate_before_map=True,
    pl=None,
    camera_position=None,
    return_plot=False,
):
    """
    Plot a mesh or point cloud, optionally with extra points and a vector field.

    ``scalars`` holds the scalar or RGB data, ``cmap`` holds the *name* of the matplotlib colormap
    used to turn scalar data into colors.
    Values may be given per vertex or per face; the domain is deduced from the array length.

    Parameters
    ------------------------------
    mesh       : object exposing ``.vertices`` (n,3) and optionally ``.faces`` (m,3) (a point cloud has no faces or faces=None)
    scalars    : np.ndarray - (n|m,) scalar values or (n|m,3) RGB values in [0,1]
    cmap       : str - matplotlib colormap name, used when ``scalars`` is 1-dimensional
    clim       : (2,) - color limits for ``scalars``
    points     : np.ndarray - extra points drawn on top of the mesh. *integer* array is
                 read as vertex indices of ``mesh.vertices``; a float ``(3,)`` or ``(p,3)``
                 array is read as coordinates.
    points_scalars : np.ndarray - (p,) scalar or (p,3) RGB values coloring ``points``.
                 When None, the points are drawn in ``points_color``.
    points_cmap : str - matplotlib colormap name for scalar ``points_scalars``
    points_clim : (2,) - color limits for ``points_scalars``
    points_color : str - color used when ``points_scalars`` is None
    point_size : float - radius of the rendered points
    vfield     : np.ndarray - (n,3) or (m,3) vector field. A per-face field is drawn at the
                 face barycenters.
    vfield_color : str - a color name for every arrow (default ``"black"``), or the special
                 value ``"magnitude"`` to color each arrow by its norm using
                 ``vfield_cmap`` as matplotlib colormap name.
    vfield_cmap : str - matplotlib colormap name, used when ``vfield_color`` is
                 ``"magnitude"``
    vfield_rescale : float - global scale factor for arrow length
    vfield_scale_by_norm : bool - if True arrow length is proportional to the vector norm,
                 otherwise every arrow has the same length
    vfield_tolerance : float - merge arrows closer than this fraction of the bounding box, to
                 draw fewer of them
    wireframe  : bool - draw the mesh edges
    line_width : float - width of the wireframe edges
    smooth     : bool - use smooth shading
    opacity    : float - opacity of the mesh
    color      : str - solid color used when ``scalars`` is None
    show_colorbar : bool - display the color bar
    interpolate_before_map : bool - interpolate scalars before mapping them to colors
    pl         : pyvista.Plotter - plotter to draw into. If None, a new one is created.
    camera_position : str or (3,3) - camera position for the plot. If None, the default is used
    return_plot : bool - return the plotter instead of showing it

    Output
    ------------------------------
    pl : pyvista.Plotter - returned when ``pl`` was given or ``return_plot`` is True.
         Otherwise the plot is shown and None is returned.
    """
    pv = _require_pyvista()

    vertices = np.asarray(mesh.vertices)
    # is_pointcloud = mesh.faces is None or len(mesh.faces) == 0

    scalars, is_rgb = _as_scalar_kind(scalars)
    pv_mesh = to_polydata(mesh, scalars=scalars)
    is_pointcloud = pv_mesh.n_faces == 0

    scalars_name = None if scalars is None else "scalars"

    show_plot = False
    if pl is None:
        show_plot = not return_plot
        pl = pv.Plotter()

    if is_pointcloud:
        pl.add_points(
            pv_mesh,
            scalars=scalars_name,
            cmap=cmap,
            clim=clim,
            rgb=is_rgb,
            color=color,
            point_size=point_size,
            render_points_as_spheres=True,
            show_scalar_bar=show_colorbar,
            opacity=opacity,
        )
    else:
        pl.add_mesh(
            pv_mesh,
            scalars=scalars_name,
            cmap=cmap,
            clim=clim,
            rgb=is_rgb,
            color=color,
            smooth_shading=smooth,
            opacity=opacity,
            show_edges=wireframe,
            line_width=line_width,
            interpolate_before_map=interpolate_before_map,
            show_scalar_bar=show_colorbar,
        )

    if points is not None:
        points = np.asarray(points)
        if np.issubdtype(points.dtype, np.integer):
            points = vertices[points.reshape(-1)]
        elif points.ndim == 1:
            points = points[None]

        points_scalars, points_is_rgb = _as_scalar_kind(points_scalars)
        pl.add_points(
            points,
            scalars=points_scalars,
            rgb=points_is_rgb,
            cmap=points_cmap,
            clim=points_clim,
            color=points_color if points_scalars is None else None,
            point_size=point_size,
            render_points_as_spheres=True,
            show_scalar_bar=show_colorbar and points_scalars is not None,
        )

    if vfield is not None:
        vfield = np.asarray(vfield)
        if vfield.shape[0] == len(vertices):
            base_points = vertices
        elif vfield.shape[0] == len(mesh.faces):
            base_points = vertices[np.asarray(mesh.faces)].mean(1)
        else:
            raise ValueError(
                f"`vfield` has length {vfield.shape[0]}, expected {len(vertices)} (per-vertex)"
                f" or {len(mesh.faces)} (per-face)"
            )

        plot_arrows(
            base_points,
            vfield,
            color=vfield_color,
            cmap=vfield_cmap,
            rescale=vfield_rescale,
            scale_by_norm=vfield_scale_by_norm,
            tolerance=vfield_tolerance,
            show_colorbar=show_colorbar and vfield_color == "magnitude",
            pl=pl,
            return_plot=True,
        )

    return _finish(pl, show_plot, cpos=camera_position)


def plot_arrows(
    points,
    vfield,
    *,
    color="black",
    cmap="viridis",
    clim=None,
    rescale=1.0,
    scale_by_norm=True,
    tolerance=None,
    show_colorbar=False,
    opacity=1.0,
    pl=None,
    return_plot=False,
):
    """
    Plot a vector field as arrows.

    Each arrow starts at a point of ``points`` and is oriented along the matching row of
    ``vfield``.

    Parameters
    ------------------------------
    points     : np.ndarray - (p,3) or (3,) base point of each arrow
    vfield     : np.ndarray - (p,3) or (3,) vector carried by each arrow
    color      : str - a fixed color name for every arrow, or the special value
                 ``"magnitude"`` to color each arrow by its norm through ``cmap``
    cmap       : str - matplotlib colormap name, used when ``color`` is ``"magnitude"``
    clim       : (2,) - color limits, used when ``color`` is ``"magnitude"``
    rescale    : float - global scale factor for arrow length
    scale_by_norm : bool - if True arrow length is proportional to the vector norm, otherwise
                 every arrow has the same length
    tolerance  : float - merge points closer than this fraction of the bounding box, to draw
                 fewer arrows
    show_colorbar : bool - display the color bar
    opacity    : float - opacity of the arrows
    pl         : pyvista.Plotter - plotter to draw into. If None, a new one is created.
    return_plot : bool - return the plotter instead of showing it

    Output
    ------------------------------
    pl : pyvista.Plotter - returned when ``pl`` was given or ``return_plot`` is True.
         Otherwise the plot is shown and None is returned.
    """
    pv = _require_pyvista()

    points = np.asarray(points)
    vfield = np.asarray(vfield)
    if points.ndim == 1:
        points = points[None]
    if vfield.ndim == 1:
        vfield = vfield[None]
    if points.shape != vfield.shape:
        raise ValueError(
            f"`points` and `vfield` must have the same shape, got {points.shape} and {vfield.shape}"
        )

    pdata = pv.PolyData(points)
    pdata.point_data.set_vectors(vfield, name="vfield")
    pdata.point_data["magnitude"] = np.linalg.norm(vfield, axis=1)
    pdata.set_active_scalars("magnitude")

    glyphs = pdata.glyph(
        orient="vfield",
        scale="magnitude" if scale_by_norm else False,
        factor=rescale,
        tolerance=tolerance,
        geom=pv.Arrow(),
    )

    by_magnitude = color == "magnitude"

    show_plot = False
    if pl is None:
        show_plot = not return_plot
        pl = pv.Plotter()

    pl.add_mesh(
        glyphs,
        scalars="magnitude" if by_magnitude else None,
        color=None if by_magnitude else color,
        cmap=cmap,
        clim=clim,
        show_scalar_bar=show_colorbar and by_magnitude,
        opacity=opacity,
    )

    return _finish(pl, show_plot)


def normalize(f, vmin=0, vmax=1):
    """
    Normalize a function or a set of functions between vmin and vmax

    Parameters
    ----------------------------
    f : (n,) or (n,p) - one or multiple functions
    vmin : minimum value for the normalized function(s)
    vmax : maximum value for the normalized function(s)

    Output
    ---------------------------
    f_normalized : (n,) or (n,p) - normalized function(s). A constant function is mapped to
                   `vmin` instead of producing a division by zero.
    """
    f = np.asarray(f, dtype=float)

    if f.ndim == 1:
        f_norm = f - np.min(f)
        scale = np.max(f_norm)
    else:
        f_norm = f - np.min(f, axis=0, keepdims=True)
        scale = np.max(f_norm, axis=0, keepdims=True)

    # A constant function has a zero range: leave it at 0 rather than dividing by zero
    scale = np.where(scale == 0, 1, scale)
    f_norm = vmin + (vmax - vmin) * f_norm / scale

    return f_norm


def vertices_to_rgb(vertices, param=(-2, -1, 3)):
    """
    Transforms (x,y,z) coordinates into RGB values using some better transformation than XYZ->RGB. It parametrized by
    a rearrangement of the array [1,2,3] up to sign flip and column reordering.
    Default parameter value is [-2, -1, 3].

    Parameters
    ----------------------------
    vertices : (n,3) - x,y,z coordinates of vertices
    param    : (3,) - rearrangement of the [1,2,3] array up to sign flip and column reordering.
               The magnitude of each entry selects the source channel, its sign inverts it.

    Output
    ----------------------------
    cmap : (n,3) RGB value for each vertex, in [0,1]
    """
    param = np.asarray(param)

    if np.any(np.sort(np.abs(param)) != np.arange(1, 4)):
        raise ValueError(
            "'param' should use a reorganization of \
                          [1,2,3] up to sign flip and column switch"
        )

    # Invert some colors and switch some channels
    cmap = np.sign(param)[None, :] * vertices
    cmap = cmap[:, np.abs(param) - 1]

    cmap = normalize(np.cos(normalize(cmap)))

    return cmap
