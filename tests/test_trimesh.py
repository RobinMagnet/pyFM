"""Characterization tests for :class:`pyFM.mesh.TriMesh`.

These pin the observable behaviour of the class so it can be refactored safely.
Values are either derived from the mesh itself or taken from the current
implementation on ``examples/data/cat-00.off``.
"""

import copy
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from pyFM.mesh import TriMesh

# Reference quantities for cat-00.off
N_VERTICES = 7207
N_FACES = 14410
N_EDGES = 21615
AREA = 0.35022939765342315


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_load_from_path(cat):
    assert cat.n_vertices == N_VERTICES
    assert cat.n_faces == N_FACES
    assert cat.vertices.shape == (N_VERTICES, 3)
    assert cat.faces.shape == (N_FACES, 3)
    assert cat.vertices.dtype == float


def test_path_and_name_recorded(cat, cat_path):
    assert cat.path == cat_path
    assert cat.name == "cat-00"


def test_build_from_arrays(cat):
    mesh = TriMesh(cat.vertices, cat.faces)
    assert mesh.n_vertices == N_VERTICES
    assert mesh.n_faces == N_FACES
    np.testing.assert_allclose(mesh.vertices, cat.vertices)


def test_arrays_are_copied(cat):
    """The constructor must not alias caller-owned arrays."""
    verts = cat.vertices.copy()
    mesh = TriMesh(verts, cat.faces)
    verts[0] = 1e6
    assert not np.allclose(mesh.vertices[0], verts[0])


def test_point_cloud_has_no_faces(cat):
    pc = TriMesh(cat.vertices)
    assert pc.faces is None
    assert pc.n_faces == 0
    assert pc.n_vertices == N_VERTICES


def test_rejects_malformed_vertices():
    with pytest.raises(ValueError):
        TriMesh(np.zeros((10, 2)))
    with pytest.raises(ValueError):
        TriMesh(np.zeros(10))


def test_rejects_malformed_faces(cat):
    with pytest.raises(ValueError):
        TriMesh(cat.vertices, np.zeros((10, 4), dtype=int))


# ---------------------------------------------------------------------------
# Constructor options
# ---------------------------------------------------------------------------


def test_area_normalize(cat_path):
    mesh = TriMesh.load(cat_path, area_normalize=True)
    assert mesh.area == pytest.approx(1.0)
    assert mesh.is_normalized


def test_center(cat_path):
    mesh = TriMesh.load(cat_path, center=True)
    np.testing.assert_allclose(mesh.center_mass, 0.0, atol=1e-12)


def test_area_normalize_preserves_center_of_mass(cat, cat_path):
    """area_normalize scales about the center of mass, so it must not move it."""
    mesh = TriMesh.load(cat_path, area_normalize=True)
    np.testing.assert_allclose(mesh.center_mass, cat.center_mass, atol=1e-12)


def test_translation_option(cat, cat_path):
    t = np.array([1.0, 2.0, 3.0])
    mesh = TriMesh.load(cat_path, translation=t)
    np.testing.assert_allclose(mesh.vertices, cat.vertices + t)


def test_rotation_option(cat, cat_path):
    theta = 0.7
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mesh = TriMesh.load(cat_path, rotation=R)
    np.testing.assert_allclose(mesh.vertices, cat.vertices @ R.T)
    # rigid motion preserves area
    assert mesh.area == pytest.approx(cat.area)


def test_rotation_rejects_non_rotation(cat_path):
    with pytest.raises(ValueError):
        TriMesh.load(cat_path, rotation=2 * np.eye(3))


# ---------------------------------------------------------------------------
# Geometric quantities
# ---------------------------------------------------------------------------


def test_area(cat):
    assert cat.area == pytest.approx(AREA)
    assert cat.sqrt_area == pytest.approx(np.sqrt(AREA))


def test_areas_are_consistent(cat):
    """Face areas, vertex areas and total area must all agree."""
    assert cat.face_areas.sum() == pytest.approx(cat.area)
    assert cat.vertex_areas.sum() == pytest.approx(cat.area)
    assert cat.face_areas.shape == (N_FACES,)
    assert cat.vertex_areas.shape == (N_VERTICES,)


def test_edges(cat):
    edges = cat.edges
    assert edges.shape == (N_EDGES, 2)
    # Euler characteristic of a closed genus-0 surface: V - E + F = 2
    assert cat.n_vertices - len(edges) + cat.n_faces == 2
    assert cat.edge_lengths.shape == (N_EDGES,)
    assert (cat.edge_lengths > 0).all()


def test_normals_are_unit(cat):
    assert cat.face_normals.shape == (N_FACES, 3)
    np.testing.assert_allclose(np.linalg.norm(cat.face_normals, axis=1), 1.0)


def test_vertex_normals_are_unit(cat):
    assert cat.vertex_normals.shape == (N_VERTICES, 3)
    # geometry.per_vertex_normal_* divides by `1e-6 + norm` as a zero-guard, so
    # vertex normals are unit only up to that epsilon.
    np.testing.assert_allclose(np.linalg.norm(cat.vertex_normals, axis=1), 1.0, atol=1e-5)


def test_vertex_normal_weighting_switch(cat):
    area_normals = cat.vertex_normals.copy()
    cat.set_vertex_normal_weighting("uniform")
    uniform_normals = cat.vertex_normals
    assert not np.allclose(area_normals, uniform_normals)
    np.testing.assert_allclose(np.linalg.norm(uniform_normals, axis=1), 1.0, atol=1e-5)


def test_vertex_normal_weighting_rejects_unknown(cat):
    with pytest.raises(AssertionError):
        cat.set_vertex_normal_weighting("gaussian")


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------


def test_operators_are_none_before_process(cat):
    """Several callers branch on `is None`; this must stay true."""
    assert cat.stiffness is None
    assert cat.mass is None
    assert cat.eigenvalues is None
    assert cat.eigenvectors is None


def test_process_computes_spectrum(cat):
    cat.process(k=10)
    assert cat.eigenvalues.shape == (10,)
    assert cat.eigenvectors.shape == (N_VERTICES, 10)
    assert cat.stiffness.shape == (N_VERTICES, N_VERTICES)
    assert cat.mass.shape == (N_VERTICES, N_VERTICES)


def test_process_returns_self(cat):
    """Notebooks chain `TriMesh(...).process(...)`."""
    assert cat.process(k=5) is cat


def test_first_eigenvalue_is_zero(cat):
    cat.process(k=10)
    assert abs(cat.eigenvalues[0]) < 1e-8
    assert (np.diff(cat.eigenvalues) >= -1e-10).all()


def test_process_k_zero_builds_operators_only(cat):
    cat.process(k=0)
    assert cat.stiffness is not None
    assert cat.mass is not None
    assert cat.eigenvalues is None


def test_process_truncates_existing_spectrum(cat):
    cat.process(k=20)
    evals = cat.eigenvalues.copy()
    cat.process(k=5)
    assert cat.eigenvalues.shape == (5,)
    np.testing.assert_allclose(cat.eigenvalues, evals[:5])


def test_eigenvectors_are_mass_orthonormal(cat):
    cat.process(k=8)
    gram = cat.eigenvectors.T @ cat.mass @ cat.eigenvectors
    np.testing.assert_allclose(gram, np.eye(8), atol=1e-8)


def test_point_cloud_spectrum(cat):
    pc = TriMesh(cat.vertices)
    pc.process(k=5)
    assert pc.eigenvectors.shape == (N_VERTICES, 5)


# ---------------------------------------------------------------------------
# Projection / inner products
# ---------------------------------------------------------------------------


def test_project_unproject_shapes(cat):
    cat.process(k=20)
    f = cat.vertices[:, 0].copy()
    coeffs = cat.project(f)
    assert coeffs.shape == (20,)
    assert cat.unproject(coeffs).shape == (N_VERTICES,)
    assert cat.project(f, k=5).shape == (5,)


def test_project_multiple_functions(cat):
    cat.process(k=15)
    coeffs = cat.project(cat.vertices)
    assert coeffs.shape == (15, 3)
    assert cat.unproject(coeffs).shape == (N_VERTICES, 3)


def test_reconstruct_improves_with_k(cat):
    cat.process(k=100)
    f = cat.vertices[:, 0].copy()

    def err(k):
        return np.linalg.norm(cat.reconstruct(f, k=k) - f)

    assert err(100) < err(10)


def test_project_rejects_too_many_modes(cat):
    cat.process(k=10)
    with pytest.raises(ValueError):
        cat.project(cat.vertices[:, 0], k=50)


def test_l2_inner_matches_mass_matrix(cat):
    cat.process(k=0)
    f = cat.vertices[:, 0].copy()
    g = cat.vertices[:, 1].copy()
    assert cat.l2_inner(f, g) == pytest.approx(f @ cat.mass @ g)
    assert cat.l2_sqnorm(f) == pytest.approx(f @ cat.mass @ f)


def test_h1_inner_matches_stiffness_matrix(cat):
    cat.process(k=0)
    f = cat.vertices[:, 0].copy()
    g = cat.vertices[:, 1].copy()
    assert cat.h1_inner(f, g) == pytest.approx(f @ cat.stiffness @ g)
    assert cat.h1_sqnorm(f) == pytest.approx(f @ cat.stiffness @ f)


def test_integrate_constant_gives_area(cat):
    cat.process(k=0)
    assert cat.integrate(np.ones(N_VERTICES)) == pytest.approx(cat.area)


# ---------------------------------------------------------------------------
# Differential operators
# ---------------------------------------------------------------------------


def test_gradient_of_constant_is_zero(cat):
    grad = cat.gradient(np.ones(N_VERTICES))
    assert grad.shape == (N_FACES, 3)
    np.testing.assert_allclose(grad, 0.0, atol=1e-12)


def test_gradient_of_linear_function(cat):
    """grad of the x coordinate is the x axis projected on each face."""
    grad = cat.gradient(cat.vertices[:, 0].copy())
    expected = np.array([1.0, 0.0, 0.0]) - cat.face_normals * cat.face_normals[:, [0]]
    np.testing.assert_allclose(grad, expected, atol=1e-8)


def test_gradient_normalize(cat):
    grad = cat.gradient(cat.vertices[:, 0].copy(), normalize=True)
    np.testing.assert_allclose(np.linalg.norm(grad, axis=1), 1.0)


def test_divergence_shape(cat):
    grad = cat.gradient(cat.vertices[:, 0].copy())
    assert cat.divergence(grad).shape == (N_VERTICES,)


def test_orientation_op_vanishes_on_its_own_function(cat):
    """Op_f g measures <grad f x grad g, n>, so Op_f f must be zero."""
    f = cat.vertices[:, 0].copy()
    op = cat.orientation_op(cat.gradient(f))
    assert op.shape == (N_VERTICES, N_VERTICES)
    assert np.abs(op @ f).max() < 1e-10


def test_orientation_op_is_antisymmetric(cat):
    """Integrating <grad f x grad g, n> is antisymmetric under swapping f and g."""
    f = cat.vertices[:, 0].copy()
    g = cat.vertices[:, 1].copy()
    ones = np.ones(N_VERTICES)
    fg = ones @ (cat.orientation_op(cat.gradient(f)) @ g)
    gf = ones @ (cat.orientation_op(cat.gradient(g)) @ f)
    assert fg == pytest.approx(-gf, rel=1e-10)


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------


def test_translate_preserves_area(cat):
    area = cat.area
    cat.translate([1.0, 2.0, 3.0])
    assert cat.area == pytest.approx(area)


def test_scale_updates_area_and_spectrum(cat):
    cat.process(k=10)
    area, evals = cat.area, cat.eigenvalues.copy()
    cat.scale(2.0)
    assert cat.area == pytest.approx(4 * area)
    np.testing.assert_allclose(cat.eigenvalues, evals / 4)


def test_scale_keeps_eigenvectors_mass_orthonormal(cat):
    cat.process(k=8)
    cat.scale(3.0)
    gram = cat.eigenvectors.T @ cat.mass @ cat.eigenvectors
    np.testing.assert_allclose(gram, np.eye(8), atol=1e-8)


def test_transformations_return_self(cat):
    assert cat.translate([0.0, 0.0, 0.0]) is cat
    assert cat.scale(1.0) is cat
    assert cat.center() is cat
    assert cat.area_normalize() is cat


def test_setting_vertices_invalidates_spectrum(cat):
    cat.process(k=10)
    cat.vertices = 2 * cat.vertices
    assert cat.stiffness is None
    assert cat.mass is None
    assert cat.eigenvalues is None
    assert cat.eigenvectors is None


# ---------------------------------------------------------------------------
# Geodesics and sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["heat", "heat_pure", "dijkstra", "fast_marching"])
def test_geodesic_from_single_source(cat, method):
    dist = cat.geodesic_from(0, method=method)
    assert dist.shape == (N_VERTICES,)
    assert dist[0] == pytest.approx(0.0, abs=1e-6)
    assert (dist >= -1e-9).all()


def test_geodesic_from_multiple_sources(cat):
    dist = cat.geodesic_from(np.array([0, 10, 20]), method="heat")
    assert dist.shape == (N_VERTICES, 3)


def test_geodesic_from_rejects_unknown_method(cat):
    with pytest.raises(ValueError):
        cat.geodesic_from(0, method="teleport")


def test_geodesic_matrix_rejects_unknown_method(cat):
    with pytest.raises(ValueError):
        cat.geodesic_matrix(method="teleport")


def test_farthest_point_sampling(cat):
    fps = cat.farthest_point_sampling(20)
    assert fps.shape == (20,)
    assert len(np.unique(fps)) == 20
    assert fps.min() >= 0 and fps.max() < N_VERTICES


def test_farthest_point_sampling_euclidean(cat):
    fps = cat.farthest_point_sampling(15, geodesic=False)
    assert len(np.unique(fps)) == 15


def test_farthest_point_sampling_sub(cat):
    sub = np.arange(0, N_VERTICES, 7)
    fps, fps_sub = cat.farthest_point_sampling_sub(10, sub, return_sub_inds=True, geodesic=False)
    assert len(np.unique(fps)) == 10
    np.testing.assert_array_equal(sub[fps_sub], fps)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_save_off_roundtrip(cat, tmp_path):
    out = tmp_path / "out.off"
    cat.save(str(out))
    reloaded = TriMesh.load(str(out))
    np.testing.assert_allclose(reloaded.vertices, cat.vertices, atol=1e-10)
    np.testing.assert_array_equal(reloaded.faces, cat.faces)


def test_save_obj_roundtrip(cat, tmp_path):
    out = tmp_path / "out.obj"
    cat.save(str(out))
    reloaded = TriMesh.load(str(out))
    np.testing.assert_allclose(reloaded.vertices, cat.vertices, atol=1e-10)
    np.testing.assert_array_equal(reloaded.faces, cat.faces)


def test_save_defaults_to_off(cat, tmp_path):
    cat.save(str(tmp_path / "noext"))
    assert (tmp_path / "noext.off").is_file()


def test_save_returns_self(cat, tmp_path):
    assert cat.save(str(tmp_path / "out.off")) is cat


def test_load_rejects_unknown_format(tmp_path):
    bad = tmp_path / "mesh.stl"
    bad.write_text("solid\n")
    with pytest.raises(ValueError, match="Cannot read"):
        TriMesh.load(str(bad))


PLY_TRIANGLE = """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
0 1 0
3 0 1 2
"""


def test_load_ply(tmp_path):
    path = tmp_path / "triangle.ply"
    path.write_text(PLY_TRIANGLE)
    mesh = TriMesh.load(str(path))
    assert mesh.n_vertices == 3
    assert mesh.n_faces == 1
    assert mesh.area == pytest.approx(0.5)
    assert mesh.name == "triangle"


def test_get_uv(cat):
    uv = cat.get_uv(0, 2, 1.0)
    assert uv.shape == (N_VERTICES, 2)


# ---------------------------------------------------------------------------
# Copying — regressions for the deepcopy bug
# ---------------------------------------------------------------------------


def test_deepcopy_is_independent(cat):
    cat.process(k=10)
    other = copy.deepcopy(cat)
    other.translate([1.0, 0.0, 0.0])
    assert not np.allclose(other.vertices, cat.vertices)
    np.testing.assert_allclose(other.eigenvalues, cat.eigenvalues)


@pytest.mark.parametrize("method", ["heat", "fast_marching", "heat_pure", "dijkstra"])
def test_deepcopy_after_geodesics(cat, method):
    """Cached solvers must never make a mesh un-copyable."""
    cat.geodesic_from(0, method=method)
    other = copy.deepcopy(cat)
    assert other.n_vertices == cat.n_vertices


@pytest.mark.parametrize("method", ["heat", "fast_marching", "heat_pure", "dijkstra"])
def test_pickle_after_geodesics(cat, method):
    cat.geodesic_from(0, method=method)
    reloaded = pickle.loads(pickle.dumps(cat))
    np.testing.assert_allclose(reloaded.vertices, cat.vertices)


def test_functional_mapping_after_fps(cat, lion):
    """The reported crash: farthest_point_sampling caches a pp3d solver, then deepcopy fails."""
    from pyFM.functional import FunctionalMapping

    cat.farthest_point_sampling(20)
    model = FunctionalMapping(cat, lion)
    assert model.mesh1.n_vertices == cat.n_vertices


def test_copied_mesh_still_computes_geodesics(cat):
    cat.geodesic_from(0, method="heat")
    other = copy.deepcopy(cat)
    dist = other.geodesic_from(0, method="heat")
    assert dist.shape == (N_VERTICES,)


# ---------------------------------------------------------------------------
# Cache invalidation — regression for the stale-faces bug
# ---------------------------------------------------------------------------


def test_setting_faces_invalidates_derived_data(cat):
    cat.process(k=10)
    n_edges_before = len(cat.edges)
    cat.faces = cat.faces[:100]

    assert cat.n_faces == 100
    assert len(cat.edges) != n_edges_before, "edges must be recomputed"
    assert cat.stiffness is None, "stiffness matrix must be invalidated"
    assert cat.mass is None, "mass matrix must be invalidated"
    assert cat.eigenvectors is None, "spectrum must be invalidated"


def test_setting_faces_invalidates_face_quantities(cat):
    _ = cat.face_normals, cat.face_areas
    cat.faces = cat.faces[:100]
    assert cat.face_normals.shape == (100, 3)
    assert cat.face_areas.shape == (100,)


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


def test_short_operator_aliases_are_silent(cat):
    """W and A are kept as permanent short forms, they must not warn."""
    cat.process(k=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert cat.W is cat.stiffness
        assert cat.A is cat.mass


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("vertlist", "vertices"),
        ("facelist", "faces"),
        ("normals", "face_normals"),
        ("faces_areas", "face_areas"),
        ("edges_lengths", "edge_lengths"),
        ("sqrtarea", "sqrt_area"),
        ("meshname", "name"),
    ],
)
def test_deprecated_attribute_aliases(cat, old, new):
    with pytest.warns(DeprecationWarning):
        value = getattr(cat, old)

    expected = getattr(cat, new)
    if isinstance(value, np.ndarray):
        np.testing.assert_array_equal(value, expected)
    else:
        assert value == expected


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("geod_from", "geodesic_from"),
        ("extract_fps", "farthest_point_sampling"),
        ("decode", "unproject"),
        ("export", "save"),
        ("get_geodesic", "geodesic_matrix"),
        ("laplacian_spectrum", "compute_spectrum"),
        ("extract_fps_sub", "farthest_point_sampling_sub"),
    ],
)
def test_deprecated_methods_exist(cat, old, new):
    assert callable(getattr(cat, old))
    assert callable(getattr(cat, new))


def test_deprecated_decode_matches_unproject(cat):
    cat.process(k=20)
    coeffs = cat.project(cat.vertices[:, 0].copy())
    with pytest.warns(DeprecationWarning):
        decoded = cat.decode(coeffs)
    np.testing.assert_allclose(decoded, cat.unproject(coeffs))


def test_deprecated_extract_fps_matches_new_name(cat):
    with pytest.warns(DeprecationWarning):
        fps = cat.extract_fps(10, geodesic=False)
    assert len(np.unique(fps)) == 10


def test_deprecated_path_constructor(cat_path, cat):
    """TriMesh("mesh.off") still works, but warns."""
    with pytest.warns(DeprecationWarning):
        mesh = TriMesh(cat_path, area_normalize=True, center=True)
    assert mesh.n_vertices == cat.n_vertices
    assert mesh.name == "cat-00"
    assert mesh.area == pytest.approx(1.0)


def test_path_and_faces_together_is_an_error(cat_path, cat):
    with pytest.raises(TypeError):
        TriMesh(cat_path, cat.faces)


def test_from_file_is_load(cat_path):
    assert TriMesh.from_file(cat_path).n_vertices == N_VERTICES


def test_load_accepts_pathlib(cat_path):
    assert TriMesh.load(Path(cat_path)).n_vertices == N_VERTICES


def test_unknown_kwarg_raises(cat):
    """The old **kwargs constructor silently ignored typos."""
    with pytest.raises(TypeError):
        TriMesh(cat.vertices, cat.faces, aera_normalize=True)


def test_normalize_shorthand(cat_path):
    mesh = TriMesh.load(cat_path, normalize=True)
    assert mesh.area == pytest.approx(1.0)
    np.testing.assert_allclose(mesh.center_mass, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Point clouds
# ---------------------------------------------------------------------------


def test_is_point_cloud(cat):
    assert not cat.is_point_cloud
    assert TriMesh(cat.vertices).is_point_cloud


@pytest.mark.parametrize(
    "operation",
    ["edges", "face_normals", "vertex_normals", "face_areas"],
)
def test_point_cloud_rejects_face_quantities(cat, operation):
    pc = TriMesh(cat.vertices)
    with pytest.raises(ValueError, match="no faces"):
        getattr(pc, operation)


def test_point_cloud_rejects_geodesics(cat):
    pc = TriMesh(cat.vertices)
    with pytest.raises(ValueError, match="no faces"):
        pc.geodesic_from(0)


def test_point_cloud_area_is_none_before_processing(cat):
    assert TriMesh(cat.vertices).area is None


def test_repr(cat):
    assert "cat-00" in repr(cat)
    assert str(N_VERTICES) in repr(cat)
    assert "point cloud" in repr(TriMesh(cat.vertices))


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def test_save_rejects_unknown_format(cat, tmp_path):
    """This used to write nothing and raise nothing."""
    with pytest.raises(ValueError, match="Cannot write"):
        cat.save(str(tmp_path / "out.ply"))
    assert not (tmp_path / "out.ply").exists()


def test_save_accepts_pathlib(cat, tmp_path):
    cat.save(tmp_path / "out.off")
    assert (tmp_path / "out.off").is_file()


def test_save_obj_with_vertex_normals(cat, tmp_path):
    out = tmp_path / "out.obj"
    cat.save(str(out), vertex_normals=True)
    assert "vn " in out.read_text()


def test_save_obj_with_uv(cat, tmp_path):
    out = tmp_path / "out.obj"
    cat.save(str(out), uv=cat.get_uv(0, 2, 1.0))
    assert "vt " in out.read_text()


def test_save_off_rejects_obj_only_attributes(cat, tmp_path):
    with pytest.raises(ValueError, match="cannot store"):
        cat.save(str(tmp_path / "out.off"), uv=cat.get_uv(0, 2, 1.0))


def test_save_obj_rejects_face_colors(cat, tmp_path):
    colors = np.zeros((cat.n_faces, 3))
    with pytest.raises(ValueError, match="cannot store face colors"):
        cat.save(str(tmp_path / "out.obj"), face_colors=colors)


def test_save_texture_requires_uv(cat, tmp_path):
    with pytest.raises(ValueError, match="uv coordinates are required"):
        cat.save(str(tmp_path / "out.obj"), texture="texture_1.jpg")


def test_off_face_colors_use_a_valid_vertex_count(cat, tmp_path):
    """The leading number on an OFF face line is its vertex count, always 3."""
    from pyFM.mesh import file_utils

    out = tmp_path / "colored.off"
    colors = np.tile(np.array([0.1, 0.2, 0.3]), (cat.n_faces, 1))
    cat.save(str(out), face_colors=colors)

    face_lines = out.read_text().splitlines()[2 + cat.n_vertices :]
    assert all(line.startswith("3 ") for line in face_lines if line)

    verts, faces, read_colors = file_utils.read_off(str(out), read_colors=True)
    np.testing.assert_array_equal(faces, cat.faces)
    assert read_colors.shape == (cat.n_faces, 3)


def test_read_off_colors_on_a_file_without_faces(tmp_path):
    """This used to raise UnboundLocalError."""
    from pyFM.mesh import file_utils

    path = tmp_path / "cloud.off"
    path.write_text("OFF\n2 0 0\n0 0 0\n1 1 1\n")
    verts, faces, colors = file_utils.read_off(str(path), read_colors=True)
    assert verts.shape == (2, 3)
    assert faces is None
    assert colors is None


# ---------------------------------------------------------------------------
# Sampling from a precomputed distance matrix
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cat_distances(cat_path):
    """A geodesic distance matrix, computed once for the whole module."""
    return TriMesh.load(cat_path).geodesic_matrix(method="dijkstra")


def test_geodesic_matrix_shape(cat_distances):
    assert cat_distances.shape == (N_VERTICES, N_VERTICES)
    np.testing.assert_allclose(np.diag(cat_distances), 0.0)


def test_fps_with_precomputed_distances(cat, cat_distances):
    fps = cat.farthest_point_sampling(20, distances=cat_distances)
    assert len(np.unique(fps)) == 20


def test_fps_deterministic_without_random_init(cat, cat_distances):
    """random_init=False needs a distance matrix, and then it is reproducible."""
    first = cat.farthest_point_sampling(15, distances=cat_distances, random_init=False)
    second = cat.farthest_point_sampling(15, distances=cat_distances, random_init=False)
    np.testing.assert_array_equal(first, second)


def test_fps_sub_with_precomputed_distances(cat, cat_distances):
    sub = np.arange(0, N_VERTICES, 7)
    fps, fps_sub = cat.farthest_point_sampling_sub(
        10, sub, return_sub_inds=True, distances=cat_distances
    )
    assert len(np.unique(fps)) == 10
    np.testing.assert_array_equal(sub[fps_sub], fps)


def test_fps_sub_respects_return_sub_inds(cat, cat_distances):
    sub = np.arange(0, N_VERTICES, 7)
    fps = cat.farthest_point_sampling_sub(10, sub, distances=cat_distances)
    assert isinstance(fps, np.ndarray)
    assert len(np.unique(fps)) == 10
