"""Tests for FunctionalMapping.fit()."""

import numpy as np
import pytest

from pyFM.functional import FunctionalMapping
from pyFM.mesh import TriMesh


@pytest.fixture(scope="module")
def model(cat_path, lion_path):
    mesh1 = TriMesh.load(cat_path, area_normalize=True, center=True)
    mesh2 = TriMesh.load(lion_path, area_normalize=True, center=True)
    model = FunctionalMapping(mesh1, mesh2)
    model.preprocess(descr_type="WKS", subsample_step=5)
    return model


@pytest.mark.parametrize(
    "weights",
    [
        {"w_descr": 1e-1, "w_lap": 1e-3, "w_dcomm": 1e0, "w_orient": 0},
        {"w_descr": 1e-1, "w_lap": 1e-3, "w_dcomm": 1e0, "w_orient": 1e0},
        {"w_descr": 1e-4, "w_lap": 1e-6, "w_dcomm": 1e-3, "w_orient": 0},
        {"w_descr": 0, "w_lap": 0, "w_dcomm": 0, "w_orient": 1e0},
    ],
)
@pytest.mark.parametrize("scale", [10, 1000])
def test_fit_is_invariant_to_weight_scaling(model, weights, scale):
    """Only the ratios of the weights matter, so a global rescaling changes nothing.

    Two things used to break this: the orientation weight was rescaled by a factor
    itself proportional to the other weights (making that term grow quadratically),
    and L-BFGS-B's stopping criteria have absolute floors (pgtol, and the max(., 1)
    denominator of the factr test) that trigger at different points once the energy
    is scaled.
    """
    model.fit(K=(20, 20), **weights)
    FM = model.FM_12.copy()

    model.fit(K=(20, 20), **{k: scale * v for k, v in weights.items()})
    FM_scaled = model.FM_12.copy()

    assert np.linalg.norm(FM_scaled - FM) / np.linalg.norm(FM) < 1e-8


# ----------------------------------------------------------------------
# Deprecated pre-1.3 API
# ----------------------------------------------------------------------


def test_FM_alias_reads_and_writes(model):
    model.fit(K=(20, 20))

    with pytest.warns(DeprecationWarning, match="FM_12"):
        assert model.FM is model.FM_12

    replacement = np.eye(20)
    with pytest.warns(DeprecationWarning, match="FM_12"):
        model.FM = replacement
    assert model.FM_12 is replacement


def test_preprocessed_alias(model):
    with pytest.warns(DeprecationWarning, match="has_spectral"):
        assert model.preprocessed == model.has_spectral


def test_project_decode_aliases(model):
    func = model.mesh1.eigenvectors[:, 1]

    with pytest.warns(DeprecationWarning, match="project"):
        coeffs = model.project(func, k=10, mesh_ind=1)
    assert coeffs.shape == (10,)

    with pytest.warns(DeprecationWarning, match="unproject"):
        back = model.decode(coeffs, mesh_ind=1)
    assert back.shape == func.shape


def test_transport_alias(model):
    model.fit(K=(20, 20))
    coeffs = np.arange(20, dtype=float)

    with pytest.warns(DeprecationWarning, match="transport"):
        assert np.allclose(model.transport(coeffs), model.FM_12 @ coeffs)
    with pytest.warns(DeprecationWarning, match="transport"):
        assert np.allclose(model.transport(coeffs, reverse=True), model.FM_12.T @ coeffs)


def test_preprocess_K_kwarg_still_accepted(cat_path, lion_path):
    """`preprocess(K=...)` moved to `fit()`, but the old keyword still works."""
    mesh1 = TriMesh.load(cat_path, area_normalize=True, center=True)
    mesh2 = TriMesh.load(lion_path, area_normalize=True, center=True)
    model = FunctionalMapping(mesh1, mesh2)

    with pytest.warns(DeprecationWarning, match="moved from `preprocess"):
        model.preprocess(descr_type="WKS", subsample_step=5, K=20)

    assert model.k1 == 20 and model.k2 == 20
