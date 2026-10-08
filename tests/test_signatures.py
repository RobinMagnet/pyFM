import numpy as np
import pytest

from pyFM.signatures.WKS_functions import auto_WKS


@pytest.mark.parametrize("landmarks", [None, np.array([0, 100, 1000])])
def test_auto_WKS_is_scale_invariant(cat, landmarks):
    """Scaling a mesh by s divides eigenvalues by s**2 and eigenvectors by s."""
    scale = 100.0
    cat.process(k=50)
    evals = cat.eigenvalues
    evects = cat.eigenvectors

    wks = auto_WKS(evals, evects, 20, landmarks=landmarks)
    wks_scaled = auto_WKS(evals / scale**2, evects / scale, 20, landmarks=landmarks)

    np.testing.assert_allclose(scale**2 * wks_scaled, wks, rtol=1e-8)
