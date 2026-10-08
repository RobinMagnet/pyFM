import numpy as np

from pyFM.mesh.geometry import _normalize_vectors


def test_normalize_vectors_keeps_zero_vectors_at_zero():
    vectors = np.zeros((4, 2, 3))
    vectors[0, 0] = [3.0, 0.0, 4.0]
    vectors[2, 1] = [1e-100, 0.0, 0.0]

    normalized = _normalize_vectors(vectors)

    np.testing.assert_allclose(normalized[0, 0], [0.6, 0.0, 0.8])
    np.testing.assert_allclose(normalized[2, 1], [1.0, 0.0, 0.0])
    assert np.isfinite(normalized).all()
    np.testing.assert_array_equal(normalized[1], 0.0)
