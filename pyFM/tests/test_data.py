def test_loading_data():
    from pyFM.mesh import TriMesh
    mesh1 = TriMesh('data/cat-00.off', area_normalize=True, center=False)
    mesh2 = TriMesh('data/lion-00.off', area_normalize=True, center=True)

    assert mesh1 is not None
    assert mesh2 is not None
    