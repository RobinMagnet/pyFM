def test_loading_data(cat_path, lion_path):
    from pyFM.mesh import TriMesh

    mesh1 = TriMesh.load(cat_path, area_normalize=True, center=False)
    mesh2 = TriMesh.load(lion_path, area_normalize=True, center=True)

    assert mesh1 is not None
    assert mesh2 is not None
