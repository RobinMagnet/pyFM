"""Tests for the Functional Map Network."""

import numpy as np
import pytest

from pyFM.FMN import FMN
from pyFM.mesh import TriMesh
from pyFM.spectral import mesh_p2p_to_FM


@pytest.fixture(scope="module")
def camel_network(data_dir):
    """Three camel poses with the reference maps between them, as (meshlist, maps, p2p)."""
    camel_dir = data_dir / "camel_gallop"
    frame_to_node = {1: 0, 3: 1, 5: 2}

    meshlist = []
    for frame in frame_to_node:
        mesh_path = camel_dir / f"camel-gallop-{frame:02d}.off"
        mesh = TriMesh.load(mesh_path, area_normalize=True, center=True)
        meshlist.append(mesh.process(k=30, intrinsic=True))

    maps = {}
    reference_p2p = {}
    for frame_1 in frame_to_node:
        for frame_2 in frame_to_node:
            if frame_1 == frame_2:
                continue
            i, j = frame_to_node[frame_1], frame_to_node[frame_2]
            p2p_ji = np.loadtxt(camel_dir / "maps" / f"{frame_2}_to_{frame_1}", dtype=int)
            reference_p2p[(i, j)] = p2p_ji
            maps[(i, j)] = mesh_p2p_to_FM(p2p_ji, meshlist[i], meshlist[j], dims=20)

    return meshlist, maps, reference_p2p


def mean_error_to_reference(fmn, meshlist, reference_p2p):
    """Return the mean Euclidean distance between the network's p2p maps and the reference."""
    errors = []
    for (i, j), p2p_ji in fmn.p2p.items():
        vertices = meshlist[i].vertices
        offsets = vertices[p2p_ji] - vertices[reference_p2p[(i, j)]]
        errors.append(np.linalg.norm(offsets, axis=1).mean())
    return np.mean(errors)


def test_complete_p2p_after_subsampled_zoomout_matches_full(camel_network):
    """With a subsample set, complete maps must still index all vertices of the source mesh."""
    meshlist, maps, reference_p2p = camel_network
    subsample_size = 200

    errors = {}
    for subsample in [subsample_size, None]:
        fmn = FMN(meshlist, maps)
        fmn.zoomout_refine(nit=2, step=2, subsample=subsample)
        fmn.compute_CCLB(m=int(0.9 * fmn.M))
        fmn.compute_p2p(complete=True)
        errors[subsample] = mean_error_to_reference(fmn, meshlist, reference_p2p)

        for (i, j), p2p_ji in fmn.p2p.items():
            assert p2p_ji.shape == (meshlist[j].n_vertices,)
        assert max(p2p_ji.max() for p2p_ji in fmn.p2p.values()) >= subsample_size

    assert errors[subsample_size] < 1.5 * errors[None]
