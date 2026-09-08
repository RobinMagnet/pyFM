"""Shared fixtures for the test suite.

Paths are resolved from this file rather than the working directory, so the
suite can be run from anywhere.
"""

from pathlib import Path

import pytest

from pyFM.mesh import TriMesh

DATA_DIR = Path(__file__).resolve().parent.parent / "examples" / "data"


@pytest.fixture(scope="session")
def data_dir():
    return DATA_DIR


@pytest.fixture(scope="session")
def cat_path():
    return str(DATA_DIR / "cat-00.off")


@pytest.fixture(scope="session")
def lion_path():
    return str(DATA_DIR / "lion-00.off")


@pytest.fixture
def cat(cat_path):
    """A fresh, unprocessed cat mesh for each test."""
    return TriMesh.load(cat_path)


@pytest.fixture
def lion(lion_path):
    """A fresh, unprocessed lion mesh for each test."""
    return TriMesh.load(lion_path)
