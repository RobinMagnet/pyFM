# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import importlib.metadata
import os
import pathlib
import re
import sys

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pyfmaps"
copyright = "2024, Robin Magnet"
author = "Robin Magnet"
try:
    release = importlib.metadata.version("pyfmaps")
except importlib.metadata.PackageNotFoundError:
    release = "0.0.0"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",  # merge type hints into descriptions (after napoleon)
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",  # copy button on code blocks
    "myst_parser",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "pyvista.ext.viewer_directive",  # renders the .vtksz scenes the scraper emits
]


# -- Example gallery (sphinx-gallery + PyVista) ------------------------------
# The gallery executes examples/gallery/plot_*.py at build time and scrapes the
# scenes their pl.show() calls produce.

import pyvista as pv  # noqa: E402
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper  # noqa: E402

pv.BUILDING_GALLERY = True
pv.OFF_SCREEN = True
pv.set_plot_theme("document")
pv.global_theme.window_size = [1024, 768]


def reset_pyvista(gallery_conf, fname):
    """Keep one example's theme/plotter state from leaking into the next."""
    pv.close_all()
    pv.set_plot_theme("document")
    pv.global_theme.window_size = [1024, 768]


sphinx_gallery_conf = {
    "examples_dirs": "../../examples/gallery",
    "gallery_dirs": "auto_examples",
    # Only plot_*.py are executed. os.sep keeps this working on Windows.
    "filename_pattern": re.escape(os.sep) + r"plot_",
    "image_scrapers": (DynamicScraper(), "matplotlib"),
    "doc_module": ("pyFM",),
    # Rendered by the ``minigallery`` directive in _templates/autosummary/.
    "backreferences_dir": "gen_modules/backreferences",
    # The default is ("matplotlib", "seaborn") -- keep matplotlib's reset, since
    # setting this key replaces the list rather than extending it.
    "reset_modules": ("matplotlib", reset_pyvista),
    "reset_modules_order": "both",
    "remove_config_comments": True,
}

# sphinx_gallery_conf holds a scraper instance and a reset callback, so Sphinx
# cannot pickle it into the config cache. Harmless, but noisy on every build.
suppress_warnings = ["config.cache"]

# Resolve :class:/:meth: roles pointing at our dependencies. A short timeout keeps
# an unreachable inventory to a warning rather than a stalled build.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pyvista": ("https://docs.pyvista.org", None),
}
intersphinx_timeout = 10

# Napoleon (numpydoc parsing)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Members are documented on their own per-object pages (numpy-style), so module
# and class pages only render the module/class docstring + summary tables.
autodoc_default_options = {"member-order": "bysource"}
autosummary_generate = True
# Honour each module's __all__ so only intended public names are documented
# (default True *ignores* __all__, which would leak imported numpy/tqdm/sklearn names).
autosummary_ignore_module_all = False

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = [".rst", ".md"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["pyfm.css"]


from sphinx.ext.autodoc import between


def setup(app):
    # Register a sphinx.ext.autodoc.between listener to ignore everything
    # between lines that contain the word IGNORE
    app.connect("autodoc-process-docstring", between("^.*IGNORE.*$", exclude=True))
    return {"parallel_read_safe": True}
