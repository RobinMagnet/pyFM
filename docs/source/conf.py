# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import importlib.metadata
import pathlib
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
    "sphinx_autodoc_typehints",  # merge type hints into descriptions (after napoleon)
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",  # copy button on code blocks
    "myst_parser",
    "sphinx_design",
]

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


from sphinx.ext.autodoc import between


def setup(app):
    # Register a sphinx.ext.autodoc.between listener to ignore everything
    # between lines that contain the word IGNORE
    app.connect("autodoc-process-docstring", between("^.*IGNORE.*$", exclude=True))
    return app
