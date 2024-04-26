# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyfmaps'
copyright = '2024, Robin Magnet'
author = 'Robin Magnet'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.duration',
              'sphinx.ext.doctest',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              "sphinx.ext.napoleon",
              'sphinx_math_dollar',
              'sphinx.ext.mathjax',
              "myst_parser",
              "sphinx_design",
              ]

# mathjax_config = {
#     'tex2jax': {
#         'inlineMath': [ ["\\(","\\)"] ],
#         'displayMath': [["\\[","\\]"] ],
#     },
# }

# mathjax3_config = {
#   "tex": {
#     "inlineMath": [['\\(', '\\)']],
#     "displayMath": [["\\[", "\\]"]],
#   }
# }

# autodoc_mock_imports = ["pyFM", "scipy", "numpy", "trimesh", "scipy.linalg", "scipy.sparse", 'potpourri3d', "robust_laplacian"]

autodoc_default_options = {
    'members': True}

templates_path = ['_templates']
exclude_patterns = []

source_suffix = [".rst", ".md"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']


from sphinx.ext.autodoc import between

def setup(app):
    # Register a sphinx.ext.autodoc.between listener to ignore everything
    # between lines that contain the word IGNORE
    app.connect('autodoc-process-docstring', between('^.*IGNORE.*$', exclude=True))
    return app
