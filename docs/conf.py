# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../src/"))

# -- Project information -------------------------------------------------------

project = "openpois"
copyright = "2026, Nathaniel Henry"
author = "Nathaniel Henry"
release = "1.1.0"

# -- General configuration -----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Mock heavy imports so Sphinx can parse docstrings without installing the
# full conda environment in CI.
autodoc_mock_imports = [
    "jax",
    "jaxlib",
    "blackjax",
    "numpy",
    "pandas",
    "geopandas",
    "shapely",
    "fiona",
    "pyosmium",
    "osmium",
    "duckdb",
    "pyiceberg",
    "pyarrow",
    "config_versioned",
    "ngeohash",
    "requests",
    "scipy",
    "matplotlib",
    "sklearn",
    "overturemaps",
    "plotnine",
    "mizani",
]

# Napoleon settings
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Wrap function signatures so each argument appears on its own line
python_maximum_signature_line_length = 1

# -- Options for HTML output ---------------------------------------------------

html_theme = "furo"

html_theme_options = {
    # Show only our custom view-this-page override (no edit button)
    "top_of_page_buttons": ["view"],
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
