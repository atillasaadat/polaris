# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath("../polaris"))  # Adjust the path to your module

# Project information
project = "polaris"
copyright = "2024, Atilla Saadat"
author = "Atilla Saadat"
release = "0.1.2"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",  # Enable autosummary
    "sphinx.ext.napoleon",  # Optional: For Google/NumPy docstring styles
    "numpydoc"
    # Add other extensions if needed
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Options for HTML output
html_theme = "pydata_sphinx_theme"  # Use PyData Sphinx Theme
html_static_path = ["_static"]

# Optional: Customize the theme
html_theme_options = {
    # "sidebar_end": [],
    "show_prev_next": True,
    "navbar_align": "left",
    "search_bar_text": "Search...",
}
