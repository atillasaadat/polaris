"""Sphinx configuration for the unified Polaris documentation site.

Builds one site from: reStructuredText pages, Python docstrings (numpydoc),
C++ API (Doxygen -> Breathe), the bibliography (sphinxcontrib-bibtex), and the
requirements + traceability matrix (sphinx-needs). Design doc Sec. 21.

The build is a CI gate: ``sphinx-build -W`` turns warnings into errors, so a
broken docstring, a missing citation, or an approved requirement without a
verifying test fails CI.
"""

from __future__ import annotations

import os

# -- Project information ------------------------------------------------------
project = "Polaris"
author = "Atilla Saadat"
copyright = (
    "2026, Atilla Saadat (PolyForm Noncommercial 1.0.0; commercial license available)"
)
release = "0.0.0"
version = "0.0"

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "breathe",
    "sphinxcontrib.bibtex",
    "sphinx_needs",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "_generated", "Thumbs.db", ".DS_Store", "design"]

# Only reStructuredText is a source format for now. The design doc lives under
# docs/design/ as Markdown and is intentionally NOT rendered into this site yet
# (no myst-parser), so it does not trigger toctree warnings.
source_suffix = {".rst": "restructuredtext"}

nitpicky = False  # tighten later once the C++/Python API is populated

# -- HTML output (PyData Sphinx Theme — same look NumPy uses) ------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "Polaris GNC"
html_theme_options = {
    "show_nav_level": 2,
    "navigation_with_keys": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/atillasaadat/polaris",
            "icon": "fa-brands fa-github",
        }
    ],
}

# -- numpydoc -----------------------------------------------------------------
numpydoc_show_class_members = False  # avoid noisy autosummary warnings
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- Bibliography (provenance — Golden Rule 7 / design doc Sec. 21.1) ----------
bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "label"

# -- Breathe (C++ API via Doxygen XML) ----------------------------------------
# Doxygen writes XML to docs/doxygen/xml (see docs/Doxyfile). Breathe directives
# are not used until lib/ has C++ code (Push 2), so an absent XML dir is fine.
breathe_projects = {
    "polaris": os.path.join(os.path.dirname(__file__), "doxygen", "xml")
}
breathe_default_project = "polaris"

# -- intersphinx --------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# =============================================================================
# sphinx-needs — requirements & bidirectional traceability (design doc Sec. 22.2)
# =============================================================================
needs_types = [
    {
        "directive": "req",
        "title": "Requirement",
        "prefix": "R_",
        "color": "#BFD8D2",
        "style": "node",
    },
    {
        "directive": "test",
        "title": "Verification",
        "prefix": "V_",
        "color": "#DCB239",
        "style": "node",
    },
    {
        "directive": "mc",
        "title": "MC Campaign",
        "prefix": "M_",
        "color": "#9856A5",
        "style": "node",
    },
]

# Allow hyphenated IDs like REQ-CONV-001 (the default regex forbids hyphens).
needs_id_regex = "^[A-Z][A-Z0-9_-]{4,}$"

# Extra attributes every requirement may carry (see docs/requirements/index.rst).
needs_extra_options = [
    "level",  # L0 (mission) / L1 (system) / L2 (subsystem)
    "rationale",
    "method",  # Test / Analysis / Inspection / Demonstration
    "allocation",  # owning component / sim model / lib module
    "value_required",  # quantitative threshold
    "margin_required",  # required margin to threshold
    "value_demonstrated",  # filled by the verifying artifact
    "margin_achieved",  # filled by the verifying artifact
    "refs",  # docs/refs.bib citekey(s)
]

# Trace links. Each option X also exposes an auto-computed back-link field X_back.
needs_extra_links = [
    {"option": "derived_from", "incoming": "derives", "outgoing": "derived from"},
    {"option": "verifies", "incoming": "verified by", "outgoing": "verifies"},
]

needs_statuses = [
    # Requirement lifecycle.
    {"name": "draft", "description": "authored, not yet reviewed"},
    {"name": "reviewed", "description": "reviewed; not yet baselined/approved"},
    {"name": "approved", "description": "baselined; must be verified"},
    {"name": "verified", "description": "verified with margin by >=1 passing artifact"},
    {"name": "deprecated", "description": "withdrawn; ID retired, never reused"},
    # Verification-artifact results (test/mc needs from the collectors).
    {"name": "passed", "description": "verifying artifact passed"},
    {"name": "failed", "description": "verifying artifact failed"},
    {"name": "skipped", "description": "verifying artifact skipped"},
]

# External verification results, written by the test collectors into
# docs/_generated/*.json (empty-but-valid until tests exist in Push 2).
needs_external_needs = [
    {
        "base_url": "../tests",
        "json_path": os.path.join(
            os.path.dirname(__file__), "_generated", "verif_pytest.json"
        ),
        "version": "1.0",
    },
    {
        "base_url": "../tests",
        "json_path": os.path.join(
            os.path.dirname(__file__), "_generated", "verif_gtest.json"
        ),
        "version": "1.0",
    },
]

# CI GATE: an approved/verified requirement with no incoming `verifies` link is a
# coverage hole and fails the build under `-W`. Phase-0 seeds are status
# `reviewed` (not approved), so this matches nothing yet and the build stays green.
needs_warnings = {
    "req_without_verification": (
        "type == 'req' and status in ['approved', 'verified'] and len(verifies_back) == 0"
    )
}
needs_warnings_always_warn = True

# Render report tables deterministically.
needs_table_style = "datatables"
