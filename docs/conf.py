"""Sphinx configuration of the pydemi documentation (https://cmslabiitk.github.io/pydemi/)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS.parent / "src"))
sys.path.insert(0, str(DOCS / "tools"))

import pydemi  # noqa: E402

import descriptors_to_myst  # noqa: E402
import tables  # noqa: E402

# the descriptor pages come from docs/descriptors.tex, the tables from docs/_generated
descriptors_to_myst.build()
tables.build()
(DOCS / "_static").mkdir(exist_ok=True)
shutil.copy(DOCS / "descriptors.pdf", DOCS / "_static" / "descriptors.pdf")

project = "pydemi"
author = "Shubham Maurya, CMS Lab, IIT Kanpur"
copyright = "2026, Shubham Maurya, CMS Lab, IIT Kanpur"
release = version = pydemi.__version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", "tools", "assets", "_generated", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist", "attrs_inline",
                          "substitution"]
myst_heading_anchors = 3
myst_substitutions = {"version": release}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": False}
# the optional extras are imported only by the modules that need them
autodoc_mock_imports = ["torch", "e3nn", "ase", "streamlit", "lightgbm", "plotly"]
napoleon_numpy_docstring = True
intersphinx_mapping = {"python": ("https://docs.python.org/3", None),
                       "numpy": ("https://numpy.org/doc/stable/", None),
                       "pandas": ("https://pandas.pydata.org/docs/", None)}

html_theme = "furo"
html_title = f"pydemi {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# plotly.js of the Python plotly that made the figures (7.x -> plotly.js 4.1.1); the figure
# scripts wait for the page load, so the order of the script tags does not matter
html_js_files = ["https://cdn.plot.ly/plotly-4.1.1.min.js", "figures.js"]
html_favicon = "_static/favicon.png"
html_theme_options = {
    "light_logo": "logo.png",
    "dark_logo": "logo_dark.png",
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/CMSLabIITK/pydemi/",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
    "light_css_variables": {"color-brand-primary": "#3d2378", "color-brand-content": "#c4502a"},
    "dark_css_variables": {"color-brand-primary": "#c3b0f0", "color-brand-content": "#f39a72"},
    "footer_icons": [{
        "name": "GitHub", "url": "https://github.com/CMSLabIITK/pydemi", "class": "",
        "html": ('<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">'
                 '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07'
                 '.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13'
                 '-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51'
                 '-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21'
                 ' 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92'
                 '.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 '
                 '1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>'),
    }],
}
pygments_style = "friendly"
pygments_dark_style = "monokai"
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
suppress_warnings = ["myst.header"]


def _escape_bars(app, what, name, obj, options, lines):
    """The docstrings write |x| for absolute values, which reST reads as a substitution."""
    for i, line in enumerate(lines):
        if "|" in line:
            lines[i] = line.replace("|", r"\|")


def setup(app):
    app.connect("autodoc-process-docstring", _escape_bars)
