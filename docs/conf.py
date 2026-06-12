"""Sphinx configuration for PySSE documentation."""

from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "docs" / "_build" / ".matplotlib"))

project = "PySSE"
author = "Manuel Fernando Sanchez Alarcon"
copyright = "2026, Manuel Fernando Sanchez Alarcon"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_baseurl = os.getenv("SPHINX_HTML_BASEURL", "").rstrip("/")

html_title = "PySSE Documentation"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_logo": "light_logo.png",
    "dark_logo": "dark_logo.png",
    "light_css_variables": {
        "color-brand-primary": "#b42318",
        "color-brand-content": "#b42318",
        "color-api-name": "#b42318",
        "color-api-pre-name": "#4b5563",
    },
    "dark_css_variables": {
        "color-brand-primary": "#ff6b6b",
        "color-brand-content": "#ff6b6b",
        "color-api-name": "#ff8787",
        "color-api-pre-name": "#cbd5e1",
    },
}
