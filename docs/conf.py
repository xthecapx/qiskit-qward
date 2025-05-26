# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""
Sphinx documentation builder
"""

# General options:
from pathlib import Path

project = "Qward"
copyright = "2022"  # pylint: disable=redefined-builtin
author = ""

_rootdir = Path(__file__).parent.parent

# The full version, including alpha/beta/rc tags
release = "0.1.0"
# The short X.Y version
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "jupyter_sphinx",
    "sphinx_autodoc_typehints",
    "reno.sphinxext",
    "nbsphinx",
    "qiskit_sphinx_theme",
    "myst_parser",
    "sphinxcontrib.mermaid",
]

# Add Markdown support
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
numfig = True
numfig_format = {"table": "Table %s"}
language = "en"
pygments_style = "colorful"
add_module_names = False
modindex_common_prefix = ["qward."]

# html theme options
html_title = f"{project} {release}"
html_theme = "qiskit-ecosystem"

# autodoc/autosummary options
autosummary_generate = True
autosummary_generate_overwrite = False
autoclass_content = "both"

# nbsphinx options (for tutorials)
nbsphinx_timeout = 180
nbsphinx_execute = "always"
nbsphinx_widgets_path = ""
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# Mermaid configuration
mermaid_version = "latest"  # Use latest version of Mermaid.js
mermaid_init_js = """
mermaid.initialize({
    startOnLoad: true,
    theme: 'default',
    themeVariables: {
        primaryColor: '#ff6b6b',
        primaryTextColor: '#333',
        primaryBorderColor: '#ff6b6b',
        lineColor: '#333',
        secondaryColor: '#4ecdc4',
        tertiaryColor: '#ffe66d'
    },
    flowchart: {
        useMaxWidth: true,
        htmlLabels: true
    },
    classDiagram: {
        useMaxWidth: true
    }
});
"""
