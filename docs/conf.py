#
# Documentation build configuration file
#

import os
import sys
import datetime

sys.path.append(os.path.abspath(".."))

# Set env flag so that we can doc functions that may otherwise not be loaded
# see for example interactive visualizations in qiskit.visualization.
os.environ["QISKIT_DOCS"] = "TRUE"

# -- Project information -----------------------------------------------------

# TODO: Set verison automatically from module/git version

# The short X.Y version
version = ""
# The full version, including alpha/beta/rc tags
release = "23.0"
project = f"PEA Summit"
copyright = f"2023-{datetime.date.today().year}, Primitives Development Team"  # pylint: disable=redefined-builtin
author = "Christopher J. Wood"


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx_design",
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.doctest",
    'nbsphinx',
    "qiskit_sphinx_theme",
]

# Add `data keys` and `style parameters` alias. Needed for `expected_*_data_keys` methods in
# visualization module and `default_style` method in `PlotStyle` respectively.
napoleon_custom_sections = [("data keys", "params_style"), ("style parameters", "params_style")]

autosummary_generate = True

autodoc_default_options = {"inherited-members": None}

# If true, figures, tables and code-blocks are automatically numbered if they
# have a caption.
numfig = True

# A dictionary mapping 'figure', 'table', 'code-block' and 'section' to
# strings that are used for format of figure numbers. As a special character,
# %s will be replaced to figure number.
numfig_format = {"table": "Table %s"}

language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "colorful"

# A boolean that decides whether module names are prepended to all object names
# (for object types where a “module” of some kind is defined), e.g. for
# py:function directives.
add_module_names = False

# A list of prefixes that are ignored for sorting the Python module index
# (e.g., if this is set to ['foo.'], then foo.bar is shown under B, not F).
# This can be handy if you document a project that consists of a single
# package. Works only for the HTML builder currently.
modindex_common_prefix = ["pea_summit."]

# -- Configuration for executing ipynbs ------------------------------------
# Refer to https://www.sphinx-doc.org/en/master/usa

# Don't execute notebooks before conversion
nbsphinx_execute = 'never'

# Determine the kernel name based on the current tox environment
tox_env = os.environ.get('TOXENV', '')  # Get the current tox environment

# Map the tox environment to the desired kernel name
kernel_name_map = {
    'py38': 'myenv38',  # Replace 'myenv38' with the name of the kernel for Python 3.8
    'py39': 'myenv39',  # Replace 'myenv38' with the name of the kernel for Python 3.9
    'py310': 'myenv310',  # Replace 'myenv38' with the name of the kernel for Python 3.10
}

# Set the nbsphinx_kernel_name based on the tox environment
nbsphinx_kernel_name = kernel_name_map.get(tox_env, 'python3')  # 'python3' as a fallback


# -- Configuration for extlinks extension ------------------------------------
# Refer to https://www.sphinx-doc.org/en/master/usage/extensions/extlinks.html


# -- Options for HTML output -------------------------------------------------

html_theme = "qiskit-ecosystem"
html_title = f"{project} {release}"
html_last_updated_fmt = "%Y/%m/%d"
html_theme_options = {
    "disable_ecosystem_logo": True,
    "light_css_variables": {
        "color-brand-primary": "var(--qiskit-color-blue)",
    }
}

autoclass_content = "both"
intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "qiskit": ("https://qiskit.org/documentation/", None),
    "qiskit-ibm-runtime": ("https://qiskit.org/ecosystem/ibm-runtime/", None),
}
