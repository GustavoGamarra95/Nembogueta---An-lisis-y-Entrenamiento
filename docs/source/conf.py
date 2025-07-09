import os
import sys

import sphinx.deprecation

sphinx.deprecation.RemovedInSphinx50Warning = None

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../src/"))

project = "Ñemongeta"
copyright = "2025"
author = "Gustavo Ariel Gamarra Rojas"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

html_theme = "sphinx_rtd_theme"
html_theme_options = {"navigation_depth": 4, "titles_only": False}

autodoc_mock_imports = [
    "src.data_collection.phrase_collection",
    "src.config",
    "src.utils",
    "src.preprocessing",
    "src.training",
    "numpy",
    "pandas",
    "torch",
    "cv2",
    "PIL",
    "src.utils.validators",
    "src.data_collection",
    "src.data_collection.word_collection",
]

language = "es"

html_static_path = ["_static"]
templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "APIReferenceDocumentation.md",
    "SetupInstructions.md",
]
warnings_are_errors = False
keep_warnings = False

suppress_warnings = [
    "myst.header",
    "autodoc.duplicate",
    "toc.not_readable",
    "ref.class",
    "ref.python",
    "docutils.duplicate_label",
    "epub.unknown_project_files",
    "autosectionlabel.*",
]

source_encoding = "utf-8-sig"

myst_all_links_external = True
myst_heading_anchors = 3
myst_parse_html = True

autodoc_default_flags = ["members", "undoc-members", "show-inheritance"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_warningiserror = False

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

myst_heading_anchors = 3

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "noindex": False,  # Changed to False
}

suppress_warnings = [
    "myst.header",
    "autodoc.duplicate",
    "toc.not_readable",
    "ref.class",
    "ref.python",
]

nitpicky = False

nitpick_ignore = [
    ("py:class", "optional"),
    ("py:class", "Any"),
    ("py:class", "src.utils.validators.VideoData"),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "description"

master_doc = "index"

toctree_only = False
