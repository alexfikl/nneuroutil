# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: CC0-1.0

# https://www.sphinx-doc.org/en/master/usage/configuration.html
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

from __future__ import annotations

import os
import sys
from importlib import metadata

# {{{ project information

m = metadata.metadata("nneuroutil")
project = m["Name"]
author = m["Author-email"]
copyright = f"2026 {author}"  # ruff:ignore[builtin-variable-shadowing]
version = m["Version"]
release = version
url = "https://github.com/alexfikl/nneuroutil"

# }}}

# {{{ github roles


def add_dataclass_annotation(app, name, obj, options, bases):
    from dataclasses import is_dataclass

    if not getattr(options, "show_inheritance", False):
        return

    if is_dataclass(obj):
        # NOTE: this needs to be a string because `dataclass` is a function, not
        # a class, so Sphinx gets confused when it tries to insert it into the docs
        bases.append(":func:`dataclasses.dataclass`")

    if object in bases:
        # NOTE: not very helpful to show inheritance from "object"
        bases.remove(object)


def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None

    modname = info["module"]
    objname = info["fullname"]

    mod = sys.modules.get(modname)
    if not mod:
        return None

    obj = mod
    for part in objname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    import inspect

    try:
        moduleparts = obj.__module__.split(".")
        filepath = f"{os.path.join(*moduleparts)}.py"
    except Exception:
        return None

    # FIXME: this checks if the module is actually the `__init__.py`. Is there
    # any easier way to figure that out?
    if mod.__name__ == obj.__module__ and mod.__spec__.submodule_search_locations:
        filepath = os.path.join(*moduleparts, "__init__.py")

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        return None
    else:
        linestart, linestop = lineno, lineno + len(source) - 1

    return f"{url}/blob/main/src/{filepath}#L{linestart}-L{linestop}"


def process_autodoc_missing_reference(app, env, node, contnode):
    """Fix missing references due to string annotations."""

    # NOTE: only classes for now, since we just need some numpy objects
    if node["reftype"] != "class":
        return None

    target = node["reftarget"]
    if target not in custom_type_links:
        return None

    from docutils.nodes import Text

    inventory, reftarget, reftype = custom_type_links[target]
    if "." in reftarget:
        module, objname = reftarget.rsplit(".", maxsplit=1)
    else:
        module = objname = reftarget

    if isinstance(contnode, Text):
        if app.config.autodoc_typehints_format == "short":
            contnode = Text(objname)
        else:
            contnode = Text(reftarget)

    if inventory:
        node.attributes["py:module"] = module
        node.attributes["reftype"] = reftype
        node.attributes["reftarget"] = reftarget

        from sphinx.ext import intersphinx

        return intersphinx.resolve_reference_in_inventory(
            env, inventory, node, contnode
        )
    else:
        target = target.split(".")[-1]
        py_domain = env.get_domain("py")
        return py_domain.resolve_xref(
            env, node["refdoc"], app.builder, reftype, target, node, contnode
        )


def setup(app) -> None:
    app.connect("autodoc-process-bases", add_dataclass_annotation)
    app.connect("missing-reference", process_autodoc_missing_reference)


# }}}

# {{{ general configuration

# needed extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
]

# extension for source files
source_suffix = {".rst": "restructuredtext"}
# name of the main (master) document
master_doc = "index"
# min sphinx version
needs_sphinx = "4.0"
# files to ignore
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# highlighting
pygments_style = "sphinx"

html_theme = "sphinx_book_theme"
html_title = project
html_theme_options = {
    "show_toc_level": 3,
    "use_source_button": True,
    "use_repository_button": True,
    "navigation_with_keys": True,
    "repository_url": "https://github.com/alexfikl/nneuroutil",
    "repository_branch": "main",
    "icon_links": [
        # {
        #     "name": "Release",
        #     "url": "https://github.com/alexfikl/nneuroutil/releases",
        #     "icon": "https://img.shields.io/github/v/release/alexfikl/nneuroutil",
        #     "type": "url",
        # },
        {
            "name": "License",
            "url": "https://github.com/alexfikl/nneuroutil/tree/main/LICENSES",
            "icon": "https://img.shields.io/badge/License-MIT-blue.svg",
            "type": "url",
        },
        {
            "name": "CI",
            "url": "https://github.com/alexfikl/nneuroutil/actions/workflows/ci.yml",
            "icon": "https://github.com/alexfikl/nneuroutil/actions/workflows/ci.yml/badge.svg",
            "type": "url",
        },
        {
            "name": "Issues",
            "url": "https://github.com/alexfikl/nneuroutil/issues",
            "icon": "https://img.shields.io/github/issues/alexfikl/nneuroutil",
            "type": "url",
        },
    ],
}

# }}}

# {{{ internationalization

language = "en"

# }}

# {{{ extension settings

autoclass_content = "class"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": None,
    "show-inheritance": None,
}

# }}}

# {{{ links

nitpick_ignore_regex = [
    ["py:class", r".*TorchDispatchMode"],
]

intersphinx_mapping = {
    "flax": ("https://flax.readthedocs.io/en/latest", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "rich": ("https://rich.readthedocs.io/en/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "torch": ("https://docs.pytorch.org/docs/main", None)
}

# fmt: off
custom_type_links = {
}
# fmt: on

# }}}
