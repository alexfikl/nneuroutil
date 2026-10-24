# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from importlib import metadata

try:
    __version__ = metadata.version("nneuroutil")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"
