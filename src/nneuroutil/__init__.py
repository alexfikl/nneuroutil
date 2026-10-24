# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from importlib import metadata

logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    __version__ = metadata.version("nneuroutil")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"
