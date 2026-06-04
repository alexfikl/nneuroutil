# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from dataclasses import Field
from typing import Any, ClassVar, Protocol, TypeAlias

from typing_extensions import TypeVar

T = TypeVar("T")
"""An unbound invariant generic type variable."""

PathLike: TypeAlias = os.PathLike[str] | str
"""A union of types supported as paths."""

ShapeT = TypeVar("ShapeT", bound=tuple[int, ...], default=tuple[Any, ...])
"""An invariant type alias for ``tuple[int, ...]``."""

Array0D = TypeVar("Array0D")
"""A type variable for a 0-dimensional array."""
Array1D = TypeVar("Array1D")
"""A type variable for a 1-dimensional array."""
Array2D = TypeVar("Array2D")
"""A type variable for a 2-dimensional array."""
Array3D = TypeVar("Array3D")
"""A type variable for a 3-dimensional array."""
Array4D = TypeVar("Array4D")
"""A type variable for a 4-dimensional array."""
ArrayND = TypeVar("ArrayND")
"""A type variable for a n-dimensional array."""


class DataclassInstance(Protocol):
    """Dataclass protocol from
    `typeshed <https://github.com/python/typeshed/blob/770724013de34af6f75fa444cdbb76d187b41875/stdlib/_typeshed/__init__.pyi#L329-L334>`__."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


DataclassInstanceT = TypeVar("DataclassInstanceT", bound=DataclassInstance)
"""An invariant :class:`~typing.TypeVar` bound to :class:`DataclassInstance`."""
