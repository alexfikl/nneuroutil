# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

import array_api_compat
import numpy as np

from nneuroutil.typing import ArrayND

# {{{ array_equal


def array_equal(
    a: ArrayND[np.number[Any]],
    b: ArrayND[np.number[Any]],
    *,
    equal_nan: bool = False,
    xp: Any = None,
) -> bool:
    """*True* if two arrays have the same shape and elements, *False* otherwise."""

    if equal_nan:
        raise NotImplementedError("'equal_nan=True")

    if xp is None:
        xp = array_api_compat.array_namespace(a, b)

    a, b = xp.asarray(a), xp.asarray(b)
    if a.shape != b.shape:
        return False

    return bool(xp.all(xp.equal(a, b)))


# }}}


# {{{ interleave


def interleave(
    x: ArrayND[np.number[Any]],
    y: ArrayND[np.number[Any]],
    /,
    *,
    axis: int = -1,
    xp: Any = None,
) -> ArrayND[np.number[Any]]:
    if x.shape != y.shape:
        raise ValueError(
            f"interleave: x and y must have the same shape: {x.shape} and {y.shape}"
        )

    if xp is None:
        xp = array_api_compat.array_namespace(x, y)

    # normalize negative axis before inserting a new one
    axis %= x.ndim

    # stack the arrays
    stacked = xp.stack([x, y], axis=axis + 1)

    # reshape everything so that it interleaves correctly
    new_shape = (*x.shape[:axis], 2 * x.shape[axis], *x.shape[axis + 1 :])
    return xp.reshape(stacked, new_shape)


# }}}


# {{{ deinterleave


def deinterleave(
    z: ArrayND[np.number[Any]],
    /,
    *,
    axis: int = -1,
    xp: Any = None,
) -> ArrayND[np.number[Any]]:
    axis %= z.ndim
    n = z.shape[axis]

    if n % 2 != 0:
        raise ValueError(f"deinterleave: size along axis {axis} must be even: {n}")

    if xp is None:
        xp = array_api_compat.array_namespace(z)

    new_shape = (*z.shape[:axis], n // 2, 2, *z.shape[axis + 1 :])
    reshaped = xp.reshape(z, new_shape)

    idx0 = (slice(None),) * axis + (slice(None), 0)
    idx1 = (slice(None),) * axis + (slice(None), 1)
    return reshaped[idx0], reshaped[idx1]


# }}}
