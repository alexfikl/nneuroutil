# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

import array_api_compat

from nneuroutil.helpers import module_logger
from nneuroutil.typing import ArrayND

log = module_logger(__name__)


# {{{ dice score


def dice_score(
    x: ArrayND,
    y: ArrayND,
    *,
    eps: float = 0.0,
    axis: int | tuple[int, ...] | None = None,
    xp: Any = None,
) -> ArrayND:
    r"""Compute the Dice Score for *x* and *y*.

    .. math::

        S(x, y) = 2 \frac{\sum_i x_i y_i}{\sum_i x_i + \sum_i y_i}

    where the summation is done along the *axis* axes. This is a "soft" Dice
    Score by default. The arrays must be converted to an appropriate boolean
    array by the caller.

    The Dice score (named after Lee Raymond Dice) is also sometimes known as
    the Sørensen-Dice coefficient or Sørensen-Dice similarity.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(x, y)

    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")

    x_n_y = xp.sum(x * y, axis=axis)
    x_plus_y = xp.sum(x, axis=axis) + xp.sum(y, axis=axis)
    x_plus_y = xp.where(x_plus_y == 0, x_n_y, x_plus_y)

    return (2.0 * x_n_y + eps) / (x_plus_y + eps)


# }}}


# {{{ f1_score


def f1_score(
    x: ArrayND,
    y: ArrayND,
    *,
    eps: float = 0.0,
    axis: int | tuple[int, ...] | None = None,
    xp: Any = None,
) -> ArrayND:
    """Equivalent to the Dice Score."""
    return dice_score(x, y, eps=eps, axis=axis, xp=xp)


# }}}


# {{{ jaccard


def jaccard_index(
    x: ArrayND,
    y: ArrayND,
    *,
    eps: float = 0.0,
    axis: int | tuple[int, ...] | None = None,
    xp: Any = None,
):
    r"""Compute the Jaccard Index for *x* and *y*.

    .. math::

        S(x, y) = 2 \frac{\sum_i x_i y_i}{\sum_i x_i + \sum_i y_i - \sum_i x_i y_i}

    where the summation is done along the *axis* axes. This is a "soft" Jaccard
    index by default. The arrays must be converted to an appropriate boolean
    array by the caller.

    The Jaccard index (named after Paul Jaccard) is also sometimes known as the
    Intersection over Union score, critical success index, the Tanimoto Index, or
    the Tanimoto coefficient.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(x, y)

    # NOTE: n is intersection and u is union :)
    x_n_y = xp.sum(x * y, axis=axis)
    x_u_y = xp.sum(x, axis=axis) + xp.sum(y, axis=axis) - x_n_y

    return (x_n_y + eps) / (x_u_y + eps)


# }}}
