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
) -> ArrayND:
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

    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")

    # NOTE: n is intersection and u is union :)
    x_n_y = xp.sum(x * y, axis=axis)
    x_u_y = xp.sum(x, axis=axis) + xp.sum(y, axis=axis) - x_n_y

    return (x_n_y + eps) / (x_u_y + eps)


# }}}


# {{{ tversky_index


def tversky_index(
    x: ArrayND,
    y: ArrayND,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
    eps: float = 0.0,
    axis: int | tuple[int, ...] | None = None,
    xp: Any = None,
) -> ArrayND:
    r"""Compute the Tversky Index for *x* and *y*.

    .. math::

        S(x, y) = \frac{\sum_i x_i y_i}
            {\sum_i x_i y_i + \alpha \sum_i x_i (1 - y_i) + \beta \sum_i (1 - x_i) y_i}

    where the summation is done along the *axis* axes. This is a "soft" Tversky
    index by default, defined for all :math:`x_i, y_i in [0, 1]`. The arrays
    must be converted to an appropriate boolean array by the caller.

    This score is a generalization of the :func:`dice_score` and the
    :func:`jaccard_index`. If we take

    * ``alpha = beta = 0.5``, we get the Dice Score.
    * ``alpha = beta = 1.0``, we get the Jaccard Index.
    * ``alpha < beta``, we penalize false negatives more,
    * ``alpha > beta``, we penalize false positives more.

    :arg alpha: weight on false positives.
    :arg beta: weight on false negatives.
    """

    if xp is None:
        xp = array_api_compat.array_namespace(x, y)

    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")

    x_n_y = xp.sum(x * y, axis=axis)
    x_m_y = xp.sum(x * (1 - y), axis=axis)
    y_m_x = xp.sum((1 - x) * y, axis=axis)

    return (x_n_y + eps) / (x_n_y + alpha * x_m_y + beta * y_m_x + eps)


# }}}


# {{{ volumetric_similarity


def volumetric_similarity(
    x: ArrayND,
    y: ArrayND,
    *,
    eps: float = 0.0,
    axis: int | tuple[int, ...] | None = None,
    xp: Any = None,
) -> ArrayND:
    r"""Compute the Volumetric Similarity between *x* and *y*.

    .. math::

        S(x, y) =
            1 - \frac{\left|\sum_i x_i - \sum_i y_i\right|}{\sum_i x_i + \sum_i y_i},

    where the summation is done along the *axis* axes. This is a "soft" volumetric
    similarity by default. The arrays must be converted to an appropriate boolean
    array by the caller.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(x, y)

    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")

    vx = xp.sum(x, axis=axis)
    vy = xp.sum(y, axis=axis)

    return 1.0 - xp.abs(vx - vy + eps) / (vx + vy + eps)


# }}}
