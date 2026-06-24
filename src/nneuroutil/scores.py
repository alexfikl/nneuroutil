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
    eps: float = 1.0e-6,
    axis: int | tuple[int, ...] | None = None,
    xp: Any = None,
) -> ArrayND:
    if xp is None:
        xp = array_api_compat.array_namespace(x, y)

    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")

    x_times_y = xp.sum(x * y, axis=axis)
    x_plus_y = xp.sum(x, axis=axis) + xp.sum(y, axis=axis)
    x_plus_y = xp.where(x_plus_y == 0, x_times_y, x_plus_y)

    return (2.0 * x_times_y + eps) / (x_plus_y + eps)


# }}}
