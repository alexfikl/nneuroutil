# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

from nneuroutil.helpers import module_logger

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)


# {{{ test_otsu_threshold_from_image


def test_otsu_threshold_from_image(xp: Any) -> None:
    from nneuroutil.vision import otsu_threshold_from_image

    rng = np.random.default_rng(seed=42)

    # two well-separated clusters: the threshold must fall between them
    dark = xp.asarray(rng.integers(0, 30, size=(8, 16)))
    bright = xp.asarray(rng.integers(120, 200, size=(8, 16)))
    img = xp.concat([dark, bright], axis=0)

    threshold = otsu_threshold_from_image(img)

    assert xp.max(dark) < threshold < xp.min(bright)


def test_otsu_threshold_from_image_symmetric(xp: Any) -> None:
    from nneuroutil.vision import otsu_threshold_from_image

    # equal numbers of 0s and 100s: the between-class variance is identical for
    # every split in between, so the first maximizer (the lowest bin center) wins
    img = xp.concat(
        [
            xp.zeros((8, 8), dtype=xp.int64),
            xp.full((8, 8), 100, dtype=xp.int64),
        ],
        axis=0,
    )

    assert otsu_threshold_from_image(img) == pytest.approx(0.5)


def test_otsu_threshold_from_image_nbins(xp: Any) -> None:
    from nneuroutil.vision import otsu_threshold_from_image

    # with only 2 bins over [0, 256) the centers are {64, 192}, so the split
    # between the 0s and 100s is picked at the coarser center 64
    img = xp.concat(
        [
            xp.zeros((8, 8), dtype=xp.int64),
            xp.full((8, 8), 100, dtype=xp.int64),
        ],
        axis=0,
    )

    assert otsu_threshold_from_image(img, nbins=2) == pytest.approx(64.0)


def test_otsu_threshold_from_image_histogram(xp: Any) -> None:
    from nneuroutil.array_api_extras import histogram
    from nneuroutil.vision import (
        otsu_threshold_from_histogram,
        otsu_threshold_from_image,
    )

    rng = np.random.default_rng(seed=42)
    img = xp.asarray(rng.integers(0, 256, size=(16, 16)))

    # the image-level function must agree with an explicit histogram
    bin_counts, bin_edges = histogram(xp.reshape(img, (-1,)), bins=256, range=(0, 256))
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    expected = otsu_threshold_from_histogram((bin_counts, bin_centers), xp=xp)

    assert otsu_threshold_from_image(img) == pytest.approx(expected)


def test_otsu_threshold_from_image_ndim(xp: Any) -> None:
    from nneuroutil.vision import otsu_threshold_from_image

    with pytest.raises(ValueError, match="2D array"):
        otsu_threshold_from_image(xp.asarray(np.zeros((4, 4, 4))))


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
