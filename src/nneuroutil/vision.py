# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import array_api_compat
import numpy as np

from nneuroutil.helpers import module_logger
from nneuroutil.typing import Array0D, Array1D, Array2D

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

log = module_logger(__name__)

# {{{ as_grayscale


def as_grayscale(
    x: Array2D[np.floating[Any]],
    *,
    dtype: DTypeLike | None = None,
    xp: Any = None,
) -> Array2D[np.integer[Any]]:
    """Convert the 2D array *x* to a grayscale image.

    This is done by normalizing the array to :math:`[0, 255]` and casting to the
    given *dtype*.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(x)

    from nneuroutil.array_api_extras import to_namespace_dtype

    if dtype is None:
        dtype = np.dtype(np.uint8)
    dtype = to_namespace_dtype(xp, dtype)

    xmin, xmax = xp.min(x), xp.max(x)
    dx = 1.0 if xmin == xmax else (xmax - xmin)

    x = 255 * (x - xmin) / dx
    x = xp.astype(xp.clip(x, 0, 255), dtype)

    return x


# }}}


# {{{ otsu_threshold


# NOTE: The implementation is very much inspired by scikit-image:
# https://github.com/scikit-image/scikit-image/blob/745baa05fc5d39c5e9679ee1ab8f19b432403ab0/src/_skimage2/filters/thresholding.py#L333


def otsu_threshold_from_image(
    img: Array2D[np.integer[Any]],
    *,
    nbins: int = 256,
    xp: Any = None,
) -> Array0D[np.floating[Any]]:
    """Compute a threshold based on
    `Otsu's Method <https://en.wikipedia.org/wiki/Otsu%27s_method>`__.

    This threshold can be used to convert the image to a binary image. The
    function computes the histogram and then uses :func:`otsu_threshold_from_histogram`.

    :arg img: an array of shape ``(n, m)`` representing a grayscale image.
    :arg nbins: number of bins used to calculate the histogram from *img*.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(img)
    else:
        assert array_api_compat.array_namespace(img) is xp

    if img.ndim != 2:
        raise ValueError(f"'img' is not a 2D array: {img.shape}")

    from nneuroutil.array_api_extras import histogram

    bin_counts, bin_edges = histogram(
        xp.reshape(img, (-1,)), bins=nbins, range=(0, 256)
    )
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    return otsu_threshold_from_histogram((bin_counts, bin_centers), xp=xp)


def otsu_threshold_from_histogram(
    hist: (
        Array1D[np.integer[Any]]
        | tuple[Array1D[np.integer[Any]], Array1D[np.floating[Any]]]
    ),
    *,
    xp: Any = None,
) -> Array0D[np.floating[Any]]:
    """Compute a threshold based on
    `Otsu's Method <https://en.wikipedia.org/wiki/Otsu%27s_method>`__.

    :arg hist: an array of ``counts`` or a tuple of ``(counts, centers)``. For
        more information, see :func:`numpy.histogram`, which returns
        ``(counts, edges)`` in a compatible manner (edges can be averaged to
        obtain the centers). If only counts are given, the centers are assumed to
        be at integer locations ``(0, 1, ..., counts.size - 1)``.
    """
    if not (array_api_compat.is_array_api_obj(hist) or isinstance(hist, tuple)):
        raise TypeError("'hist' must be an array or a ``(bins, centers)`` tuple")

    if isinstance(hist, tuple):
        counts, centers = hist
        if xp is None:
            xp = array_api_compat.array_namespace(counts, centers)
        else:
            assert array_api_compat.array_namespace(counts, centers) is xp
    else:
        counts = hist
        if xp is None:
            xp = array_api_compat.array_namespace(counts)
        else:
            assert array_api_compat.array_namespace(counts) is xp

        centers = xp.arange(array_api_compat.size(counts))  # ty: ignore[no-matching-overload]

    # https://en.wikipedia.org/wiki/Otsu%27s_method#Python_implementation

    # class probabilities
    weight1 = xp.cumsum(counts, axis=0)
    weight2 = xp.flip(xp.cumsum(xp.flip(counts), axis=0))

    # class means
    value = counts * centers
    mean1 = xp.cumsum(value, axis=0) / xp.where(weight1 > 0, weight1, 1.0)
    mean2 = xp.flip(
        xp.cumsum(xp.flip(value), axis=0) / xp.flip(xp.where(weight2 > 0, weight2, 1.0))
    )

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = xp.argmax(variance12)
    return centers[idx]


# }}}
