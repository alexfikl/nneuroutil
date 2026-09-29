# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

from nneuroutil.classifier import classify_linear_discriminant_analysis
from nneuroutil.helpers import module_logger

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)


# {{{ test_classify_linear_discriminant_analysis


def test_classify_linear_discriminant_analysis(xp: Any) -> None:
    rng = np.random.default_rng(seed=42)

    device = xp.empty(0).device
    dtype = xp.float64

    # {{{ 1. multi-class separable clusters (k=3, dim=4)

    c0 = xp.asarray(
        rng.standard_normal((20, 4)) + np.array([10.0, 0.0, 0.0, 0.0]),
        dtype=dtype,
        device=device,
    )
    c1 = xp.asarray(
        rng.standard_normal((25, 4)) + np.array([0.0, 10.0, 0.0, 0.0]),
        dtype=dtype,
        device=device,
    )
    c2 = xp.asarray(
        rng.standard_normal((30, 4)) + np.array([0.0, 0.0, 10.0, 0.0]),
        dtype=dtype,
        device=device,
    )

    features = (c0, c1, c2)
    labels = (0, 1, 2)

    clf = classify_linear_discriminant_analysis(features, labels, xp=xp)

    # check estimator attributes and shapes
    assert clf.G.shape == (4, 2)
    assert clf.centroids.shape == (4,)
    assert clf.proj_centroids.shape == (3, 2)
    assert clf.labels.shape == (3,)

    # predict per-class
    pred0 = clf.predict(c0)
    pred1 = clf.predict(c1)
    pred2 = clf.predict(c2)
    assert bool(xp.all(pred0 == 0))
    assert bool(xp.all(pred1 == 1))
    assert bool(xp.all(pred2 == 2))

    # predict batch of combined data
    all_data = xp.concat([c0, c1, c2], axis=0)
    all_labels = xp.concat([
        xp.zeros(len(c0), dtype=dtype, device=device),
        xp.ones(len(c1), dtype=dtype, device=device),
        2 * xp.ones(len(c2), dtype=dtype, device=device),
    ])
    preds = clf.predict(all_data)
    assert bool(xp.all(preds == all_labels))

    # predict on class centroids themselves
    cs = xp.stack([xp.mean(x, axis=0) for x in features])
    cs_preds = clf.predict(cs)
    assert bool(xp.all(cs_preds == clf.labels))

    # }}}

    # {{{ 2. binary classification (k=2, dim=3)

    b0 = xp.asarray(
        rng.standard_normal((15, 3)) + np.array([-5.0, 0.0, 0.0]),
        dtype=dtype,
        device=device,
    )
    b1 = xp.asarray(
        rng.standard_normal((15, 3)) + np.array([5.0, 0.0, 0.0]),
        dtype=dtype,
        device=device,
    )
    clf_binary = classify_linear_discriminant_analysis((b0, b1), (10, 20), xp=xp)
    assert clf_binary.G.shape == (3, 1)
    assert clf_binary.proj_centroids.shape == (2, 1)
    assert bool(xp.all(clf_binary.predict(b0) == 10))
    assert bool(xp.all(clf_binary.predict(b1) == 20))

    # }}}

    # {{{ 3. input validation / error cases

    # mismatched features and labels
    with pytest.raises(ValueError, match="do not match"):
        classify_linear_discriminant_analysis((c0, c1), (0, 1, 2), xp=xp)

    # <= 1 classes
    with pytest.raises(ValueError, match="0 or 1 classes"):
        classify_linear_discriminant_analysis((c0,), (0,), xp=xp)

    with pytest.raises(ValueError, match="0 or 1 classes"):
        classify_linear_discriminant_analysis((), (), xp=xp)

    # mismatched feature dimensions
    c_mismatched = xp.asarray(rng.standard_normal((10, 2)), dtype=dtype, device=device)
    with pytest.raises(ValueError, match="same number of features"):
        classify_linear_discriminant_analysis((c0, c_mismatched), (0, 1), xp=xp)

    # non-positive eps
    with pytest.raises(ValueError, match="must be positive"):
        classify_linear_discriminant_analysis((c0, c1), (0, 1), eps=0.0, xp=xp)

    with pytest.raises(ValueError, match="must be positive"):
        classify_linear_discriminant_analysis((c0, c1), (0, 1), eps=-1e-5, xp=xp)

    # }}}


# }}}
