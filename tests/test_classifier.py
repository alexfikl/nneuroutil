# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from typing import Any

import array_api_compat
import numpy as np
import pytest

from nneuroutil.classifier import (
    classify_linear_discriminant_analysis,
    classify_support_vector_machine,
    solve_svm_fista,
    solve_svm_jax,
    solve_svm_numpy,
    solve_svm_torch,
)
from nneuroutil.helpers import module_logger
from nneuroutil.typing import Array0D, Array2D

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


# {{{ test_classify_support_vector_machine


def test_classify_support_vector_machine(xp: Any) -> None:
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

    clf = classify_support_vector_machine(features, labels, xp=xp)

    # check estimator attributes and shapes
    assert clf.W.shape == (5, 3)
    assert clf.mu.shape == (4,)
    assert clf.bias == 1.0  # ruff: ignore[float-equality-comparison]
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

    # predict 1D single sample
    assert clf.predict(c0[0]) == 0
    assert clf.predict(c1[0]) == 1
    assert clf.predict(c2[0]) == 2

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
    clf_binary = classify_support_vector_machine((b0, b1), (10, 20), xp=xp)
    assert clf_binary.W.shape == (4, 2)
    assert clf_binary.mu.shape == (3,)
    assert bool(xp.all(clf_binary.predict(b0) == 10))
    assert bool(xp.all(clf_binary.predict(b1) == 20))

    # }}}

    # {{{ 3. explicit solvers

    from functools import partial

    # solve_svm_fista works across all backends with precomputed L0
    from nneuroutil.classifier import _svm_estimate_lipschitz_constant  # ruff: ignore[import-private-name]

    A_binary = xp.concat([b0, b1], axis=0)
    L0_est = _svm_estimate_lipschitz_constant(A_binary, xp=xp)
    clf_fista = classify_support_vector_machine(
        (b0, b1), (10, 20), solver=partial(solve_svm_fista, L0=L0_est), xp=xp
    )
    assert bool(xp.all(clf_fista.predict(b0) == 10))
    assert bool(xp.all(clf_fista.predict(b1) == 20))

    if array_api_compat.is_numpy_array(b0):
        clf_np = classify_support_vector_machine(
            (b0, b1), (10, 20), solver=solve_svm_numpy, xp=xp
        )
        assert bool(xp.all(clf_np.predict(b0) == 10))
        assert bool(xp.all(clf_np.predict(b1) == 20))
    elif array_api_compat.is_torch_array(b0):
        clf_torch = classify_support_vector_machine(
            (b0, b1), (10, 20), solver=solve_svm_torch, xp=xp
        )
        assert bool(xp.all(clf_torch.predict(b0) == 10))
        assert bool(xp.all(clf_torch.predict(b1) == 20))
    elif array_api_compat.is_jax_array(b0):
        clf_jax = classify_support_vector_machine(
            (b0, b1), (10, 20), solver=partial(solve_svm_jax, L0=L0_est), xp=xp
        )
        assert bool(xp.all(clf_jax.predict(b0) == 10))
        assert bool(xp.all(clf_jax.predict(b1) == 20))

        # test pytree flattening/unflattening
        import jax

        leaves, treedef = jax.tree_util.tree_flatten(clf_jax)
        clf_reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        assert bool(xp.all(clf_reconstructed.W == clf_jax.W))
        assert bool(xp.all(clf_reconstructed.mu == clf_jax.mu))

    # }}}

    # {{{ 4. input validation / error cases

    # mismatched features and labels
    with pytest.raises(ValueError, match="do not match"):
        classify_support_vector_machine((c0, c1), (0, 1, 2), xp=xp)

    # <= 1 classes
    with pytest.raises(ValueError, match="0 or 1 classes"):
        classify_support_vector_machine((c0,), (0,), xp=xp)

    with pytest.raises(ValueError, match="0 or 1 classes"):
        classify_support_vector_machine((), (), xp=xp)

    # mismatched feature dimensions
    c_mismatched = xp.asarray(rng.standard_normal((10, 2)), dtype=dtype, device=device)
    with pytest.raises(ValueError, match="same number of features"):
        classify_support_vector_machine((c0, c_mismatched), (0, 1), xp=xp)

    # non-positive C
    with pytest.raises(ValueError, match="must be positive"):
        classify_support_vector_machine((c0, c1), (0, 1), C=0.0, xp=xp)

    with pytest.raises(ValueError, match="must be positive"):
        classify_support_vector_machine((c0, c1), (0, 1), C=-1.0, xp=xp)

    # non-positive bias
    with pytest.raises(ValueError, match="must be positive"):
        classify_support_vector_machine((c0, c1), (0, 1), bias=0.0, xp=xp)

    with pytest.raises(ValueError, match="must be positive"):
        classify_support_vector_machine((c0, c1), (0, 1), bias=-0.5, xp=xp)

    # L0 missing in solve_svm_fista direct call
    def dummy_func(
        x: Array2D[np.floating[Any]],
    ) -> tuple[Array0D[np.floating[Any]], Array2D[np.floating[Any]]]:
        return xp.sum(x), x

    dummy_x0 = xp.zeros((2, 2), dtype=dtype, device=device)
    with pytest.raises(ValueError, match="Lipschitz constant 'L0' must be provided"):
        solve_svm_fista(dummy_func, dummy_x0, 0.0, 1.0)

    # }}}


# }}}
