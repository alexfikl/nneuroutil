# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from nneuroutil.helpers import module_logger
from nneuroutil.visualization import set_plotting_defaults

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)
set_plotting_defaults()

# {{{ test_dmd_classic


def test_dmd_classic_linear() -> None:
    from nneuroutil.dmd import build_dmd_classic

    rng = np.random.default_rng(seed=42)
    ndim = 8
    nsnapshots = 64

    # construct a random stable-ish linear map and evolve an initial condition
    A = rng.standard_normal((ndim, ndim)) / ndim

    X = np.empty((nsnapshots, ndim))
    X[0] = rng.standard_normal(ndim)
    for i in range(1, nsnapshots):
        X[i] = A @ X[i - 1]

    eps = 1.0e-10
    dmd = build_dmd_classic(X, eps=eps)

    X = rng.standard_normal(ndim)
    X_ref = A @ X
    X_dmd = dmd.decode(dmd.evolve(dmd.encode(X)))

    error = np.linalg.norm(X_dmd - X_ref) / np.linalg.norm(X_ref)
    log.info("DMD classic relative error: %.3e (rank=%d)", error, dmd.reduced_size)
    assert error < eps


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
