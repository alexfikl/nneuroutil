# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from nneuroutil.helpers import module_logger, set_jax_config
from nneuroutil.visualization import set_plotting_defaults

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)
set_plotting_defaults()

# {{{ test_dmd_classic


def test_dmd_classic_linear_numpy() -> None:
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

    dmd = build_dmd_classic(X[:-1], X[1:])

    x = rng.standard_normal(ndim)
    x_ref = A @ x
    x_dmd = dmd.decode(dmd.evolve(dmd.encode(x)))

    error = np.linalg.norm(x_dmd - x_ref) / np.linalg.norm(x_ref)
    log.info("DMD classic relative error: %.3e (rank=%d)", error, dmd.reduced_size)
    assert error < 1.0e-14


def test_dmd_classic_linear_jax() -> None:
    pytest.importorskip("jax")
    set_jax_config()

    import jax.numpy as jnp
    from jax.tree_util import register_dataclass

    rng = np.random.default_rng(seed=42)
    ndim = 8
    nsnapshots = 64

    from nneuroutil.dmd import DMD, build_dmd_classic

    register_dataclass(DMD)

    # construct a random stable-ish linear map and evolve an initial condition
    A = jnp.asarray(rng.standard_normal((ndim, ndim)) / ndim)
    x0 = jnp.asarray(rng.standard_normal(ndim))

    import jax

    def step(x, _):
        return A @ x, x

    _, X = jax.lax.scan(step, x0, xs=None, length=nsnapshots)

    # build DMD approximation
    build_dmd_classic_jit = jax.jit(build_dmd_classic, static_argnames=("rank",))
    dmd = build_dmd_classic_jit(X[:-1], X[1:])

    # check DMD approximation
    x = jnp.asarray(rng.standard_normal(ndim))
    x_ref = A @ x
    x_dmd = dmd.decode(dmd.evolve(dmd.encode(x)))

    error = jnp.linalg.norm(x_dmd - x_ref) / jnp.linalg.norm(x_ref)
    log.info("DMD classic relative error: %.3e (rank=%d)", error, dmd.reduced_size)
    assert float(error) < 1.0e-14


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
