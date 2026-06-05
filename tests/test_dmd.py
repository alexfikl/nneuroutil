# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import array_api_compat
import numpy as np
import pytest

from nneuroutil.helpers import module_logger, set_jax_config
from nneuroutil.visualization import set_plotting_defaults

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)
set_plotting_defaults()

# {{{ test_dmd_classic


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_dmd_classic_linear(backend: str) -> None:
    from nneuroutil.dmd import DMD, build_dmd_classic

    rng = np.random.default_rng(seed=42)
    ndim = 8
    nsnapshots = 64

    if backend == "jax":
        pytest.importorskip("jax")
        set_jax_config()

        import jax.numpy as jnp
        from jax.tree_util import register_dataclass

        xp = jnp
        register_dataclass(DMD)
    elif backend == "torch":
        xp = pytest.importorskip("torch")
    elif backend == "numpy":
        xp = np
    else:
        raise ValueError(f"unknown backend: {backend!r}")

    # construct a random stable-ish linear map and evolve an initial condition
    A = xp.asarray(rng.standard_normal((ndim, ndim)) / ndim)
    xp = array_api_compat.array_namespace(A)

    xs = [xp.asarray(rng.standard_normal(ndim))]
    for _ in range(nsnapshots - 1):
        xs.append(A @ xs[-1])
    X = xp.stack(xs)

    # build DMD approximation
    dmd = build_dmd_classic(X[:-1], X[1:])

    # ensure the implementation can be compiled
    if backend == "jax":
        import jax

        _ = jax.jit(build_dmd_classic, static_argnames=("rank", "xp"))(
            X[:-1], X[1:], xp=xp
        )
    elif backend == "torch":
        import torch

        _ = torch.compile(build_dmd_classic)(X[:-1], X[1:], xp=xp)

    # check DMD approximation on a random state
    x = xp.asarray(rng.standard_normal(ndim))
    x_ref = A @ x
    x_dmd = dmd.decode(dmd.evolve(dmd.encode(x)))

    error = xp.linalg.norm(x_dmd - x_ref) / xp.linalg.norm(x_ref)
    log.info(
        "[%s] DMD classic relative error: %.3e (rank=%d)",
        backend,
        float(error),
        dmd.reduced_size,
    )
    assert float(error) < 1.0e-14


# }}}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
