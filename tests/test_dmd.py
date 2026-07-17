# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from typing import Any

import array_api_compat
import numpy as np
import pytest

from nneuroutil.helpers import module_logger, spectrum_error

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)

# {{{ test_dmd_classic


@pytest.mark.parametrize("tls", [True, False])
def test_dmd_classic_linear(xp: Any, tls: bool) -> None:  # ruff:ignore[boolean-type-hint-positional-argument]
    rng = np.random.default_rng(seed=42)
    ndim = 8
    nsnapshots = 64

    from nneuroutil.dmd import build_dmd, total_least_squares

    # construct a random stable-ish linear map and evolve an initial condition
    A = xp.asarray(rng.standard_normal((ndim, ndim)) / ndim)

    xs = [xp.asarray(rng.standard_normal(ndim))]
    for _ in range(nsnapshots - 1):
        xs.append(A @ xs[-1])
    X = xp.stack(xs)

    # build DMD approximation
    X1 = X[:-1]
    X2 = X[1:]
    if tls:
        X1, X2 = total_least_squares(X1, X2, xp=xp)
    dmd = build_dmd(X1, X2, xp=xp)

    # ensure the implementation can be compiled
    if array_api_compat.is_jax_namespace(xp):
        import jax

        _ = jax.jit(build_dmd, static_argnames=("rank", "xp"))(X1, X2, xp=xp)
    elif array_api_compat.is_torch_namespace(xp):
        import torch  # ty: ignore[unresolved-import,unused-ignore-comment]

        _ = torch.compile(build_dmd)(X1, X2, xp=xp)

    # check DMD approximation on a random state
    x = xp.asarray(rng.standard_normal(ndim))
    x_ref = A @ x
    x_dmd = dmd.decode(dmd.evolve(dmd.encode(x)))

    error = float(xp.linalg.norm(x_dmd - x_ref) / xp.linalg.norm(x_ref))
    log.info(
        "[%s] DMD classic relative error: %.3e (rank=%d)",
        xp.__name__,
        error,
        dmd.reduced_size,
    )
    assert error < 1.0e-14

    # check eigenvalues
    lambdas, _ = dmd.eigendecomposition()
    assert xp.all(xp.abs(lambdas) < 1.0 + 1.0e-10)
    log.info("[%s] Largest eigenvalue: %.3e", xp.__name__, xp.max(xp.abs(lambdas)))


# }}}


@pytest.mark.parametrize("sigma", [0.1, 1.0])
def test_dmd_tls(sigma: float) -> None:
    rng = np.random.default_rng(seed=42)
    ndim = 8
    maxit = 8
    nruns = 16

    # construct a random stable-ish linear map and evolve an initial condition
    A = rng.standard_normal((ndim, ndim)) / 2.0

    xs = []
    for _ in range(nruns):
        xi = [rng.standard_normal(ndim)]
        for _ in range(maxit):
            xi.append(A @ xi[-1])
        xs.extend(xi)
    X = np.stack(xs)

    # add some noise to the whole thing
    sigma *= np.linalg.norm(X) / np.sqrt(X.size)
    X += sigma * rng.standard_normal(X.shape)

    # solve
    from nneuroutil.dmd import build_dmd, total_least_squares

    rank = ndim - 2
    X1 = X[:-1]
    X2 = X[1:]
    dmd_ref = build_dmd(X1, X2, rank=rank)

    X1, X2 = total_least_squares(X1, X2, rank=rank)
    dmd_tls = build_dmd(X1, X2, rank=rank)

    # compare eigenvalues
    eig_ref = np.linalg.eigvals(A)
    eig_dmd, _ = dmd_ref.eigendecomposition()
    eig_tls, _ = dmd_tls.eigendecomposition()

    eig_ref_index = np.argsort(-np.abs(eig_ref))
    error_dmd = spectrum_error(eig_dmd, eig_ref[eig_ref_index[:rank]])
    error_tls = spectrum_error(eig_tls, eig_ref[eig_ref_index[:rank]])

    log.info("Noise %.3f Error DMD %.5e TLS %.5e", sigma, error_dmd, error_tls)
    assert error_tls < error_dmd


# {{{


# }}}

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
