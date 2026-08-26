# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from typing import Any, Literal

import array_api_compat
import numpy as np
import pytest

from nneuroutil.helpers import module_logger, spectrum_error
from nneuroutil.typing import Array2D

TEST_FILENAME = pathlib.Path(__file__)
TEST_DIRECTORY = TEST_FILENAME.parent

log = module_logger(__name__)

# {{{ test_dmd_classic


@pytest.mark.parametrize("tls", [True, False])
def test_dmd_classic_linear(xp: Any, *, tls: bool) -> None:
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
    error_dmd = spectrum_error(eig_dmd, eig_ref[eig_ref_index[:rank]])  # ty: ignore[invalid-argument-type]
    error_tls = spectrum_error(eig_tls, eig_ref[eig_ref_index[:rank]])  # ty: ignore[invalid-argument-type]

    log.info("Noise %.3f Error DMD %.5e TLS %.5e", sigma, error_dmd, error_tls)
    assert error_tls < error_dmd


# {{{ test_build_full_dmd


@pytest.mark.parametrize("use_complex", [False, True])
def test_build_full_dmd_pinv(xp: Any, *, use_complex: bool) -> None:
    rng = np.random.default_rng(seed=42)
    n, d = 16, 5
    noise = 0.01

    if use_complex:
        X = rng.standard_normal((n, d)) + 1j * rng.standard_normal((n, d))
    else:
        X = rng.standard_normal((n, d))

    A_true = rng.standard_normal((d, d))
    Y = X @ A_true + noise * rng.standard_normal((n, d))

    from nneuroutil.dmd import build_full_dmd

    X = xp.asarray(X)
    Y = xp.asarray(Y)

    for eps in [1.0e-12, 0.1]:
        A = build_full_dmd(X, Y, method="pinv", eps=eps, xp=xp)
        A_ref = np.linalg.pinv(np.asarray(X), rcond=eps) @ np.asarray(Y)

        error = float(np.linalg.norm(np.asarray(A) - A_ref))
        log.info(
            "[%s] pinv error: %.3e (complex=%s eps=%.1e)",
            xp.__name__,
            error,
            use_complex,
            eps,
        )
        assert error < 1.0e-12


@pytest.mark.parametrize("use_complex", [False, True])
def test_build_full_dmd_ridge(xp: Any, *, use_complex: bool) -> None:
    rng = np.random.default_rng(seed=42)
    n, d = 16, 5
    noise = 0.01

    if use_complex:
        X = rng.standard_normal((n, d)) + 1j * rng.standard_normal((n, d))
    else:
        X = rng.standard_normal((n, d))

    A_true = rng.standard_normal((d, d))
    Y = X @ A_true + noise * rng.standard_normal((n, d))

    from nneuroutil.dmd import build_full_dmd

    X = xp.asarray(X)
    Y = xp.asarray(Y)

    for eps in [1.0e-12, 1.0e-4]:
        A = build_full_dmd(X, Y, method="ridge", eps=eps, xp=xp)

        X_ref = np.asarray(X)
        Y_ref = np.asarray(Y)
        X_aug = np.concatenate([X_ref, eps**0.5 * np.eye(d, dtype=X_ref.dtype)])
        Y_aug = np.concatenate([Y_ref, np.zeros((d, d), dtype=Y_ref.dtype)])
        A_ref = np.linalg.lstsq(X_aug, Y_aug, rcond=None)[0]

        error = float(np.linalg.norm(np.asarray(A) - A_ref))
        log.info(
            "[%s] ridge error: %.3e (complex=%s eps=%.1e)",
            xp.__name__,
            error,
            use_complex,
            eps,
        )
        assert error < 1.0e-12


@pytest.mark.parametrize("method", ["pinv", "ridge"])
def test_build_full_dmd_default_eps(xp: Any, method: Literal["pinv", "ridge"]) -> None:
    rng = np.random.default_rng(seed=42)
    nsnapshots, ndim = 16, 5

    X = xp.asarray(rng.standard_normal((nsnapshots, ndim)))
    Y = xp.asarray(rng.standard_normal((nsnapshots, ndim)))

    from nneuroutil.dmd import build_full_dmd

    A = build_full_dmd(X, Y, method=method, xp=xp)
    A_ref = np.linalg.lstsq(np.asarray(X), np.asarray(Y), rcond=None)[0]

    error = float(np.linalg.norm(np.asarray(A) - A_ref))
    log.info("[%s] default eps error: %.3e (method=%s)", xp.__name__, error, method)
    assert error < 1.0e-12


def test_build_full_dmd_errors(xp: Any) -> None:
    from nneuroutil.dmd import build_full_dmd

    X = xp.asarray(np.random.default_rng(42).standard_normal((16, 5)))

    with pytest.raises(ValueError, match="must be of shape"):
        build_full_dmd(X[..., None], X, xp=xp)
    with pytest.raises(ValueError, match="must be of shape"):
        build_full_dmd(X, X[..., None], xp=xp)
    with pytest.raises(ValueError, match="different shapes"):
        build_full_dmd(X, X[:-1], xp=xp)
    with pytest.raises(ValueError, match="unknown method"):
        build_full_dmd(X, X, method="garbage", xp=xp)  # ty: ignore[invalid-argument-type]


# }}}


# {{{ test_build_full_extended_dmd


@pytest.mark.parametrize("method", ["pinv", "ridge"])
@pytest.mark.parametrize("use_trajectory", [True, False])
def test_build_full_extended_dmd(
    xp: Any, method: Literal["pinv", "ridge"], *, use_trajectory: bool
) -> None:
    x0 = 0.7
    # NOTE: keep the trajectory short, as the [x, x^2] features over a
    # doubling map grow quadratically and make the lifted matrix ill-conditioned.
    xs = [x0]
    for _ in range(12):
        xs.append(2.0 * xs[-1])
    S = xp.asarray(xs, dtype=xp.float64)[:, None]

    def identity(x: Array2D[np.floating[Any]]) -> Array2D[np.floating[Any]]:
        return x

    def square(x: Array2D[np.floating[Any]]) -> Array2D[np.floating[Any]]:
        return x**2

    from nneuroutil.dmd import build_full_extended_dmd

    observables = [identity, square]
    if use_trajectory:
        A, C = build_full_extended_dmd(observables, S, method=method, xp=xp)
    else:
        A, C = build_full_extended_dmd(observables, S[:-1], S[1:], method=method, xp=xp)

    # the lifted dynamics [x, x^2] -> [2x, 4x^2] are exactly linear
    A_ref = xp.asarray([[2.0, 0.0], [0.0, 4.0]], dtype=A.dtype)
    C_ref = xp.asarray([[1.0], [0.0]], dtype=C.dtype)
    assert A.shape == (2, 2)
    assert C.shape == (2, 1)
    assert xp.all(xp.abs(A - A_ref) < 1.0e-9)
    assert xp.all(xp.abs(C - C_ref) < 1.0e-9)

    # predict: lift -> evolve -> decode
    z = xp.asarray([[x0, x0**2]], dtype=A.dtype)
    for _ in range(10):
        z = z @ A  # ruff: ignore[non-augmented-assignment]
    x_pred = float((z @ C)[0, 0])
    x_ref = 2.0**10 * x0

    error = abs(x_pred - x_ref)
    log.info(
        "[%s] extended DMD 10-step error: %.3e (method=%s, use_trajectory=%s)",
        xp.__name__,
        error,
        method,
        use_trajectory,
    )
    assert error < 1.0e-10


def test_build_full_extended_dmd_errors(xp: Any) -> None:
    from nneuroutil.dmd import build_full_extended_dmd

    X = xp.asarray(np.random.default_rng(42).standard_normal((16, 3)))

    with pytest.raises(ValueError, match="must be of shape"):
        build_full_extended_dmd([], X[..., None], xp=xp)
    with pytest.raises(ValueError, match="no 'observables'"):
        build_full_extended_dmd([], X, xp=xp)
    with pytest.raises(ValueError, match="different shapes"):
        build_full_extended_dmd([lambda z: z], X, X[:-1], xp=xp)


# }}}

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
