# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal

import array_api_compat
import numpy as np

from nneuroutil.helpers import module_logger, register_dataclass
from nneuroutil.typing import Array1D, Array2D, ArrayND, ScalarTypeT

log = module_logger(__name__)


# {{{ DMDBase


@register_dataclass
@dataclass(frozen=True)
class DMDBase(ABC, Generic[ScalarTypeT]):
    A: Array2D[ScalarTypeT]
    """Linear approximation on the lifted coordinates."""

    @property
    def dtype(self) -> np.dtype[ScalarTypeT]:
        """The :class:`~numpy.dtype` of this operator."""
        return self.A.dtype

    @property
    def lifted_dim(self) -> int:
        """The dimension of the lifted space."""
        return self.A.shape[0]

    @property
    @abstractmethod
    def physical_dim(self) -> int:
        """The dimension of the physical space."""

    @abstractmethod
    def encode(self, x: ArrayND[ScalarTypeT]) -> ArrayND[ScalarTypeT]:
        """Project the physical state *x* into the lifted space."""

    @abstractmethod
    def decode(self, xhat: ArrayND[ScalarTypeT]) -> ArrayND[ScalarTypeT]:
        """Project the lifted state *xhat* into the physical space."""

    @abstractmethod
    def evolve(self, xhat: ArrayND[ScalarTypeT]) -> ArrayND[ScalarTypeT]:
        """Evolve the system in the lifted space."""

    def predict(
        self,
        x0: ArrayND[ScalarTypeT],
        maxit: int,
        *,
        full: bool = False,
    ) -> ArrayND[ScalarTypeT]:
        xp = array_api_compat.array_namespace(x0)

        assert x0.shape[-1] == self.physical_dim
        xhat = self.encode(x0)

        if full:
            result = [x0]
            for _ in range(maxit):
                xhat = self.evolve(xhat)
                result.append(self.decode(xhat))

            result = xp.stack(result, axis=0)
        else:
            for _ in range(maxit):
                xhat = self.evolve(xhat)

            result = self.decode(xhat)

        return result


# }}}


# {{{ reduced DMD


@register_dataclass
@dataclass(frozen=True)
class ReducedDMD(DMDBase[ScalarTypeT]):
    U: Array2D[ScalarTypeT]
    """Temporal modes as an array of shape :math:`(n - 1, r)`."""
    S: Array1D[ScalarTypeT]
    """Singular values as an array of shape :math:`(r,)`."""
    Vh: Array2D[ScalarTypeT]
    """Spatial modes as an array of shape :math:`(r, d)`."""

    @property
    def physical_dim(self) -> int:
        return self.Vh.shape[1]

    def encode(self, x: ArrayND[ScalarTypeT]) -> ArrayND[ScalarTypeT]:
        assert x.shape[-1] == self.physical_dim

        xp = array_api_compat.array_namespace(x, self.Vh)
        return xp.einsum("...j,ij->...i", x, self.Vh)

    def decode(self, xhat: ArrayND[ScalarTypeT]) -> ArrayND[ScalarTypeT]:
        assert xhat.shape[-1] == self.lifted_dim

        xp = array_api_compat.array_namespace(xhat, self.Vh)
        return xp.einsum("...i,ij->...j", xhat, xp.conj(self.Vh))

    def evolve(self, xhat: ArrayND[ScalarTypeT]) -> ArrayND[ScalarTypeT]:
        assert xhat.shape[-1] == self.lifted_dim

        xp = array_api_compat.array_namespace(xhat, self.A)
        return xp.einsum("...j,ij->...i", xhat, self.A)


def build_dmd(
    X: Array2D[ScalarTypeT],
    Y: Array2D[ScalarTypeT] | None = None,
    *,
    rank: int | None = None,
    eps: float | None = None,
    xp: Any = None,
) -> ReducedDMD[ScalarTypeT]:
    """Construct a DMD approximation of the system with snapshots *X* and outputs *Y*.

    For robust results, it is recommended to apply the :func:`total_least_squares`
    algorithm to the snapshots, so that any noise is handled consistently.

    :arg X: system snapshots of shape ``(nsnapshots, ndim)``.
    :arg Y: system outputs of shape ``(nsnapshots, ndim)``.
    :arg rank: if given, the desired fixed rank of the approximation.
    :arg eps: a minimum absolute tolerance for singular values. Note that this
        is a data-dependent slice and some frameworks (e.g. ``jax``) will not
        be able to compile it.
    """
    if X.ndim != 2:
        raise ValueError(
            f"inputs 'X' must be of shape ``(nsnapshots, dim)``: {X.shape}"
        )

    if Y is None:
        Y = X[:-1, :]
        X = X[1:, :]

    if Y.ndim != 2:
        raise ValueError(
            f"outputs 'Y' must be of shape ``(nsnapshots, dim)``: {Y.shape}"
        )

    _, dx = X.shape
    _, dy = Y.shape
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"inputs 'X' and outputs 'Y' have different shapes: {X.shape} and {Y.shape}"
        )

    if rank is not None and not 0 < rank < min(dx, dy):
        raise ValueError(f"'rank' must be in (0, {min(dx, dy)}): {rank}")

    if eps is not None and eps < 0:
        raise ValueError(f"'eps' must be positive: {eps}")

    if xp is None:
        xp = array_api_compat.array_namespace(X)

    U, S, Vh = xp.linalg.svd(X, full_matrices=False)
    S = xp.astype(S, X.dtype)

    if rank is not None:
        U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]

    if eps is not None:
        mask = xp.abs(S) > eps
        U, S, Vh = U[:, mask], S[mask], Vh[mask, :]

    # construct reduced order model
    Ahat = Vh @ xp.conj(Y).T @ (U / S)
    assert Ahat.ndim == 2
    assert Ahat.shape[0] == Ahat.shape[1]

    return ReducedDMD(A=Ahat, U=U, S=S, Vh=Vh)


# }}}


# {{{ build_full_dmd


@register_dataclass
@dataclass(frozen=True)
class FullDMD(DMDBase[ScalarTypeT]):
    @property
    def physical_dim(self) -> int:
        return self.A.shape[0]

    def encode(self, x: ArrayND[ScalarTypeT]) -> ArrayND[ScalarTypeT]:
        assert x.shape[-1] == self.physical_dim
        return x

    def decode(self, xhat: ArrayND[ScalarTypeT]) -> ArrayND[ScalarTypeT]:
        assert xhat.shape[-1] == self.lifted_dim
        return xhat

    def evolve(self, xhat: ArrayND[ScalarTypeT]) -> ArrayND[ScalarTypeT]:
        assert xhat.shape[-1] == self.lifted_dim

        xp = array_api_compat.array_namespace(xhat, self.A)
        return xp.einsum("...j,ji->...i", xhat, self.A)


def build_full_dmd(
    X: Array2D[ScalarTypeT],
    Y: Array2D[ScalarTypeT] | None = None,
    *,
    # NOTE: using 'ridge' as the default here because (1) it's differentiable
    # and (2) seems to be recommended for Extended DMD. Might reconsider..
    method: Literal["pinv", "ridge"] = "ridge",
    eps: float | None = None,
    xp: Any = None,
) -> FullDMD[ScalarTypeT]:
    r"""Compute the full DMD operator using a pseudo-inverse.

    This is very inefficient for large system, but can work for toy examples. We
    want to solve :math:`Y = X A` for the operator :math:`A`. The implemented
    methods are:

    1. `pinv`: using the pseudo-inverse :math:`A^* = X^\dagger Y`. This is more
        accurate and numerically stable for ill-conditioned :math:`X`.
    2. `ridge`: using a ridge regression on the normal equations. This is more
        efficient and differentiable.

    :arg eps: tolerance used to regularize the pseudo-inverse. This has different
        meanings based on the method being used: (1) a relative tolerance on the
        singular values; (2) a ridge parameter.
    """
    if X.ndim != 2:
        raise ValueError(
            f"inputs 'X' must be of shape ``(nsnapshots, dim)``: {X.shape}"
        )

    if Y is None:
        Y = X[:-1, :]
        X = X[1:, :]

    if Y.ndim != 2:
        raise ValueError(
            f"outputs 'Y' must be of shape ``(nsnapshots, dim)``: {Y.shape}"
        )

    n, dx = X.shape
    n, dy = Y.shape
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"inputs 'X' and outputs 'Y' have different shapes: {X.shape} and {Y.shape}"
        )

    if xp is None:
        xp = array_api_compat.array_namespace(X, Y)
    else:
        assert array_api_compat.array_namespace(X, Y) is xp

    if eps is None:
        eps = max(n, dx, dy) * xp.finfo(X.dtype).eps

    if eps < 0:
        raise ValueError(f"'eps' must be positive: {eps}")

    if method == "pinv":
        # NOTE: this essentially does a least squares fit for `Y = X A`. We
        # don't construct the pseudo-inverse directly to avoid the extra cost.
        U, S, Vh = xp.linalg.svd(X, full_matrices=False)

        S = xp.where(eps * S[0] < S, 1.0 / S, xp.zeros_like(S))

        UY = xp.conj(U).T @ Y
        VS = xp.conj(Vh).T * S
        A = VS @ UY
    elif method == "ridge":
        # NOTE: this tries to solve the regularized optimization problem
        #   A^* = argmin |X A - Y|^2 + \epsilon |A|^2
        # Taking the gradient and setting it to zero gives the normal equations
        #   (X^T X + \epsilon I) A = X^T Y
        # which have size (d, d). To avoid squaring all those ill-conditioned
        # matrices, we instead solve
        #   A^* = argmin |[X, \sqrt{\epsilon} I] A - [Y, O]|^2

        I = xp.eye(dx, dx, dtype=X.dtype, device=X.device)  # ruff: ignore[ambiguous-variable-name]
        O = xp.zeros((dx, dy), dtype=Y.dtype, device=Y.device)  # ruff: ignore[ambiguous-variable-name]

        # FIXME: make eps relative as well? a bit expensive..
        X = xp.concat([X, eps**0.5 * I], axis=0)
        Y = xp.concat([Y, O], axis=0)

        Q, R = xp.linalg.qr(X, mode="reduced")
        A = xp.linalg.solve(R, xp.conj(Q).T @ Y)
    else:
        raise ValueError(f"unknown method: {method!r}")

    return FullDMD(A)


# }}}


# {{{ build_full_extended_dmd


@register_dataclass
@dataclass(frozen=True)
class FullExtendedDMD(DMDBase[ScalarTypeT]):
    C: Array2D[ScalarTypeT]
    observables: tuple[Callable[[ArrayND[ScalarTypeT]], ArrayND[ScalarTypeT]], ...]

    @property
    def physical_dim(self) -> int:
        return self.C.shape[0]

    def encode(self, x: ArrayND[ScalarTypeT]) -> ArrayND[ScalarTypeT]:
        assert x.shape[-1] == self.physical_dim

        xp = array_api_compat.array_namespace(x, self.A)
        return xp.concat(
            [xp.reshape(g(x), (*x.shape[:-1], -1)) for g in self.observables], axis=-1
        )

    def decode(self, xhat: ArrayND[ScalarTypeT]) -> ArrayND[ScalarTypeT]:
        assert xhat.shape[-1] == self.lifted_dim

        xp = array_api_compat.array_namespace(xhat, self.C)
        return xp.einsum("...j,ji->...i", xhat, self.C)

    def evolve(self, xhat: ArrayND[ScalarTypeT]) -> ArrayND[ScalarTypeT]:
        assert xhat.shape[-1] == self.lifted_dim

        xp = array_api_compat.array_namespace(xhat, self.A)
        return xp.einsum("...j,ji->...i", xhat, self.A)


def build_full_extended_dmd(
    observables: Sequence[Callable[[ArrayND[ScalarTypeT]], ArrayND[ScalarTypeT]]],
    X: Array2D[ScalarTypeT],
    Y: Array2D[ScalarTypeT] | None = None,
    *,
    method: Literal["pinv", "ridge"] = "ridge",
    first_observable_is_state: bool = False,
    eps: float | None = None,
    xp: Any = None,
) -> FullExtendedDMD[ScalarTypeT]:
    r"""Construct a DMD approximation of the system in the space of the *observables*.

    Each observable :math:`g` is evaluated on the snapshots and its output is
    appended to the feature axis, lifting the system into a space of shape
    ``(nsnapshots, sum(d_g))``. The returned operator :math:`A` acts on this
    lifted space.

    To evolve a state :math:`x` with the resulting operator, lift it and
    apply :math:`A` from the right, then finally project it back to the state space.

    .. code:: python

        z = xp.concat([g(x[None, :]) for g in observables], axis=1)
        z = z @ A
        X = z @ C

    :arg observables: sequence of maps :math:`g(x)`, each returning an array
        of shape ``(nsnapshots, d_g)``.
    :arg X: system snapshots of shape ``(nsnapshots, ndim)``.
    :arg Y: optional outputs of the same shape as *X*. If given, the operator
        is fit on the pairs ``(X, Y)``; otherwise *X* is treated as a single
        trajectory and the pairs ``(X[:-1], X[1:])`` are used.

    :returns: a tuple of ``(A, C)`` matrices, where the :math:`A` matrix can be
        used to evolve the system in the lifted space and :math:`C` can be used
        to project back to the state space.
    """
    if X.ndim != 2:
        raise ValueError(
            f"inputs 'X' must be of shape ``(nsnapshots, dim)``: {X.shape}"
        )

    if Y is not None and Y.ndim != 2:
        raise ValueError(
            f"outputs 'Y' must be of shape ``(nsnapshots, dim)``: {Y.shape}"
        )

    if Y is not None and X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"inputs 'X' and outputs 'Y' have different shapes: {X.shape} and {Y.shape}"
        )

    if not observables:
        raise ValueError("no 'observables' provided (use at least the identity map)")

    def lift(x: Array2D[ScalarTypeT], *, xp: Any) -> Array2D[ScalarTypeT]:
        return xp.concat(
            [xp.reshape(g(x), (x.shape[0], -1)) for g in observables], axis=1
        )

    if Y is None:
        if xp is None:
            xp = array_api_compat.array_namespace(X)

        X_lift = lift(X, xp=xp)
        Y_lift = X_lift[1:, :]
        X_lift = X_lift[:-1, :]
        X = X[:-1]
    else:
        if xp is None:
            xp = array_api_compat.array_namespace(X, Y)

        X_lift = lift(X, xp=xp)
        Y_lift = lift(Y, xp=xp)

    A = build_full_dmd(X_lift, Y_lift, method=method, eps=eps, xp=xp).A
    if first_observable_is_state:
        C = xp.eye(X_lift.shape[1], X.shape[1], dtype=X.dtype, device=X.device)
    else:
        C = build_full_dmd(X_lift, X, method=method, eps=eps, xp=xp).A

    return FullExtendedDMD(A=A, C=C, observables=tuple(observables))


# }}}


# {{{ total-least-squares


def total_least_squares(
    X: Array2D[ScalarTypeT],
    Y: Array2D[ScalarTypeT] | None = None,
    *,
    rank: int | None = None,
    eps: float | None = None,
    xp: Any = None,
) -> tuple[Array2D[ScalarTypeT], Array2D[ScalarTypeT]]:
    """Apply Total Least Squares de-biasing to the dataset.

    :arg X: system snapshots of shape ``(nsnapshots, ndim)``.
    :arg Y: system outpyts of shape ``(nsnapshots, ndim)``.
    :arg rank: if given, the desired fixed rank of the approximation.
    :arg eps: a minimum absolute tolerance for singular values. Note that this
        is a data-dependent slice and some frameworks (e.g. ``jax``) will not
        be able to compile it.
    """
    if X.ndim != 2:
        raise ValueError(
            f"inputs 'X' must be of shape ``(nsnapshots, dim)``: {X.shape}"
        )

    if Y is None:
        Y = X[:-1, :]
        X = X[1:, :]

    if Y.ndim != 2:
        raise ValueError(
            f"outputs 'Y' must be of shape ``(nsnapshots, dim)``: {Y.shape}"
        )

    n, dx = X.shape
    n, dy = Y.shape
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"inputs 'X' and outputs 'Y' have different shapes: {X.shape} and {Y.shape}"
        )

    if xp is None:
        xp = array_api_compat.array_namespace(X, Y)

    if rank is not None and not 0 < rank < min(dx, dy):
        raise ValueError(f"'rank' must be in (0, {min(dx, dy)}): {rank}")

    if eps is not None and eps < 0:
        raise ValueError(f"'eps' must be positive: {eps}")

    Z = xp.concat([X, Y], axis=1)
    assert Z.shape == (n, dx + dy)

    U, S, Vh = xp.linalg.svd(Z, full_matrices=False)
    S = xp.astype(S, X.dtype)

    if rank is not None:
        U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]

    if eps is not None:
        mask = xp.abs(S) > eps
        U, S, Vh = U[:, mask], S[mask], Vh[mask, :]

    US = U * S
    X = US @ Vh[:, :dx]
    Y = US @ Vh[:, dx:]

    return X, Y


# }}}


# {{{ diagnostics


def relative_forecast_error(
    dmd: DMDBase[ScalarTypeT],
    X: Array2D[ScalarTypeT],
) -> ArrayND[np.floating[Any]]:
    """Per-step relative error of :meth:`DMDBase.predict` against snapshots *X*."""
    xp = array_api_compat.array_namespace(X)

    X_pred = dmd.predict(X[0], X.shape[0] - 1, full=True)
    return xp.linalg.norm(X_pred - X, axis=-1) / xp.linalg.norm(X)


def fit_residual(
    dmd: DMDBase[ScalarTypeT],
    X1: Array2D[ScalarTypeT],
    X2: Array2D[ScalarTypeT],
) -> float:
    """Relative residual of the one-step prediction on the training pairs."""
    xp = array_api_compat.array_namespace(X1, X2)

    X_fit = dmd.decode(dmd.evolve(dmd.encode(X1)))
    return float(xp.linalg.norm(X_fit - X2) / xp.linalg.norm(X2))


def cumulative_energy(S: Array1D[ScalarTypeT]) -> Array1D[np.floating[Any]]:
    """Normalized cumulative energy of the singular values *S*."""
    xp = array_api_compat.array_namespace(S)

    S2 = xp.abs(S) ** 2
    return xp.cumulative_sum(S2) / xp.sum(S2)


# }}}
