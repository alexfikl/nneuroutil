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
from nneuroutil.typing import Array0D, Array1D, Array2D, ArrayND, ScalarTypeT

log = module_logger(__name__)


# {{{ DMDBase


@register_dataclass
@dataclass(frozen=True)
class DMDBase(ABC, Generic[ScalarTypeT]):
    """Base class for linear approximations of a dynamical system in a lifted space.

    All states are assumed to have their spatial dimension on the last axis,
    e.g. snapshots of shape ``(nsnapshots, ndim)``. The dynamics are given by

    .. code:: python

        xhat_next = self.evolve(self.encode(x))

    where :meth:`encode` maps the physical state into the lifted space and
    :attr:`~nneuroutil.dmd.DMDBase.A` acts on its last dimension.
    """

    A: Array2D[ScalarTypeT]
    """Linear evolution operator acting on the lifted space."""

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
        """Project the physical state ``(..., d)`` into the lifted space."""

    @abstractmethod
    def decode(self, xhat: ArrayND[ScalarTypeT]) -> ArrayND[ScalarTypeT]:
        """Project the lifted state ``(..., r)`` into the physical space."""

    @abstractmethod
    def evolve(self, xhat: ArrayND[ScalarTypeT]) -> ArrayND[ScalarTypeT]:
        """Advance the lifted state ``(..., r)`` by a single time step."""

    def predict(
        self,
        x0: ArrayND[ScalarTypeT],
        maxit: int,
        *,
        full: bool = False,
    ) -> ArrayND[ScalarTypeT]:
        """Evolve the initial condition *x0* forward for *maxit* steps.

        :arg x0: initial condition with its spatial dimension on the last axis.
        :arg maxit: number of time steps to take.
        :arg full: if ``True``, also include intermediate steps and *x0* itself;
            the result is stacked along a new leading axis. Otherwise, only the
            final state is returned.
        """
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
    """Rank-truncated DMD model built from the SVD of the snapshot matrix.

    The lifted space is the space spanned by the leading right singular
    vectors :attr:`Vh`.
    """

    U: Array2D[ScalarTypeT]
    """Left singular vectors as an array of shape :math:`(n, r)`."""
    S: Array1D[ScalarTypeT]
    """Singular values as an array of shape :math:`(r,)`."""
    Vh: Array2D[ScalarTypeT]
    """Right singular vectors as an array of shape :math:`(r, d)`."""

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

    if n >= 2 * dx:
        # NOTE: precondition with a QR to avoid SVD-ing very tall matrices
        Q, R = xp.linalg.qr(X, mode="reduced")
        Ur, S, Vh = xp.linalg.svd(R, full_matrices=False)

        if rank is not None:
            Ur, S, Vh = Ur[:, :rank], S[:rank], Vh[:rank, :]

        if eps is not None:
            mask = xp.abs(S) > eps
            Ur, S, Vh = Ur[:, mask], S[mask], Vh[mask, :]

        # construct reduced order model
        QTY = xp.conj(Q).T @ Y
        Ahat = Vh @ (xp.conj(QTY).T @ (Ur / S))
        U = Q @ Ur
    else:
        U, S, Vh = xp.linalg.svd(X, full_matrices=False)

        if rank is not None:
            U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]

        if eps is not None:
            mask = xp.abs(S) > eps
            U, S, Vh = U[:, mask], S[mask], Vh[mask, :]

        # construct reduced order model
        # (r, dy) @ (dy, n) @ (n @ r): association avoids two large matrix multiplies
        Ahat = Vh @ (xp.conj(Y).T @ (U / S))

    assert Ahat.ndim == 2
    assert Ahat.shape[0] == Ahat.shape[1]

    return ReducedDMD(A=Ahat, U=U, S=S, Vh=Vh)


# }}}


# {{{ build_exact_dmd


@register_dataclass
@dataclass(frozen=True)
class ExactDMD(DMDBase[ScalarTypeT]):
    r"""DMD model for which the lifted space is the physical space itself.

    The encoding and decoding steps are both the identity and
    :attr:`~nneuroutil.dmd.DMDBase.A` acts directly on the physical space.

    Note that this implementation fits the dense :math:`d \times d` operator
    directly. Use :class:`ReducedDMD` to construct a low-rank approximation
    instead.
    """

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


def build_exact_dmd(
    X: Array2D[ScalarTypeT],
    Y: Array2D[ScalarTypeT] | None = None,
    *,
    # NOTE: using 'ridge' as the default here because (1) it's differentiable
    # and (2) seems to be recommended for Extended DMD. Might reconsider..
    method: Literal["pinv", "ridge"] = "ridge",
    eps: float | None = None,
    xp: Any = None,
) -> ExactDMD[ScalarTypeT]:
    r"""Fit a linear model :math:`Y = X A` on snapshot pairs.

    This is very inefficient for large systems, but can work for toy examples.
    The implemented methods are:

    1. ``pinv``: using the pseudo-inverse :math:`A^* = X^\dagger Y`. This is more
       accurate and numerically stable for ill-conditioned :math:`X`.
    2. ``ridge``: using a ridge regression on the normal equations. This is more
       efficient and differentiable.

    :arg eps: tolerance used to regularize the pseudo-inverse. This has different
        meanings based on the method being used: (1) a relative tolerance on the
        singular values; (2) a ridge parameter. If not given, it is chosen as
        ``max(n, d) * eps_machine``.

    :returns: an approximation of the full-state dynamics.
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

    if eps is None:
        eps = max(n, dx, dy) * xp.finfo(X.dtype).eps

    if eps < 0:
        raise ValueError(f"'eps' must be positive: {eps}")

    if method == "pinv":
        # NOTE: this essentially does a least squares fit for `Y = X A`. We
        # don't construct the pseudo-inverse directly to avoid the extra cost.

        if n >= 2 * dx:
            # NOTE: if X is very tall (n >= 2 dx), doing a direct SVD is costly.
            # We do a thin QR first and then an SVD on the small (dx, dx) R factor
            # to save memory workspace and compute time.

            Q, R = xp.linalg.qr(X, mode="reduced")
            Ur, S, Vh = xp.linalg.svd(R, full_matrices=False)
            S = xp.where(eps * S[0] < S, 1.0 / S, xp.zeros_like(S))

            UY = xp.conj(Ur).T @ (xp.conj(Q).T @ Y)
            VS = xp.conj(Vh).T * S
            A = VS @ UY
        else:
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
        # which have size (d, d). To avoid squaring condition numbers while
        # avoiding large augmented matrix allocations, we use a two-stage QR:
        #   1. X = Q1 R1,  B1 = Q1^* Y
        #   2. [R1; \sqrt{\epsilon} I] = Q2 R
        #   3. R A = Q2[:d]^* B1
        # NOTE: do not be inclined to use an SVD here, even it could be faster for
        # n < 2 d, because it ruins differentiability, which was the whole point.

        # Step 1: Orthogonal reduction of X -> preserves kappa(X)
        Q1, R1 = xp.linalg.qr(X, mode="reduced")
        B1 = xp.conj(Q1).T @ Y
        del Q1

        # Step 2: Regularized QR on the small (2 * dx, dx) stacked R1 factor
        I = xp.eye(dx, dx, dtype=X.dtype, device=X.device)  # ruff: ignore[ambiguous-variable-name]
        R_aug = xp.concat([R1, eps**0.5 * I], axis=0)
        del R1

        Q2, R = xp.linalg.qr(R_aug, mode="reduced")
        del R_aug

        # Q2[:dx] acts on B1, Q2[dx:] acts on 0
        rhs = xp.conj(Q2[:dx]).T @ B1
        del Q2, B1

        # FIXME: would be nice if the Array API would have a triangular solv
        # for this, but it shouldn't be an issue (the two QRs above dominate)
        A = xp.linalg.solve(R, rhs)
    else:
        raise ValueError(f"unknown method: {method!r}")

    return ExactDMD(A)


# }}}


# {{{ build_exact_extended_dmd


@register_dataclass
@dataclass(frozen=True)
class ExactExtendedDMD(DMDBase[ScalarTypeT]):
    r"""DMD model in the space of nonlinear observables.

    The lifted state is obtained by concatenating the outputs of
    :attr:`observables` along the last axis and the physical state is
    recovered by applying the decoder :attr:`C`.

    Note that this implementation fits the dense :math:`m \times m` lifted
    operator directly.
    """

    C: Array2D[ScalarTypeT]
    """Decoder mapping the lifted space back to the physical space."""
    observables: tuple[Callable[[ArrayND[ScalarTypeT]], ArrayND[ScalarTypeT]], ...]
    """Sequence of maps lifting the physical state into the observable space."""

    @property
    def physical_dim(self) -> int:
        return self.C.shape[1]

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


def build_exact_extended_dmd(
    observables: Sequence[Callable[[ArrayND[ScalarTypeT]], ArrayND[ScalarTypeT]]],
    X: Array2D[ScalarTypeT],
    Y: Array2D[ScalarTypeT] | None = None,
    *,
    method: Literal["pinv", "ridge"] = "ridge",
    first_observable_is_state: bool = False,
    rank: int | None = None,
    eps: float | None = None,
    xp: Any = None,
) -> ExactExtendedDMD[ScalarTypeT]:
    r"""Construct a DMD approximation of the system in the space of the *observables*.

    Each observable :math:`g` is evaluated on the snapshots and its output is
    appended to the last axis, lifting the system into a space of shape
    ``(nsnapshots, sum(d_g))``. The returned operator acts on this lifted space.

    Evolution of a state :math:`x` can be performed with
    :meth:`DMDBase.predict`, or manually as

    .. code:: python

        z = xp.concat([g(x) for g in observables], axis=-1)
        z = xp.einsum("...j,ji->...i", z, dmd.A)
        x_next = xp.einsum("...j,ji->...i", z, dmd.C)

    :arg observables: sequence of maps :math:`g(x)`, each taking states with the
        spatial dimension on the last axis and returning an array of shape
        ``(..., d_g)``.
    :arg Y: optional outputs of the same shape as *X*. If given, the operator
        is fit on the pairs ``(X, Y)``; otherwise *X* is treated as a single
        trajectory and the pairs ``(X[:-1], X[1:])`` are used.
    :arg first_observable_is_state: if ``True``, use an exact (rectangular)
        identity decoder built from the first observable, instead of fitting
        one by regression.

    :returns: a DMD approximation of the lifted dynamics together with its
        decoder back to the physical space.
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

    if rank is not None:
        X_lift, Y_lift = total_least_squares(X_lift, Y_lift, rank=rank, eps=eps)

    if first_observable_is_state:
        A = build_exact_dmd(X_lift, Y_lift, method=method, eps=eps, xp=xp).A
        C = xp.eye(X_lift.shape[1], X.shape[1], dtype=X.dtype, device=X.device)
    else:
        # NOTE: this should avoid performing two SVDs on X_lift (for A and C)
        YX_lift = xp.concat([Y_lift, X], axis=1)
        result = build_exact_dmd(X_lift, YX_lift, method=method, eps=eps, xp=xp).A

        A = result[:, : Y_lift.shape[1]]
        C = result[:, Y_lift.shape[1] :]

    return ExactExtendedDMD(A=A, C=C, observables=tuple(observables))


# }}}


# {{{ forward-backward dmd


@register_dataclass
@dataclass(frozen=True)
class ForwardBackwardDMD(ExactDMD[ScalarTypeT]):
    """DMD model with forward-backward debiasing of the fitted operator.

    The operator :attr:`~nneuroutil.dmd.DMDBase.A` is the geometric mean of a
    forward and a backward least-squares fit, whose systematic
    errors-in-variables biases have opposite signs and cancel to first order.
    """

    A_forward: Array2D[ScalarTypeT]
    """Uncorrected operator fit on the pairs ``(X, Y)``."""
    A_backward: Array2D[ScalarTypeT]
    """Uncorrected operator fit on the pairs ``(Y, X)``."""


def build_forward_backward_dmd(
    X: Array2D[ScalarTypeT],
    Y: Array2D[ScalarTypeT] | None = None,
    *,
    method: Literal["pinv", "ridge"] = "ridge",
    eps: float | None = None,
    xp: Any = None,
) -> ForwardBackwardDMD[ScalarTypeT]:
    r"""Construct a DMD approximation with forward-backward debiasing based on
    [Dawson2016]_.

    This is implemented in Algorithm 3 from [Dawson2016]_. The algorithm fits
    the forward operator :math:`A_f` on the pairs ``(X, Y)`` and the backward
    operator :math:`A_b` on the pairs ``(Y, X)``. It then combines them into the
    geometric mean

    .. math::

        A = A_f^{1/2} A_b^{-1/2}

    which cancels the first-order eigenvalue shrinkage that least-squares
    fits exhibit on noisy snapshot pairs. In such setups, it may work better than
    :class:`ExactDMD` or :class:`ExactExtendedDMD`.

    .. [Dawson2016] S. T. M. Dawson, M. S. Hemati, M. O. Williams, C. W. Rowley,
        *Characterizing and Correcting for the Effect of Sensor Noise in the
        Dynamic Mode Decomposition*,
        Experiments in Fluids, Vol. 57, pp. 42--42, 2016,
        `doi:10.1007/s00348-016-2127-7 <https://doi.org/10.1007/s00348-016-2127-7>`__.

    :arg method: regularized solver used for both fits.
    :arg eps: tolerance for the regularized solver, see :func:`build_exact_dmd`.

    :returns: a model whose operator is the debiased forward-backward mean.
    """
    if Y is None:
        Y = X[1:]
        X = X[:-1]

    if xp is None:
        xp = array_api_compat.array_namespace(X, Y)

    A_f = build_exact_dmd(X, Y, method=method, eps=eps, xp=xp).A
    A_b = build_exact_dmd(Y, X, method=method, eps=eps, xp=xp).A

    lambda_f, v_f = xp.linalg.eig(A_f)
    lambda_b, v_b = xp.linalg.eig(A_b)

    A_f_sqrt = (v_f * xp.sqrt(lambda_f)) @ xp.linalg.inv(v_f)
    A_b_sqrt = (v_b * (1.0 / xp.sqrt(lambda_b))) @ xp.linalg.inv(v_b)
    A_fb = A_f_sqrt @ A_b_sqrt

    # NOTE: the principal square root of a real matrix is real, but the
    # eigendecomposition based construction may leave a small imaginary part
    if xp.isdtype(X.dtype, "complex floating"):
        A_fb = xp.real(A_fb)

    return ForwardBackwardDMD(A=A_fb, A_forward=A_f, A_backward=A_b)


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

    # NOTE: for (classic?) TLS, we want to project Y into the range of X. In
    # that case, the rank cannot be larger than dx, because that would mean Y
    # cannot be represented in that subspace. This seems reasonable..
    if rank is not None and not 0 < rank <= min(n, dx):
        raise ValueError(f"'rank' must be in (0, {min(n, dx)}]: {rank}")

    if eps is not None and eps < 0:
        raise ValueError(f"'eps' must be positive: {eps}")

    Z = xp.concat([X, Y], axis=1)
    assert Z.shape == (n, dx + dy)
    d = dx + dy

    if n >= 2 * d:
        # NOTE: for tall matrices (n >= 2 d), compute thin QR first and perform
        # SVD on the small (d, d) factor R. Then project without materializing U.
        Q, R = xp.linalg.qr(Z, mode="reduced")
        Ur, S, Vh = xp.linalg.svd(R, full_matrices=False)

        if rank is not None:
            Ur, S, Vh = Ur[:, :rank], S[:rank], Vh[:rank, :]

        if eps is not None:
            mask = xp.abs(S) > eps
            Ur, S, Vh = Ur[:, mask], S[mask], Vh[mask, :]

        W = Ur * S
        Z = Q @ (W @ Vh)
        X = Z[:, :dx]
        Y = Z[:, dx:]
    else:
        U, S, Vh = xp.linalg.svd(Z, full_matrices=False)

        if rank is not None:
            U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]

        if eps is not None:
            mask = xp.abs(S) > eps
            U, S, Vh = U[:, mask], S[mask], Vh[mask, :]

        Z = (U * S) @ Vh
        X = Z[:, :dx]
        Y = Z[:, dx:]

    return X, Y


# }}}


# {{{ diagnostics


def relative_forecast_error(
    dmd: DMDBase[ScalarTypeT],
    X: Array2D[ScalarTypeT],
    Xpred: Array2D[ScalarTypeT] | None = None,
    *,
    maxit: int | None = None,
    xp: Any = None,
) -> Array1D[np.floating[Any]]:
    r"""Compute the per-step relative error of a forecast against *X*.

    The error at step :math:`k` is given by

    .. math::

        E = \frac{\|x_k - \hat{x}_k\|}{\|X\|},

    where the denominator is the norm of the whole trajectory to keep the
    errors bounded as amplitudes decay. By default, the forecast is generated
    from ``X[0]`` with :meth:`DMDBase.predict`.

    :arg Xpred: optional precomputed forecast trajectory. If not provided, it is
        computed with :meth:`DMDBase.predict` from ``X[0]``.
    :arg maxit: number of forecast steps to evaluate. If *None*, the maximum
        number of available steps is used.
    """
    max_maxit = X.shape[0] if Xpred is None else min(Xpred.shape[0], X.shape[0])
    if maxit is None:
        maxit = max_maxit - 1

    if not 0 < maxit < max_maxit:
        raise ValueError(f"'maxit' must be in (0, {max_maxit}): {maxit}")

    if xp is None:
        xp = (
            array_api_compat.array_namespace(X)
            if Xpred is None
            else array_api_compat.array_namespace(X, Xpred)
        )

    if Xpred is None:
        Xpred = dmd.predict(X[0], maxit, full=True)

    error = xp.linalg.norm(Xpred[: maxit + 1] - X[: maxit + 1], axis=-1)
    xnorm = xp.linalg.norm(X[: maxit + 1], axis=-1)

    xnorm = xp.linalg.norm(xnorm, axis=0, keepdims=True)
    xnorm = xp.where(xnorm < 100 * xp.finfo(X.dtype).eps, 1.0, xnorm)

    return error / xnorm


def fit_residual(
    dmd: DMDBase[ScalarTypeT],
    X1: Array2D[ScalarTypeT],
    X2: Array2D[ScalarTypeT],
    *,
    xp: Any = None,
) -> Array0D[np.floating[Any]]:
    r"""Compute the relative one-step residual of the fitted model.

    Measures how well the model explains the training pairs by comparing
    ``dmd.decode(dmd.evolve(dmd.encode(X1)))`` against *X2*, normalized by the
    norm of *X2*.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(X1, X2)

    X_fit = dmd.decode(dmd.evolve(dmd.encode(X1)))

    x2norm = xp.linalg.norm(X2)
    if abs(x2norm) < 100 * xp.finfo(X2.dtype).eps:
        x2norm = 1.0

    return xp.linalg.norm(X_fit - X2) / x2norm


def cumulative_energy(
    S: Array1D[ScalarTypeT],
    *,
    xp: Any = None,
) -> Array1D[np.floating[Any]]:
    r"""Compute the normalized cumulative energy of singular values *S*."""
    if xp is None:
        xp = array_api_compat.array_namespace(S)

    S2 = xp.abs(S) ** 2
    return xp.cumulative_sum(S2) / xp.sum(S2)


# }}}
