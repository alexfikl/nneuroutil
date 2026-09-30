# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import array_api_compat
import numpy as np

from nneuroutil.helpers import module_logger, register_dataclass
from nneuroutil.typing import Array0D, Array1D, Array2D

log = module_logger(__name__)

# {{{ classify_linear_discriminant_analysis


@register_dataclass
@dataclass(frozen=True)
class LinearDiscriminandAnalysisClassifier:
    """Estimator obtained from :func:`classify_linear_discriminant_analysis`."""

    G: Array2D[np.floating[Any]]
    """Projection matrix to the reduced :math:`k - 1` dimensional subspace."""
    centroids: Array1D[np.floating[Any]]
    """Global centroid of shape :math:`(nfeatures,)`."""
    proj_centroids: Array2D[np.floating[Any]]
    """Projected class centroids of shape :math:`(k, k - 1)`."""
    labels: Array1D[np.floating[Any]]
    """Array of class labels of shape :math:`(k,)`."""

    def predict(
        self,
        x: Array1D[np.floating[Any]] | Array2D[np.floating[Any]],
        *,
        metric: Literal["l2", "cos"] = "l2",
        xp: Any = None,
    ) -> Array1D[np.floating[Any]]:
        """Predict class labels for given sample(s) *x*.

        :arg metric: classification metric used in the reduced subspace:
            * ``"l2"``: assigns to the centroid with minimum Euclidean distance.
            * ``"cos"``: assigns to the centroid with maximum cosine similarity.
        """
        if x.ndim == 1:
            return self.predict(x[None, :], xp=xp)[0]

        if xp is None:
            xp = array_api_compat.array_namespace(x, self.G)

        # NOTE: this largely implements Algorithm 3 from [Howland2004]_
        cm = self.centroids
        cz = self.proj_centroids
        Z = (x - cm) @ self.G

        if metric == "l2":
            d = (
                xp.sum(Z**2, axis=1)[:, None]
                - 2 * Z @ cz.T  # ty: ignore[unresolved-attribute]
                + xp.sum(cz**2, axis=1)[None]
            )

            return self.labels[xp.argmin(d, axis=1)]
        elif metric == "cos":
            Zn = Z / xp.linalg.norm(Z, axis=1, keepdims=True)
            Cn = cz / xp.linalg.norm(cz, axis=1, keepdims=True)
            return self.labels[xp.argmax(Zn @ Cn.T, axis=1)]
        else:
            raise ValueError(f"unknown metric: {metric!r}")


def classify_linear_discriminant_analysis(
    features: tuple[Array2D[np.floating[Any]], ...],
    labels: tuple[Any, ...],
    *,
    eps: float | None = None,
    xp: Any = None,
) -> LinearDiscriminandAnalysisClassifier:
    """Classify the given dataset using standard LDA (mainly based on
    Algorithm 1 from [Howland2004]_).

    .. [Howland2004] P. Howland, H. Park,
        *Generalizing Discriminant Analysis Using the Generalized Singular Value
        Decomposition*,
        IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 26,
        pp. 995--1006, 2004,
        `doi:10.1109/tpami.2004.46 <https://doi.org/10.1109/tpami.2004.46>`__.

    :arg features: sequence of 2D arrays containing sample features for each
        class, each of shape ``(n_samples_i, nfeatures)``.
    :arg labels: sequence of labels corresponding to each class in *features*.
    :arg eps: tolerance used for rank determination in the SVD truncation.
    """

    if len(features) != len(labels):
        raise ValueError(
            f"'features' and 'labels' do not match: {len(features)} and {len(labels)}"
        )

    if len(features) <= 1:
        raise ValueError(
            "cannot perform classification on 'features' with 0 or 1 classes"
        )

    _, nfeatures = features[0].shape
    dtype = features[0].dtype
    device = array_api_compat.device(features[0])

    if any(f.ndim != 2 or f.shape[-1] != nfeatures for f in features[1:]):
        raise ValueError(
            f"'features' do not have the same number of features: '{nfeatures}'"
        )

    if xp is None:
        xp = array_api_compat.array_namespace(*features)

    if eps is None:
        eps = 10.0 * xp.finfo(dtype).eps

    if eps <= 0:
        raise ValueError(f"'eps' must be positive: {eps}")

    # compute class sizes: Equation (1) in [Howland2004]_
    ns = xp.asarray([len(x) for x in features], dtype=dtype, device=device)
    n = xp.sum(ns)
    k = len(labels)

    # compute centroids: c^{(i)} and c in [Howland2004]_
    cs = xp.stack([xp.mean(x, axis=0) for x in features])
    c_m = ns @ cs / n

    # compute H_B and H_W from 'features' and form K = [H_B, H_W]^T
    K = xp.vstack([
        xp.sqrt(ns)[:, None] * (cs - c_m),
        *(x - c_i for x, c_i in zip(features, cs, strict=True)),
    ])

    # compute the SVD
    P, R, Qh = xp.linalg.svd(K, full_matrices=False)

    rank = xp.sum(R[0] * (k + n) * eps < R)
    P, S, Qh = P[:, :rank], R[:rank], Qh[:rank]

    _, _, Wt = xp.linalg.svd(P[:k], full_matrices=False)
    G = Qh.T @ (Wt.T[:, : k - 1] / S[:, None])

    return LinearDiscriminandAnalysisClassifier(
        G=G,
        centroids=c_m,
        proj_centroids=(cs - c_m) @ G,
        labels=xp.asarray(labels, dtype=dtype, device=device),
    )


# }}}


# {{{ classify_support_vector_machine


class SupportVectorMachineFunction(Protocol):
    """Protocol for the dual objective function returning ``(value, gradient)``."""

    def __call__(
        self, x: Array2D[np.floating[Any]], /
    ) -> tuple[Array0D[np.floating[Any]], Array2D[np.floating[Any]]]: ...


class SupportVectorMachineSolver(Protocol):
    """Protocol for box-constrained solvers minimizing *fun* in ``[lower,
    upper]`` starting from *x0*.

    The problem for SVM is a quadratic, so special solvers could be used.
    """

    def __call__(
        self,
        fun: SupportVectorMachineFunction,
        x0: Array2D[np.floating[Any]],
        lower: float,
        upper: float,
        /,
    ) -> Array2D[np.floating[Any]]:
        pass


def _svm_estimate_lipschitz_constant(
    A: Array2D[np.floating[Any]],
    *,
    niters: int = 8,
    xp: Any = None,
) -> Array0D[np.floating[Any]]:
    """Estimate the Lipschitz constant of the dual gradient using power iteration.

    The constant is given by the eigenvalue of the Gram matrix :math:`K = A A^T`.
    """
    if xp is None:
        xp = array_api_compat.array_namespace(A)

    device = array_api_compat.device(A)
    v = xp.ones((A.shape[1], 1), dtype=A.dtype, device=device)
    v = v / xp.sqrt(xp.sum(v * v))

    for _ in range(niters):
        v = A.T @ (A @ v)  # ty: ignore[unresolved-attribute]
        norm = xp.sqrt(xp.sum(v * v))
        v = xp.where(norm > 0, v / norm, v)

    Av = A @ v
    eps = xp.asarray(1.0e-8, dtype=A.dtype, device=device)
    return xp.maximum(xp.sum(Av * Av), eps)


def _solve_svm_jax(
    func: SupportVectorMachineFunction,
    x0: Array2D[np.floating[Any]],
    lower: float,
    upper: float,
    /,
    *,
    maxiter: int = 500,
    atol: float = 1.0e-3,
    L0: float | Array0D[np.floating[Any]] | None = None,
) -> Array2D[np.floating[Any]]:
    """Solve the box-constrained dual problem using JIT-compiled FISTA in JAX."""

    # NOTE: on performance:
    # 1. jax.scipy.optimize.mimize.BFGS seems to be a lot slower. Also, because
    #    it is not the low-storage L-BFGS variant, it uses a lot of memory.
    # 2. Jitting inside this function makes everything a lot faster. The user can
    #    still jit classify_support_vector_machine without an issue.
    # 3. This implements the same algorithm as _solve_svm_fista, but using jax
    #    constructs to get better performance.

    if L0 is None:
        raise ValueError("Lipschitz constant 'L0' must be provided to 'solve_svm_jax'")

    import jax
    import jax.numpy as jnp

    @jax.jit
    def _solve_fista(x0: Array2D[np.floating[Any]]) -> Array2D[np.floating[Any]]:
        L = jnp.asarray(L0, dtype=x0.dtype)

        def cond(val: tuple[Any, ...]) -> Any:
            _, _, _, it, d_norm = val
            return (it < maxiter) & (L * d_norm >= atol)

        def body(val: tuple[Any, ...]) -> tuple[Any, ...]:
            x, z, t, it, _ = val
            _, gz = func(z)

            znew = jnp.clip(z - gz / L, lower, upper)
            d = znew - z

            t1 = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
            z_next = znew + ((t - 1.0) / t1) * (znew - x)

            return (znew, z_next, t1, it + 1, jnp.max(jnp.abs(d)))

        val0 = (x0, x0, 1.0, 0, 1.0)
        final_val = jax.lax.while_loop(cond, body, val0)
        return final_val[0]

    return _solve_fista(x0)


def _solve_svm_torch(
    func: SupportVectorMachineFunction,
    x0: Array2D[np.floating[Any]],
    lower: float,
    upper: float,
    /,
    *,
    maxiter: int = 500,
) -> Array2D[np.floating[Any]]:
    """Solve the box-constrained dual problem using L-BFGS with sigmoid
    reparameterization in PyTorch.
    """
    import torch

    # NOTE: pytorch does not seem to have a constrained optimizer that would work
    # here. We reparameterize using a sigmoid, but that has problems
    # 1. A value of zero/one means u -> ±infty
    # 2. The optimization is slower than it needs to be due to it trying hard to
    #    find these zero/one solutions.

    device = x0.device
    dtype = x0.dtype
    scale = upper - lower

    u = torch.zeros(x0.shape, dtype=dtype, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [u],
        max_iter=maxiter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        with torch.no_grad():
            sig = torch.sigmoid(u)
            x = lower + scale * sig
            val, g = func(x)
            u.grad = g * (scale * sig * (1.0 - sig))
        return val

    optimizer.step(closure)

    with torch.no_grad():
        return lower + scale * torch.sigmoid(u)


def _solve_svm_numpy(
    func: SupportVectorMachineFunction,
    x0: Array1D[np.floating[Any]],
    lower: float,
    upper: float,
    /,
    *,
    maxiter: int = 500,
) -> Array2D[np.floating[Any]]:
    """Solve the box-constrained dual problem using SciPy's L-BFGS-B."""
    from scipy.optimize import Bounds, minimize

    def f(
        v: Array1D[np.floating[Any]],
    ) -> tuple[Array0D[np.floating[Any]], Array1D[np.floating[Any]]]:
        val, g = func(np.reshape(v, x0.shape))
        return val, np.ravel(g)

    res = minimize(  # ty: ignore[no-matching-overload]
        f,
        np.ravel(x0),
        jac=True,
        method="L-BFGS-B",
        bounds=Bounds(lower, upper),
        options={"maxiter": maxiter},
    )

    return np.reshape(res.x, x0.shape)


def _solve_svm_fista(
    func: SupportVectorMachineFunction,
    x0: Array2D[np.floating[Any]],
    lower: float,
    upper: float,
    /,
    *,
    maxit: int = 1000,
    atol: float = 1.0e-3,
    L0: float | Array0D[np.floating[Any]] | None = None,
) -> Array2D[np.floating[Any]]:
    """Solve the box-constrained dual problem using FISTA across array backends."""

    # NOTE: we require the user give the Lipschitz constant here because it can
    # be accurately computed with _svm_estimate_lipschitz_constant. It's a
    # parameter so that a bit of slack can maybe be added too, if necessary.

    if L0 is None:
        raise ValueError(
            "Lipschitz constant 'L0' must be provided to 'solve_svm_fista'"
        )

    xp = array_api_compat.array_namespace(x0)
    device = array_api_compat.device(x0)

    L = xp.asarray(L0, dtype=x0.dtype, device=device)

    x, z, t = x0, x0, 1.0
    for _ in range(maxit):
        _, gz = func(z)
        znew = xp.clip(z - gz / L, lower, upper)
        d = znew - z

        if L * xp.max(xp.abs(d)) < atol:
            return znew

        t1 = (1.0 + (1.0 + 4.0 * t * t) ** 0.5) / 2.0
        z = znew + ((t - 1.0) / t1) * (znew - x)
        x, t = znew, t1

    return x


@register_dataclass
@dataclass(frozen=True)
class SupportVectorMachine:
    """Trained linear Support Vector Machine classifier."""

    W: Array2D[np.floating[Any]]
    """Weight matrix of shape :math:`(nfeatures + 1, nclasses)` containing
    primal weights and bias coefficients for each class.
    """
    mu: Array1D[np.floating[Any]]
    """Mean feature vector of shape :math:`(nfeatures,)` subtracted before
    scoring.
    """
    bias: float
    """Bias scaling constant appended to centered features."""
    labels: Array1D[np.floating[Any]]
    """Array of class labels of shape :math:`(nclasses,)`."""

    def predict(
        self,
        x: Array1D[np.floating[Any]] | Array2D[np.floating[Any]],
        *,
        xp: Any = None,
    ) -> Array1D[np.floating[Any]]:
        r"""Predict class labels for given sample(s) *x*.

        For a sample :math:`x`, decision scores are computed for each class
        as :math:`(x - \mu) W_{:-1} + \text{bias} \cdot W_{-1}`, and the
        label with the highest score is returned.

        :arg x: 1D array of a single sample of shape ``(nfeatures,)`` or 2D array
            of samples of shape ``(nsamples, nfeatures)``.
        :returns: predicted class label(s).
        """
        if x.ndim == 1:
            return self.predict(x[None, :], xp=xp)[0]

        if xp is None:
            xp = array_api_compat.array_namespace(x, self.W)

        scores = (x - self.mu) @ self.W[:-1] + self.bias * self.W[-1]
        idx = xp.argmax(scores, axis=1)

        return self.labels[idx]


def classify_support_vector_machine(
    features: tuple[Array2D[np.floating[Any]], ...],
    labels: tuple[Any, ...],
    *,
    C: float = 1.0,
    bias: float = 1.0,
    solver: SupportVectorMachineSolver | None = None,
    xp: Any = None,
) -> SupportVectorMachine:
    r"""Train a linear Support Vector Machine (SVM) classifier for multi-class data
    (mainly based on Chapter 12 in [Hastie2013]_).

    Multi-class classification is handled using a **one-vs-rest** (OvR) scheme:
    for :math:`K` classes, :math:`K` binary linear SVMs are trained
    simultaneously. Each model separates one class from all other classes.

    The primal problem incorporates an explicit bias by augmenting the centered
    feature matrix with a constant column :math:`[\mathbf{x} - \mu,
    \text{bias}]`. This absorbs the intercept into the weight vector and
    regularizes it along with the feature weights, which removes the dual
    equality constraint :math:`\sum_i y_i \alpha_i = 0`. The dual optimization
    then reduces to a simpler quadratic program with standard box constraints
    :math:`0 \le \alpha_i \le C`.

    .. [Hastie2013] T. Hastie, R. Tibshirani, J. Friedman,
        *Elements of Statistical Learning - Data Mining, Inference, and Prediction*,
        Springer London, Limited, 2013.

    :arg features: sequence of 2D arrays containing sample features for each
        class, each of shape ``(n_samples_i, nfeatures)``.
    :arg labels: sequence of labels corresponding to each class in *features*.
    :arg C: soft-margin regularization parameter (:math:`C > 0`). Controls the
        trade-off between maximizing the margin and minimizing classification
        errors. Larger values penalize margin violations heavily (harder margin),
        while smaller values allow more violations (softer margin, more regularization).
    :arg bias: positive constant appended as an extra feature dimension to include
        an intercept term.
    :arg solver: custom box-constrained quadratic solver matching
        :class:`SupportVectorMachineSolver`. If ``None``, a backend-appropriate
        solver is chosen automatically.
    """

    if len(features) != len(labels):
        raise ValueError(
            f"'features' and 'labels' do not match: {len(features)} and {len(labels)}"
        )

    if len(features) <= 1:
        raise ValueError(
            "cannot perform classification on 'features' with 0 or 1 classes"
        )

    _, nfeatures = features[0].shape
    dtype = features[0].dtype
    device = array_api_compat.device(features[0])

    if any(f.ndim != 2 or f.shape[-1] != nfeatures for f in features[1:]):
        raise ValueError(
            f"'features' do not have the same number of features: '{nfeatures}'"
        )

    if C <= 0:
        raise ValueError(f"'C' weight must be positive: {C}")

    if bias <= 0:
        raise ValueError(f"'bias' must be positive: {bias}")

    if xp is None:
        xp = array_api_compat.array_namespace(*features)

    # construct the full feature array
    A = xp.concat(features, axis=0)
    mu = xp.mean(A, axis=0)
    n = A.shape[0]

    # remove mean and add bias to the array
    # NOTE: the bias is added here to so that we don't have to care about
    # 1. the KKT conditions (12.14-12.16 in [Hastie2013]_).
    # 2. the additional equality constraint `sum(y @ alpha) = 0`
    A = xp.concat([A - mu, xp.full((n, 1), bias, dtype=A.dtype, device=device)], axis=1)

    # construct the ±1 label matrix for the whole thing
    E = xp.eye(len(features), dtype=A.dtype, device=device)
    S = (
        2
        * xp.concat(
            [
                xp.broadcast_to(E[j], (x.shape[0], E.shape[1]))
                for j, x in enumerate(features)
            ],
            axis=0,
        )
        - 1
    )

    # {{{ optimize

    # precompute the Gram matrix for the dual calculation
    K = A @ A.T

    def dual(
        x: Array2D[np.floating[Any]], /
    ) -> tuple[Array0D[np.floating[Any]], Array2D[np.floating[Any]]]:
        # NOTE: dual functional is given by 12.13 in [Hastie2013]_
        Sx = S * x
        F = K @ Sx
        return (
            # NOTE: this is the quadratic function
            #   1/2 * (S * x).T @ K @ (S * x) - x @ 1
            0.5 * xp.sum(Sx * F) - xp.sum(x),
            S * F - 1,
        )

    if solver is None:
        from functools import partial

        if array_api_compat.is_jax_array(features[0]):
            solver = partial(
                _solve_svm_jax, L0=_svm_estimate_lipschitz_constant(A, xp=xp)
            )
        elif array_api_compat.is_torch_array(features[0]):
            solver = _solve_svm_torch
        elif array_api_compat.is_numpy_array(features[0]):
            solver = _solve_svm_numpy
        else:
            solver = partial(
                _solve_svm_fista, L0=_svm_estimate_lipschitz_constant(A, xp=xp)
            )

    alpha0 = xp.zeros(S.shape, dtype=A.dtype, device=device)
    alpha = solver(dual, alpha0, 0.0, C)

    # }}}

    W = A.T @ (S * alpha)
    return SupportVectorMachine(
        W=W,
        mu=mu,
        bias=bias,
        labels=xp.asarray(labels, dtype=dtype, device=device),
    )


# }}}
