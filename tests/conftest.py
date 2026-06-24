# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import array_api_compat
import pytest

if TYPE_CHECKING:
    from _pytest.fixtures import SubRequest


# {{{ xp


xp_known_backends = [
    "numpy",
    "jax",
    "torch",
]


@pytest.fixture(scope="session", params=xp_known_backends)
def xp(request: SubRequest) -> Any:
    if request.param == "numpy":
        np = pytest.importorskip("numpy")

        dummy = np.empty(0)
    elif request.param == "jax":
        jnp = pytest.importorskip("jax.numpy")

        from nneuroutil.flax_extras import set_jax_config

        set_jax_config()
        dummy = jnp.empty(0)
    elif request.param == "torch":
        torch = pytest.importorskip("torch")

        torch.set_float32_matmul_precision("high")
        dummy = torch.empty(0)
    else:
        raise ValueError(f"unknown backend: {request.param!r}")

    return array_api_compat.array_namespace(dummy)


# }}}
