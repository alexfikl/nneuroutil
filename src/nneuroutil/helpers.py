# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import array_api_compat
import numpy as np

from nneuroutil.typing import Array1D, DataclassInstanceT

if TYPE_CHECKING:
    from types import TracebackType

# {{{ environment


# fmt: off
BOOLEAN_STATES = {
    1: True, "1": True, "yes": True, "true": True, "on": True, "y": True,
    0: False, "0": False, "no": False, "false": False, "off": False, "n": False,
}
# fmt: on


def get_environ_boolean(name: str) -> bool:
    value = os.environ.get(name)
    return BOOLEAN_STATES.get(value.lower(), False) if value else False


def on_ci() -> bool:
    """
    :returns: *True* if the current running system is recognized as a CI.
    """
    return (
        # NOTE: seems to be some new standard for CI?
        get_environ_boolean("CI")
        # NOTE: GitHub CI
        or os.environ.get("GITHUB_ACTIONS") is not None
        # NOTE: Gitlab CI
        or os.environ.get("CI_PROJECT_DIR") is not None
    )


def enable_test_plotting() -> bool:
    """Check if plotting is enabled.

    This is controlled by the ``NNEUROUTIL_ENABLE_TEST_PLOTTING`` environment
    variable. The name can change, so use this helper function instead, if possible.
    """

    return get_environ_boolean("NNEUROUTIL_ENABLE_TEST_PLOTTING")


# }}}


# {{{ logging


def module_logger(
    module: str,
    level: int | str | None = None,
    *,
    root_name: str = "nneuroutil",
) -> logging.Logger:
    """Create a new logging for the module *module*.

    The logger is created using a :class:`rich.logging.RichHandler` for fancy
    highlighting. The ``NO_COLOR`` environment variable can be used to
    disable colors.

    :arg module: a name for the module to create a logger for.
    :arg level: if *None*, the default value is taken to from the
        ``NNEUROUTIL_LOGGING_LEVEL`` environment variable and falls back to the
        ``INFO`` level if it does not exist (see :mod:`logging`).
    """
    if level is None:
        level = os.environ.get("NNEUROUTIL_LOGGING_LEVEL", "INFO").upper()

    if isinstance(level, str):
        level = getattr(logging, level.upper())

    assert isinstance(level, int)

    # NOTE: insist on putting everything under the root
    path = pathlib.Path(module)
    if path.exists():
        module = f"{root_name}.{path.stem}".replace("-", "_")

    # set up the root logger
    root = logging.getLogger(root_name)
    root.propagate = False

    if not root.handlers:
        from rich.highlighter import NullHighlighter
        from rich.logging import RichHandler

        no_color = "NO_COLOR" in os.environ
        handler = RichHandler(
            level,
            show_time=True,
            omit_repeated_times=False,
            show_level=True,
            show_path=True,
            highlighter=NullHighlighter() if no_color else None,
            markup=True,
        )

        root.addHandler(handler)
        root.setLevel(level)

    _, *rest = module.split(".", maxsplit=1)
    return root.getChild(rest[0]) if rest else root


log = module_logger(__name__)

# }}}


# {{{ TicTocTimer


@dataclass
class TicTocTimer:
    """A simple timer that tries to copy MATLAB's ``tic`` and ``toc`` functions.

    .. code:: python

        timer = TicTocTimer()
        timer.tic()

        # ... do some work ...

        elapsed = timer.toc()
        print(timer)

    Note that, unlike MATLAB's function, this class also remembers previous
    calls and gathers statistics. These can be shown with :meth:`stats`.
    """

    t_wall_start: float = field(default=0.0, init=False)
    t_wall: float = field(default=0.0, init=False)

    n_calls: int = field(default=0, init=False)
    t_total: float = field(default=0.0, init=False)
    t_avg: float = field(default=0.0, init=False)
    t_sqr: float = field(default=0.0, init=False)

    def tic(self) -> None:
        """Start the timer."""
        self.t_wall = 0.0
        self.t_wall_start = time.perf_counter()

    def toc(self) -> float:
        """Stop the timer and update internal statistics."""
        self.t_wall = time.perf_counter() - self.t_wall_start

        # statistics
        self.n_calls += 1
        self.t_total += self.t_wall

        delta0 = self.t_wall - self.t_avg
        self.t_avg += delta0 / self.n_calls
        delta1 = self.t_wall - self.t_avg
        self.t_sqr += delta0 * delta1

        return self.t_wall

    def __str__(self) -> str:
        # NOTE: this matches how MATLAB shows the time from `toc`.
        return f"Elapsed time is {self.t_wall:.5f} seconds"

    def stats(self) -> str:
        """Aggregate statistics across multiple calls to :meth:`toc`."""
        import math

        # NOTE: n_calls == 0 => toc was not called yet, so stddev is zero
        #       n_calls == 1 => only one call to toc, so the stddev is zero
        t_std = math.sqrt(self.t_sqr / (self.n_calls - 1)) if self.n_calls > 1 else 0.0

        return f"avg {self.t_avg:.3f}s ± {t_std:.3f}s"

    def short(self) -> str:
        """A shorter string for the last :meth:`tic`-:meth:`toc` cycle."""
        return f"wall {self.t_wall:.5f}s"


@contextmanager
def tictoc(name: str = "timing") -> Iterator[None]:
    tt = TicTocTimer()
    tt.tic()

    try:
        yield
    finally:
        tt.toc()
        log.info("%s: %s", name.upper(), tt)


# }}}


# {{{ BlockTimer


@dataclass
class BlockTimer:
    """A context manager for timing blocks of code.

    .. code:: python

        with BlockTimer("my-code-block") as bt:
            # ... do some work ...

        print(bt)
    """

    name: str = "block"
    """An identifier used to differentiate the timer."""

    t_wall_start: float = field(init=False)
    t_wall: float = field(init=False)
    """Total wall time (set after ``__exit__``), obtained from
    :func:`time.perf_counter`.
    """

    t_proc_start: float = field(init=False)
    t_proc: float = field(init=False)
    """Total process time (set after ``__exit__``), obtained from
    :func:`time.process_time`.
    """

    @property
    def t_cpu(self) -> float:
        """Total CPU time, obtained from ``t_proc / t_wall``."""
        return self.t_proc / self.t_wall

    def __enter__(self) -> BlockTimer:
        self.t_wall = self.t_proc = 0.0
        self.t_wall_start = time.perf_counter()
        self.t_proc_start = time.process_time()

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.t_wall = time.perf_counter() - self.t_wall_start
        self.t_proc = time.process_time() - self.t_proc_start

    def __str__(self) -> str:
        import datetime

        t_wall = datetime.timedelta(seconds=round(self.t_wall))
        return f"{self.name}: {t_wall} wall, {self.t_cpu:.3f}x cpu"

    def pretty(self) -> str:
        # NOTE: this matches how MATLAB shows the time from `toc`.
        return f"[{self.name}] Elapsed time is {self.t_wall:.5f} seconds."


# }}}


# {{{ slugify


def slugify(stem: str, separator: str = "_") -> str:
    """
    :returns: an ASCII slug representing *stem*, with all the unicode cleaned up
        and all non-standard separators replaced.
    """
    import re
    import unicodedata

    stem = unicodedata.normalize("NFKD", stem)
    stem = stem.encode("ascii", "ignore").decode().lower()
    stem = re.sub(r"[^a-z0-9]+", separator, stem)
    stem = re.sub(rf"[{separator}]+", separator, stem.strip(separator))

    return stem


# }}}


# {{{ FuzzyChoiceAction


def _fuzzy_choice_matcher(options: Sequence[str]) -> Callable[[str], str]:
    def match(value: str) -> str:
        matches = [o for o in options if o.startswith(value)]
        if not matches:
            raise argparse.ArgumentTypeError(
                f"invalid choice {value!r} (choose from {options})"
            )

        return matches[0]

    return match


class FuzzyChoiceAction(argparse.Action):
    """A custom action for :mod:`argparse` that handles suffix choices.

    For example, when using ``("cpu", "cuda:0", "cuda:1")`` as choices and passing
    ``--device cuda``, the action will select the first match.
    """

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        choices: Iterable[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if choices is None:
            raise ValueError(f"'choices' not provided for {type(self).__name__!r}")

        choices = list(choices)
        metavar = ",".join(choices)

        kwargs.setdefault("metavar", f"{{{metavar}}}")
        kwargs.setdefault("type", _fuzzy_choice_matcher(choices))
        super().__init__(option_strings, dest, choices=None, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[str] | None,
        option_string: str | None = None,
    ) -> None:
        setattr(namespace, self.dest, values)


# }}}


# {{{ jax


_PENDING_JAX_REGISTER_DATACLASS = []


def set_jax_config() -> None:
    """Set up any :mod:`jax` related functionality.

    This should be called after JAX is imported. It will mainly enable ``float64``
    mode, register any required PyTrees, etc.
    """
    if "jax" not in sys.modules:
        return

    import jax

    jax.config.update("jax_enable_x64", val=True)
    while _PENDING_JAX_REGISTER_DATACLASS:
        jax.tree_util.register_dataclass(_PENDING_JAX_REGISTER_DATACLASS.pop())


def register_dataclass(cls: type[DataclassInstanceT]) -> type[DataclassInstanceT]:
    if "jax" in sys.modules:
        import jax

        jax.tree_util.register_dataclass(cls)
    else:
        _PENDING_JAX_REGISTER_DATACLASS.append(cls)

    # TODO: Anyone else need to register dataclasses?

    return cls


# }}}

# {{{ match_spectrum


def spectrum_error(eig_a: Array1D, eig_b: Array1D, *, xp: Any | None = None) -> float:
    if xp is None:
        xp = array_api_compat.array_namespace(eig_a, eig_b)

    from scipy.optimize import linear_sum_assignment

    M = xp.abs(eig_a[:, None] - eig_b[None, :])
    row, col = linear_sum_assignment(M)

    return xp.linalg.norm(eig_a[row] - eig_b[col]) / xp.linalg.norm(eig_b)


# }}}


# {{{ to_real + to_complex


_REAL_TO_COMPLEX_DTYPE = {
    np.dtype(np.float32): np.dtype(np.complex64),
    np.dtype(np.float64): np.dtype(np.complex128),
    np.dtype(np.longdouble): np.dtype(np.clongdouble),
}
_COMPLEX_TO_REAL_DTYPE = {v: k for k, v in _REAL_TO_COMPLEX_DTYPE.items()}


def to_real(dtype: Any) -> np.dtype[np.floating[Any]]:
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return dtype

    try:
        return _COMPLEX_TO_REAL_DTYPE[dtype]
    except KeyError:
        raise TypeError(f"no real counterpart for dtype {dtype!r}") from None


def to_complex(dtype: Any) -> np.dtype[np.complexfloating[Any]]:
    dtype = np.dtype(dtype)
    if dtype.kind == "c":
        return dtype  # ty: ignore[invalid-return-type]

    try:
        return _REAL_TO_COMPLEX_DTYPE[dtype]  # ty: ignore[invalid-return-type]
    except KeyError:
        raise TypeError(f"no complex counterpart for dtype {dtype!r}") from None


# }}}
