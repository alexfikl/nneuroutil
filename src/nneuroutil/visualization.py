# SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from nneuroutil.helpers import BOOLEAN_STATES, module_logger, on_ci
from nneuroutil.typing import PathLike

if TYPE_CHECKING:
    import matplotlib.pyplot as mp

log = module_logger(__name__)

# {{{ load_scienceplots_styles


class Styles:
    def __init__(self, styles: dict[str, str]) -> None:
        self.styles = styles

    def get(self, styles: str | Sequence[str]) -> tuple[str, ...]:
        if isinstance(styles, str):
            styles = [styles]

        # NOTE: this silently does nothing if the styles are incorrectly named or
        # if scienceplots does not actually exist. This should be fine..
        return tuple(self.styles[style] for style in styles if style in self.styles)


def load_scienceplots_styles() -> Styles:
    from importlib.util import find_spec

    spec = find_spec("scienceplots")
    if spec is None:
        return Styles({})

    if spec.submodule_search_locations is None:
        return Styles({})

    styles_root = pathlib.Path(spec.submodule_search_locations[0]) / "styles"
    return Styles({f.stem: str(f) for f in styles_root.rglob("*") if f.is_file()})


# }}}


# {{{ set_plotting_defaults


def _check_usetex(*, s: bool) -> bool:
    try:
        import matplotlib
    except ImportError:
        return False

    try:
        return bool(matplotlib.checkdep_usetex(s))  # ty: ignore[unresolved-attribute]
    except AttributeError:
        # NOTE: simplified version from matplotlib
        # https://github.com/matplotlib/matplotlib/blob/ec85e725b4b117d2729c9c4f720f31cf8739211f/lib/matplotlib/__init__.py#L439=L456

        import shutil

        if not shutil.which("tex"):
            return False

        if not shutil.which("dvipng"):
            return False

        if not shutil.which("gs"):  # ruff:ignore[needless-bool]
            return False

        return True


def set_plotting_defaults(
    mplstyle: pathlib.Path | str | None = None,
    *,
    use_tex: bool | None = None,
    dark: bool | None = None,
    savefig_format: str | None = None,
) -> None:
    """Set custom :mod:`matplotlib` parameters.

    These are mainly used in the tests and examples to provide a uniform style
    to the results using `SciencePlots <https://github.com/garrettj403/SciencePlots>`__.
    For other applications, it is recommended to use local settings (e.g. in
    `matplotlibrc`).

    :arg mplstyle: a path to a :mod:`matplotlib` style configuration file. If
        not provided, a default set is used.
    :arg use_tex: if *True*, LaTeX labels are enabled. By default, this checks
        if LaTeX is available on the system and only enables it if possible.
    :arg dark: if *True*, a dark default theme is selected instead of the
        default light one. If *None*, this takes its values from the ``NNEUROUTIL_DARK``
        boolean environment variable.
    :arg savefig_format: the format used when saving figures. By default, this
        uses the ``NNEUROUTIL_SAVEFIG`` environment variable and falls back to
        the :mod:`matplotlib` parameter ``savefig.format``.
    """
    if on_ci():
        return

    try:
        import matplotlib.pyplot as mp
    except ImportError:
        return

    # start off by resetting the defaults
    import matplotlib as mpl

    mpl.rcParams.update(mpl.rcParamsDefault)

    import os

    if use_tex is None:
        use_tex = "GITHUB_REPOSITORY" not in os.environ and _check_usetex(s=True)

    if not use_tex:
        log.warning("'use_tex' is disabled on this system.")

    if dark is None:
        tmp = os.environ.get("NNEUROUTIL_DARK", "off").lower()
        dark = BOOLEAN_STATES.get(tmp, False)

    if savefig_format is None:
        savefig_format = os.environ.get(
            "NNEUROUTIL_SAVEFIG", mp.rcParams["savefig.format"]
        ).lower()

    # NOTE: preserve existing colors (the ones in "science" are ugly)
    prop_cycle = mp.rcParams["axes.prop_cycle"]

    # try to load scienceplots styles
    scienceplots = load_scienceplots_styles()
    mp.style.use(scienceplots.get(["science", "ieee"]))

    # NOTE: the 'petroff10' style is available for version >= 3.10.0 and changes
    # the 'prop_cycle' to the 10 colors that are more accessible
    if "petroff10" in mp.style.available:
        mp.style.use("petroff10")
        prop_cycle = mp.rcParams["axes.prop_cycle"]

    if mplstyle is not None:
        mp.style.use(mplstyle)

        mp.rc("savefig", format=savefig_format)
        mp.rc("text", usetex=use_tex)
    else:
        mp.rc("figure", figsize=(8, 8), dpi=300)
        mp.rc("figure.constrained_layout", use=True)
        mp.rc("savefig", format=savefig_format)
        mp.rc("text", usetex=use_tex)
        mp.rc("legend", fontsize=20, frameon=True, fancybox=False, edgecolor="black")
        mp.rc("lines", linewidth=2, markersize=10)
        mp.rc("axes", labelsize=28, titlesize=28, grid=True, prop_cycle=prop_cycle)
        mp.rc("axes.grid", axis="both", which="both")
        mp.rc("xtick", labelsize=20, direction="in")
        mp.rc("ytick", labelsize=20, direction="in")
        mp.rc("xtick.major", size=6.5, width=1.5)
        mp.rc("ytick.major", size=6.5, width=1.5)
        mp.rc("xtick.minor", size=4.0)
        mp.rc("ytick.minor", size=4.0)

        if dark:
            # NOTE: this is the black color used by the sphinx-book theme
            black = "111111"
            gray = "28313D"
            mp.rc("text", color="white")
            mp.rc("axes", labelcolor="white", facecolor=gray, edgecolor="white")
            mp.rc("xtick", color="white")
            mp.rc("ytick", color="white")
            mp.rc("figure", facecolor=black, edgecolor=black)
            mp.rc("savefig", facecolor=black, edgecolor=black)


# }}}


# {{{ figure context manager


def with_savefig_suffix(filename: PathLike) -> pathlib.Path:
    """Adds the default :mod:`matplotlib` ``savefig.format`` extension to the path."""
    filename = pathlib.Path(filename)

    ext = mp.rcParams["savefig.format"]
    return filename.with_suffix(f".{ext}").resolve()


@contextmanager
def figure(
    filename: PathLike | None = None,
    nrows: int = 1,
    ncols: int = 1,
    *,
    pane_fill: bool = False,
    projection: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> Iterator[Any]:
    """A small wrapper context manager around :class:`matplotlib.figure.Figure`.

    :arg nrows: number of rows of subplots.
    :arg ncols: number of columns of subplots.
    :arg projection: a projection for all the axes in this figure, see
        :mod:`matplotlib.projections`.
    :arg figsize: the size of the resulting figure, set to
        ``(L * ncols, L * nrows)`` by default.
    :arg kwargs: Additional arguments passed to :func:`savefig`.
    :returns: the :class:`~matplotlib.figure.Figure` that was constructed. On exit
        from the context manager, the figure is saved to *filename* and closed.
    """
    import matplotlib.pyplot as mp

    fig = mp.figure()
    for i in range(nrows * ncols):
        fig.add_subplot(nrows, ncols, i + 1, projection=projection)

    # FIXME: get size of one figure
    if figsize is None:
        width, height = mp.rcParams["figure.figsize"]
        figsize = (width * ncols, height * nrows)
    fig.set_size_inches(*figsize)

    if projection == "3d":
        from mpl_toolkits.mplot3d.axes3d import Axes3D

        for ax in fig.axes:
            assert isinstance(ax, Axes3D)
            ax.xaxis.pane.fill = pane_fill  # ty: ignore[unresolved-attribute]
            ax.yaxis.pane.fill = pane_fill  # ty: ignore[unresolved-attribute]
            ax.zaxis.pane.fill = pane_fill

    try:
        yield fig
    finally:
        if projection == "3d":
            for ax in fig.axes:
                assert isinstance(ax, Axes3D)
                ax.set_box_aspect((4, 4, 4), zoom=1.1)

        if filename is not None:
            savefig(fig, filename, **kwargs)
        else:
            mp.show(block=True)

        fig.clear()
        mp.close(fig)


# }}}


# {{{ savefig wrapper


def savefig(
    fig: Any,
    filename: PathLike,
    *,
    bbox_inches: str = "tight",
    pad_inches: float = 0.075,
    normalize: bool = False,
    facecolor: str = "white",
    transparent: bool = False,
    overwrite: bool = True,
    **kwargs: Any,
) -> None:
    """A wrapper around :meth:`~matplotlib.figure.Figure.savefig`.

    :arg filename: a file name where to save the figure. If the file name does
        not have an extension, the default format from ``savefig.format`` is
        used.
    :arg normalize: if *True*, use :func:`slugify` to normalize the file name.
        Note that this will slugify any extensions as well and replace them
        with the default extension. If a certain extension is desired, it should
        probably be set in ``savefig.format``.
    :arg overwrite: if *True*, any existing files are overwritten.
    :arg kwargs: renaming arguments are passed directly to ``savefig``.
    """
    import matplotlib.pyplot as mp

    ext = mp.rcParams["savefig.format"]
    filename = pathlib.Path(filename)

    if normalize:
        from nneuroutil.helpers import slugify

        # NOTE: slugify(name) will clubber any prefixes, so we special-case a
        # few of them here to help out the caller
        if filename.suffix in {".png", ".jpg", ".jpeg", ".pdf", ".eps", ".tiff"}:
            filename = filename.with_stem(slugify(filename.stem))
        else:
            filename = filename.with_name(slugify(filename.name)).with_suffix(f".{ext}")

    if not filename.suffix:
        filename = filename.with_suffix(f".{ext}").resolve()

    if not overwrite and filename.exists():
        raise FileExistsError(f"output file '{filename}' already exists")

    bbox_extra_artists = []
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            bbox_extra_artists.append(legend)

    log.info("Saving '%s'", filename)
    fig.savefig(
        filename,
        bbox_extra_artists=tuple(bbox_extra_artists),
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor=facecolor,
        transparent=transparent,
        **kwargs,
    )


# }}}


# {{{ make_colorbar_axis


def make_colorbar_axes(
    ax: mp.Axes,
    *,
    position: str = "right",
    size: str = "5%",
    pad: float = 0.1,
) -> mp.Axes:
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size=size, pad=pad)

    return cax


# }}}
