"""Shared plotting utilities for this repo: a common color palette and a
loader for the shared Matplotlib style (``plotting.mplstyle``, in this same
directory), so figures across scripts stay visually consistent.
"""
import os

import matplotlib.pyplot as plt

_STYLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plotting.mplstyle")


def get_palette() -> dict[str, str]:
    """Return the shared color palette used across this repo's figures.

    Returns
    -------
    dict of str to str
        Maps color names (e.g. ``"green"``, ``"light_blue"``) to hex codes.
        Each hue has a base, ``light_``, and ``pale_`` variant (lightest to
        darkest: pale < light < base), plus a few standalone ``dark_``
        variants and ``light_grey``/``dark_grey``.
    """
    return {'green': '#7AA974', 'light_green': '#BFD598',
            'pale_green': '#DCECCB', 'yellow': '#EAC264',
            'light_yellow': '#F3DAA9', 'pale_yellow': '#FFEDCE',
            'blue': '#738FC1', 'light_blue': '#A9BFE3',
            'pale_blue': '#C9D7EE', 'red': '#D56C55', 'light_red': '#E8B19D',
            'pale_red': '#F1D4C9', 'purple': '#AB85AC',
            'light_purple': '#D4C2D9', 'dark_green': '#7E9D90', 'dark_brown': '#905426',
            'dark_blue': '#535D87', 'dark_grey': '#363737', 'light_grey': '#D3D3D3', 'dark_purple': '#887191'}


def set_plotting_style(style_path: str | None = None) -> None:
    """Apply the shared Matplotlib style to all subsequent plots.

    Parameters
    ----------
    style_path : str, optional
        Path to a Matplotlib style file. Defaults to ``plotting.mplstyle``
        in this same directory.
    """
    plt.style.use(style_path or _STYLE_PATH)
