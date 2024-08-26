"""Utility functions for working with colors and color palettes."""

from collections.abc import Mapping

import colour
import numpy as np
from colorcet import glasbey
from matplotlib.colors import to_rgba, to_rgb, ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from seaborn import color_palette

colour.set_domain_range_scale("1")


def get_spaced_colors(
    n, light=0.5, sat=0.35, space="Oklab", base_space="sRGB", norm_method="clip"
):
    """
    Generate evenly spaced, maximally different colors in a specified color space.

    Parameters
    ----------
    n : int
        The number of colors to generate.
    light : float, optional
        The lightness value of the colors. Default is 0.5.
    sat : float, optional
        The saturation value of the colors. Default is 0.35.
    space : str, optional
        The color space to generate the colors in. Default is "Oklab".
    base_space : str, optional
        The base color space to convert colors into. Default is "sRGB".
    norm_method : str, optional
        The normalization method to use. Default is "clip".

    Returns
    -------
    numpy.ndarray
        An array of shape (n, 3) containing the generated colors.

    Raises
    ------
    ValueError
        If an unknown normalization method is provided.

    """
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    polar = np.stack(
        [light * np.ones(n), sat * np.cos(theta), sat * np.sin(theta)], axis=1
    )
    out = colour.convert(polar, space, base_space)
    if norm_method == "clip":
        out = np.minimum(1, np.maximum(0, out))
    elif norm_method == "scale":
        out = colour.algebra.normalise_maximum(out, axis=1)
    elif norm_method == "none" or norm_method is None:
        pass
    else:
        raise ValueError(f"Unknown norm_method: {norm_method}")
    return out


def get_color_array(n_colors, colors=None):
    """Generate an array of RGB colors, picking defaults if necessary.

    Parameters
    ----------
    n_colors
        number to generate
    colors, optional
        colors to use, either as a list of RGB tuples or color strings

    Returns
    -------
        np.array of RGB colors, shape (n_colors, 3)
    """
    if colors is None:
        colors = get_spaced_colors(n_colors)
    elif isinstance(colors[0], str):
        colors = np.array([to_rgb(x) for x in colors])
    else:
        colors = np.array(colors)
        assert colors.shape == (
            n_colors,
            3,
        ), "colors must be a list of RGB tuples or color strings"
    return colors


def combine_scaled_colors(colors, coeffs, base_space="sRGB"):
    """
    Combine scaled colors using coefficients.

    Parameters
    ----------
    colors : array_like
        The input colors to be combined. It should be an array of shape (N, 3),
        where N is the number of colors and 3 represents the RGB values.
    coeffs : array_like
        The coefficients used to combine the colors. It should be an array of shape (M, N),
        where M is the number of coefficients and N is the number of colors.
    base_space : str, optional
        The color space of the input colors. Default is "sRGB".

    Returns
    -------
    out : ndarray
        The combined colors. It is an array of shape (M, 3), where M is the number of coefficients
        and 3 represents the RGB values.

    Notes
    -----
    This function converts the input colors from the base color space to the CIE xyY color space,
    combines the colors using the coefficients, and then converts the combined colors back to the
    base color space. The resulting colors are clipped to the range [0, 1].

    """
    interp_space = "CIE xyY"
    origin = np.array([[1 / 3, 1 / 3, 0]])
    colors = colour.convert(colors, base_space, interp_space)
    combination = np.dot(coeffs, colors - origin) + origin
    out = colour.convert(combination, interp_space, base_space)
    out = np.minimum(1, np.maximum(0, out))
    return out


def mappable_for_colorbar(c, vmin=0, vmax=1):
    sm = ScalarMappable(
        cmap=ListedColormap(c),
        norm=Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    return sm


def generate_palette(categories, palette=glasbey, hue_label=None, **items):
    """Generate a color palette dict for a given list of categories.

    Parameters
    ----------
    categories : list of str
        List of category names
    palette : dict, mappable, list of colors, or string
        Colors to use to create palette, or string to pass to sns.color_palette
        if None, use default taxonomy palette
    hue_label : str, {'class', 'subclass', 'supertype', 'cluster'}
        Taxonomy level to generate a palette for if palette is None
    **items : additional category=color pairs to be combined with palette

    Returns
    -------
    palette : dict of (str, RGB tuples)
    """
    if isinstance(palette, Mapping):
        palette = {x: palette[x] for x in categories if x in palette}
    else:
        # generate a palette from a list of colors or palette name string
        sns_palette = color_palette(palette, n_colors=len(categories))
        palette = dict(zip(categories, sns_palette))
    palette.update(**items)
    return palette


def palette_to_rgba_lookup(palette, index):
    """Convert a color palette dict to an RGBA lookup table.

    Parameters
    ----------
    palette : dict of (str, )
        Dictionary of CCF region names and their corresponding colors
    index : pd.Series
        Series of CCF region names, with the index as the CCF region IDs

    Returns
    -------
    rgba_lookup : np.ndarray
        2D array of RGBA color values, where the row indices correspond to the
        the CCF region IDs and the column indices are RGBA values
    """
    # rgba_lookup = index.map(lambda x: to_rgb(palette[x]))
    # rgba_lookup = rgba_lookup.reindex(range(max_val), fill_value=0)
    max_val = np.max(index.index)
    rgba_lookup = np.zeros((max_val+1, 4))
    # fill only values in index and also in palette
    # rest remain transparent (alpha=0)
    for i in index.index:
        name = index[i]
        if name in palette:
            rgba_lookup[i, :] = to_rgba(palette[name])
    rgba_lookup[0, :] = [1, 1, 1, 0]
    return rgba_lookup


__all__ = [
    get_spaced_colors.__name__,
    combine_scaled_colors.__name__,
    generate_palette.__name__,
    palette_to_rgba_lookup.__name__,
    get_color_array.__name__,
    mappable_for_colorbar.__name__,
]
