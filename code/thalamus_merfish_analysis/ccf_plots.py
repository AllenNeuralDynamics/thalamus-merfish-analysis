from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from colorcet import glasbey
from matplotlib.colors import to_rgba

from . import abc_load as abc
from . import ccf_images as cci


# TODO: make thalamus-specific versions with thalamus names default nuclei
# ------------------------- Multi-Section Plotting ------------------------- #
def plot_ccf_overlay(
    obs,
    ccf_images,
    sections=None,
    point_hue="CCF_acronym",
    section_col="section",
    x_col="cirro_x",
    y_col="cirro_y",
    point_palette=None,  # controls foreground cells
    categorical=True,
    min_group_count=10,
    min_section_count=20,
    bg_cells=None,  # controls background cells
    ccf_names=None,
    ccf_highlight=[],
    ccf_level="substructure",  # controls CCF regions
    boundary_img=None,
    face_palette=None,
    edge_color="grey",
    s=2,
    custom_xy_lims=[],  # xy coords
    axes=False,
    legend="cells",  # plot formatting
):
    """
    Parameters
    ----------
    obs : pd.DataFrame
        DataFrame of cells to display
    ccf_images : np.ndarray
        3D array of CCF parcellation regions
    sections : list of numbers, optional
        Section(s) to display. Must be a list even for single section.
        If None, all sections that contain cells in obs are displayed
    ccf_names : list of str, optional
        List of CCF region names to display
    point_hue : str
        Column name in obs to color cells by
    section_col : str, {'z_section', 'z_reconstructed', 'z_realigned'}
        Column name in obs for the section numbers in 'sections'; must be a col
        that allows for indexing into ccf_images
    x_col, y_col : str
        Column names in obs for the x and y coordinates of cells
    point_palette : dict, optional
        Dictionary of point_hue categories and colors
    categorical : bool, default=True
        Whether point_hue is a categorical variable
    s : int, default=2
        Size of foreground cell scatter points (background cells set to 0.8*s)
    min_group_count : int
        Minimum number of cells in a group to be displayed
    bg_cells : pd.DataFrame, optional
        DataFrame of background cells to display
    ccf_highlight : list of str, optional
        List of CCF region names to highlight with a darker outline for ccf_names
    ccf_level : str, {'substructure', 'structure'}, default='substructure'
        Level of CCF to be displayed
    boundary_img : np.ndarray, optional
        3D array of CCF parcellation boundaries; if None, calculated on the fly
    face_palette : {None, dict, 'glasbey'}, default=None
        Sets face color of CCF region shapes in plot_ccf_section(), see that
        function's docstring for more details
    edge_color : str, default='grey'
        Sets outline/boundary color of CCF region shapes in plot_ccf_section()
    min_section_count : int
        Minimum number of cells in a section to be displayed
    custom_xy_lims : list of float, [xmin, xmax, ymin, ymax]
        Custom x and y limits for the plot
    axes : bool
        Whether to display the axes and spines
    legend : str, {'ccf', 'cells', 'both', None}
        Whether to display a legend for the CCF region shapes, the cell types,
        both, or neither
    """
    # Set variables not specified by user
    if ccf_names is None:
        ccf_names = abc.get_ccf_names(level=ccf_level)
    if sections is None:
        # TODO: intersect with sections for ccf_names if provided?
        sections = (
            obs[section_col].value_counts().loc[lambda x: x > min_section_count].index
        )
    obs = obs[obs[section_col].isin(sections)]

    if categorical:
        # Clean up point hue column: must happen after filtering by section
        obs = preprocess_categorical_plot(
            obs,
            point_hue,
            section_col=section_col,
            min_group_count=min_group_count,
        )
        # Set color palette for cell scatter points
        point_palette = _generate_palette(
            obs[point_hue].unique().tolist(), 
            hue_label=point_hue,
            palette=point_palette,
            other=OTHER_CATEGORY_COLOR,
        )

    # add background cells with NA values
    if bg_cells is not None:
        obs = _integrate_background_cells(obs, point_hue, bg_cells)

    # Display each section as a separate plot
    figs = [
        plot_section_overlay(
            obs,
            ccf_images,
            section,
            boundary_img=boundary_img,
            ccf_names=ccf_names,
            section_col=section_col,
            x_col=x_col,
            y_col=y_col,
            point_hue=point_hue,
            point_palette=point_palette,
            face_palette=face_palette,
            edge_color=edge_color,
            ccf_highlight=ccf_highlight,
            ccf_level=ccf_level,
            custom_xy_lims=custom_xy_lims,
            axes=axes,
            legend=legend,
        )
        for section in sections
    ]
    return figs


def plot_section_overlay(
    obs,
    ccf_images,
    section,
    section_col="section",
    x_col="cirro_x",
    y_col="cirro_y",
    point_hue="CCF_acronym",
    point_palette=None,  
    face_palette=None,
    edge_color="grey",
    ccf_names=None,
    ccf_highlight=[],
    ccf_level="substructure",  
    custom_xy_lims=None,  
    axes=False,
    legend="cells",  # plot formatting
    show=True,
    ax=None,
    colorbar=False,
    scatter_args={},
    cb_args={},
):
    ax, fig = _get_figure_handles(ax)
    secdata = obs.loc[lambda df: (df[section_col] == section)]
    if custom_xy_lims is not None:
        # TODO: apply at dataset level instead?
        secdata = _filter_by_xy_lims(secdata, x_col, y_col, custom_xy_lims)

    # display CCF shapes
    plot_ccf_section(
        ccf_images,
        section,
        boundary_img=boundary_img,
        ccf_region_names=ccf_names,
        highlight_region_names=ccf_highlight,
        ccf_level=ccf_level,
        face_palette=face_palette,
        edge_color=edge_color,
        legend=(legend == "ccf"),
        ax=ax,
    )

    _plot_cells_scatter(
        secdata,
        x_col=x_col,
        y_col=y_col,
        point_hue=point_hue,
        point_palette=point_palette,
        legend=(legend in ["cells", "both"]),
        custom_xy_lims=custom_xy_lims,
        ax=ax,
        **scatter_args,
    )
    # plot formatting
    if legend is not None:
        # cell type names require more horizontal space
        # TODO: detect this from label text
        ncols = 4 if (legend == "ccf") else 2
        plt.legend(
            ncols=ncols, loc="upper center", bbox_to_anchor=(0.5, 0), frameon=False
        )
    if colorbar:
        img = ax.imshow(np.array([[0, 1]]), cmap="viridis")
        img.set_visible(False)
        plt.colorbar(img, orientation="vertical", **cb_args)
        
    _format_image_axes(ax=ax, axes=axes, custom_xy_lims=custom_xy_lims)
    ax.set_title("z=" + str(section) + "\n" + point_hue)
    if show:
        plt.show()
    return fig


def _plot_cells_scatter(
    secdata,
    x_col,
    y_col,
    point_hue,
    point_palette=None,
    s=2,
    legend=True,
    ax=None,
    **kwargs,
):
    sns.scatterplot(
        secdata,
        x=x_col,
        y=y_col,
        hue=point_hue,
        s=s,
        palette=point_palette,
        linewidth=0,
        legend=legend,
        ax=ax,
        **kwargs,
    )
    bg_s = s * 0.8 if (s <= 2) else 2
    # TODO: add background cells to legend?
    sns.scatterplot(
        secdata.loc[secdata[point_hue].isna()],
        x=x_col,
        y=y_col,
        c=BACKGROUND_POINT_COLOR,
        s=bg_s,
        linewidth=0,
    )


def plot_expression_ccf(
    adata,
    gene,
    ccf_images,
    sections=None,
    nuclei=None,
    highlight=[],
    s=0.5,
    cmap="Blues",
    show_outline=False,
    axes=False,
    edge_color="lightgrey",
    section_col="section",
    x_col="cirro_x",
    y_col="cirro_y",
    boundary_img=None,
    custom_xy_lims=[],
    cb_vmin_vmax=[None, None],
    **kwargs,
):
    # TODO: allow sections arg to be single section not list?
    if sections is None:
        sections = adata.obs[section_col].unique()
    # Plot
    figs = []
    for section in sections:
        fig = plot_expression_ccf_section(
            adata,
            gene,
            ccf_images,
            section,
            nuclei=nuclei,
            highlight=highlight,
            s=s,
            cmap=cmap,
            show_outline=show_outline,
            axes=axes,
            edge_color=edge_color,
            section_col=section_col,
            x_col=x_col,
            y_col=y_col,
            boundary_img=boundary_img,
            custom_xy_lims=custom_xy_lims,
            cb_vmin_vmax=cb_vmin_vmax,
        )
        figs.append(fig)
    return figs



def _get_counts_label(adata, gene):
    # if adata from load_adata(), counts_transform is recorded in .uns
    if 'counts_transform' in adata.uns:
        label = f'gene counts ({adata.uns['counts_transform']})'
    # if we don't have .uns['counts_transform'], check if we have raw counts or not
    else:
        if all(i.is_integer() for i in adata[gene]):  # no [] around loop == stops at 1st non-integer encounter
            label = 'gene counts (raw)'
        else:
            label = 'gene counts (unknown transform)'
    return label

# TODO: is single-section function needed?
def plot_expression_ccf_section(
    adata,
    gene,
    ccf_images,
    section,
    # TODO: rename the below to be consistent with other functions?
    nuclei=None,
    highlight=[],
    s=0.5,
    cmap="Blues",
    axes=False,
    edge_color="lightgrey",
    section_col="section",
    x_col="cirro_x",
    y_col="cirro_y",
    boundary_img=None,
    custom_xy_lims=[],
    cb_vmin_vmax=[None, None],
    label=None,
    colorbar=True,
    ax=None,
    **kwargs,
):
    fig, ax = _get_figure_handles(ax)
    obs = preprocess_gene_plot(adata, gene)
    scatter_args = dict(hue_norm=cb_vmin_vmax, **kwargs)
    if label is None:
        label = _get_counts_label(adata, gene)
    cb_args = dict(label=label, fraction=0.046, pad=0.01)
    
    plot_section_overlay(
        obs,
        ccf_images,
        section,
        boundary_img=boundary_img,
        section_col=section_col,
        x_col=x_col,
        y_col=y_col,
        point_hue=gene,
        point_palette=cmap,
        edge_color=edge_color,
        ccf_names=ccf_names,
        ccf_highlight=ccf_highlight,
        ccf_level=ccf_level,
        custom_xy_lims=custom_xy_lims,
        axes=axes,
        legend=None,
        colorbar=colorbar,
        scatter_args=scatter_args,
        cb_args=cb_args,
        ax=ax,
        s=s,
    )
    ax.set_title(gene)
    return fig


def plot_metrics_ccf(
    ccf_img,
    metric_series,
    sections,
    structure_index=None,
    cmap="viridis",
    cb_label="metric",
    vmin=None,
    vmax=None,
    axes=False,
):
    if structure_index is None:
        structure_index = abc.get_ccf_index()
    vmin = metric_series.min() if vmin is None else vmin
    vmax = metric_series.max() if vmax is None else vmax
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps.get_cmap(cmap)
    palette = metric_series.apply(lambda x: cmap(norm(x))).to_dict()

    figs = []
    for section_z in sections:
        print(section_z)
        fig, ax = plt.subplots(figsize=(8, 5))
        # hidden image just to generate colorbar
        img = ax.imshow(np.array([[vmin, vmax]]), cmap=cmap)
        img.set_visible(False)
        plt.colorbar(img, orientation="vertical", label=cb_label, shrink=0.75)

        plot_ccf_section(
            ccf_img,
            section_z,
            palette,
            structure_index=structure_index,
            legend=False,
            ax=ax,
        )
        _format_image_axes(ax=ax, axes=axes)
        figs.append(fig)
    return figs

def plot_ccf_section(
    ccf_img,
    section_z,
    ccf_region_names=None,
    highlight_region_names=None,
    face_palette=None,
    edge_color="grey",
    boundary_img=None,
    ccf_level="substructure",
    z_resolution=200e-3,
    legend=True,
    ax=None,
):
    """Display CCF parcellations for a single section from an
    image volume of region labels

    Parameters
    ----------
    ccf_img : np.ndarray
        3D array of CCF parcellations
    section_z : int
        Section number
    ccf_region_names : list of str, optional
        Subset of CCF regions to display
    highlight_region_names : list of str, optional
        Subset of CCF regions to highlight with darkened edges
    face_palette : {None, dict, 'glasbey'}
        Sets face color of CCF region shapes;
        None to have no face color, or a dictionary of CCF region names and
        colors, or 'glasbey' to generate a color palette from the
        colorcet.glasbey color map
    edge_color : str, default='grey'
        Sets outline/boundary color of all CCF region shapes; any valid
        matplotlib color
    boundary_img : np.ndarray, optional
        2D array of CCF parcellation boundaries; if None, calculated on the fly
    z_resolution : float, default=200e-3
        Resolution of the CCF in the z-dimension
    legend : bool, default=True
        Whether to display a legend for the CCF region shapes
    ax : matplotlib.axes.Axes, optional
        Existing Axes to plot on; if None, a new figure and axes are created
    """

    if isinstance(section_z, str):
        raise TypeError(
            "str type detected for section var, numeric value required to plot rasterized CCF volumes."
        )
    # subset to just this section
    index_z = int(np.rint(section_z / z_resolution))
    img = ccf_img[:, :, index_z].T
    boundary_img = boundary_img[:, :, index_z].T if boundary_img is not None else None

    # select CCF regions to plot
    # TODO: could allow names from other levels and translate all to specified level...
    structure_index = abc.get_ccf_index(level=ccf_level)
    section_region_names = [
        structure_index[i] for i in np.unique(img) if i in structure_index.index
    ]

    if ccf_region_names is not None:
        section_region_names = list(
            set(section_region_names).intersection(ccf_region_names)
        )

    face_palette = _generate_palette(section_region_names, palette=face_palette)

    edge_palette = {
        x: EDGE_COLOR_HIGHLIGHT if x in highlight_region_names else edge_color
        for x in section_region_names
    }

    plot_ccf_shapes(
        img,
        structure_index,
        boundary_img=boundary_img,
        face_palette=face_palette,
        edge_palette=edge_palette,
        ax=ax,
        legend=legend,
    )


def plot_ccf_shapes(
    imdata,
    index,
    boundary_img=None,
    face_palette=None,
    edge_palette=None,
    edge_width=1,
    alpha=1,
    ax=None,
    resolution=10e-3,
    legend=True,
):
    """Plot face & boundary for CCF region shapes specified

    Parameters
    ----------
    imdata : np.ndarray
        2D array of CCF parcellations
    index : pd.Series
        Series of CCF region names and their corresponding IDs
    boundary_img : np.ndarray, optional
        2D array of CCF parcellation boundaries; if None, calculated on the fly
    regions : list of str, optional
        List of CCF region names to display; if None, display all
    face_palette : dict, optional
        Dictionary of CCF region names and colors; if None, faces are not colored
    edge_palette : dict, optional
        Dictionary of CCF region names and colors; if None, edges are not colored
    edge_width : int, default=1
        Width of the CCF region shape outlines (used only if boundary_img is None)
    alpha : float, default=1
        Opacity of the CCF region shapes' face and edge colors
    ax : matplotlib.axes.Axes, optional
        Existing Axes to plot on; if None, a new figure and axes are created
    resolution : float, default=10e-3
        Resolution of the CCF in the image plane, used to set correct image extent
    legend : bool, default=True
        Whether to display a legend for the CCF region shapes
    """
    # TODO: use xarray for applying palette, plotting, storing boundaries
    extent = resolution * (np.array([0, imdata.shape[0], imdata.shape[1], 0]) - 0.5)
    imshow_args = dict(extent=extent, interpolation="none", alpha=alpha)

    # Plot face shapes of CCF regions
    if face_palette is not None:
        rgba_lookup = _palette_to_rgba_lookup(face_palette, index)
        im_regions = rgba_lookup[imdata, :]
        ax.imshow(im_regions, **imshow_args)

    # Plot boundaries of CCF regions
    if edge_palette is not None:
        if boundary_img is None:
            boundary_img = cci.label_erosion(
                imdata, edge_width, fill_val=0, return_edges=True
            )
        rgba_lookup = _palette_to_rgba_lookup(edge_palette, index)
        im_edges = rgba_lookup[boundary_img, :]
        ax.imshow(im_edges, **imshow_args)

    # generate "empty" matplotlib handles for legend
    if legend and (face_palette is not None):
        for name, color in face_palette.items():
            plt.plot([], marker="o", ls="", color=color, label=name)


# ------------------------- DataFrame Preprocessing ------------------------- #
def preprocess_gene_plot(adata, gene):
    obs = adata.obs.copy()
    obs[gene] = adata[gene]
    return obs


def preprocess_categorical_plot(
    obs,
    type_col,
    section_col="z_section",
    min_group_count=10,
    min_group_count_section=5,
):
    """Preprocess a DataFrame for plotting by filtering out sections with
    low counts and relabeling outlier cell types in each section.

    Parameters
    ----------
    obs : pd.DataFrame
        DataFrame to preprocess
    type_col : str
        Column name in data for the categorical variable to filter
    section_col : str
        Column name in data for the section numbers
    min_group_count : int
        Minimum number of cells in a group to be displayed
    min_section_count : int
        Minimum number of cells in a section to be displayed
    min_group_count_section : int
        Minimum number of cells in a group within a section to be displayed

    Returns
    -------
    obs : pd.DataFrame
        Preprocessed DataFrame (copy of original)
    """
    obs = obs.copy()
    obs = abc.label_outlier_celltypes(obs, type_col, min_group_count=min_group_count)
    # Min group count by section shouldn't be larger than overall min_group_count
    # Set to the minimum so user can set min_group_count=0 to see all groups
    min_group_count_section = min(min_group_count_section, min_group_count)
    obs = obs.groupby(section_col).apply(
        lambda x: abc.label_outlier_celltypes(
            x, type_col, min_group_count=min_group_count_section
        )
    )
    return obs


def _filter_by_xy_lims(obs, x_col, y_col, custom_xy_lims):
    """Filter a DataFrame by custom x and y limits.

    Need to explicitly filter the DataFrame. Can't rely only on set_xlim/ylim,
    because that only masks out the data points but still plots them, so they
    are present & taking up space in a savefig() pdf output (i.e. generates
    large files that are laggy in Illustrator due to having many indiv elements).

    Parameters
    ----------
    obs : pd.DataFrame
        DataFrame to filter
    x_col, y_col : str
        Column names in data for the x and y coordinates of cells
    custom_xy_lims : list of float, [xmin, xmax, ymin, ymax]
        Custom x and y limits for the plot
    """
    # need to detect min & max because of how the abc atlas coordinates are
    # oriented, i.e. abs(ymin) > abs(ymax) but ymin=bottom & ymax=top
    xmin = min(custom_xy_lims[:2])
    xmax = max(custom_xy_lims[:2])
    ymin = min(custom_xy_lims[2:])
    ymax = max(custom_xy_lims[2:])

    obs = abc.filter_by_coordinate_range(obs, x_col, xmin, xmax)
    obs = abc.filter_by_coordinate_range(obs, y_col, ymin, ymax)

    return obs

def _integrate_background_cells(obs, point_hue, bg_cells):
    """Add background cells to the DataFrame of cells to display,
    with NA values for the point_hue column."""
    obs = pd.concat(
        [
            obs,
            bg_cells.loc[bg_cells.index.difference(obs.index)].assign(
                **{point_hue: np.nan}
            ),
        ]
    )
    return obs
# ------------------------- Color Palette Handling ------------------------- #

# Pre-set edge_colors for common situations
EDGE_COLOR_HIGHLIGHT = "black"
EDGE_COLOR_HIGHLIGHT = "black"
OTHER_CATEGORY_COLOR = "grey"
BACKGROUND_POINT_COLOR = "lightgrey"


def _generate_palette(categories, palette=glasbey, hue_label=None, **items):
    """Generate a color palette dict for a given list of categories.

    Parameters
    ----------
    categories : list of str
        List of category names

    Returns
    -------
    palette : dict of (str, RGB tuples)
    """
    # if hue is a taxonomy level, no need to pass in pre-generated
    # palette as a parameter, just calculate it on the fly
    if palette is None and hue_label in ["class", "subclass", "supertype", "cluster"]:
        palette = abc.get_taxonomy_palette(hue_label)
    try:
        palette = {x: palette[x] for x in palette if x in categories}
    except:
        sns_palette = sns.color_palette(palette, n_colors=len(categories))
        palette = dict(zip(categories, sns_palette))
    palette.update(**items)
    return palette


def _palette_to_rgba_lookup(palette, index):
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
    rgba_lookup = np.zeros((max_val, 4))
    # fill only values in index and also in palette
    # rest remain transparent (alpha=0)
    for i in index.index:
        name = index[i]
        if name in palette:
            rgba_lookup[i, :] = to_rgba(palette[name])
    rgba_lookup[0, :] = [1, 1, 1, 0]
    return rgba_lookup

# ----------------------------- Plot Formatting ----------------------------- #

def _get_figure_handles(ax):
    """Get the current figure and axes handles, or create new ones if ax is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = plt.gcf()
    return ax, fig

def _format_image_axes(ax, axes=False, set_lims="whole", custom_xy_lims=[]):
    """Format the axes of a plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to format
    axes : bool
        Whether to display the axes and spines
    set_lims : bool or str, {'whole', 'left_hemi', 'right_hemi'}
        Whether to set the x and y limits of the plot to the whole brain,
        the left hemisphere, or the right hemisphere
    custom_xy_lims : list of float, [xmin, xmax, ymin, ymax]
        Custom x and y limits for the plot, supercedes defaults from set_lims
    """
    ax.axis("image")
    if not axes:
        sns.despine(left=True, bottom=True)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    # (set_lims==True) is for backwards compatibility
    # TODO: allow for easier whole-coronal vs TH+ZI axis formatting
    if (set_lims == "whole") | (set_lims == True):
        ax.set_xlim([2.5, 8.5])
        ax.set_ylim([7, 4])
    elif set_lims == "left_hemi":
        ax.set_xlim([2.5, 6])
        ax.set_ylim([7, 4])
    elif set_lims == "right_hemi":
        ax.set_xlim([5, 8.5])
        ax.set_ylim([7, 4])
    # custom_xy_lims supercedes set_lims
    if custom_xy_lims != []:
        if len(custom_xy_lims) == 4:
            ax.set_xlim(custom_xy_lims[:2])
            ax.set_ylim(custom_xy_lims[2:])
        else:
            print(
                "incorrect custom_xy_lims detected, must be [x_min,x_max,y_min,y_max]"
            )
