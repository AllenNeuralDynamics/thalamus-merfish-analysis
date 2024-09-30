import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid

from . import abc_load as abc
from . import ccf_images as cci
from . import color_utils as cu

CCF_REGIONS_DEFAULT = None

# thalamus-specific constants
TH_EXAMPLE_SECTION_LABELS = [
    "C57BL6J-638850.44",
    "C57BL6J-638850.40",
    "C57BL6J-638850.36",
]  # anterior to posterior order
TH_EXAMPLE_Z_SECTIONS = [8.0, 7.2, 6.4]  # anterior to posterior order
XY_LIMS_TH_LEFT_HEMI = [2.8, 5.8, 7, 4]  # limits plots to thalamus left hemisphere

# TODO: make plotting class to cache indices, col names, etc?


def plot_ccf_overlay(
    obs,
    ccf_images=None,
    sections=None,
    # column names
    section_col="brain_section_label",
    x_col="x_section",
    y_col="y_section",
    point_hue="CCF_acronym",
    # point props
    point_palette=None,
    point_size=2,
    # point selection
    categorical=True,
    min_group_count=10,
    min_section_count=20,
    bg_cells=None,
    # shape props
    face_palette=None,
    edge_color="grey",
    # shape selection
    ccf_names=None,
    ccf_highlight=(),
    ccf_level="substructure",
    # formatting
    legend="cells",
    custom_xy_lims=None,
    show_axes=False,
    separate_figs=True,
    n_rows=1,
    figsize=(8, 4),
    boundary_img=None,
):
    """
    Parameters
    ----------
    obs : pd.DataFrame
        DataFrame of cells to display
    ccf_images : np.ndarray, optional
        3D array of CCF parcellation regions
    sections : list of numbers, optional
        Section(s) to display. Must be a list even for single section.
        If None, all sections that contain cells in obs are displayed
    section_col : str, default='brain_section_label'
        Column name in obs for the section labels in 'sections'
    x_col, y_col : str
        Column names in obs for the x and y coordinates of cells
    point_hue : str
        Column name in obs to color cells by
    point_palette : dict, optional
        Dictionary of point_hue categories and colors
    point_size : int, default=2
        Size of foreground cell scatter points (background cells set to min([0.8*point_size,2]))
    categorical : bool, default=True
        Whether point_hue is a categorical variable
    min_group_count : int
        Minimum number of cells in a group to be displayed
    min_section_count : int
        Minimum number of cells in a section to be displayed
    bg_cells : pd.DataFrame, optional
        DataFrame of background cells to display
    face_palette : {None, dict, 'glasbey'}, default=None
        Sets face color of CCF region shapes in plot_ccf_section(), see that
        function's docstring for more details
    edge_color : str, default='grey'
        Sets outline/boundary color of CCF region shapes in plot_ccf_section()
    ccf_names : list of str, optional
        List of CCF region names to display
    ccf_highlight : list of str, optional
        List of CCF region names to highlight with a darker outline
    ccf_level : str, {'substructure', 'structure'}, default='substructure'
        Level of CCF to be displayed
    legend : str, {'ccf', 'cells', 'both', None}
        Whether to display a legend for the CCF region shapes, the cell types,
        both, or neither
    custom_xy_lims : list of float, [xmin, xmax, ymin, ymax]
        Custom x and y limits for the plot
    show_axes : bool
        Whether to display the axes and spines
    boundary_img : np.ndarray, optional
        3D array of CCF parcellation boundaries; if None, calculated on the fly
    """
    # Set variables not specified by user
    if sections is None:
        sections = _get_sections_to_plot(
            obs, section_col, ccf_names, ccf_highlight, ccf_images, ccf_level, n0=min_section_count
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
        if point_palette is None and point_hue in ["class", "subclass", "supertype", "cluster"]:
            point_palette = abc.get_taxonomy_palette(point_hue)
        point_palette = cu.generate_palette(
            obs[point_hue].unique().tolist(),
            hue_label=point_hue,
            palette=point_palette,
            other=OTHER_CATEGORY_COLOR,
        )

    # add background cells as NA values
    if bg_cells is not None:
        obs = _integrate_background_cells(obs, point_hue, bg_cells)

    # Display each section as a separate plot by default
    if not separate_figs:
        grid = _create_axis_grid(len(sections), n_rows, figsize=figsize)
        # TODO: could use this pattern for other multi-section plots
        # probably best to use a class to share code?
    figs = [
        plot_section_overlay(
            obs,
            ccf_images,
            section,
            section_col=section_col,
            x_col=x_col,
            y_col=y_col,
            point_hue=point_hue,
            point_palette=point_palette,
            point_size=point_size,
            face_palette=face_palette,
            edge_color=edge_color,
            ccf_names=ccf_names,
            ccf_highlight=ccf_highlight,
            ccf_level=ccf_level,
            boundary_img=boundary_img,
            custom_xy_lims=custom_xy_lims,
            show_axes=show_axes,
            legend=legend,
            ax=None if separate_figs else grid[i],
            figsize=figsize,
        )
        for i, section in enumerate(sections)
    ]
    if not separate_figs and legend is not None:
        _combine_subplot_legends(figs[0], title=ccf_level if legend == "ccf" else point_hue)

    return figs


def _create_axis_grid(n_total, n_rows, figsize, axes_pad=0.1, **kwargs):
    return ImageGrid(
        plt.figure(figsize=figsize),
        111,  # similar to subplot(111)
        nrows_ncols=(n_rows, int(np.ceil(n_total / n_rows))),
        axes_pad=axes_pad,
        **kwargs,
    )


def _combine_subplot_legends(fig, ncol=4, **legend_args):
    args = dict(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=ncol, frameon=False)
    args.update(**legend_args)
    labels = []
    handles = []
    for ax in fig.axes:
        if ax.get_legend() is None:
            # could also explicitly skip colorbar axes
            continue
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                labels.append(label)
                handles.append(handle)
        ax.get_legend().remove()
    labels = np.array(labels)
    handles = np.array(handles)
    try:
        order = np.argsort(labels.astype(int))
    except ValueError:
        order = np.argsort(labels)
    fig.legend(handles[order], labels[order], **args)


def get_sections_for_ccf_regions(
    ccf_images, ccf_names, ccf_level="substructure", section_col="z_section"
):
    """Get the sections that contain cells from a list of CCF regions."""
    structure_index = abc.get_ccf_index_reverse_lookup(level=ccf_level)
    ccf_ids = structure_index[ccf_names].values
    sections = []
    for i in range(ccf_images.shape[2]):
        if np.any(np.isin(ccf_images[:, :, i], ccf_ids)):
            sections.append(i)
    section_index = abc.get_section_index(section_col=section_col)
    ind = pd.Series(section_index.index.values, index=section_index)
    section_names = ind[ind.index.intersection(sections)].values
    return section_names


def plot_section_overlay(
    obs,
    ccf_images,
    section,
    # column names
    section_col="brain_section_label",
    x_col="x_section",
    y_col="y_section",
    point_hue="CCF_acronym",
    # point props
    point_palette=None,
    point_size=2,
    # shape props
    face_palette=None,
    edge_color="grey",
    # shape selection
    ccf_names=None,
    ccf_highlight=(),
    ccf_level="substructure",
    # formatting
    legend="cells",
    custom_xy_lims=None,
    show_axes=False,
    colorbar=False,
    boundary_img=None,
    zoom_to_highlighted=False,
    scatter_args=None,
    cb_args=None,
    ax=None,
    figsize=(8, 4),
):
    if cb_args is None:
        cb_args = {}
    if scatter_args is None:
        scatter_args = {}
    fig, ax = _get_figure_handles(ax, figsize=figsize)
    secdata = obs.loc[lambda df: (df[section_col] == section)]
    if custom_xy_lims is not None:
        # TODO: apply at dataset level instead?
        secdata = _filter_by_xy_lims(secdata, x_col, y_col, custom_xy_lims)

    # display CCF shapes
    if ccf_images is not None:
        plot_ccf_section(
            ccf_images,
            section,
            boundary_img=boundary_img,
            ccf_names=ccf_names,
            ccf_highlight=ccf_highlight,
            section_col=section_col,
            ccf_level=ccf_level,
            face_palette=face_palette,
            edge_color=edge_color,
            legend=(legend == "ccf"),
            zoom_to_highlighted=zoom_to_highlighted,
            show_axes=show_axes,
            ax=ax,
        )
    # need to keep zoom set by plot_ccf_section
    if zoom_to_highlighted:
        custom_xy_lims = [*ax.get_xlim(), *ax.get_ylim()]

    _plot_cells_scatter(
        secdata,
        x_col=x_col,
        y_col=y_col,
        point_hue=point_hue,
        point_palette=point_palette,
        point_size=point_size,
        legend=(legend == "cells"),
        ax=ax,
        **scatter_args,
    )
    # plot formatting
    label = ccf_level if legend == "ccf" else point_hue
    if legend is not None:
        # cell type names require more horizontal space
        # TODO: detect this from label text
        _add_legend(ax, ncols=4 if (legend == "ccf") else 2, title=label, markerscale=2)
    if colorbar:
        _add_colorbar(ax, **cb_args)

    _format_image_axes(ax=ax, show_axes=show_axes, custom_xy_lims=custom_xy_lims)
    title = f"Section {section}"
    if not legend:
        title += f"\n by {label}"
    ax.set_title(title)
    return fig


def _add_legend(ax=None, ncols=2, **kwargs):
    args = dict(ncols=ncols, loc="upper center", bbox_to_anchor=(0.5, 0), frameon=False)
    args.update(**kwargs)
    if ax is None:
        ax = plt.gca()
    ax.legend(**args)


def _add_colorbar(ax, cb_vmin_vmax=(0, 1), cmap="viridis", **kwargs):
    sm = ax.scatter([], [], c=[], cmap=cmap, vmin=cb_vmin_vmax[0], vmax=cb_vmin_vmax[1])
    args = dict(orientation="vertical")
    args.update(**kwargs)
    ax.figure.colorbar(sm, **kwargs)


def _plot_cells_scatter(
    secdata,
    x_col,
    y_col,
    point_hue,
    point_palette=None,
    point_size=2,
    legend=True,
    ax=None,
    **kwargs,
):
    if len(secdata) == 0:
        return
    # remove missing types from legend
    if legend and secdata[point_hue].dtype.name == "category":
        secdata = secdata.copy()
        secdata[point_hue] = secdata[point_hue].cat.remove_unused_categories()

    # default to no marker outlines, but allow user to override
    if ("linewidth" not in kwargs) and ("linewidths" not in kwargs):
        kwargs["linewidth"] = 0

    sns.scatterplot(
        secdata,
        x=x_col,
        y=y_col,
        hue=point_hue,
        s=point_size,
        palette=point_palette,
        legend=legend,
        ax=ax,
        **kwargs,
    )
    bg_s = np.min([point_size * 0.8, 2])
    # TODO: add background cells to legend?
    sns.scatterplot(
        secdata.loc[secdata[point_hue].isna()],
        x=x_col,
        y=y_col,
        c=BACKGROUND_POINT_COLOR,
        s=bg_s,
        linewidth=0,
        zorder=-1,
        ax=ax,
    )


def _get_counts_label(adata, gene):
    # if adata from load_adata(), counts_transform is recorded in .uns
    if "counts_transform" in adata.uns:
        label = f"gene counts ({adata.uns['counts_transform']})"
    # if we don't have .uns['counts_transform'], check if we have raw counts or not
    else:
        if all(
            i.is_integer() for i in adata[gene]
        ):  # no [] around loop -> stops at 1st non-integer encounter
            label = "gene counts (raw)"
        else:
            label = "gene counts (unknown transform)"
    return label


def plot_expression_ccf(
    adata,
    gene,
    ccf_images=None,
    sections=None,
    nuclei=None,
    highlight=(),
    ccf_level="substructure",
    point_size=1.5,
    cmap="Blues",
    edge_color="lightgrey",
    section_col="brain_section_label",
    x_col="x_section",
    y_col="y_section",
    boundary_img=None,
    custom_xy_lims=None,
    cb_vmin_vmax=None,
    label=None,
    colorbar=True,
    show_axes=False,
    zoom_to_highlighted=False,
    figsize=(8, 4),
    **scatter_args,
):
    # TODO: rename these to be consistent with other functions
    ccf_names = nuclei
    ccf_highlight = highlight

    obs = preprocess_gene_plot(adata, gene)
    if cb_vmin_vmax is None:
        cb_vmin_vmax = (0, obs[gene].max())
    scatter_args = dict(hue_norm=cb_vmin_vmax, **scatter_args)
    if label is None:
        label = _get_counts_label(adata, gene)
    cb_args = dict(cmap=cmap, cb_vmin_vmax=cb_vmin_vmax, label=label, fraction=0.046, pad=0.01)

    if sections is None:
        sections = _get_sections_to_plot(obs, section_col, ccf_names, ccf_highlight, ccf_images, ccf_level)

    # Plot
    figs = []
    for section in sections:
        fig, ax = plt.subplots(figsize=figsize)
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
            show_axes=show_axes,
            legend=None,
            colorbar=colorbar,
            scatter_args=scatter_args,
            cb_args=cb_args,
            ax=ax,
            point_size=point_size,
            zoom_to_highlighted=zoom_to_highlighted,
        )
        ax.set_title(f"Section {section}\n{gene}")
        figs.append(fig)
        plt.show()
    return figs


def plot_hcr(
    adata,
    genes,
    sections=None,
    colors=None,
    section_col="brain_section_label",
    x_col="x_section",
    y_col="y_section",
    dark_background=True,
    normalize_sections=False,
    figsize=(14, 2),
    ccf_images=None,
    boundary_img=None,
    **kwargs
):
    """Display separate, and overlay, expression of multiple genes in multiple sections.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing cells to display with gene expression in .X and spatial coordinates in .obs.
    genes : list of str
        List of genes to display.
    sections : list of float or str, optional
        List of sections to display. If not provided, all unique sections in `adata.obs[section_col]` will be used.
    colors : list of str, optional
        List of colors to use for each gene. If not provided, default colors will be used.
    section_col : str, optional
        Column in `adata.obs` that contains the section values. Default is "brain_section_label".
    x_col, y_col : str, optional
        Columns in `adata.obs` that contain the x- and y-coordinates. Default is "x_section" and "y_section" respectively.
    dark_background : bool, optional
        Whether to use a dark background for the plots. Default is True.
    normalize_sections : bool, optional
        Whether to normalize gene expression by section. If False, gene expression will be normalized across all sections.
        Default is False.
    figsize : tuple, optional
        Figure size in inches. Default is (14, 2).

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        List of matplotlib figure objects, each representing a plot for a specific section.
    """
    # set variable(s) not specified at input
    if sections is None:
        sections = adata.obs[section_col].unique()

    obs = preprocess_gene_plot(adata, genes)
    if not normalize_sections:
        # normalize by gene across all sections
        obs[genes] /= obs[genes].max(axis=0)
        normalize_by = None
        colorbar = False
    else:
        # let plot_multichannel_overlay() normalize by gene by section
        normalize_by = "channels"
        colorbar = True

    # plot
    figs = []
    counts_label = _get_counts_label(adata, genes[0])
    for section in sections:
        fig = plot_multichannel_overlay(
            obs,
            genes,
            section,
            section_col=section_col,
            x_col=x_col,
            y_col=y_col,
            colors=colors,
            dark_background=dark_background,
            normalize_by=normalize_by,
            colorbar=colorbar,
            single_channel_subplots=True,
            figsize=figsize,
            ccf_images=ccf_images,
            boundary_img=boundary_img,
        )
        with matplotlib.style.context("dark_background" if dark_background else "default"):
            fig.suptitle(f"Section {section}\n{counts_label}", y=1.2)
        figs.append(fig)
    return figs


def plot_multichannel_overlay(
    obs,
    columns,
    section,
    colors=None,
    section_col="brain_section_label",
    x_col="x_section",
    y_col="y_section",
    dark_background=True,
    figsize=None,
    single_channel_subplots=False,
    legend=None,
    ccf_images=None,
    boundary_img=None,
    normalize_by="channels",
    colorbar=False,
    point_size=5,
):
    """
    Display overlay of multiple channels in a single section.

    Parameters
    ----------
    obs : pandas.DataFrame
        The dataframe containing the data.
    columns : list
        The list of column names representing the channels to be plotted.
    section : float or string
        The section to display.
    colors : list, optional
        The list of colors to be used for each channel. If not provided, default colors will be used.
    section_col : str, optional
        The column in `obs` that contains the section values. Default is "brain_section_label".
    x_col, y_col : str, optional
        The columns in `obs` that contain the x- and y-coordinates. Default is "x_section" and "y_section" respectively.
    dark_background : bool, optional
        Whether to use a dark background for the plot. Default is True.
    figsize : tuple, optional
        The figure size. Default is None.
    single_channel_subplots : bool, optional
        Whether to create subplots for each channel. Default is False.
    legend : bool, optional
        Whether to show the legend. If not provided, the legend will be shown if `single_channel_subplots` is False.
    ccf_images : numpy.ndarray, optional
        The array of CCF images. Default is None.
    boundary_img : numpy.ndarray, optional
        The array of CCF boundaries. Default is None.
    normalize_by : str, optional
        The normalization method. Can be "channels", "all", or None. Default is "channels".
    colorbar : bool, optional
        Whether to show the colorbar. Default is False.
    point_size : int, optional
        The size of the scatter plot points. Default is 5.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    """
    n_channel = len(columns)
    if legend is None:
        legend = not single_channel_subplots

    df = obs.loc[obs[section_col] == section]
    coeffs = df[columns].values
    # normalize either by channel or over all channels together
    if normalize_by == "channels":
        scale = coeffs.max(axis=0, keepdims=True)
    elif normalize_by == "all":
        scale = coeffs.max()
    else:
        assert normalize_by is None, "normalize_by must be 'channels', 'all', or None"
        scale = 1
    coeffs = coeffs / scale

    colors = cu.get_color_array(n_channel, colors=colors)

    with matplotlib.style.context("dark_background" if dark_background else "default"):
        if single_channel_subplots:
            cbar_mode = "each" if colorbar else None
            axes_pad = (0.15, 0.1) if colorbar else 0.05
            ax_subplots = _create_axis_grid(
                n_channel + 1,
                1,
                figsize=figsize,
                cbar_mode=cbar_mode,
                axes_pad=axes_pad,
            )
            for i in range(n_channel):
                ax = ax_subplots[i]
                c = cu.combine_scaled_colors(colors[[i], :], coeffs[:, [i]])
                ax.scatter(
                    *df[[x_col, y_col]].values.T,
                    s=point_size,
                    marker=".",
                    color=c,
                )
                ax.set_title(columns[i])
                _format_image_axes(ax)
                if colorbar:
                    # create a colorbar from list of shades of a single color
                    coeffs_cbar = np.linspace(0, 1, 256)[:, None]
                    c = cu.combine_scaled_colors(colors[[i], :], coeffs_cbar)
                    plt.colorbar(cu.mappable_for_colorbar(c, vmax=scale[0, i]), cax=ax.cax)
            ax = ax_subplots[-1]
            if colorbar:
                # hide colorbar from overlay plot
                _format_image_axes(ax.cax)
        else:
            _, ax = plt.subplots(figsize=figsize)
        c = cu.combine_scaled_colors(colors[:n_channel], coeffs)
        ax.scatter(
            *df[[x_col, y_col]].values.T,
            s=point_size,
            marker=".",
            linewidth=0,
            color=c,
        )
        if legend:
            for i in range(n_channel):
                plt.scatter([], [], color=colors[i], label=columns[i])
            ax.legend(markerscale=1.5)
        if ccf_images is not None:
            plot_ccf_section(
                ccf_images,
                section=section,
                section_col=section_col,
                edge_color="darkgrey",
                boundary_img=boundary_img,
                legend=False,
                ax=ax,
            )
        _format_image_axes(ax)
        if single_channel_subplots:
            ax.set_title("Overlay")
            ax.figure.suptitle(f"Section {section}", color="white")

    return ax.figure


def plot_metrics_ccf(
    ccf_img,
    metric_series,
    sections,
    section_col="z_section",
    ccf_level="substructure",
    cmap="viridis",
    cb_label="metric",
    vmin=None,
    vmax=None,
    show_axes=False,
    figsize=(8, 5),
):
    vmin = metric_series.min() if vmin is None else vmin
    vmax = metric_series.max() if vmax is None else vmax
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps.get_cmap(cmap)
    palette = metric_series.apply(lambda x: cmap(norm(x))).to_dict()

    figs = []
    for section_z in sections:
        print(section_z)
        fig, ax = plt.subplots(figsize=figsize)
        _add_colorbar(ax, cb_vmin_vmax=[vmin, vmax], cmap=cmap, label=cb_label, shrink=0.75)

        plot_ccf_section(
            ccf_img,
            section_z,
            face_palette=palette,
            ccf_level=ccf_level,
            section_col=section_col,
            legend=False,
            ax=ax,
        )
        _format_image_axes(ax=ax, show_axes=show_axes)
        figs.append(fig)
    return figs


# TODO: make multi-section option?
def plot_ccf_section(
    ccf_img,
    section,
    ccf_names=None,
    ccf_highlight=(),
    face_palette=None,
    edge_color="grey",
    boundary_img=None,
    section_col="z_section",
    ccf_level="substructure",
    legend=True,
    zoom_to_highlighted=False,
    show_axes=False,
    ax=None,
):
    """Display CCF parcellations for a single section from an
    image volume of region labels.
    Generates palettes to show highlights and hide regions not specified,
    then calls plot_ccf_shapes() to display the shapes.

    Parameters
    ----------
    ccf_img : np.ndarray
        3D array of CCF parcellations
    section : string or float
        Name of section to display (based on section_col)
    ccf_names : list of str, optional
        Subset of CCF regions to display
    ccf_highlight : list of str, optional
        Subset of CCF regions to highlight with darkened edges
    face_palette : {None, dict, list of colors, string}, default=None
        Sets face color of CCF region shapes;
        None to have no face color, or a dictionary of CCF region names and
        colors, or 'glasbey' to generate a color palette from the
        colorcet.glasbey color map
    edge_color : str, default='grey'
        Sets outline/boundary color of all CCF region shapes; any valid
        matplotlib color
    boundary_img : np.ndarray, optional
        2D array of CCF parcellation boundaries; if None, calculated on the fly
    section_col : str, default='z_section'
        Type of section names to use, based on columns in cell metadata
    ccf_level : str, {'substructure', 'structure'}, default='substructure'
        Level of CCF to be displayed
    legend : bool, default=True
        Whether to display a legend for the CCF region shapes
    zoom_to_highlighted : bool, default=False
        Whether to zoom the plot to the bounding box of the highlighted regions
    ax : matplotlib.axes.Axes, optional
        Existing Axes to plot on; if None, a new figure and Axes are created
    """
    section_index = abc.get_section_index(section_col=section_col)
    _, ax = _get_figure_handles(ax)
    # subset to just this section
    section_i = section_index[section]
    img = ccf_img[:, :, section_i].T
    boundary_img = boundary_img[:, :, section_i].T if boundary_img is not None else None

    # select CCF regions to plot
    # TODO: could allow names from other levels and translate all to specified level...
    structure_index = abc.get_ccf_index(level=ccf_level)
    # reindex in case image has extra regions not in structure_index
    structure_index = structure_index.reindex(np.unique(img)).sort_values()
    section_region_names = structure_index.values

    if ccf_names is None:
        ccf_names = CCF_REGIONS_DEFAULT  # may still be None
    if ccf_names is not None:
        section_region_names = list(set(section_region_names).intersection(ccf_names))

    face_palette = (
        cu.generate_palette(section_region_names, palette=face_palette)
        if face_palette is not None
        else None
    )

    edge_palette = {
        x: EDGE_HIGHLIGHT_COLOR if x in ccf_highlight else edge_color for x in section_region_names
    }

    plot_ccf_shapes(
        img,
        structure_index,
        boundary_img=boundary_img,
        face_palette=face_palette,
        edge_palette=edge_palette,
        ax=ax,
        legend=legend,
        show_axes=show_axes,
    )
    if zoom_to_highlighted:
        try:
            bbox = abc.X_RESOLUTION * get_bbox_for_regions(img, ccf_highlight, ccf_level)
            _format_image_axes(ax=ax, show_axes=show_axes, custom_xy_lims=bbox)
        except ValueError:
            pass
    ax.set_title(f"Section {section}")


def get_bbox_for_regions(img, ccf_names, ccf_level, buffer=10):
    structure_index = abc.get_ccf_index_reverse_lookup(level=ccf_level)
    ccf_ids = structure_index[ccf_names].values
    x_inds = np.flatnonzero(np.any(np.isin(img, ccf_ids), axis=0))
    y_inds = np.flatnonzero(np.any(np.isin(img, ccf_ids), axis=1))
    if len(x_inds) == 0 or len(y_inds) == 0:
        raise ValueError("Specified regions not found in image")
    # reverse order for y due to CCF orientation
    bbox = np.concatenate([x_inds[[0, -1]], y_inds[[-1, 0]]])
    if buffer > 0:
        bbox[[0, -1]] = np.maximum(bbox[[0, -1]] - buffer, 0)
        bbox[[1]] = np.minimum(bbox[[1]] + buffer, img.shape[0])
        bbox[[2]] = np.minimum(bbox[[2]] + buffer, img.shape[1])
    return bbox


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
    show_axes=False,
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
        Existing Axes to plot on; if None, a new figure and Axes are created
    resolution : float, default=10e-3
        Resolution of the CCF in the image plane, used to set correct image extent
    legend : bool, default=True
        Whether to display a legend for the CCF region shapes
    """
    _, ax = _get_figure_handles(ax)
    # TODO: use xarray for applying palette, plotting, storing boundaries
    extent = resolution * (np.array([0, imdata.shape[0], imdata.shape[1], 0]) - 0.5)
    imshow_args = dict(extent=extent, interpolation="none", alpha=alpha)

    # Plot face shapes of CCF regions
    if face_palette is not None:
        rgba_lookup = cu.palette_to_rgba_lookup(face_palette, index)
        im_regions = rgba_lookup[imdata, :]
        ax.imshow(im_regions, **imshow_args)

    # Plot boundaries of CCF regions
    if edge_palette is not None:
        if boundary_img is None:
            # TODO: keep and use ccf_level to merge if needed before erosion?
            boundary_img = cci.label_erosion(imdata, edge_width, fill_val=0, return_edges=True)
        rgba_lookup = cu.palette_to_rgba_lookup(edge_palette, index)
        im_edges = rgba_lookup[boundary_img, :]
        ax.imshow(im_edges, **imshow_args)

    # generate "empty" matplotlib handles for legend
    if legend and (face_palette is not None):
        for name, color in face_palette.items():
            ax.plot([], marker="o", ls="", color=color, label=name)
        _add_legend(ax, ncols=4)
    _format_image_axes(ax=ax, show_axes=show_axes)


# ------------------------- DataFrame Preprocessing ------------------------- #
def preprocess_gene_plot(adata, gene):
    obs = adata.obs.copy()
    obs[gene] = adata[:, gene].X.toarray()
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
    # obs = obs.copy()
    obs = abc.label_outlier_celltypes(obs, type_col, min_group_count=min_group_count)
    # Min group count by section shouldn't be larger than overall min_group_count
    # Set to the minimum so user can set min_group_count=0 to see all groups
    min_group_count_section = min(min_group_count_section, min_group_count)
    obs = obs.groupby(section_col, group_keys=False, observed=False).apply(
        lambda x: abc.label_outlier_celltypes(
            x, type_col, min_group_count=min_group_count_section
        )
    )
    obs[type_col] = obs[type_col].cat.remove_unused_categories()
    return obs


def _get_sections_to_plot(obs, section_col, ccf_names, ccf_highlight, ccf_images, ccf_level, n0=0):
    if n0 > 0:
        sections = obs[section_col].value_counts().loc[lambda x: x > n0].index
    else:
        sections = obs[section_col].unique()
    target_regions = ccf_highlight if len(ccf_highlight) > 0 else ccf_names
    if target_regions is not None and len(target_regions) > 0:
        sections = set(sections).intersection(
            get_sections_for_ccf_regions(
                ccf_images,
                target_regions,
                ccf_level=ccf_level,
                section_col=section_col,
            )
        )
    return sorted(sections)


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
            bg_cells.loc[bg_cells.index.difference(obs.index)].assign(**{point_hue: np.nan}),
        ]
    )
    return obs


# ------------------------- Color Palette Handling ------------------------- #

# Pre-set edge_colors for common situations
EDGE_HIGHLIGHT_COLOR = "black"
OTHER_CATEGORY_COLOR = "grey"
BACKGROUND_POINT_COLOR = "lightgrey"


# ----------------------------- Plot Formatting ----------------------------- #


def _get_figure_handles(ax, figsize=(8, 4)):
    """Get the figure handle for a set of axes, or create new ones if ax is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.gcf()
    return fig, ax


def _format_image_axes(ax, show_axes=False, set_lims="whole", custom_xy_lims=None):
    """Format the axes of a plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to format
    show_axes : bool
        Whether to display the axes and spines
    set_lims : bool or str, {'whole', 'left_hemi', 'right_hemi'}
        Whether to set the x and y limits of the plot to the whole brain,
        the left hemisphere, or the right hemisphere
    custom_xy_lims : list of float, [xmin, xmax, ymin, ymax]
        Custom x and y limits for the plot, supercedes defaults from set_lims
    """
    # TODO: separate limits from axis formatting
    ax.axis("image")
    if not show_axes:
        sns.despine(left=True, bottom=True)
        ax.set_xticks([])
        ax.set_yticks([])
        
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    # (set_lims==True) is for backwards compatibility
    # TODO: allow for easier whole-coronal vs TH+ZI axis formatting
    if (set_lims == "whole") | (set_lims is True):
        ax.set_xlim([2.5, 8.5])
        ax.set_ylim([7, 4])
    elif set_lims == "left_hemi":
        ax.set_xlim([2.5, 6])
        ax.set_ylim([7, 4])
    elif set_lims == "right_hemi":
        ax.set_xlim([5, 8.5])
        ax.set_ylim([7, 4])
    # custom_xy_lims supercedes set_lims
    if custom_xy_lims is not None:
        if len(custom_xy_lims) == 4:
            ax.set_xlim(custom_xy_lims[:2])
            ax.set_ylim(custom_xy_lims[2:])
        else:
            print("incorrect custom_xy_lims detected, must be [x_min,x_max,y_min,y_max]")
