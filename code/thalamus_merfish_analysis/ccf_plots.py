import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from colorcet import glasbey
from matplotlib.colors import to_rgba

from . import abc_load as abc
from . import ccf_images as cci

CCF_REGIONS_DEFAULT = None

# ------------------------- Multi-Section Plotting ------------------------- #
def plot_ccf_overlay(
    obs,
    ccf_images,
    sections=None,
    # column names
    section_col="section",
    x_col="cirro_x",
    y_col="cirro_y",
    point_hue="CCF_acronym",
    # point props
    point_palette=None,
    s=2,  # TODO: rename 's' arg!
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
    ccf_highlight=[],
    ccf_level="substructure",
    # formatting
    legend="cells",
    custom_xy_lims=None,
    show_axes=False,
    boundary_img=None,
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
    section_col : str, {'z_section', 'z_reconstructed', 'z_realigned'}
        Column name in obs for the section numbers in 'sections'; must be a col
        that allows for indexing into ccf_images
    x_col, y_col : str
        Column names in obs for the x and y coordinates of cells
    point_hue : str
        Column name in obs to color cells by
    point_palette : dict, optional
        Dictionary of point_hue categories and colors
    s : int, default=2
        Size of foreground cell scatter points (background cells set to 0.8*s)
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
        List of CCF region names to highlight with a darker outline for ccf_names
    ccf_level : str, {'substructure', 'structure'}, default='substructure'
        Level of CCF to be displayed
    legend : str, {'ccf', 'cells', 'both', None}
        Whether to display a legend for the CCF region shapes, the cell types,
        both, or neither
    custom_xy_lims : list of float, [xmin, xmax, ymin, ymax]
        Custom x and y limits for the plot
    show_axes : bool
        Whether to display the show_axes and spines
    boundary_img : np.ndarray, optional
        3D array of CCF parcellation boundaries; if None, calculated on the fly
    """
    # Set variables not specified by user
    if sections is None:
        sections = (
            obs[section_col].value_counts().loc[lambda x: x > min_section_count].index
        )
        if ccf_names is not None:
            sections = sections.intersection(
                get_sections_for_ccf_regions(ccf_images, ccf_names)
            )
        sections = sorted(sections)
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

    # add background cells as NA values
    if bg_cells is not None:
        obs = _integrate_background_cells(obs, point_hue, bg_cells)

    # Display each section as a separate plot
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
            s=s,
            face_palette=face_palette,
            edge_color=edge_color,
            ccf_names=ccf_names,
            ccf_highlight=ccf_highlight,
            ccf_level=ccf_level,
            boundary_img=boundary_img,
            custom_xy_lims=custom_xy_lims,
            show_axes=show_axes,
            legend=legend,
        )
        for section in sections
    ]
    return figs


def get_sections_for_ccf_regions(
    ccf_images,
    ccf_names,
    ccf_level="substructure",
    z_resolution=200e-3,
):
    """Get the sections that contain cells from a list of CCF regions."""
    structure_index = abc.get_ccf_index_reverse_lookup(level=ccf_level)
    ccf_ids = structure_index[ccf_names].values
    sections = []
    for i in range(ccf_images.shape[2]):
        if np.any(np.isin(ccf_images[:, :, i], ccf_ids)):
            sections.append(np.round(i * z_resolution, 1))
    return sections


def plot_section_overlay(
    obs,
    ccf_images,
    section,
    # column names
    section_col="section",
    x_col="cirro_x",
    y_col="cirro_y",
    point_hue="CCF_acronym",
    # point props
    point_palette=None,
    s=2,
    # shape props
    face_palette=None,
    edge_color="grey",
    # shape selection
    ccf_names=None,
    ccf_highlight=[],
    ccf_level="substructure",
    # formatting
    legend="cells",
    custom_xy_lims=None,
    show_axes=False,
    colorbar=False,
    boundary_img=None,
    zoom_to_highlighted=False,
    scatter_args={},
    cb_args={},
    ax=None,
):
    fig, ax = _get_figure_handles(ax)
    secdata = obs.loc[lambda df: (df[section_col] == section)]
    if custom_xy_lims is not None:
        # TODO: apply at dataset level instead?
        secdata = _filter_by_xy_lims(secdata, x_col, y_col, custom_xy_lims)

    # display CCF shapes
    plot_ccf_section(
        ccf_images,
        section,
        boundary_img=boundary_img,
        ccf_names=ccf_names,
        ccf_highlight=ccf_highlight,
        ccf_level=ccf_level,
        face_palette=face_palette,
        edge_color=edge_color,
        legend=(legend == "ccf"),
        zoom_to_highlighted=zoom_to_highlighted,
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
        s=s,
        legend=(legend in ["cells", "both"]),
        ax=ax,
        **scatter_args,
    )
    # plot formatting
    if legend is not None:
        # cell type names require more horizontal space
        # TODO: detect this from label text
        _add_legend(ncols = 4 if (legend == "ccf") else 2)
    if colorbar: 
        _add_colorbar(ax, **cb_args)

    _format_image_axes(ax=ax, show_axes=show_axes, custom_xy_lims=custom_xy_lims)
    ax.set_title("z=" + str(section) + "\n" + point_hue)
    plt.show()
    return fig

def _add_legend(ncols=2, **kwargs):
    args = dict(
            ncols=ncols, loc="upper center", bbox_to_anchor=(0.5, 0), frameon=False
        )
    args.update(**kwargs)
    plt.legend(**args)

def _add_colorbar(ax, cb_vmin_vmax=[0,1], cmap="viridis", **kwargs):
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
    s=2,
    legend=True,
    ax=None,
    **kwargs,
):
    if len(secdata)==0:
        return
    # remove missing types from legend
    if legend and secdata[point_hue].dtype.name == "category":
        secdata = secdata.copy()
        secdata[point_hue] = secdata[point_hue].cat.remove_unused_categories()

    sc = sns.scatterplot(
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
        zorder=-1,
    )


def _get_counts_label(adata, gene):
    # if adata from load_adata(), counts_transform is recorded in .uns
    if "counts_transform" in adata.uns:
        label = f"gene counts ({adata.uns['counts_transform']})"
    # if we don't have .uns['counts_transform'], check if we have raw counts or not
    else:
        if all(
            i.is_integer() for i in adata[gene]
        ):  # no [] around loop == stops at 1st non-integer encounter
            label = "gene counts (raw)"
        else:
            label = "gene counts (unknown transform)"
    return label


def plot_expression_ccf(
    adata,
    gene,
    ccf_images,
    sections=None,
    nuclei=None,
    highlight=[],
    ccf_level="substructure",
    s=1.5,
    cmap="Blues",
    edge_color="lightgrey",
    section_col="section",
    x_col="cirro_x",
    y_col="cirro_y",
    boundary_img=None,
    custom_xy_lims=None,
    cb_vmin_vmax=None,
    label=None,
    colorbar=True,
    show_axes=False,
    zoom_to_highlighted=False,
    figsize=(8,4),
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

    # TODO: allow sections arg to be single section not list?
    if sections is None:
        sections = adata.obs[section_col].unique()
        if ccf_names is not None:
            sections = set(sections).intersection(
                get_sections_for_ccf_regions(
                    ccf_images, ccf_highlight if zoom_to_highlighted else ccf_names
                )
            )
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
            s=s,
            zoom_to_highlighted=zoom_to_highlighted,
        )
        ax.set_title(f"z={section}\n{gene}")
        figs.append(fig)
        plt.show()
    return figs

def plot_hcr(adata, genes, sections=None, section_col='section', 
             x_col='cirro_x', y_col='cirro_y', bg_color='white'):
    '''Display separate, and overlay, expression of 3 genes in multiple sections.
    
    Parameters
    ----------
    adata : AnnData
        cells to display; gene expression in .X and spatial coordinates in .obs
    genes : list of str
        list of genes to display
    section : list of float
        sections to display.
        if passing in a single section, still must be in a list, e.g. [7.2].
    section_col : str
        column in adata.obs that contains the section values
    x_col, y_col : str
        columns in adata.obs that contains the x- & y-coordinates
    bg_color : str, default='white'
        background color of the plot. Can use any color str that is recognized  
        by matplotlib; passed on to plt.subplots(..., facecolor=bg_color) and
        ax.set_facecolor(bg_color).
        'black' / 'k' / '#000000' changes font colors to 'white'.

    Returns
    -------
    figs : list of matplotlib.figure.Figure
    '''
    # set variable(s) not specified at input
    if sections is None:
        sections = adata.obs[section_col].unique()
    # plot
    figs = []
    for section in sections:
        fig = plot_hcr_section(adata, genes, section, section_col=section_col, 
                               x_col=x_col, y_col=y_col, bg_color=bg_color)
        figs.append(fig)
    return figs

def plot_hcr(adata, genes, sections=None, section_col='section', 
             x_col='cirro_x', y_col='cirro_y', bg_color='white'):
    '''Display separate, and overlay, expression of 3 genes in multiple sections.
    
    Parameters
    ----------
    adata : AnnData
        cells to display; gene expression in .X and spatial coordinates in .obs
    genes : list of str
        list of genes to display
    section : list of float
        sections to display.
        if passing in a single section, still must be in a list, e.g. [7.2].
    section_col : str
        column in adata.obs that contains the section values
    x_col, y_col : str
        columns in adata.obs that contains the x- & y-coordinates
    bg_color : str, default='white'
        background color of the plot. Can use any color str that is recognized  
        by matplotlib; passed on to plt.subplots(..., facecolor=bg_color) and
        ax.set_facecolor(bg_color).
        'black' / 'k' / '#000000' changes font colors to 'white'.

    Returns
    -------
    figs : list of matplotlib.figure.Figure
    '''
    # set variable(s) not specified at input
    if sections is None:
        sections = adata.obs[section_col].unique()
    # plot
    figs = []
    for section in sections:
        fig = plot_hcr_section(adata, genes, section, section_col=section_col, 
                               x_col=x_col, y_col=y_col, bg_color=bg_color)
        figs.append(fig)
    return figs


def plot_hcr_section(adata, genes, section, section_col='section',
                     x_col='cirro_x', y_col='cirro_y', bg_color='white'):
    '''Display separate, and overlay, expression of 3 genes in a single section.
    
    Parameters
    ---------
    adata : AnnData
        cells to display; gene expression in .X and spatial coordinates in .obs
    genes : list of str
        list of genes to display
    section : float
        section to display
    section_col : str
        column in adata.obs that contains the section values
    x_col, y_col : str
        columns in adata.obs that contains the x- & y-coordinates
    bg_color : str, default='white'
        background color of the plot. Can use any color str that is recognized  
        by matplotlib; passed on to plt.subplots(..., facecolor=bg_color) and
        ax.set_facecolor(bg_color).
        'black' / 'k' / '#000000' changes font colors to 'white'.

    Returns
    -------
    fig : matplotlib.figure.Figure
    '''
    
    # Subset based on requested section
    sec_adata = adata[adata.obs[section_col]==section]

    # Set font color based on bg_color
    if (bg_color=='black') | (bg_color=='k') | (bg_color=='#000000'):
        font_color = 'white'
    else:
        font_color = 'black'
    fontsize = 16
    
    # Get normalized expression of each gene
    gene1_norm = sec_adata[:,genes[0]].X / sec_adata[:,genes[0]].X.max()
    gene2_norm = sec_adata[:,genes[1]].X / sec_adata[:,genes[1]].X.max()
    gene3_norm = sec_adata[:,genes[2]].X / sec_adata[:,genes[2]].X.max()

    # Convert each genes normalized expression into an RGB value
    zeros = np.zeros([len(sec_adata),1])
    colorR = np.concatenate((gene1_norm, zeros, zeros),axis=1)
    colorG = np.concatenate((zeros, gene2_norm, zeros),axis=1)
    colorB = np.concatenate((zeros, zeros, gene3_norm),axis=1)
    # combine each gene into a single RGB color for overlay
    colorRGB = np.concatenate((gene1_norm, gene2_norm, gene3_norm),axis=1)
    # add overlay to list of colors & gene labels
    cell_colors = (colorR, colorG, colorB, colorRGB)
    genes.append('Overlay') # Append for labeling purposes

    # Plot spatial expression for each channel (3 genes + overlay), 
    fig, axes = plt.subplots(1,4, figsize=(24,3), dpi=80, facecolor=bg_color)
    axes = axes.flatten()
    for i, cell_color in enumerate(cell_colors):
        ax = axes[i]
        ax.scatter(sec_adata.obs[x_col],
                   sec_adata.obs[y_col],
                   s=10, marker='.', color=cell_color)
        ax.set_title(genes[i], color=font_color, fontsize=fontsize)
        _format_image_axes(ax)
        ax.set_facecolor(bg_color)  # must be set AFTER _format_image_axes to take effect

    counts_str = adata.uns['counts_transform']    
    plt.suptitle(f'{section=}\ncounts={counts_str}', y=1.2, 
                 color=font_color, fontsize=fontsize)
    plt.show()
    
    return fig


def plot_hcr(adata, genes, sections=None, section_col='section', 
             x_col='cirro_x', y_col='cirro_y', bg_color='white'):
    '''Display separate, and overlay, expression of 3 genes in multiple sections.
    
    Parameters
    ----------
    adata : AnnData
        cells to display; gene expression in .X and spatial coordinates in .obs
    genes : list of str
        list of genes to display
    section : list of float
        sections to display.
        if passing in a single section, still must be in a list, e.g. [7.2].
    section_col : str
        column in adata.obs that contains the section values
    x_col, y_col : str
        columns in adata.obs that contains the x- & y-coordinates
    bg_color : str, default='white'
        background color of the plot. Can use any color str that is recognized  
        by matplotlib; passed on to plt.subplots(..., facecolor=bg_color) and
        ax.set_facecolor(bg_color).
        'black' / 'k' / '#000000' changes font colors to 'white'.

    Returns
    -------
    figs : list of matplotlib.figure.Figure
    '''
    # set variable(s) not specified at input
    if sections is None:
        sections = adata.obs[section_col].unique()
    # plot
    figs = []
    for section in sections:
        fig = plot_hcr_section(adata, genes, section, section_col=section_col, 
                               x_col=x_col, y_col=y_col, bg_color=bg_color)
        figs.append(fig)
    return figs


def plot_metrics_ccf(
    ccf_img,
    metric_series,
    sections,
    structure_index=None,
    cmap="viridis",
    cb_label="metric",
    vmin=None,
    vmax=None,
    show_axes=False,
    figsize=(8, 5),
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
        fig, ax = plt.subplots(figsize=figsize)
        _add_colorbar(ax, cb_vmin_vmax=[vmin, vmax], cmap=cmap, shrink=0.75)

        plot_ccf_section(
            ccf_img,
            section_z,
            palette,
            structure_index=structure_index,
            legend=False,
            ax=ax,
        )
        _format_image_axes(ax=ax, show_axes=show_axes)
        figs.append(fig)
    return figs


# TODO: make multi-section option?
def plot_ccf_section(
    ccf_img,
    section_z,
    ccf_names=None,
    ccf_highlight=[],
    face_palette=None,
    edge_color="grey",
    boundary_img=None,
    ccf_level="substructure",
    z_resolution=200e-3,
    legend=True,
    zoom_to_highlighted=False,
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
    ccf_names : list of str, optional
        Subset of CCF regions to display
    ccf_highlight : list of str, optional
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
    ccf_level : str, {'substructure', 'structure'}, default='substructure'
        Level of CCF to be displayed
    z_resolution : float, default=200e-3
        Resolution of the CCF in the z-dimension
    legend : bool, default=True
        Whether to display a legend for the CCF region shapes
    ax : matplotlib.axes.Axes, optional
        Existing Axes to plot on; if None, a new figure and Axes are created
    """
    fig, ax = _get_figure_handles(ax)
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

    if ccf_names is None: ccf_names = CCF_REGIONS_DEFAULT
    if ccf_names is not None:
        section_region_names = list(set(section_region_names).intersection(ccf_names))

    face_palette = (
        _generate_palette(section_region_names, palette=face_palette)
        if face_palette is not None
        else None
    )

    edge_palette = {
        x: EDGE_HIGHLIGHT_COLOR if x in ccf_highlight else edge_color
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
    ax.set_title("z=" + str(section_z) + "\n" + ccf_level)
    if zoom_to_highlighted:
        resolution=10e-3
        bbox = resolution*get_bbox_for_regions(img, ccf_highlight, ccf_level)
        _format_image_axes(ax=ax, show_axes=True, custom_xy_lims=bbox)
    
def get_bbox_for_regions(img, ccf_names, ccf_level, buffer=10):
    structure_index = abc.get_ccf_index_reverse_lookup(level=ccf_level)
    ccf_ids = structure_index[ccf_names].values
    bbox = np.concatenate([
        np.flatnonzero(np.any(np.isin(img, ccf_ids), axis=0))[[0,-1]],
        # reverse order for y
        np.flatnonzero(np.any(np.isin(img, ccf_ids), axis=1))[[-1,0]]
    ])
    if buffer>0:
        bbox[[0,-1]] = np.maximum(bbox[[0,-1]]-buffer,0)
        bbox[[1]] = np.minimum(bbox[[1]]+buffer,img.shape[0])
        bbox[[2]] = np.minimum(bbox[[2]]+buffer,img.shape[1])
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
    fig, ax = _get_figure_handles(ax)
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
        _add_legend(ncols=4)
    _format_image_axes(ax=ax)


# ------------------------- DataFrame Preprocessing ------------------------- #
def preprocess_gene_plot(adata, gene):
    obs = adata.obs.copy()
    obs[gene] = adata[:,gene].X.toarray()
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
    obs[type_col] = obs[type_col].cat.remove_unused_categories()
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
EDGE_HIGHLIGHT_COLOR = "black"
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
        # TODO: allow smaller palette?
        palette.update(**items)
        palette = {x: palette[x] for x in categories}
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


def _get_figure_handles(ax, figsize=(8, 4)):
    """Get the current figure and show_axes handles, or create new ones if ax is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.gcf()
    return fig, ax


def _format_image_axes(ax, show_axes=False, set_lims="whole", custom_xy_lims=None):
    """Format the show_axes of a plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to format
    show_axes : bool
        Whether to display the show_axes and spines
    set_lims : bool or str, {'whole', 'left_hemi', 'right_hemi'}
        Whether to set the x and y limits of the plot to the whole brain,
        the left hemisphere, or the right hemisphere
    custom_xy_lims : list of float, [xmin, xmax, ymin, ymax]
        Custom x and y limits for the plot, supercedes defaults from set_lims
    """
    ax.axis("image")
    if not show_axes:
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
    if custom_xy_lims is not None:
        if len(custom_xy_lims) == 4:
            ax.set_xlim(custom_xy_lims[:2])
            ax.set_ylim(custom_xy_lims[2:])
        else:
            print(
                "incorrect custom_xy_lims detected, must be [x_min,x_max,y_min,y_max]"
            )
