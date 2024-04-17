from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from colorcet import glasbey
from matplotlib.colors import ListedColormap, to_rgba, to_rgb
from scipy.ndimage import binary_dilation

from .abc_load import get_thalamus_names, get_ccf_index, get_color_dictionary, CURRENT_VERSION
from . import ccf_images as cci


# ------------------------- Multi-Section Plotting ------------------------- #
def plot_ccf_overlay(obs, ccf_images, sections=None, 
                     point_hue='CCF_acronym', point_palette=None, # controls foreground cells
                     categorical=True, s=2, min_group_count=10,
                     bg_cells=None, # controls background cells
                     ccf_names=None, ccf_highlight=[], ccf_level='substructure', # controls CCF regions
                     boundary_img=None, face_palette=None, edge_color='grey',
                     section_col='section', min_section_count=20, # sections
                     x_col='cirro_x', y_col='cirro_y', custom_xy_lims=[], # xy coords
                     axes=False, legend='cells' # plot formatting
                     # highlight=[], TODO: re-implement highlight for plot_ccf_overlay
                     ):
    ''' 
    Parameters
    ----------
    obs : pd.DataFrame
        DataFrame of cells to display
    ccf_images : np.ndarray
        3D array of CCF parcellation regions
    sections : list of numbers, optional
        Section(s) to display. Must be a list even for single section. 
        If None, all sections in obs[section_col].unique() are displayed
    point_hue : str
        Column name in obs to color cells by
    point_palette : dict, optional
        Dictionary of point_hue categories and colors
    categorical : bool
        Whether point_hue is a categorical variable
    s : int
        Size of foreground cell scatter points (background cells set to 0.8*s)
    min_group_count : int
        Minimum number of cells in a group to be displayed
    bg_cells : pd.DataFrame, optional
        DataFrame of background cells to display
    ccf_names : list of str
        List of CCF region names to display
    ccf_highlight : list of str
        List of CCF region names to highlight with a darker outline for ccf_names
    ccf_level : str, {'substructure', 'structure'}
        Level of CCF to be displayed
    boundary_img : np.ndarray, optional
        3D array of CCF parcellation boundaries; if None, calculated on the fly
    face_palette : {None, dict, 'glasbey'}
        Sets face color of CCF region shapes in plot_ccf_section(), see that 
        function's docstring for more details
    edge_color : str
        Sets outline/boundary color of CCF region shapes in plot_ccf_section()
    section_col : str, {'z_section', 'z_reconstructed', 'z_realigned'}
        Column name in obs for the section numbers in 'sections'; must be a col 
        that allows for indexing into ccf_images
    min_section_count : int
        Minimum number of cells in a section to be displayed
    x_col, y_col : str
        Column names in obs for the x and y coordinates of cells
    custom_xy_lims : list of float, [xmin, xmax, ymin, ymax]
        Custom x and y limits for the plot
    axes : bool
        Whether to display the axes and spines
    legend : str, {'ccf', 'cells', 'both', None}
        Whether to display a legend for the CCF region shapes, the cell types,
        both, or neither
    '''
    obs = obs.copy()
    # Set variables not specified by user
    # TODO: update logic to specify all brain regions vs just thalamus regions
    if ccf_names is None:
        ccf_names = get_thalamus_names(level=ccf_level)
    if sections is None:
        sections = sorted(obs[section_col].unique())

    if isinstance(sections[0], str):
        raise Exception('str type detected for ''sections''. You must use ''z_section'' OR ''z_reconstructed'' as your ''section_col'' in order to plot the rasterized CCF volumes.')
        
    # Clean up point hue column    
    # string type (not categorical) allows adding 'other' to data slice by slice
    if categorical:
        obs[point_hue] = obs[point_hue].astype(str)
        # drop groups below min_group_count and sort by size
        # currently across all of obs, not just selected sections
        point_group_names = obs[point_hue].value_counts().loc[lambda x: x>min_group_count].index
        obs = obs[obs[point_hue].isin(point_group_names)]
        
        # Set color palette for cell scatter points
        if point_palette is None:
            # if point_hue is a taxonomy level, no need to pass in pre-generated
            # palette as a parameter, just calculate it on the fly
            if point_hue in ['class','subclass','supertype','cluster']:
                hue_categories = obs[point_hue].unique().tolist() #.cat.categories.tolist()
                point_palette = get_color_dictionary(hue_categories, 
                                                     point_hue, 
                                                     version=CURRENT_VERSION)
            else:
                point_palette = _generate_palette(point_group_names)
        else:
            point_palette = point_palette.copy()
        point_palette.update(other='grey')
    
    if bg_cells is not None:
        bg_cells = bg_cells.loc[bg_cells.index.difference(obs.index)]
    
    # Display each section as a separate plot
    figs = []
    for section in sections:
        secdata = obs.loc[lambda df: (df[section_col]==section)].copy()
        if len(secdata) < min_section_count:
            continue
        fig, ax = plt.subplots(figsize=(8,4))
        ax.set_title('z='+str(section)+'\n'+point_hue)
        
        # display CCF shapes
        plot_ccf_section(ccf_images, section, boundary_img=boundary_img,
                         ccf_region_names=ccf_names, 
                         structure_index=get_ccf_index(level=ccf_level), 
                         face_palette=face_palette, edge_color=edge_color,
                         legend=(legend=='ccf'), ax=ax)
        if ccf_highlight!=[]:
            plot_ccf_section(ccf_images, section, boundary_img=boundary_img,
                             ccf_region_names=ccf_highlight, 
                             structure_index=get_ccf_index(level=ccf_level), 
                             face_palette=None, edge_color=EDGE_COLOR_HIGHLIGHT,
                             ax=ax)

        # display background cells in grey
        if bg_cells is not None:
            bg_s = s*0.8 if (s<=2) else 2
            if custom_xy_lims!=[]:
                bg_cells = _filter_by_xy_lims(bg_cells, x_col, y_col, custom_xy_lims)
            sns.scatterplot(bg_cells.loc[lambda df: (df[section_col]==section)], 
                            x=x_col, y=y_col, c='grey', s=bg_s, alpha=0.5, 
                            linewidth=0)
        # lump small groups if legend list is too long
        if categorical:
            sec_group_counts = secdata[point_hue].value_counts(ascending=True)
            if len(sec_group_counts) > 10:
                point_groups_section = sec_group_counts.loc[lambda x: x>min_group_count].index
                secdata.loc[lambda df: ~df[point_hue].isin(point_groups_section), point_hue] = 'other'
            secdata[point_hue] = pd.Categorical(secdata[point_hue])
        # display foreground cells according to point_hue
        if len(secdata) > 0:
            if custom_xy_lims!=[]:
                secdata = _filter_by_xy_lims(secdata, x_col, y_col, custom_xy_lims)
            sns.scatterplot(secdata, x=x_col, y=y_col, hue=point_hue,
                            s=s, palette=point_palette, linewidth=0,
                            legend=(legend in ['cells', 'both']))
        # plot formatting
        if legend is not None:
            ncols = 4 if (legend=='ccf') else 2 # cell type names require more horizontal space
            plt.legend(ncols=ncols, loc='upper center', bbox_to_anchor=(0.5, 0),
                       frameon=False)
        _format_image_axes(ax=ax, axes=axes, custom_xy_lims=custom_xy_lims)
        plt.show()
        figs.append(fig)
    return figs
            
            
def plot_nucleus_cluster_comparison_slices(obs, ccf_images, nuclei, 
                                           bg_cells=None, 
                                           legend='cells', section_col='section',
                                           x_col='cirro_x', y_col='cirro_y', **kwargs):
    sections_points = obs[section_col].value_counts().loc[lambda x: x>10].index
    nuclei = [nuclei] if type(nuclei) is str else nuclei
    if type(ccf_images) is np.ndarray:
        sections_nuclei = sections_points
    else:
        sections_nuclei = ccf_images.index.get_level_values('name')[
                                ccf_images.index.isin(nuclei, level='name')
                                ].unique()
    sections = sorted(sections_nuclei.union(sections_points))
    # ABC dataset uses 'cluster', internal datasets used 'cluster_label'
    if 'cluster' in obs.columns:
        hue_column = 'cluster'
    else:
        hue_column = 'cluster_label'
    plot_ccf_overlay(obs, ccf_images, sections, point_hue=hue_column, 
                     legend=legend, ccf_names=nuclei, bg_cells=bg_cells, 
                     section_col=section_col, 
                     x_col=x_col, y_col=y_col, **kwargs)   


def plot_expression_ccf(adata, gene, ccf_images, 
                        sections=None, nuclei=None, highlight=[], 
                        s=0.5, cmap='Blues', show_outline=False, 
                        axes=False, edge_color='lightgrey',  
                        section_col='section', x_col='cirro_x',y_col='cirro_y',
                        boundary_img=None, custom_xy_lims=[], 
                        cb_vmin_vmax=[None,None],
                        **kwargs):
    # set variables not specified by user
    if sections is None:
        sections = adata.obs[section_col].unique()
    # Plot
    figs = []
    for section in sections:
        fig = plot_expression_ccf_section(adata, gene, ccf_images, 
                        section, nuclei=nuclei, highlight=highlight, 
                        s=s, cmap=cmap, show_outline=show_outline, 
                        axes=axes, edge_color=edge_color,
                        section_col=section_col, x_col=x_col, y_col=y_col,
                        boundary_img=boundary_img, custom_xy_lims=custom_xy_lims,
                        cb_vmin_vmax=cb_vmin_vmax)
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


def plot_metrics_ccf(ccf_img, metric_series, sections, 
                            structure_index=None,
                            cmap='viridis', cb_label='metric', 
                            vmin=None, vmax=None,
                            axes=False):
    if structure_index is None:
        structure_index = get_ccf_index()
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
        plt.colorbar(img, orientation='vertical', label=cb_label, shrink=0.75)
        
        plot_ccf_section(ccf_img, section_z, palette, 
                                structure_index=structure_index, legend=False, ax=ax)
        _format_image_axes(ax=ax, axes=axes)
        figs.append(fig)
    return figs


# ----------------------- Single-Section Plot Elements ----------------------- #

def plot_expression_ccf_section(adata_or_obs, gene, ccf_images, 
                        section, nuclei=None, highlight=[], 
                        s=0.5, cmap='Blues', show_outline=False, 
                        axes=False,  edge_color='lightgrey',
                        section_col='section', x_col='cirro_x', y_col='cirro_y',
                        boundary_img=None, custom_xy_lims=[], 
                        cb_vmin_vmax=[None,None],
                        label=None, colorbar=True, ax=None,
                        **kwargs):
    if nuclei is None:
        nuclei = get_thalamus_names()

    is_obs_df = type(adata_or_obs) is pd.DataFrame
    obs = adata_or_obs if is_obs_df else adata_or_obs.obs
    # need to parse both string & num sections so can't use query()
    sec_obs = obs[obs[section_col]==section] 
    section_z = sec_obs['z_section'].iloc[0]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,4))
    else:
        plt.sca(ax)
        fig = plt.gcf()

    # Plot ccf annotation in front of gene expression
    plot_ccf_section(ccf_images, section_z, ccf_region_names=nuclei, 
                     face_palette=None, edge_color=edge_color,
                     boundary_img=boundary_img, ax=ax, **kwargs)
    if highlight!=[]:
        plot_ccf_section(ccf_images, section_z, ccf_region_names=highlight, 
                         face_palette=None, edge_color=EDGE_COLOR_HIGHLIGHT, 
                         ax=ax, **kwargs)
    
    # if you rely solely on set_xlim/ylim, the data is just masked but is
    # still actually present in the pdf savefig
    if custom_xy_lims!=[]:
        sec_obs = _filter_by_xy_lims(sec_obs, x_col, y_col, 
                                        custom_xy_lims)
    # Plot gene expression
    if is_obs_df:
        c = sec_obs[gene].values
    else:
        c = adata_or_obs[sec_obs.index, gene].X.toarray().squeeze()
    sc = ax.scatter(x=sec_obs[x_col], y=sec_obs[y_col], c=c, 
                    s=s, cmap=cmap, vmin=cb_vmin_vmax[0], vmax=cb_vmin_vmax[1], 
                    zorder=-1) # force sc to very bottom of plot
    if colorbar:
        if label is None:
            # if adata from load_adata(), counts_transform is recorded in .uns
            if hasattr(adata_or_obs, 'uns') & ('counts_transform' in adata_or_obs.uns):
                label = 'gene counts ('+adata_or_obs.uns['counts_transform']+')'
            # if we don't have .uns['counts_transform'], check if we have raw counts or not
            else:
                if all(i.is_integer() for i in c):  # no [] around loop == stops at 1st non-integer encounter
                    label = 'gene counts (raw)'
                else:
                    label = 'gene counts (unknown transform)'
        plt.colorbar(sc, label=label, fraction=0.046, pad=0.01)
    
    ax.set_title(gene)
    _format_image_axes(ax=ax, custom_xy_lims=custom_xy_lims)
    return fig


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
    colorR = np.concatenate((gene1_norm, np.zeros([gene1_norm.shape[0],2])),axis=1)
    colorG = np.concatenate((np.zeros([gene1_norm.shape[0],1]),gene2_norm,np.zeros([gene1_norm.shape[0],1])),axis=1)
    colorB = np.concatenate((np.zeros([gene1_norm.shape[0],2]),gene3_norm),axis=1)
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


def plot_ccf_section(ccf_img, section_z, ccf_region_names=None,
                     face_palette=None, edge_color='grey',
                     boundary_img=None, structure_index=None, 
                     z_resolution=200e-3, legend=True, ax=None):
    ''' Display CCF parcellations for a single section
    Parameters
    ----------
    ccf_img : np.ndarray
        3D array of CCF parcellations
    section_z : int
        Section number
    ccf_region_names : list of str
        List of CCF region names to display
    face_palette : {None, dict, 'glasbey'}
        Sets face color of CCF region shapes; 
        None to have no face color, or a dictionary of CCF region names and 
        colors, or 'glasbey' to generate a color palette from the 
        colorcet.glasbey color map
    edge_color : str
        Sets outline/boundary color of all CCF region shapes; any valid 
        matplotlib color
    boundary_img : np.ndarray, optional
        2D array of CCF parcellation boundaries; if None, calculated on the fly
    structure_index : pd.Series
        Series of CCF region names and their corresponding IDs
    z_resolution : float
        Resolution of the CCF in the z-dimension
    legend : bool
        Whether to display a legend for the CCF region shapes
    ax : matplotlib.axes.Axes, optional
        Existing Axes to plot on; if None, a new figure and axes are created
    '''
    # subset to just this section
    index_z = int(np.rint(section_z/z_resolution))
    img = ccf_img[:,:, index_z].T
    boundary_img = boundary_img[:,:, index_z].T if boundary_img is not None else None
    
    # selection CCF regions to plot
    if structure_index is None:
        structure_index = get_ccf_index()
    region_nums = np.unique(img)
    section_region_names = [structure_index[i] for i in region_nums if i in structure_index.index]
    if (ccf_region_names is None) or ((isinstance(ccf_region_names, str)) 
                                      and (ccf_region_names=='all')):
        ccf_region_names = list(set(section_region_names).intersection(get_thalamus_names()))
    else:
        ccf_region_names = list(set(section_region_names).intersection(ccf_region_names))

    if face_palette=='glasbey':
        face_palette = _generate_palette(ccf_region_names)
    
    regions = ccf_region_names if face_palette is None else [x for x in ccf_region_names 
                                                             if x in face_palette
                                                            ]
    
    # plot CCF shapes
    plot_ccf_shapes(img, structure_index, boundary_img=boundary_img, 
                    regions=regions, face_palette=face_palette, 
                    edge_color=edge_color, ax=ax)
    
    # generate "empty" matplotlib handles to be used by plt.legend() call in 
    # plot_ccf_overlay() (NB: that supercedes any call to plt.legend here)
    if legend and (face_palette is not None):
        handles = [plt.plot([], marker="o", ls="", color=face_palette[name], label=name)[0] 
                   for name in regions]

    
def plot_ccf_shapes(imdata, index, boundary_img=None, regions=None, 
                    face_palette=None, edge_color='black', edge_width=1,  
                    alpha=1, ax=None, resolution=10e-3):
    ''' Plot face & boundary for CCF region shapes specified

    Parameters
    ----------
    imdata : np.ndarray
        2D array of CCF parcellations
    index : pd.Series
        Series of CCF region names and their corresponding IDs
    boundary_img : np.ndarray, optional
        2D array of CCF parcellation boundaries; if None, calculated on the fly
    regions : list of str, optional
        List of CCF region names to display; if None, ??????
    face_palette : dict, optional
        Dictionary of CCF region names and colors
    edge_color : str, optional
        Sets outline/boundary color of all CCF region shapes; any valid
        matplotlib color
    edge_width : int, optional
        Width of the CCF region shape outlines
    alpha : float, optional
        Opacity of the CCF region shapes' face and edge colors
    ax : matplotlib.axes.Axes, optional
        Existing Axes to plot on; if None, a new figure and axes are created
    resolution : float, optional
        Resolution of the CCF in the z-dimension
    '''
    # TODO: move index logic and boundary_img creation out to plot_ccf_section, pass rgba lookups
    extent = resolution * (np.array([0, imdata.shape[0], imdata.shape[1], 0]) - 0.5)
    kwargs = dict(extent=extent, interpolation="none", alpha=alpha)
    
    # Plot face shapes of CCF regions
    if face_palette is not None:
        if regions:
            face_palette = {x: y for x, y in face_palette.items() if x in regions}
        rgba_lookup = _palette_to_rgba_lookup(face_palette, index)
        im_regions = rgba_lookup[imdata, :]
        ax.imshow(im_regions, **kwargs)
    
    # Plot boundaries of CCF regions    
    if edge_color is not None:
        if boundary_img is None:
            boundary_img = cci.label_erosion(imdata, edge_width, fill_val=0, 
                                             return_edges=True)
            
        # filter edges by face_palette or CCF regions if present
        if face_palette is not None:
            im_edges = rgba_lookup[boundary_img, 3] != 0
        elif regions is not None:
            edge_palette = {x: 'k' for x in regions}
            rgba_lookup = _palette_to_rgba_lookup(edge_palette, index)
            im_edges = rgba_lookup[boundary_img, 3] != 0
        else:
            im_edges = boundary_img
        
        im_edges = np.where(im_edges[:,:,None]!=0, np.array(to_rgba(edge_color), ndmin=3), 
                            np.zeros((1,1,4)))
        ax.imshow(im_edges, **kwargs)


# ------------------------- DataFrame Preprocessing ------------------------- #

def _filter_by_xy_lims(data, x_col, y_col, custom_xy_lims):
    ''' Filter a DataFrame by custom x and y limits.
    
    Need to explicitly filter the DataFrame. Can't rely only on set_xlim/ylim,
    because that only masks out the data points but still plots them, so they
    are present & taking up space in a savefig() pdf output (i.e. generates  
    large files that are laggy in Illustrator due to having many indiv elements).
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to filter
    x_col, y_col : str
        Column names in data for the x and y coordinates of cells
    custom_xy_lims : list of float, [xmin, xmax, ymin, ymax]
        Custom x and y limits for the plot
    '''
    # need to detect min & max because of how the abc atlas coordinates are 
    # oriented, i.e. abs(ymin) > abs(ymax) but ymin=bottom & ymax=top
    xmin = min(custom_xy_lims[:2])
    xmax = max(custom_xy_lims[:2])
    ymin = min(custom_xy_lims[2:])
    ymax = max(custom_xy_lims[2:])
    
    # can handle both adata or obs
    if hasattr(data, 'obs'):
        adata = data
        subset = ((adata.obs[x_col] >= xmin) & (adata.obs[x_col] <= xmax) &
                  (adata.obs[y_col] >= ymin) & (adata.obs[y_col] <= ymax)
                 )
    else:
        obs = data
        subset = ((obs[x_col] >= xmin) & (obs[x_col] <= xmax) &
                  (obs[y_col] >= ymin) & (obs[y_col] <= ymax)
                 )
        
    return data[subset]


# ------------------------- Color Palette Handling ------------------------- #

# Pre-set edge_colors for common situations
EDGE_COLOR_HIGHLIGHT = 'black'

def _generate_palette(ccf_names):
    ''' Generate a color palette dict for a given list of CCF regions.
    
    Parameters
    ----------
    ccf_names : list of str
        List of CCF region names
        
    Returns
    -------
    palette_dict : dict of (str, RGB tuples)
        Dictionary of CCF region names and colors
    '''
    sns_palette = sns.color_palette(glasbey, n_colors=len(ccf_names))
    palette_dict = dict(zip(ccf_names, sns_palette))
    
    return palette_dict


def _palette_to_rgba_lookup(palette, index):
    ''' Convert a color palette dict to an RGBA lookup table.
    
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
    '''
    # rgba_lookup = index.map(lambda x: to_rgb(palette[x]))
    # rgba_lookup = rgba_lookup.reindex(range(max_val), fill_value=0)
    max_val = np.max(index.index)
    rgba_lookup = np.zeros((max_val, 4))
    # fill only values in index and also in palette
    # rest remain transparent (alpha=0)
    for i in index.index:
        name = index[i]
        if name in palette:
            rgba_lookup[i,:] = to_rgba(palette[name])
    rgba_lookup[0,:] = [1, 1, 1, 0]
    return rgba_lookup


# ----------------------------- Plot Formatting ----------------------------- #

def _format_image_axes(ax, axes=False, set_lims='whole', custom_xy_lims=[]):
    ''' Format the axes of a plot.
    
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
        Custom x and y limits for the plot
    '''
    ax.axis('image')
    if not axes:
        sns.despine(left=True, bottom=True)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    # (set_lims==True) is for backwards compatibility
    # TODO: allow for easier whole-coronal vs TH+ZI axis formatting
    if (set_lims=='whole') | (set_lims==True):
        ax.set_xlim([2.5, 8.5])
        ax.set_ylim([7, 4])
    elif set_lims=='left_hemi':
        ax.set_xlim([2.5, 6])
        ax.set_ylim([7, 4])
    elif set_lims=='right_hemi':
        ax.set_xlim([5, 8.5])
        ax.set_ylim([7, 4])
    # custom_xy_lims supercedes set_lims
    if custom_xy_lims!=[]:
        if len(custom_xy_lims)==4:
            ax.set_xlim(custom_xy_lims[:2])
            ax.set_ylim(custom_xy_lims[2:])
        else:
            print('incorrect custom_xy_lims detected, must be [x_min,x_max,y_min,y_max]')