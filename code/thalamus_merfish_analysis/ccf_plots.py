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

from .abc_load import get_thalamus_names, get_ccf_index
from . import ccf_images as cci


def plot_ccf_overlay(obs, ccf_polygons, sections=None, ccf_names=None, 
                     point_hue='CCF_acronym', legend='cells', 
                     min_group_count=10, min_section_count=20, 
                     highlight=[], shape_palette=None, 
                     point_palette=None, bg_cells=None, bg_shapes=True, s=2,
                     axes=False, section_col='section', x_col='cirro_x', 
                     y_col='cirro_y', categorical=True, ccf_level='substructure',
                     boundary_img=None, custom_xy_lims=[]):
    obs = obs.copy()
    # Set variables not specified by user
    if sections is None:
        sections = sorted(obs[section_col].unique())
    if ccf_names is None:
        ccf_names = get_thalamus_names(level=ccf_level)
    if shape_palette is None:
        shape_palette = _generate_palette(ccf_names)
    
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
            # make sure point palette matches shape palette
            match_palette = ((point_hue in ['CCF_acronym', 'parcellation_substructure']) 
                and shape_palette not in ('bw','dark_outline','light_outline'))
            point_palette = _generate_palette(point_group_names, palette_to_match=None)
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
        plot_ccf_section_raster(ccf_polygons, section, boundary_img=boundary_img,
                                ccf_region_names=ccf_names, palette=shape_palette, 
                                structure_index=get_ccf_index(level=ccf_level),
                                legend=(legend=='ccf'), ax=ax)

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
            
            
def plot_nucleus_cluster_comparison_slices(obs, ccf_polygons, nuclei, 
                                           bg_cells=None, bg_shapes=True, 
                                           legend='cells', section_col='section',
                                           x_col='cirro_x', y_col='cirro_y', **kwargs):
    sections_points = obs[section_col].value_counts().loc[lambda x: x>10].index
    nuclei = [nuclei] if type(nuclei) is str else nuclei
    if type(ccf_polygons) is np.ndarray:
        sections_nuclei = sections_points
    else:
        sections_nuclei = ccf_polygons.index.get_level_values('name')[
                                ccf_polygons.index.isin(nuclei, level='name')
                                ].unique()
    sections = sorted(sections_nuclei.union(sections_points))
    # ABC dataset uses 'cluster', internal datasets used 'cluster_label'
    if 'cluster' in obs.columns:
        hue_column = 'cluster'
    else:
        hue_column = 'cluster_label'
    plot_ccf_overlay(obs, ccf_polygons, sections, point_hue=hue_column, 
                     legend=legend, ccf_names=nuclei, bg_cells=bg_cells, 
                     bg_shapes=bg_shapes, section_col=section_col, 
                     x_col=x_col, y_col=y_col, **kwargs)   


def plot_expression_ccf(adata, gene, ccf_polygons, 
                        sections=None, nuclei=None, highlight=[], 
                        s=0.5, cmap='Blues', show_outline=False, 
                        bg_shapes=False, axes=False,  
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
        fig = plot_expression_ccf_section(adata, gene, ccf_polygons, 
                        section, nuclei=nuclei, highlight=highlight, 
                        s=s, cmap=cmap, show_outline=show_outline, 
                        bg_shapes=bg_shapes, axes=axes,  
                        section_col=section_col, x_col=x_col, y_col=y_col,
                        boundary_img=boundary_img, custom_xy_lims=custom_xy_lims,
                        cb_vmin_vmax=cb_vmin_vmax)
        figs.append(fig)
    return figs

def plot_expression_ccf_section(adata_or_obs, gene, ccf_polygons, 
                        section, nuclei=None, highlight=[], 
                        s=0.5, cmap='Blues', show_outline=False, 
                        bg_shapes=False, axes=False,  
                        section_col='section', x_col='cirro_x', y_col='cirro_y',
                        boundary_img=None, custom_xy_lims=[], 
                        cb_vmin_vmax=[None,None],
                        label=None, colorbar=True, ax=None,
                        **kwargs):
    if nuclei is None:
        nuclei = get_thalamus_names()

    obs_df = type(adata_or_obs) is pd.DataFrame
    obs = adata_or_obs if obs_df else adata_or_obs.obs
    # need to parse both string & num sections so can't use query()
    sec_obs = obs[obs[section_col]==section] 
    section_z = sec_obs['z_section'].iloc[0]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,4))
    else:
        plt.sca(ax)
        fig = plt.gcf()

    # Plot ccf annotation in front of gene expression
    if highlight==[]:
        plot_ccf_section_raster(ccf_polygons, section_z, palette='dark_outline',
                                ccf_region_names=nuclei, boundary_img=boundary_img, ax=ax, **kwargs)
    elif highlight!=[]:
        plot_ccf_section_raster(ccf_polygons, section_z, palette='light_outline',
                                ccf_region_names=nuclei, ax=ax, **kwargs)
        plot_ccf_section_raster(ccf_polygons, section_z, palette='dark_outline',
                                ccf_region_names=highlight, ax=ax, **kwargs)
    
    # if you rely solely on set_xlim/ylim, the data is just masked but is
    # still actually present in the pdf savefig
    if custom_xy_lims!=[]:
        sec_obs = _filter_by_xy_lims(sec_obs, x_col, y_col, 
                                        custom_xy_lims)
    # Plot gene expression
    if obs_df:
        c = sec_obs[gene].values
    else:
        c = adata_or_obs[sec_obs.index, gene].X.toarray().squeeze()
    sc = ax.scatter(x=sec_obs[x_col], y=sec_obs[y_col], c=c, 
                    s=s, cmap=cmap, vmin=cb_vmin_vmax[0], vmax=cb_vmin_vmax[1], 
                    zorder=-1) # force sc to very bottom of plot
    # are we plotting raw counts or log2p counts?
    if colorbar:
        if label is None:
            if all([i.is_integer() for i in c]):
                label="CPM"
            else:
                label="log2(CPM+1)"
        plt.colorbar(sc, label=label, fraction=0.046, pad=0.01)
    
    ax.set_title(gene)
    _format_image_axes(ax=ax, custom_xy_lims=custom_xy_lims)
    return fig


def plot_ccf_section_raster(ccf_img, section_z, 
                            palette, boundary_img=None,
                            structure_index=None, 
                            ccf_region_names=None, z_resolution=200e-3, legend=True, 
                            ax=None):
    # subset to just this section
    index_z = int(np.rint(section_z/z_resolution))
    img = ccf_img[:,:, index_z].T
    boundary_img = boundary_img[:,:, index_z].T if boundary_img is not None else None
    
    if structure_index is None:
        structure_index = get_ccf_index()
    region_nums = np.unique(img)
    section_region_names = [structure_index[i] for i in region_nums if i in structure_index.index]
    if (ccf_region_names is None) or ((isinstance(ccf_region_names, str)) 
                                      and (ccf_region_names=='all')):
        ccf_region_names = list(set(section_region_names).intersection(get_thalamus_names()))
    else:
        ccf_region_names = list(set(section_region_names).intersection(ccf_region_names))

    palette, edgecolor, alpha = _expand_palette(palette, ccf_region_names)
    
    regions = ccf_region_names if palette is None else [x for x in ccf_region_names if x in palette]
        
    plot_raster_all(img, structure_index, boundary_img=boundary_img, palette=palette, regions=regions,
                            edgecolor=edgecolor, alpha=alpha, ax=ax)
    if legend and palette is not None:
        # generate "empty" matplotlib handles to be used by plt.legend() call in 
        # plot_ccf_overlay() (NB: that supercedes any call to plt.legend here)
        handles = [plt.plot([], marker="o", ls="", color=palette[name], label=name)[0] 
                   for name in regions]
    return

    
def plot_raster_all(imdata, index, boundary_img=None, palette=None, regions=None, resolution=10e-3,
                       edgecolor='black', edge_width=1, alpha=1, ax=None):
    # TODO: move index logic and boundary_img creation out to plot_ccf_section_raster, pass rgba lookups
    extent = resolution * (np.array([0, imdata.shape[0], imdata.shape[1], 0]) - 0.5)
    kwargs = dict(extent=extent, interpolation="none", alpha=alpha)
    
    if palette is not None:
        if regions:
            palette = {x: y for x, y in palette.items() if x in regions}
        rgba_lookup = _palette_to_rgba_lookup(palette, index)
        im_regions = rgba_lookup[imdata, :]
        ax.imshow(im_regions, **kwargs)
        
    if edgecolor is not None:
        if boundary_img is None:
            boundary_img = cci.label_erosion(imdata, edge_width, fill_val=0, return_edges=True)
            
        # filter edges by palette or regions if present
        if palette is not None:
            im_edges = rgba_lookup[boundary_img, 3] != 0
        elif regions is not None:
            edge_palette = {x: 'k' for x in regions}
            rgba_lookup = _palette_to_rgba_lookup(edge_palette, index)
            im_edges = rgba_lookup[boundary_img, 3] != 0
        else:
            im_edges = boundary_img
        
        im_edges = np.where(im_edges[:,:,None]!=0, np.array(to_rgba(edgecolor), ndmin=3), 
                            np.zeros((1,1,4)))
        ax.imshow(im_edges, **kwargs)


def plot_metrics_ccf_raster(ccf_img, metric_series, sections, 
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
        
        plot_ccf_section_raster(ccf_img, section_z, palette, 
                                structure_index=structure_index, legend=False, ax=ax)
        _format_image_axes(ax=ax, axes=axes)
        figs.append(fig)
    return figs


# ------------------------- DataFrame Preprocessing ------------------------- #

def _filter_by_xy_lims(data, x_col, y_col, custom_xy_lims):
    '''custom_xy_lims =[xlim, xlim, ylim, ylim]'''
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

def _expand_palette(palette, ccf_names):
    edgecolor = None
    alpha = 0.6
    if palette is None:
        palette = dict(zip(ccf_names, sns.color_palette(glasbey, n_colors=len(ccf_names))))
    elif palette=='greyscale':
        palette = {x: '#BBBBBB' for x in ccf_names}
        edgecolor = 'grey'
        alpha = 1
    elif palette=='allen_reference_atlas':
        palette = {x: '#FE8084' for x in ccf_names} # lighter==#FF90A0, darker==#FE8084
        edgecolor = 'black'
        alpha = 1
    elif palette=='dark_outline':
        palette = None
        edgecolor = 'grey'
        alpha = 1
    elif palette=='light_outline':
        palette = None
        edgecolor = 'lightgrey'
        alpha = 1
    else:
        edgecolor = 'grey'
        alpha = 1
    return palette, edgecolor, alpha


def _generate_palette(names, palette_to_match=None):
    sns_palette = sns.color_palette(glasbey, n_colors=len(names))
    if palette_to_match is not None:
        point_palette = palette_to_match.copy()
        extra_names = names.difference(palette_to_match.keys())
        extra_palette = dict(zip(extra_names, sns_palette[-len(extra_names):]))
        point_palette.update(extra_palette)
    else:
        point_palette = dict(zip(names, sns_palette))
    return point_palette


def _palette_to_rgba_lookup(palette, index):
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
    ax.axis('image')
    if not axes:
        sns.despine(left=True, bottom=True)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    # (set_lims==True) is for backwards compatibility
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