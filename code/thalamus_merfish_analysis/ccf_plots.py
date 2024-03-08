from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from colorcet import glasbey
from shapely.plotting import plot_polygon
from matplotlib.colors import ListedColormap, to_rgba, to_rgb
from scipy.ndimage import binary_dilation

# TODO: get rid of this import
# from ccf_polygons import get_outline_polygon, CCF_TH_NAMES
from .abc_load import get_thalamus_names, get_ccf_index
from . import ccf_images as cci

def plot_shape(shape: shapely.Polygon | shapely.GeometryCollection, edgecolor='black', **kwargs):    
    """Plot shapely geometry, 
    wrapping shapely.plotting.plot_polygon to accept either a Polygon or GeometryCollection.
    
    All **kwargs are passed through to plot_polygon, then to matplotlib.Patch
    """
    
    if type(shape) is shapely.GeometryCollection:
        for subpoly in shape.geoms:
            patch = plot_polygon(subpoly, add_points=False, edgecolor=edgecolor, **kwargs)
    else:
        patch = plot_polygon(shape, add_points=False, edgecolor=edgecolor, **kwargs)
    return shape

def expand_palette(palette, ccf_names):
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

def plot_ccf_section(ccf_polygons, section, highlight=[], palette=None, 
                     labels=True, bg_shapes=True, ax=None):
    ccf_names = ccf_polygons.index.get_level_values('name')
    palette, edgecolor, alpha = expand_palette(palette, ccf_names)
    if highlight=='all':
        highlight = ccf_names
    if palette is None: 
        palette = {}
    patches = []
    # could simplify this to single loop over polygons
    if bg_shapes:
        for name in ccf_names:
            if section in ccf_polygons.loc[name].index and name not in highlight:
                patches.append(plot_shape(ccf_polygons.loc[(name, section), "geometry"], 
                                          facecolor=palette.get(name), ax=ax, 
                                alpha=0.1, label=name if labels else None))
    for name in highlight:
        if section in ccf_polygons.loc[name].index:
            patches.append(plot_shape(ccf_polygons.loc[(name, section), "geometry"],
                                          facecolor=palette.get(name), ax=ax, 
                           alpha=alpha, edgecolor=edgecolor, label=name if labels else None))
    return patches

def generate_palette(names, palette_to_match=None):
    sns_palette = sns.color_palette(glasbey, n_colors=len(names))
    if palette_to_match is not None:
        point_palette = palette_to_match.copy()
        extra_names = names.difference(palette_to_match.keys())
        extra_palette = dict(zip(extra_names, sns_palette[-len(extra_names):]))
        point_palette.update(extra_palette)
    else:
        point_palette = dict(zip(names, sns_palette))
    return point_palette


def plot_ccf_overlay(obs, ccf_polygons, sections=None, ccf_names=None, 
                     point_hue='CCF_acronym', legend='cells', 
                     min_group_count=10, min_section_count=20, 
                     highlight=[], shape_palette=None, 
                     point_palette=None, bg_cells=None, bg_shapes=True, s=2,
                     axes=False, section_col='section', x_col='cirro_x', 
                     y_col='cirro_y', categorical=True, ccf_level='structure',
                     boundary_img=None, custom_xy_lims=[]):
    obs = obs.copy()
    # Set variables not specified by user
    if sections is None:
        sections = sorted(obs[section_col].unique())
    if ccf_names is None:
        ccf_names = get_thalamus_names()
    if shape_palette is None:
        shape_palette = generate_palette(ccf_names)
    
    # Determine if we have rasterized CCF volumes or polygons-from-cells
    raster_regions = type(ccf_polygons) is np.ndarray
    if raster_regions & isinstance(sections[0], str):
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
            point_palette = generate_palette(point_group_names, palette_to_match=None)
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
        if raster_regions:
            plot_ccf_section_raster(ccf_polygons, section, boundary_img=boundary_img,
                                    ccf_region_names=ccf_names, palette=shape_palette, 
                                    structure_index=get_ccf_index(level=ccf_level),
                                    legend=(legend=='ccf'), ax=ax)
        else:
            plot_ccf_section(ccf_polygons, section, highlight=highlight, 
                             palette=shape_palette, bg_shapes=bg_shapes,
                             labels=legend in ['ccf', 'both'], ax=ax)

        # display background cells in grey
        if bg_cells is not None:
            bg_s = s*0.8 if (s<=2) else 2
            if custom_xy_lims!=[]:
                bg_cells = filter_by_xy_lims(bg_cells, x_col, y_col, custom_xy_lims)
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
                secdata = filter_by_xy_lims(secdata, x_col, y_col, custom_xy_lims)
            sns.scatterplot(secdata, x=x_col, y=y_col, hue=point_hue,
                            s=s, palette=point_palette, linewidth=0,
                            legend=(legend in ['cells', 'both']))
        # plot formatting
        if legend is not None:
            ncols = 4 if (legend=='ccf') else 2 # cell type names require more horizontal space
            plt.legend(ncols=ncols, loc='upper center', bbox_to_anchor=(0.5, 0),
                       frameon=False)
        format_image_axes(ax=ax, axes=axes, custom_xy_lims=custom_xy_lims)
        plt.show()
        figs.append(fig)
    return figs

def format_image_axes(ax, axes=False, set_lims='whole', custom_xy_lims=[]):
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
            
def filter_by_xy_lims(data, x_col, y_col, custom_xy_lims):
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

        
def plot_section_outline(outline_polygons, sections, axes=False, 
                         facecolor='none', edgecolor='black', alpha=0.05):
    ''' Displays the per-section outline_polygons from get_outline_polygon() for
    the specified sections'''
    if isinstance(sections, str):
        sections = [sections]
    
    for section in sections:
        plot_shape(outline_polygons[section],facecolor=facecolor,
                   edgecolor=edgecolor,alpha=alpha)
        if not axes:
            plt.gca().set_aspect('equal')
            plt.box(False)
            plt.xticks([])
            plt.yticks([])
            
            
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
        fig = plot_expression_ccf(adata, gene, ccf_polygons, 
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
    # determine if we have rasterized CCF volumes or polygons-from-cells
    raster_regions = type(ccf_polygons) is np.ndarray
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
        if raster_regions:
            plot_ccf_section_raster(ccf_polygons, section_z, palette='dark_outline',
                                    ccf_region_names=nuclei, boundary_img=boundary_img, ax=ax, **kwargs)
        else:
            plot_ccf_section(ccf_polygons, section, 
                                highlight=nuclei, 
                                bg_shapes=bg_shapes, 
                                ax=ax, palette='dark_outline', **kwargs)
    elif highlight!=[]:
        if raster_regions:
            plot_ccf_section_raster(ccf_polygons, section_z, palette='light_outline',
                                    ccf_region_names=nuclei, ax=ax, **kwargs)
            plot_ccf_section_raster(ccf_polygons, section_z, palette='dark_outline',
                                    ccf_region_names=highlight, ax=ax, **kwargs)
        else:
            plot_ccf_section(ccf_polygons, section, highlight=nuclei, 
                                palette='light_outline', bg_shapes=bg_shapes, 
                                ax=ax, **kwargs)
            plot_ccf_section(ccf_polygons, section, highlight=highlight, 
                                palette='dark_outline', bg_shapes=bg_shapes, 
                                ax=ax, **kwargs)
    
    # if you rely solely on set_xlim/ylim, the data is just masked but is
    # still actually present in the pdf savefig
    if custom_xy_lims!=[]:
        sec_obs = filter_by_xy_lims(sec_obs, x_col, y_col, 
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
    format_image_axes(ax=ax, custom_xy_lims=custom_xy_lims)
    return fig


def get_colormap_color(value, cmap='viridis', vmin=0, vmax=1):
    # norm = plt.Normalize(vmin, vmax)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps.get_cmap(cmap)  # PiYG
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    color = matplotlib.colors.rgb2hex(rgb)
    return color

def plot_metrics_ccf(obs, ccf_polygons, metric_series, sections=None, 
                     cmap='viridis', cb_label='metric',
                     highlight=[], legend='cells', bg_shapes=True, s=2, axes=False):
    obs = obs.copy()
    if sections is None:
        sections = obs['section'].unique()
    else:
        ccf_polygons = ccf_polygons[ccf_polygons.index.isin(sections, level="section")]
    ccf_names = ccf_polygons.index.get_level_values('name')
    
    # convert metric to color palette
    vmin = np.min(metric_series.values)
    vmax = np.max(metric_series.values)
    metric_colors = [get_colormap_color(value, cmap=cmap, vmin=vmin, vmax=vmax) 
                     for (name, value) in pd.Series.items(metric_series)]
    shape_palette = dict(zip(metric_series.index, metric_colors))
    
    for section in sections:
        print(section)
        fig, ax = plt.subplots(figsize=(8,4))
        
        patches = plot_ccf_section(ccf_polygons, section, highlight=highlight, palette=shape_palette, bg_shapes=bg_shapes,
                                   labels=legend in ['ccf', 'both'], ax=ax)
        # if legend:
        #     plt.legend(ncols=2, loc='upper center', bbox_to_anchor=(0.5, 0))
        #     # plt.legend(ncols=1, loc='center left', bbox_to_anchor=(0.98, 0.5), fontsize=16)
        
        format_image_axes(ax=ax, axes=axes)
        # hidden image just to generate colorbar
        img = ax.imshow(np.array([[vmin,vmax]]), cmap=cmap)
        img.set_visible(False)
        plt.colorbar(img, orientation='vertical', label=cb_label, shrink=0.75)
        plt.show()

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
    section_region_names = structure_index[region_nums]
    if (ccf_region_names is None) or ((isinstance(ccf_region_names, str)) 
                                      and (ccf_region_names=='all')):
        ccf_region_names = list(set(section_region_names).intersection(get_thalamus_names()))
    else:
        ccf_region_names = list(set(section_region_names).intersection(ccf_region_names))

    palette, edgecolor, alpha = expand_palette(palette, ccf_region_names)
    
    regions = ccf_region_names if palette is None else [x for x in ccf_region_names if x in palette]
        
    plot_raster_all(img, structure_index, boundary_img=boundary_img, palette=palette, regions=regions,
                            edgecolor=edgecolor, alpha=alpha, ax=ax)
    if legend and palette is not None:
        # generate "empty" matplotlib handles to be used by plt.legend() call in 
        # plot_ccf_overlay() (NB: that supercedes any call to plt.legend here)
        handles = [plt.plot([], marker="o", ls="", color=palette[name], label=name)[0] 
                   for name in regions]
    return

def fill_nan(img):
    return np.where(img, 1, np.nan)

def palette_to_rgba_lookup(palette, index):
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
    
def plot_raster_all(imdata, index, boundary_img=None, palette=None, regions=None, resolution=10e-3,
                       edgecolor='black', edge_width=1, alpha=1, ax=None):
    # TODO: move index logic and boundary_img creation out to plot_ccf_section_raster, pass rgba lookups
    extent = resolution * (np.array([0, imdata.shape[0], imdata.shape[1], 0]) - 0.5)
    kwargs = dict(extent=extent, interpolation="none", alpha=alpha)
    
    if palette is not None:
        if regions:
            palette = {x: y for x, y in palette.items() if x in regions}
        rgba_lookup = palette_to_rgba_lookup(palette, index)
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
            rgba_lookup = palette_to_rgba_lookup(edge_palette, index)
            im_edges = rgba_lookup[boundary_img, 3] != 0
        else:
            im_edges = boundary_img
        
        im_edges = np.where(im_edges[:,:,None]!=0, np.array(to_rgba(edgecolor), ndmin=3), 
                            np.zeros((1,1,4)))
        ax.imshow(im_edges, **kwargs)
    
    
def plot_raster_region(imdata, region_val, resolution=10e-3, facecolor='grey', 
                       edgecolor='black', edge_width=2, alpha=1, ax=None):
    extent = (np.array([0, imdata.shape[0], imdata.shape[1], 0]) - 0.5) * resolution
    kwargs = dict(extent=extent, interpolation="none", alpha=alpha)
    im_region = (imdata==region_val)
    # transpose for x,y oriented image
    ax.imshow(fill_nan(im_region), cmap=ListedColormap([facecolor]), **kwargs)
    im_bound = binary_dilation(im_region, iterations=edge_width) & ~im_region
    ax.imshow(fill_nan(im_bound), cmap=ListedColormap([edgecolor]), **kwargs)

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
        format_image_axes(ax=ax, axes=axes)
        figs.append(fig)
    return figs