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
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_dilation

# TODO: get rid of this import
# from ccf_polygons import get_outline_polygon, CCF_TH_NAMES
from abc_load import get_thalamus_substructure_names, get_ccf_substructure_index


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
    elif palette=='dark_outline':
        palette = {x: 'none' for x in ccf_names}
        edgecolor = 'grey'
        alpha = 1
    elif palette=='light_outline':
        palette = {x: 'none' for x in ccf_names}
        edgecolor = 'lightgrey'
        alpha = 1
    else:
        edgecolor = 'grey'
        alpha = 1
    return palette, edgecolor, alpha

def plot_ccf_section(ccf_polygons, section, highlight=[], palette=None, labels=True, bg_shapes=True, ax=None):
    ccf_names = ccf_polygons.index.get_level_values('name')
    palette, edgecolor, alpha = expand_palette(palette, ccf_names)
    if highlight=='all':
        highlight = ccf_names
    patches = []
    # could simplify this to single loop over polygons
    if bg_shapes:
        for name in ccf_names:
            if section in ccf_polygons.loc[name].index and name not in highlight:
                patches.append(plot_shape(ccf_polygons.loc[(name, section), "geometry"], facecolor=palette[name], ax=ax, 
                                alpha=0.1, label=name if labels else None))
    for name in highlight:
        if section in ccf_polygons.loc[name].index:
            patches.append(plot_shape(ccf_polygons.loc[(name, section), "geometry"], facecolor=palette[name], ax=ax, 
                           alpha=alpha, edgecolor=edgecolor, label=name if labels else None))
    return patches

def plot_ccf_overlay(obs, ccf_polygons, sections=None, ccf_names=None, 
                     point_hue='CCF_acronym', legend='cells', 
                     min_group_count=10, highlight=[], shape_palette=None, 
                     point_palette=None, bg_cells=None, bg_shapes=True, s=2,
                     axes=False, section_col='section'):
    obs = obs.copy()
    if sections is None:
        sections = obs[section_col].unique()
    # else:
    #     ccf_polygons = ccf_polygons[ccf_polygons.index.isin(sections, level="section")]
    # ccf_names = ccf_polygons.index.get_level_values('name')
    if ccf_names is None:
        ccf_names = get_thalamus_substructure_names()
    if shape_palette is None:
        shape_palette = dict(zip(ccf_names, sns.color_palette(glasbey, n_colors=len(ccf_names))))
    raster_regions = type(ccf_polygons) is np.ndarray
    if raster_regions:
        substructure_index = get_ccf_substructure_index()
    # string type allows adding 'other' to data slice by slice
    obs[point_hue] = obs[point_hue].astype(str)
    # drop groups below min_group_count 
    point_group_names = obs[point_hue].value_counts().loc[lambda x: x>min_group_count].index
    obs = obs[obs[point_hue].isin(point_group_names)]
    
    if point_palette is None:
        if point_hue == 'CCF_acronym' and shape_palette not in ('bw','dark_outline','light_outline'):
            # make sure point palette matches shape palette
            point_palette = shape_palette.copy()
            extra_names = point_group_names.difference(ccf_names)
            extra_palette = dict(zip(extra_names, sns.color_palette(glasbey, n_colors=len(point_group_names))[-len(extra_names):]))
            point_palette.update(extra_palette)
        else:
            point_palette = dict(zip(point_group_names, sns.color_palette(glasbey, n_colors=len(point_group_names))))
    else:
        point_palette = point_palette.copy()
    point_palette.update(other='grey')
    
    for section in sections:
        secdata = obs.loc[lambda df: (df[section_col]==section)].copy()
        if len(secdata) < min_group_count:
            continue
        # print(section)
        fig, ax = plt.subplots(figsize=(8,4))
        if raster_regions:
            plot_ccf_section_raster(ccf_polygons, section, substructure_index, 
                                    regions=ccf_names, palette=shape_palette, 
                                    legend=(legend=='ccf'), ax=ax)
        else:
            plot_ccf_section(ccf_polygons, section, highlight=highlight, 
                             palette=shape_palette, bg_shapes=bg_shapes,
                             labels=legend in ['ccf', 'both'], ax=ax)
        ax.set_title('z='+str(section)+'\n'+point_hue)
        
        if bg_cells is not None:
            sns.scatterplot(bg_cells.loc[lambda df: (df[section_col]==section)], 
                            x='cirro_x', y='cirro_y', c='grey', 
                            s=2, alpha=0.5, linewidth=0)
        # lump small groups if legend list is too long
        sec_group_counts = secdata[point_hue].value_counts(ascending=True)
        if len(sec_group_counts) > 10:
            point_groups_section = sec_group_counts.loc[lambda x: x>min_group_count].index
            secdata.loc[lambda df: ~df[point_hue].isin(point_groups_section), point_hue] = 'other'
        secdata[point_hue] = pd.Categorical(secdata[point_hue])
        if len(secdata) > 0:
            sns.scatterplot(secdata, x='cirro_x', y='cirro_y', hue=point_hue,
                            s=s, palette=point_palette, linewidth=0,
                            legend=legend in ['cells', 'both'])
        if legend:
            plt.legend(ncols=2, loc='upper center', bbox_to_anchor=(0.5, 0),
                       frameon=False)
            # plt.legend(ncols=1, loc='center left', bbox_to_anchor=(0.98, 0.5), fontsize=16)
        format_image_axes(axes)
        plt.show()

def format_image_axes(axes=False, set_lims=True):
    plt.axis('image')
    if not axes:
        sns.despine(left=True, bottom=True)
        plt.xticks([])
        plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    if set_lims:
        plt.gca().set_xlim([2.5, 8.5])
        plt.gca().set_ylim([7, 4])
        
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
            
            
def plot_nucleus_cluster_comparison_slices(obs, ccf_polygons, nuclei, bg_cells=None, bg_shapes=True, legend='cells', section_col='section', **kwargs):
    sections_points = obs[section_col].value_counts().loc[lambda x: x>10].index
    nuclei = [nuclei] if type(nuclei) is str else nuclei
    sections_nuclei = ccf_polygons.index.get_level_values('name')[ccf_polygons.index.isin(nuclei, level='name')].unique()
    sections = sorted(sections_nuclei.union(sections_points))
    # ABC dataset uses 'cluster', internal datasets used 'cluster_label'
    if 'cluster' in obs.columns:
        hue_column = 'cluster'
    else:
        hue_column = 'cluster_label'
    plot_ccf_overlay(obs, ccf_polygons, sections, point_hue=hue_column, legend=legend, 
                     highlight=nuclei, bg_cells=bg_cells, bg_shapes=bg_shapes, section_col=section_col, **kwargs)   


def plot_expression_ccf(adata_neuronal, section, gene, polygons, nuclei=[], bg_shapes=False, axes=False, 
                        cmap='magma', show_outline=False, highlight=[]):
    subset = adata_neuronal[adata_neuronal.obs.query(f"section=='{section}'").index]
    fig, ax = plt.subplots(figsize=(8,4))
    
    # # plot ccf annotation behind gene expression
    # plot_ccf_section(polygons, section, highlight=nuclei, bg_shapes=bg_shapes, ax=ax, palette='greyscale')
    
    # plot gene expression
    # x, y = subset.obsm['spatial_cirro'].T
    c = subset[:,gene].X.toarray().squeeze()
    im = plt.scatter(x=subset.obs['cirro_x'], y=subset.obs['cirro_y'], c=c, 
                     s=1, cmap=cmap)
    plt.colorbar(label="log2(CPM+1)", fraction=0.046, pad=0.01)
    plt.axis('image')
    plt.title(gene)
    if not axes:
        plt.xticks([])
        plt.yticks([])
        plt.box(False)
    plt.xlabel('')
    plt.ylabel('')
    
    # plot TH outline
    if show_outline:
        th_outline_polygons = get_outline_polygon(subset.obs)
        plot_section_outline(th_outline_polygons, sections=section, alpha=0.15)
    
    # plot ccf annotation in front of gene expression
    if highlight==[]:
        plot_ccf_section(polygons, section, highlight=nuclei, bg_shapes=bg_shapes, ax=ax, palette='dark_outline')
    elif highlight!=[]:
        plot_ccf_section(polygons, section, highlight=nuclei, bg_shapes=bg_shapes, ax=ax, palette='light_outline')
        plot_ccf_section(polygons, section, highlight=highlight, bg_shapes=bg_shapes, ax=ax, palette='dark_outline')
    
    plt.gca().set_aspect('equal')
    plt.show()
    
    return fig, ax


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
        
        format_image_axes(axes)
        # hidden image just to generate colorbar
        img = ax.imshow(np.array([[vmin,vmax]]), cmap=cmap)
        img.set_visible(False)
        plt.colorbar(img, orientation='vertical', label=cb_label, shrink=0.75)
        plt.show()

def plot_ccf_section_raster(ccf_img, section_z, structure_index, palette, regions=None, z_resolution=200e-3, legend=True, ax=None):
    index_z = int(np.rint(section_z/z_resolution))
    img = ccf_img[:,:, index_z]
    region_nums = np.unique(img)
    if regions is None:
        regions = [structure_index[i] for i in region_nums]
    palette, edgecolor, alpha = expand_palette(palette, regions)
    # could do in single image, but looping allows selecting highlight set etc...
    
    for i in region_nums:
        name = structure_index[i]
        if name in palette:
            plot_raster_region(img, i, resolution=10e-3, facecolor=palette[name], edgecolor=edgecolor, alpha=alpha, ax=ax)
    if legend:
        handles = [plt.plot([], marker="o", ls="", color=color)[0] for name, color in palette.items() if name in regions]
        plt.legend(regions)
    return

def fill_nan(img):
    return np.where(img, 1, np.nan)

def plot_raster_region(imdata, region_val, resolution=10e-3, facecolor='grey', edgecolor='black', 
                    edge_width=2, alpha=1, ax=None):
    extent = (np.array([0, imdata.shape[1], imdata.shape[0], 0]) - 0.5) * resolution
    kwargs = dict(extent=extent, interpolation="none")
    im_region = imdata==region_val
    # transpose for x,y oriented image
    ax.imshow(fill_nan(im_region).T, cmap=ListedColormap([facecolor]), **kwargs)
    im_bound = binary_dilation(im_region, iterations=edge_width) & ~im_region
    ax.imshow(fill_nan(im_bound).T, cmap=ListedColormap([edgecolor]), **kwargs)

def plot_metrics_ccf_raster(ccf_img, metric_series, sections, structure_index,
                     cmap='viridis', cb_label='metric', axes=False):
    vmin, vmax = (metric_series.min(), metric_series.max())
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps.get_cmap(cmap)
    palette = metric_series.apply(lambda x: cmap(norm(x))).to_dict()
    
    for section_z in sections:
        print(section_z)
        fig, ax = plt.subplots(figsize=(8, 5))
        # hidden image just to generate colorbar
        img = ax.imshow(np.array([[vmin, vmax]]), cmap=cmap)
        img.set_visible(False)
        plt.colorbar(img, orientation='vertical', label=cb_label, shrink=0.75)
        
        plot_ccf_section_raster(ccf_img, section_z, structure_index, palette, legend=False, ax=ax)
        format_image_axes(axes)
        plt.show()