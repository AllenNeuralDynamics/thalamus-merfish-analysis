from __future__ import annotations

import pandas as pd
import shapely
import seaborn as sns
import matplotlib.pyplot as plt
from colorcet import glasbey
from shapely.plotting import plot_polygon

from ccf_polygons import get_outline_polygon


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

def plot_ccf_section(ccf_polygons, section, highlight=[], palette=None, labels=True, bg_shapes=True, ax=None):
    ccf_names = ccf_polygons.index.get_level_values('name')
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

def plot_ccf_overlay(obs, ccf_polygons, sections=None, point_hue='CCF_acronym', legend='cells', min_group_count=10, highlight=[], 
                     shape_palette=None, point_palette=None, bg_cells=None, bg_shapes=True, s=2, axes=False):
    obs = obs.copy()
    if sections is None:
        sections = obs['section'].unique()
    else:
        ccf_polygons = ccf_polygons[ccf_polygons.index.isin(sections, level="section")]
    ccf_names = ccf_polygons.index.get_level_values('name')
    if shape_palette is None:
        shape_palette = dict(zip(ccf_names, sns.color_palette(glasbey, n_colors=len(ccf_names))))
    
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
            extra_palette = {name: 'grey' for name in extra_names}
            # extra_palette = dict(zip(extra_names, sns.color_palette(glasbey, n_colors=len(point_group_names))[-len(extra_names):]))
            point_palette.update(extra_palette)
        else:
            point_palette = dict(zip(point_group_names, sns.color_palette(glasbey, n_colors=len(point_group_names))))
    else:
        point_palette = point_palette.copy()
    point_palette.update(other='grey')
    
    for section in sections:
        secdata = obs.loc[lambda df: (df['section']==section)].copy()
        if len(secdata) < min_group_count:
            continue
        print(section)
        fig, ax = plt.subplots(figsize=(8,4))
        
        patches = plot_ccf_section(ccf_polygons, section, highlight=highlight, palette=shape_palette, bg_shapes=bg_shapes,
                                   labels=legend in ['ccf', 'both'], ax=ax)
        
        if bg_cells is not None:
            sns.scatterplot(bg_cells.loc[lambda df: (df['section']==section)], x='cirro_x', y='cirro_y', c='grey', s=2, alpha=0.5)
        # lump small groups if legend list is too long
        sec_group_counts = secdata[point_hue].value_counts(ascending=True)
        if len(sec_group_counts) > 10:
            point_groups_section = sec_group_counts.loc[lambda x: x>min_group_count].index
            secdata.loc[lambda df: ~df[point_hue].isin(point_groups_section), point_hue] = 'other'
        secdata[point_hue] = pd.Categorical(secdata[point_hue])
        if len(secdata) > 0:
            sns.scatterplot(secdata, x='cirro_x', y='cirro_y', hue=point_hue, s=s, palette=point_palette, legend=legend in ['cells', 'both'])
        if legend:
            plt.legend(ncols=2, loc='upper center', bbox_to_anchor=(0.5, 0))
            # plt.legend(ncols=1, loc='center left', bbox_to_anchor=(0.98, 0.5), fontsize=16)
        
        plt.axis('image')
        if not axes:
            sns.despine(left=True, bottom=True)
            plt.xticks([])
            plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')
        plt.show()
        
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
            
            
def plot_nucleus_cluster_comparison_slices(obs, ccf_polygons, nuclei, bg_cells=None, bg_shapes=True, legend='cells', **kwargs):
    sections_points = obs['section'].value_counts().loc[lambda x: x>10].index
    nuclei = [nuclei] if type(nuclei) is str else nuclei
    sections_nuclei = ccf_polygons.index.get_level_values('section')[ccf_polygons.index.isin(nuclei, level='name')].unique()
    sections = sorted(sections_nuclei.union(sections_points))
    plot_ccf_overlay(obs, ccf_polygons, sections, point_hue='cluster', legend=legend, 
                     highlight=nuclei, bg_cells=bg_cells, bg_shapes=bg_shapes, **kwargs)   


def plot_expression_ccf(adata_neuronal, section, gene, polygons, nuclei=[], bg_shapes=False, axes=False, 
                        cmap='magma', show_outline=False, highlight=[]):
    subset = adata_neuronal[adata_neuronal.obs.query(f"section=='{section}'").index]
    fig, ax = plt.subplots(figsize=(8,4))
    
    # # plot ccf annotation behind gene expression
    # plot_ccf_section(polygons, section, highlight=nuclei, bg_shapes=bg_shapes, ax=ax, palette='greyscale')
    
    # plot gene expression
    x, y = subset.obs[['cirro_x', 'cirro_y']].values.T
    c = subset[:,gene].X.toarray().squeeze()
    im = plt.scatter(x=x, y=y, c=c, s=1, cmap=cmap)
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