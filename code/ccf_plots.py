import numpy as np
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import shapely
import shapely.plotting as splot

import hdbscan
from collections import defaultdict
from shapely.ops import unary_union
from sklearn.cluster import k_means #, OPTICS, DBSCAN


def get_ccf_th_names():
    '''
    Returns a list of thalamic CCF subregion names.
    '''
    
    ccf_th_names = ['AD', 'AMd', 'AMv', 'AV', 'CL', 'CM', 'Eth', 'FF', 'IAD', 
                    'IAM', 'IGL', 'IMD', 'IntG', 'LD', 'LGd-co', 'LGd-ip', 
                    'LGd-sh', 'LGv', 'LH', 'LP', 'MD', 'MGd', 'MGm', 'MGv', 
                    'MH', 'PCN', 'PF', 'PIL', 'PO', 'POL', 'PP', 'PR', 'PT', 
                    'PVT', 'PoT', 'RE', 'RH', 'RT', 'SGN', 'SMT', 'SPA', 'SPFp',
                    'SubG', 'VAL', 'VM', 'VPL', 'VPLpc', 'VPM', 'VPMpc', 'Xi', 
                    'ZI']
    return ccf_th_names

################################################################################
################################################################################
def poly_from_points(X, min_points=0, allow_holes=False):
    if X.shape[0] < min_points:
        return None
    poly = shapely.concave_hull(shapely.multipoints(X), allow_holes=allow_holes, ratio=0.3)
    if type(poly) is shapely.Polygon:
        return poly
    else:
        return None
    
################################################################################
################################################################################
def split_cells_clustering(X):
    clusters = hdbscan.HDBSCAN(min_samples=10).fit_predict(X)
    return [X[clusters==i, :] for i in set(clusters) if not i==-1]

def split_cells_midline(X, midline_gap=100):
    centroids, _, _ = k_means(X, 2, n_init=5)
    midpoint = np.mean(centroids, 0)[0]
    if np.min(np.abs(X[:,0] - midpoint)) > midline_gap:
        left = X[X[:,0] < midpoint]
        right = X[X[:,0] > midpoint]
        return [left, right]
    else:
        return [X]

def get_polygon_from_obs(df, min_points=50, split_clusters=True):
    X = df[['cirro_x','cirro_y']].values
    if X.shape[0] > min_points:
        if split_clusters:
            groups = split_cells_clustering(X)
        else:
            groups = split_cells_midline(X)
        if len(groups)>0:
            return unary_union([poly_from_points(x, allow_holes=not split_clusters) for x in groups])
        else:
            return None
                
def get_ccf_polygons(data, min_points=50, midline_gap=100):
    ccf_th_names = get_ccf_th_names()
    ccf_polygons = defaultdict(dict)  
    for (name, section), df in data.groupby(['CCF_acronym', 'section']):
        poly = get_polygon_from_obs(df, min_points, split_clusters=name not in ccf_th_names)
        if poly is not None:
            ccf_polygons[name][section] = poly
    return ccf_polygons


def get_outline_polygon(data, min_points=50, midline_gap=100):
    ''' Takes all the data points per section and generates a concave hull polygon 
    that encompasses all points
    '''
    outline_polygons = defaultdict(dict) 
    for section, df in data.groupby(['section']):
        poly = poly_from_points(df[['cirro_x','cirro_y']].values, min_points=min_points, allow_holes=True)
        if poly is not None:
            outline_polygons[section] = poly
    return outline_polygons

################################################################################
################################################################################
def plot_shape(poly, edgecolor='black', **kwargs):
    if type(poly) is shapely.GeometryCollection:
        for subpoly in poly.geoms:
            patch = splot.plot_polygon(subpoly, add_points=False, edgecolor=edgecolor, **kwargs)
    else:
        patch = splot.plot_polygon(poly, add_points=False, edgecolor=edgecolor, **kwargs)
    return patch

import colorcet as cc
def plot_ccf_section(ccf_polygons, section, highlight=[], palette=None, labels=True, bg_shapes=True, ax=None):
    alpha=0.6
    ccf_names = ccf_polygons.keys()
    if palette is None:
        palette = dict(zip(ccf_names, sns.color_palette(cc.glasbey, n_colors=len(ccf_names))))
    elif palette=='bw':
        palette = {x: '#BBBBBB' for x in ccf_names}
    elif palette=='dark_outline':
        palette = {x: 'none' for x in ccf_names}
        alpha=1
    elif palette=='light_outline':
        palette = {x: 'none' for x in ccf_names}
        alpha=0.3
        
    if highlight=='all':
        highlight = ccf_names
    patches = []
    if bg_shapes:
        for i, name in enumerate(ccf_names):
            if section in ccf_polygons[name] and name not in highlight:
                patches.append(plot_shape(ccf_polygons[name][section], color=palette[name], ax=ax, 
                                alpha=0.1, label=name if labels else None))
    for name in highlight:
        if section in ccf_polygons[name]:
            patches.append(plot_shape(ccf_polygons[name][section], facecolor=palette[name], ax=ax, 
                           alpha=alpha, label=name if labels else None))
    return patches

def plot_ccf_overlay(obs, ccf_polygons, sections=None, point_hue='CCF_acronym', legend='cells', min_group_count=10, highlight=[], 
                     outlines_only=False, bg_cells=None, bg_shapes=True, axes=False):
    obs = obs.copy()
    if sections is None:
        sections = obs['section'].unique()
    else:
        ccf_polygons = {x: y for x, y in ccf_polygons.items() 
                        if len(sections & y.keys()) > 0}
    ccf_names = ccf_polygons.keys()
    shape_palette = 'bw' if outlines_only else dict(zip(ccf_names, sns.color_palette(cc.glasbey, n_colors=len(ccf_names))))
    
#     string rep allows adding 'other'
    obs[point_hue] = obs[point_hue].astype(str)
    point_group_names = obs[point_hue].value_counts().loc[lambda x: x>min_group_count].index
    obs = obs.loc[lambda df: df[point_hue].isin(point_group_names)]
    
    if not outlines_only:
        if point_hue == 'CCF_acronym':
            point_palette = shape_palette.copy()
            extra_names = point_group_names.difference(ccf_names)
            extra_palette = dict(zip(extra_names, sns.color_palette(cc.glasbey, n_colors=len(point_group_names))[-len(extra_names):]))
            point_palette.update(extra_palette)
        else:
            point_palette = dict(zip(point_group_names, sns.color_palette(cc.glasbey, n_colors=len(point_group_names))))
        point_palette.update(other='grey')
    
    for section in sections:
        secdata = obs.loc[lambda df: (df['section']==section)].copy() #& df['CCF_acronym'].isin(ccf_names)]
        if len(secdata) < min_group_count:
            continue
        print(section)
        fig, ax = plt.subplots(figsize=(8,4))
        
        patches = plot_ccf_section(ccf_polygons, section, highlight=highlight, palette=shape_palette, bg_shapes=bg_shapes,
                                   labels=legend in ['ccf', 'both'], ax=ax)
        if not outlines_only:
            if bg_cells is not None:
                sns.scatterplot(bg_cells.loc[lambda df: (df['section']==section)], x='cirro_x', y='cirro_y', c='grey', s=2, alpha=0.5)
            # lump small groups if legend list is too long
            sec_group_counts = secdata[point_hue].value_counts(ascending=True)
            if len(sec_group_counts) > 10:
                point_groups_section = sec_group_counts.loc[lambda x: x>min_group_count].index
                secdata.loc[lambda df: ~df[point_hue].isin(point_groups_section), point_hue] = 'other'
            secdata[point_hue] = pd.Categorical(secdata[point_hue])

            sns.scatterplot(secdata, x='cirro_x', y='cirro_y', hue=point_hue, s=2, palette=point_palette, legend=legend in ['cells', 'both'])
            if legend:
                # plt.legend(ncols=2, loc='upper center', bbox_to_anchor=(0.5, 0))
                plt.legend(ncols=1, loc='center left', bbox_to_anchor=(0.98, 0.5), fontsize=16)
        
        plt.axis('image')
        if not axes:
            sns.despine(left=True, bottom=True)
            plt.xticks([])
            plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')
        plt.show()
        
def plot_th_outline(outline_polygons, sections=None, axes=False, facecolor='none', edgecolor='black', alpha=0.05):
    ''' Takes the per-section outline_polygons from get_outline_polygon() and plots the outlines only
    '''

    if sections is None:
        sections = obs['section'].unique()
    elif isinstance(sections, str):
        sections = [sections]
    
    for section in sections:
        # fig, ax = plt.subplots(figsize=(8,4))
        
        patches = []
        plot_shape(outline_polygons[section], facecolor=facecolor, edgecolor=edgecolor, alpha=alpha);
        
        if not axes:
            sns.despine(left=True, bottom=True)
            plt.xticks([])
            plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')
        # plt.show()
        
################################################################################
################################################################################
def get_obs_from_annotated_clusters(name, adata, nuclei_df):
    clusters = nuclei_df.loc[name, "annotated clusters"].split(', ')
    obs = adata.obs.loc[lambda df: df['cluster_label'].str[:4].isin(clusters)]
    return obs

def plot_nucleus_cluster_comparison_slices(obs, ccf_polygons, nuclei, bg_cells=None, bg_shapes=True, legend='cells', **kwargs):
    sections_points = obs['section'].value_counts().loc[lambda x: x>10].index
    nuclei = [nuclei] if type(nuclei) is str else nuclei
    sections_nuclei = set.union(*[set(ccf_polygons[x].keys()) for x in nuclei])
    sections = sorted(sections_nuclei.union(sections_points))
    plot_ccf_overlay(obs, ccf_polygons, sections, point_hue='cluster_label', legend=legend, highlight=nuclei, bg_cells=bg_cells, bg_shapes=bg_shapes, **kwargs)

################################################################################
################################################################################
def plot_expression_ccf(adata_neuronal, ccf_polygons, section, gene, nuclei=[], bg_shapes=False, axes=False, 
                        cmap='magma', show_outline=False, highlight=[]):
    subset = adata_neuronal[adata_neuronal.obs.query(f"section=='{section}'").index]
    fig, ax = plt.subplots(figsize=(8,4))
    
    # # plot ccf annotation behind gene expression
    # plot_ccf_section(ccf_polygons, section, highlight=nuclei, bg_shapes=bg_shapes, ax=ax, palette='bw')
    
    # plot gene expression
    x, y = subset.obsm['spatial_cirro'].T
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
        th_outline_polygons = get_outline_polygon(adata_neuronal.obs)
        plot_th_outline(th_outline_polygons, sections=section, alpha=0.15)
    
    # plot ccf annotation in front of gene expression
    if highlight==[]:
        plot_ccf_section(ccf_polygons, section, highlight=nuclei, bg_shapes=bg_shapes, ax=ax, palette='dark_outline')
    elif highlight!=[]:
        plot_ccf_section(ccf_polygons, section, highlight=nuclei, bg_shapes=bg_shapes, ax=ax, palette='light_outline')
        plot_ccf_section(ccf_polygons, section, highlight=highlight, bg_shapes=bg_shapes, ax=ax, palette='dark_outline')
    
    plt.gca().set_aspect('equal')
    plt.show()
    
    return fig, ax
    