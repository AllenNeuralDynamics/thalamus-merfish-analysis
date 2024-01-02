from __future__ import annotations

from collections import defaultdict

import geopandas as gp
import hdbscan
import numpy as np
import shapely
from sklearn.cluster import k_means
from shapely.ops import unary_union


CCF_TH_NAMES = ['AD', 'AMd', 'AMv', 'AV', 'CL', 'CM', 'Eth', 'FF', 'IAD', 'IAM',
                'IGL', 'IMD', 'IntG', 'LD', 'LGd-co', 'LGd-ip', 'LGd-sh', 'LGv', 
                'LH', 'LP', 'MD', 'MGd', 'MGm', 'MGv', 'MH', 'PCN', 'PF', 'PIL', 
                'PO', 'POL', 'PP', 'PR', 'PT', 'PVT', 'PoT', 'RE', 'RH', 'RT', 
                'SGN', 'SMT', 'SPA', 'SPFp', 'SubG', 'VAL', 'VM', 'VPL', 
                'VPLpc', 'VPM', 'VPMpc', 'Xi', 'ZI']

def poly_from_points(X: np.ndarray, 
                     min_points: int=0, 
                     allow_holes: bool=False,
                     ratio: float=0.3
                     ) -> shapely.Polygon | None:
    """Create a polygon for the convave hull (alpha shape) of a set of 2D point
    coordinates, with sensible defaults

    Parameters
    ----------
    X
        coordinate array of points, shape (N, 2)
    min_points, optional
        point count threshold below which to fail, defaults to 0, by default 0
    allow_holes, optional
        whether to create a polygon with holes, by default False
    ratio, optional
        passed to shapely.concave_hull, by default 0.3

    Returns
    -------
        polygon outlining the points, or None on failure
    """
    if X.shape[0] < min_points:
        return None
    poly = shapely.concave_hull(shapely.multipoints(X), allow_holes=allow_holes, ratio=0.3)
    if type(poly) is shapely.Polygon:
        return poly
    else:
        return None
    
# def poly_from_binary_mask(mask_img,):
    

def split_cells_hdbscan(X):
    """Split a point array into separate arrays for every separate 'island' of points

    Parameters
    ----------
    X
        coordinate array of points, shape (N, 2)

    Returns
    -------
        list of point arrays
    """
    clusters = hdbscan.HDBSCAN(min_samples=10).fit_predict(X)
    return [X[clusters==i, :] for i in set(clusters) if not i==-1]

def split_cells_midline(X, midline_gap=0.5):
    """Split a point array into separate arrays for each hemisphere, if there is a midline gap

    Parameters
    ----------
    X
        coordinate array of points, shape (N, 2)
    midline_gap, optional
        minimum midline gap to split (in units of X), by default 100

    Returns
    -------
        list of point arrays
    """
    centroids, _, _ = k_means(X, 2, n_init=5)
    midpoint = np.mean(centroids, 0)[0]
    if np.min(np.abs(X[:,0] - midpoint)) > midline_gap:
        left = X[X[:,0] < midpoint]
        right = X[X[:,0] > midpoint]
        return [left, right]
    else:
        return [X]

def get_polygon_from_obs(obs_df, strategy='hdbscan',
                         x_field='cirro_x', y_field='cirro_y'):
    """Create a shapely geometry (possibly composite) for a dataframe of points,
    first splitting contiguous groups using the selected strategy

    Parameters
    ----------
    obs_df
        observation (points) dataframe (ie adata.obs)
    strategy, optional
        'hdbscan' or 'midline', by default 'hdbscan'
    x_field, optional
        x coord field name, by default 'cirro_x'
    y_field, optional
        y coord field name, by default 'cirro_y'

    Returns
    -------
        shapely GeometryCollection or Polygon
    """
    X = obs_df[[x_field, y_field]].values
    if strategy=='hdbscan':
        allow_holes = False
        groups = split_cells_hdbscan(X)
    elif strategy=='midline':
        allow_holes = False
        groups = split_cells_midline(X)
    else:
        raise NotImplementedError()
    # allowing holes is important if using the midline strategy, probably not if using hdbscan
    return unary_union([poly_from_points(x, allow_holes=allow_holes) for x in groups])
                
def get_ccf_polygons(data, min_points=50,
                     region_name_field='CCF_acronym', section_id_field='section'):
    """Generates a nested dictionary of shape polygons organized by brain region and section

    Parameters
    ----------
    data
        observations dataframe (eg adata.obs); expects to find columns 'cirro_x'
        and 'cirro_y, which are generated from adata.obsm['spatial_cirro'][:,0]
        and adata.obsm['spatial_cirro'][:,1], respectively
    min_points, optional
        threshold point count below which to ignore a given point set, by default 50

    Returns
    -------
        GeoDataFrame of polygons organized by their names and section ids
    """
    # ccf_polygons = defaultdict(dict) 
    names = []
    sections = []
    geometry = []
    
    # check that 'data' parameter contains cirro_x, cirro_y columns
    if ('cirro_x' not in data.columns) | ('cirro_y' not in data.columns):
        raise Warning('Columns cirro_x and cirro_y not found in data. Call to '
                +'get_polygon_from_obs() will error out.\nPlease add columns '
                +'to data, e.g. data.cirro_x=adata.obsm.spatial_cirro[:,0] and '
                +'data.cirro_y=adata.obsm.spatial_cirro[:,1], and try again.')
    
    # Warning: as of pandas 2.1.0, the default of df.groupby(.., observed=False)
    # is deprecated and will be changed to True in a future version of pandas.
    #  - Setting observed=False to retain current behavior (show all values for 
    #    categorical groupers), but we may want to consider changing to new 
    #    default observed=True (only show observed values for categorical groupers)
    for (name, section), df in data.groupby([region_name_field, section_id_field], observed=False):
        if df.shape[0] > min_points:
            poly = get_polygon_from_obs(df)
            if poly is not None:
                # ccf_polygons[name][section] = poly
                geometry.append(poly)
                names.append(name)
                sections.append(section)
    ccf_polygons = gp.GeoDataFrame(
        dict(name=names, section=sections, geometry=geometry)
    ).set_index(['name', 'section'])
                
    return ccf_polygons


def get_outline_polygon(obs_data, min_points=50, coordinate_type='cirro'):
    ''' Take all the XY cell coordinates in each section and generate a concave
    hull Polygon that encompasses all points in that section'''
    outline_polygons = defaultdict(dict)
    # Warning: as of pandas 2.1.0, the default of df.groupby(.., observed=False)
    # is deprecated and will be changed to True in a future version of pandas.
    #  - Setting observed=False to retain current behavior (show all values for 
    #    categorical groupers), but we may want to consider changing to new 
    #    default observed=True (only show observed values for categorical groupers)
    for section, df in obs_data.groupby('section', observed=False):
        if coordinate_type=='napari':
            XY_n = df[['napari_x_brain1and3','napari_y_brain1and3']].values
            XY = np.asarray([[coord[0], -coord[1]] for coord in XY_n])
        elif coordinate_type=='cirro':
            XY = df[['cirro_x','cirro_y']].values
        poly = poly_from_points(XY, min_points=min_points)
        if poly is not None:
            # groupby is returning key as a number, but on my workstation
            # it returns it as a tuple with one entry here?? despite
            # identical code & versions (pandas=1.5.3)
            # Here I use [section], there I use [section[0]]
            outline_polygons[section] = poly
    return outline_polygons