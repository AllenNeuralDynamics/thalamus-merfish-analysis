import spatialdata as sd
import anndata as ad
import pandas as pd
import dask.dataframe as dd
import numpy as np

from . import abc_load as abc
from . import ccf_registration as ccf
from scipy.ndimage import map_coordinates

def get_target_grid_center(source, target, inv_transform):
    target_z = target['z'].max().compute()
    # calculate xy center by pullback from image center
    source_center = np.array([len(source[i])/2 for i in 'xyz'])
    mat = inv_transform.inverse().to_affine_matrix('xyz','xyz')
    target_center = ccf.apply_affine_left(mat, source_center)[:,0]
    target_center[2] = target_z
    return target_center

def get_sample_points_centered(sdata, source, target, scale=None, ngrid=1100):

    # xyz grid (pixel centers)
    grid = np.stack((*np.mgrid[0:ngrid, 0:ngrid][::-1], np.zeros((ngrid, ngrid))), axis=0) + 0.5
    grid_points = sd.models.PointsModel.parse(grid.reshape(3, -1).T)
    
    inv_transform = sd.transformations.get_transformation_between_coordinate_systems(sdata, target, source)
    if scale is None:
        scale = np.linalg.det(inv_transform.inverse().to_affine_matrix('xyz', 'xyz'))**(1/3)
        
    target_center = get_target_grid_center(source, target, inv_transform)
    target_grid_transform = sd.transformations.Sequence([
        sd.transformations.Translation(np.array([-ngrid/2, -ngrid/2, 0]) - 0.5, 'xyz'),
        sd.transformations.Scale(scale*np.array([1, 1, 1]), 'xyz'),
        sd.transformations.Translation(target_center, 'xyz')
    ])
    source_points = sd.transform(
        grid_points,
        sd.transformations.Sequence([target_grid_transform, inv_transform]),
        )
    return source_points, target_grid_transform

def get_sample_points_square(sdata, source, target, scale=None, ngrid=1100):
    inv_transform = sd.transformations.get_transformation_between_coordinate_systems(sdata, target, source)
    target_z = target['z'].max().compute()
    # xyz grid (pixel centers)
    grid = np.stack((*np.mgrid[0:ngrid, 0:ngrid][::-1], np.zeros((ngrid, ngrid))), axis=0)
    grid_points = sd.models.PointsModel.parse(grid.reshape(3, -1).T)
    if scale is None:
        scale = np.linalg.det(inv_transform.inverse().to_affine_matrix('xyz', 'xyz'))**(1/3)
    target_grid_transform = sd.transformations.Sequence([
        sd.transformations.Scale(scale*np.array([1, 1, 1]), 'xyz'),
        sd.transformations.Translation(np.array([0, 0, target_z]), 'xyz'),
    ])
    source_points = sd.transform(
        grid_points,
        sd.transformations.Sequence([target_grid_transform, inv_transform]),
        )
    return source_points, target_grid_transform

def map_image_to_slice(sdata, imdata, source, target, centered=True, scale=None, ngrid=None):
    # expand to diagonal dimension to allow arbitrary rotation
    if ngrid is None:
        ngrid = int(np.linalg.norm(np.array(source.shape)))
    if centered:
        source_points, target_grid_transform = get_sample_points_centered(sdata, source, target, scale, ngrid)
    else:
        source_points, target_grid_transform = get_sample_points_square(sdata, source, target, scale, ngrid)
    # scipy.ndimage evaluates at integers not pixel centers
    points = source_points[['z','y','x']].values.T - 0.5
    target_img = map_coordinates(imdata, points, prefilter=False, order=0).reshape(ngrid, ngrid)
    # target_img = np.flipud(target_img)
    return target_img, target_grid_transform

def get_normalizing_transform(min_xy, max_xy, flip_y=True):
    min_xy, max_xy = min_xy[:2], max_xy[:2]
    if flip_y:
        norm_transform = sd.transformations.Sequence([
            sd.transformations.Translation(np.array([-1*min_xy[0], -1*max_xy[1]]), 'xy'),
            sd.transformations.Scale(1/(max_xy-min_xy) * np.array([1, -1]), 'xy')
            ])
    else:
        sd.transformations.Translation(-1*min_xy, 'xy'),
        sd.transformations.Scale(1/(max_xy-min_xy), 'xy')
    return norm_transform

def parse_cells_by_section(df, transforms_by_section, norm_transform, coords, slice_label='slice_int'):
    cells_by_section = dict()
    for name, df_section in df.groupby(slice_label, observed=True):
        ccf_transform = sd.transformations.Affine(transforms_by_section[name].T, 'xyz', 'xyz')
    #! converting to dask here is essential to preserve dtypes
        cells_by_section[str(name)] = sd.models.PointsModel.parse(
            dd.from_pandas(df_section, npartitions=1), coordinates=dict(zip('xyz', coords)),
            transformations={'ccf': sd.transformations.Sequence([norm_transform, ccf_transform]),
                            str(name): sd.transformations.Identity()})
    return cells_by_section

def load_ccf_metadata_table(regions):
    # load ccf region metadata
    # (not included/updated in 20230830 version)
    ccf_df = (
            pd.read_csv(
            abc.ABC_ROOT/f"metadata/Allen-CCF-2020/20230630/parcellation_to_parcellation_term_membership.csv",
            dtype={'parcellation_term_acronym': 'category'})
            .query("parcellation_term_set_name=='substructure'")
    )
    instance_key='parcellation_index'
    region_key='annotated_element'
    # need to repeat table for every annotated element
    obs = pd.concat([ccf_df.assign(**{region_key: x}) for x in regions])
    ccf_annotation = sd.models.TableModel.parse(ad.AnnData(obs=obs), region=regions, region_key=region_key, instance_key=instance_key)
    return ccf_annotation