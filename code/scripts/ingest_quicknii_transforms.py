import sys
sys.path.append('/code/')
import os.path
import spatialdata as sd
from spatialdata import SpatialData
import pandas as pd
import abc_load as abc
import ccf_registration as ccf
from ccf_transforms import get_normalizing_transform, parse_cells_by_section

df_full = abc.get_combined_metadata()
df = abc.label_thalamus_spatial_subset(df_full, flip_y=False, distance_px=20, 
                                  cleanup_mask=True, drop_end_sections=True,
                                  filter_cells=True)

coords = ['x_section', 'y_section', 'z_section']
slice_label = 'slice_int'
df[slice_label] = df['z_section'].apply(lambda x: int(x*10))

transforms_by_section = ccf.read_quicknii_file(
    os.path.expanduser("/code/resources/adjusted_10-10_final.json"), scale=25)
minmax = pd.read_csv("/code/resources/brain3_thalamus_coordinate_bounds.csv", index_col=0)

# load to spatialdata
norm_transform = get_normalizing_transform(df, coords, 
                                           min_xy=minmax.loc['min'].values, 
                                           max_xy=minmax.loc['min'].values, 
                                           flip_y=True)
cells_by_section = parse_cells_by_section(df, transforms_by_section, norm_transform, coords, slice_label=slice_label)
sdata = SpatialData.from_elements_dict(cells_by_section)

# transform
transformed_points = pd.concat((
    df.compute() for df in 
    sdata.transform_to_coordinate_system('ccf').points.values()
    ))

# update dataframe
suffix = "_realigned"
df = df.join(transformed_points[list('xyz')].rename(columns=lambda x: f"{x}_ccf{suffix}"))

# add parcellation index
imdata = abc.get_ccf_labels_image(resampled=False)
new_coords = [f"{x}_ccf{suffix}" for x in 'zyx'] # zyx order 
df['parcellation_index'+suffix] = imdata[ccf.image_index_from_coords(df[new_coords])]

# add parcellation metadata
ccf_df = pd.read_csv(abc.ABC_ROOT/f"metadata/Allen-CCF-2020/20230630/parcellation_to_parcellation_term_membership.csv")
ccf_df = ccf_df.pivot(index='parcellation_index', columns='parcellation_term_set_name', values='parcellation_term_acronym').astype('category')
df = df.join(ccf_df[['division','structure','substructure']].rename(columns=lambda x: f"parcellation_{x}{suffix}"),
             on='parcellation_index'+suffix)

df.to_parquet("/results/abc_realigned_metadata_thalamus-boundingbox.parquet")

# saving resampled images
img_transform = sd.transformations.Scale(10e-3*np.ones(3), 'xyz')
labels = sd.models.Labels3DModel.parse(imdata, dims='zyx', transformations={'ccf': img_transform})
sdata.add_labels('ccf_regions', labels)

ngrid = 1100
nz = 76
z_res = 2
img_stack = np.zeros((ngrid, ngrid, nz))
for section in sdata.points.keys():
    i = int(section)/z_res
    target = sdata[section]
    source = sdata['ccf_regions']
    scale = 10e-3
    target_img, target_grid_transform = ccft.map_image_to_slice(
        sdata, imdata, source, target, scale=scale, ngrid=ngrid
        )
    img_stack[:,:,i] = target_img.T

import nibabel
nifti_img = nibabel.Nifti1Image(img_stack, affine=np.eye(4))
nibabel.save(nifti_img, '/results/abc_realigned_ccf_labels.nii.gz')

