import sys
sys.path.append('/code/')
from pathlib import Path
from spatialdata import SpatialData
import pandas as pd
import abc_load as abc
import ccf_registration as ccf
from ccf_transforms import get_normalizing_transform, parse_cells_by_section

df_full = abc.get_combined_metadata()

coords = ['x_section', 'y_section', 'z_section']
slice_label = 'slice_int'
ref_subset = abc.get_thalamus_reference_ids()
df = ccf.subset_to_ref_bounds(df_full, coords, ref_subset)
df[slice_label] = df['z_section'].apply(lambda x: int(x*10))

transforms_by_section = ccf.read_quicknii_file(
    Path(__file__).parent/"../../resources/adjusted_10-10_final.json", scale=25)

# load to spatialdata
norm_transform = get_normalizing_transform(df, coords)
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
imdata = abc.get_ccf_labels_image()
new_coords = [f"{x}_ccf{suffix}" for x in 'xyz']
df['parcellation_index'+suffix] = imdata[ccf.image_index_from_coords(df[new_coords])]

# add parcellation metadata
ccf_df = pd.read_csv(abc.ABC_ROOT/f"metadata/Allen-CCF-2020/20230630/parcellation_to_parcellation_term_membership.csv")
ccf_df = ccf_df.pivot(index='parcellation_index', columns='parcellation_term_set_name', values='parcellation_term_acronym').astype('category')
df = df.join(ccf_df[['division','structure','substructure']].rename(columns=lambda x: f"parcellation_{x}{suffix}"),
             on='parcellation_index'+suffix)

df.to_parquet("/results/abc_realigned_metadata_thalamus-boundingbox.parquet")

# only if saving to spatialdata
# img_transform = sd.transformations.Scale(10e-3*np.ones(3), 'xyz')
# labels = sd.models.Labels3DModel.parse(imdata, dims='zyx', transformations={'ccf': img_transform})
# regions = [f"{section}_ccf" for section in cells_by_section.keys()]
# ccf_annotation = load_ccf_metadata_table(regions)
# sdata = sd.SpatialData.from_elements_dict(dict(ccf_regions=labels, table=ccf_annotation, **cells_by_section))

# imdata = abc.get_ccf_labels_image()
# for section in sdata.points.keys():
#     target = sdata[section]
#     source = sdata['ccf_regions']
#     scale = 10e-3
#     target_img, target_grid_transform = map_image_to_slice(imdata, source, target, scale=scale)
#     section_labels = sd.models.Labels2DModel.parse(target_img, dims='yx', 
#                              transformations={section: target_grid_transform})
#     sdata.add_labels(f"{section}_ccf", section_labels)