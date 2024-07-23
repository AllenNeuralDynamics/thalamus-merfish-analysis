from importlib_resources import files
import sys

sys.path.append("/code/")
import spatialdata as sd
import pandas as pd
import numpy as np
import nibabel
from thalamus_merfish_analysis import abc_load as abc
from thalamus_merfish_analysis import ccf_registration as ccf
from thalamus_merfish_analysis import ccf_transforms as ccft

df_full = abc.get_combined_metadata(drop_unused=False)
# permissive spatial subset using published alignment
# (previously using manual subset?)
df = abc.label_thalamus_spatial_subset(
    df_full,
    flip_y=False,
    distance_px=25,
    cleanup_mask=True,
    drop_end_sections=True,
    filter_cells=True,
)

coords = ["x_section", "y_section", "z_section"]
slice_label = "slice_int"
df[slice_label] = df["z_section"].apply(lambda x: int(x * 10))

transforms_by_section = ccf.read_quicknii_file(
    files("thalamus_merfish_analysis.resources") / "quicknii_refined_20240228.json", scale=25
)
minmax = pd.read_csv(
    files("thalamus_merfish_analysis.resources") / "brain3_thalamus_coordinate_bounds.csv",
    index_col=0,
)

# load to spatialdata
norm_transform = ccft.get_normalizing_transform(
    min_xy=minmax.loc["min"].values, max_xy=minmax.loc["max"].values, flip_y=True
)
cells_by_section = ccft.parse_cells_by_section(
    df, transforms_by_section, norm_transform, coords, slice_label=slice_label
)
sdata = sd.SpatialData.from_elements_dict(cells_by_section)
sdata.write("/scratch/abc_atlas_realigned.zarr")

# transform
transformed_points = pd.concat(
    (df.compute() for df in sdata.transform_to_coordinate_system("ccf").points.values())
)

# update dataframe
new_coords = [f"{x}_ccf_realigned" for x in "xyz"]  # xyz order
df = df.join(transformed_points[list("xyz")].rename(columns=dict(zip("xyz", new_coords))))


ngrid = 1100
nz = 76
z_res = 2


# img_stack = np.zeros((ngrid, ngrid, nz))
# for section in sdata.points.keys():
def transform_section(section, imdata=None, fname=None):
    i = int(np.rint(int(section) / z_res))
    target = sdata[section]
    source = sdata[fname]
    scale = 10e-3
    target_img, target_grid_transform = ccft.map_image_to_slice(
        sdata, imdata, source, target, scale=scale, ngrid=ngrid, centered=False
    )
    return target_img


from multiprocessing import Pool
import functools


def save_resampled_image(imdata, fname):
    # saving resampled images
    img_transform = sd.transformations.Scale(10e-3 * np.ones(3), "xyz")
    labels = sd.models.Labels3DModel.parse(
        imdata, dims="xyz", transformations={"ccf": img_transform}
    )
    sdata.add_labels(fname, labels)

    # img_stack[:,:,i] = target_img.T
    # with Pool(processes=8) as p:
    #     out = p.map(functools.partial(transform_section, imdata=imdata, fname=fname),
    #                 sdata.points.keys())
    out = map(functools.partial(transform_section, imdata=imdata, fname=fname), sdata.points.keys())
    img_stack = np.stack(out, axis=-1)

    nifti_img = nibabel.Nifti1Image(img_stack, affine=np.eye(4), dtype="int64")
    nibabel.save(nifti_img, f"/results/{fname}.nii.gz")


# CCFv3
# imdata = abc.get_ccf_labels_image(resampled=False)
# df['parcellation_index_realigned'] = imdata[ccf.image_index_from_coords(df[new_coords])]
# save_resampled_image(imdata, 'abc_realigned_ccf_labels')

# # add parcellation metadata
# ccf_df = ccf_df.pivot(index='parcellation_index', columns='parcellation_term_set_name', values='parcellation_term_acronym').astype('category')
# df = df.join(ccf_df[['division','structure','substructure']].rename(columns=lambda x: f"parcellation_{x}_realigned"),
#              on='parcellation_index_realigned')

# Kim Lab DevCCF
img = nibabel.load("/data/KimLabDevCCFv001/KimLabDevCCFv001_Annotations_ASL_Oriented_10um.nii.gz")
imdata = np.array(img.dataobj).astype(int)
df["parcellation_index_realigned_devccf"] = imdata[ccf.image_index_from_coords(df[new_coords])]
save_resampled_image(imdata, "abc_realigned_devccf_labels")

devccf_index = pd.read_csv(
    "/data/KimLabDevCCFv001/KimLabDevCCFv001_MouseOntologyStructure.csv",
    dtype={"ID": int, "Parent ID": str},
)
# some quotes have both single and double
for x in ["Acronym", "Name"]:
    devccf_index[x] = devccf_index[x].str.replace("'", "")
df["parcellation_devccf"] = df["parcellation_index_realigned_devccf"].map(
    devccf_index.set_index("ID").to_dict()
)

df.to_parquet("/results/abc_realigned_metadata_thalamus-boundingbox.parquet")
