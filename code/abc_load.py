from pathlib import Path
import json
import numpy as np
import pandas as pd
import anndata as ad
import nibabel

_CIRRO_COLUMNS = {
    'x':'cirro_x',
    'y':'cirro_y',
    'x_section':'cirro_x',
    'y_section':'cirro_y',
    'brain_section_label':'section',
    # 'cluster':'cluster_label',
    'parcellation_substructure':'CCF_acronym'
}

# older data asset subsetted to thalamus/ZI outline
_THALAMUS_ANNDATA_PATH = "/data/merfish_638850_AIT17.custom_CCF_annotated_TH_ZI_only_2023-05-04_00-00-00/atlas_brain_638850_AIT17_custom_CCF_annotated_TH_ZI_only.h5ad"

def get_thalamus_reference_ids():
    adata_ref = ad.read_h5ad(_THALAMUS_ANNDATA_PATH, backed='r')
    ref_ids = adata_ref.obs_names.astype('string')
    return ref_ids

ABC_ROOT = Path("/data/abc_atlas/")
CURRENT_VERSION = "20230830"

def get_ccf_labels_image(resampled=False):
    if resampled:
        path = "image_volumes/MERFISH-C57BL6J-638850-CCF/20230630/resampled_annotation.nii.gz"
    else:
        path = "image_volumes/Allen-CCF-2020/20230630/annotation_10.nii.gz"
    img = nibabel.load(ABC_ROOT/path)
    # could maybe keep the lazy dataobj and not convert to numpy?
    imdata = np.array(img.dataobj)
    return imdata

def get_combined_metadata(drop_unused=True, cirro_names=False, flip_y=False, version=CURRENT_VERSION):
    """Load the cell metadata csv, with memory/speed improvements
    Selects correct dtypes and optionally renames and drops columns
    
    Parameters
    ----------
    drop_unused, optional
        don't load uninformative or unused columns (color etc), by default True
    cirro_names, optional
        rename columns to match older cirro anndata names, by default False
    flip_y, optional
        flip section and reconstructed y coords so up is positive, by default False
    version, optional
        version string, by default CURRENT_VERSION

    Returns
    -------
        cells_df pandas dataframe
    """
    float_columns = [
        'average_correlation_score', 
        'x_ccf', 'y_ccf', 'z_ccf', 
        'x_section', 'y_section', 'z_section', 
        'x_reconstructed', 'y_reconstructed', 'z_reconstructed',
        ]
    cat_columns = [
        'brain_section_label', 'cluster_alias', 
        'neurotransmitter', 'class',
        'subclass', 'supertype', 'cluster', 
        'parcellation_index',  
        'parcellation_division',
        'parcellation_structure', 'parcellation_substructure'
        # 'parcellation_organ', 'parcellation_category',
        ]
    dtype = dict(cell_label='string', 
                **{x: 'float' for x in float_columns}, 
                **{x: 'category' for x in cat_columns})
    usecols = list(dtype.keys()) if drop_unused else None

    # if as_dask:
    #     cells_df = dd.read_csv(
    #         ABC_ROOT/f"metadata/MERFISH-C57BL6J-638850-CCF/{version}/views/cell_metadata_with_parcellation_annotation.csv", 
    #                         dtype=dtype, usecols=usecols, blocksize=100e6)
    # else:
    cells_df = pd.read_csv(
            ABC_ROOT/f"metadata/MERFISH-C57BL6J-638850-CCF/{version}/views/cell_metadata_with_parcellation_annotation.csv", 
                            dtype=dtype, usecols=usecols, index_col='cell_label', engine='c')
    if flip_y:
        cells_df[['y_section', 'y_reconstructed']] *= -1
    if cirro_names:
        cells_df = cells_df.rename(columns=_CIRRO_COLUMNS)
    return cells_df


def load_adata(with_metadata=True, transform='log2', cirro_names=False, version=CURRENT_VERSION, subset_to_ref=True):
    adata = ad.read_h5ad(ABC_ROOT/f"expression_matrices/MERFISH-C57BL6J-638850/{version}/C57BL6J-638850-{transform}.h5ad", backed='r')
    if subset_to_ref:
        ref_ids = get_thalamus_reference_ids()
        adata = adata[adata.obs_names.intersection(ref_ids)]
    adata = adata.to_memory()
    # access genes by short symbol vs longer names
    adata.var_names = adata.var['gene_symbol']
    if with_metadata:
        cells_df = get_combined_metadata(cirro_names=cirro_names)
        adata.obs = adata.obs.join(cells_df[cells_df.columns.difference(adata.obs.columns)])
    return adata

def label_thalamus_spatial_subset(cells_df, distance_px=10, filter=False, drop_end_sections=True):
    coords = ['x_reconstructed','y_reconstructed','z_reconstructed']
    resolutions = np.array([10e-3, 10e-3, 200e-3])
    field_name='thalamus_dataset'
    
    ccf_df = pd.read_csv(
            ABC_ROOT/"metadata/Allen-CCF-2020/20230630/parcellation_to_parcellation_term_membership.csv"
            )
    th_zi_ind = np.hstack(
            (ccf_df.loc[ccf_df['parcellation_term_acronym']=='TH', 
                        'parcellation_index'].unique(),
                ccf_df.loc[ccf_df['parcellation_term_acronym']=='ZI', 
                        'parcellation_index'].unique())
    )
    ## resampled ccf (z resolution limited to merscope slices)
    ccf_img = get_ccf_labels_image(resampled=True)
    ccf_img.shape
    # takes about 5 sec
    th_mask = np.isin(ccf_img, th_zi_ind)
    mask_img = sectionwise_dilation(th_mask, distance_px, true_radius=False)
    cells_df = label_masked_cells(cells_df, mask_img, coords, resolutions, field_name=field_name)
    if drop_end_sections:
        cells_df[field_name] = cells_df[field_name] & (4.81 < cells_df['z_reconstructed']) & (cells_df['z_reconstructed'] < 8.39)
    if filter:
        return cells_df[cells_df[field_name]].copy().drop(columns=[field_name])

from scipy.ndimage import binary_dilation
def sectionwise_dilation(mask_img, distance_px, true_radius=False):
    out = np.zeros_like(mask_img)
    coords = np.mgrid[-distance_px:distance_px+1, -distance_px:distance_px+1]
    struct = np.linalg.norm(coords, axis=0) <= distance_px
    for i in range(mask_img.shape[2]):
        if true_radius:
            out[:,:,i] = binary_dilation(mask_img[:,:,i], structure=struct)
        else:
            out[:,:,i] = binary_dilation(mask_img[:,:,i], iterations=distance_px)
    return out

def label_masked_cells(cells_df, mask_img, coords, resolutions, 
                                field_name='thalamus_dataset'):
    coords_index = np.rint(cells_df[coords].values / resolutions).astype(int)
    # tuple() makes this like calling mask_img[coords_index[:,0], coords_index[:,1], coords_index[:,2]]
    cells_df[field_name] = mask_img[tuple(coords_index.T)]
    return cells_df