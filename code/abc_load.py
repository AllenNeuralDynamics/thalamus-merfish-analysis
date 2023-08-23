from pathlib import Path
import json

import pandas as pd
import anndata as ad

_CIRRO_COLUMNS = {
    'x':'cirro_x',
    'y':'cirro_y',
    'x_section':'cirro_x',
    'y_section':'cirro_y',
    'brain_section_label':'section',
    # 'cluster':'cluster_label',
    'parcellation_substructure':'CCF_acronym'
}

_REFERENCE_ANNDATA_PATH = "/data/merfish_638850_AIT17.custom_CCF_annotated_TH_ZI_only_2023-05-04_00-00-00/atlas_brain_638850_AIT17_custom_CCF_annotated_TH_ZI_only.h5ad"

abc_root = Path("/data/abc_atlas/")

with open(abc_root/"releases/20230630/manifest.json") as file:
    manifest = json.load(file)

def _md_dict_merfish():
    return manifest['file_listing']['MERFISH-C57BL6J-638850']['metadata']

def _md_dict_ccf():
    return manifest['file_listing']['Allen-CCF-2020']['metadata']

def get_combined_metadata(cirro_names=True, flip_y=True):
    # this info is redundant but includes more cells (QC'ed out of CCF process?) 
    # cells_df = pd.read_csv(
    #     abc_root/"metadata/MERFISH-C57BL6J-638850/20230630/views/cell_metadata_with_cluster_annotation.csv",
    #                        dtype={'cell_label':str}, index_col=0)
    # cells_df.rename(columns={'x':'x_section','y':'y_section','z':'z_section'}, inplace=True)

    # could save space by not merging color etc
    ccf_df = pd.read_csv(
        abc_root/"metadata/MERFISH-C57BL6J-638850-CCF/20230630/views/cell_metadata_with_parcellation_annotation.csv", 
                         dtype={'cell_label':str}, index_col=0)
    if flip_y:
        ccf_df[['y_section', 'y_reconstructed']] *= -1
        # cells_df['y_section'] *= -1
        
    # cells_df = cells_df.join(ccf_df[ccf_df.columns.difference(cells_df.columns)])
    cells_df = ccf_df
    if cirro_names:
        cells_df.rename(columns=_CIRRO_COLUMNS, inplace=True)
    return cells_df


def load_adata(with_metadata=True, transform='log2', cirro_names=True, subset_to_ref=True):
    adata = ad.read_h5ad(abc_root/f"expression_matrices/MERFISH-C57BL6J-638850/20230630/C57BL6J-638850-{transform}.h5ad", backed='r')
    if subset_to_ref:
        adata_ref = ad.read_h5ad(_REFERENCE_ANNDATA_PATH, backed='r')
        adata = adata[adata.obs_names.intersection(adata_ref.obs_names)]
    adata = adata.to_memory()
    adata.var_names = adata.var['gene_symbol']
    if with_metadata:
        cells_df = get_combined_metadata(cirro_names=cirro_names)
        adata.obs = adata.obs.join(cells_df[cells_df.columns.difference(adata.obs.columns)])
    return adata