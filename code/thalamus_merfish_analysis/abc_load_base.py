import json
from functools import lru_cache
from pathlib import Path

import anndata as ad
import networkx as nx
import nibabel
import numpy as np
import pandas as pd
import scipy.ndimage as ndi

from .ccf_images import (cleanup_mask_regions, image_index_from_coords,
                         sectionwise_dilation)

'''
Functions for loading (subsets of) the ABC Atlas MERFISH dataset.
'''

ABC_ROOT = Path("/data/abc_atlas/")
CURRENT_VERSION = "20230830"
BRAIN_LABEL = 'C57BL6J-638850'

# hardcoded non-neuronal class categories for v20230830
NN_CLASSES = ['30 Astro-Epen', '31 OPC-Oligo', '33 Vascular',
                        '34 Immune']

def load_adata(version=CURRENT_VERSION, transform='log2cpv', 
               load_metadata=True, flip_y=True, round_z=True, 
               from_metadata=None):
    '''Load ABC Atlas MERFISH dataset as an anndata object.
    
    Parameters
    ----------
    version : str, default=CURRENT_VERSION
        which release version of the ABC Atlas to load
    transform : {'log2cpv', 'log2cpm', 'raw', 'both'}, default='log2'
        which transformation of the gene counts to load from the expression matrices.
        if 'both', log2 is stored in X and log2 & raw are stored in adata.layers.
        use both if writing to a permanent file or performing mapping on the 
        output; log2 for a smaller object for plotting & most other analyses.
    load_metadata : bool, default=True
        include cell metadata in adata
    flip_y : bool, default=True
        flip y-axis coordinates so positive is up (coronal section appears 
        right-side up as expected)
    round_z : bool, default=True
        rounds z_section, z_reconstructed coords to nearest 10ths place to
        correct for overprecision in a handful of z coords
    from_metadata : DataFrame, default=None
        preloaded metadata DataFrame to merge into AnnData, loading cells in this 
        DataFrame only
        
    Results
    -------
    adata
        anndata object containing the ABC Atlas MERFISH dataset
    '''
    if transform == 'both':
        # takes ~4 min + ~9 GB of memory to load both 
        adata_log2 = ad.read_h5ad(ABC_ROOT/f"expression_matrices/MERFISH-{BRAIN_LABEL}/{version}/{BRAIN_LABEL}-log2.h5ad", 
                                  backed='r')
        adata_raw = ad.read_h5ad(ABC_ROOT/f"expression_matrices/MERFISH-{BRAIN_LABEL}/{version}/{BRAIN_LABEL}-raw.h5ad", 
                                 backed='r')
        # store log2 counts in X
        adata = adata_log2.to_memory()
        # add both log2 & raw counts to layers
        adata.layers['log2p'] = adata.X
        adata.layers['raw'] = adata_raw.X
        # clean up to reduce memory usage
        del adata_log2
        del adata_raw
    else:
        # takes ~2 min + ~3 GB of memory to load one set of counts
        adata = ad.read_h5ad(ABC_ROOT/f"expression_matrices/MERFISH-{BRAIN_LABEL}/{version}/{BRAIN_LABEL}-{transform}.h5ad",
                             backed='r')
        
    if load_metadata or (from_metadata is not None):
        if from_metadata is not None:
            cells_md_df = from_metadata
        else:
            cells_md_df = get_combined_metadata(flip_y=flip_y,
                                                round_z=round_z,
                                                version=version)
        cell_labels = adata.obs_names.intersection(cells_md_df.index)
        adata = adata[cell_labels]
        adata = adata.to_memory()
        # add metadata to obs
        adata.obs = adata.obs.join(cells_md_df.loc[cell_labels, cells_md_df.columns.difference(adata.obs.columns)])
    
    if adata.isbacked:
        adata = adata.to_memory()
    
    if transform == 'log2cpm':
        adata_glut_log2CPM.X = np.asarray(
            np.log2(1 + adata_th_zi_glut.X
                    *1e6/np.sum(adata_th_zi_glut.X.toarray(), axis=1, keepdims=True)
                    ))
    # access genes by short symbol vs longer names
    adata.var_names = adata.var['gene_symbol']

    # note what counts transform is in X
    adata.uns['counts_transform'] = transform
    
    return adata

def filter_by_ccf_region(obs, regions, buffer=0,
                         realigned=False, include_children=True):
    """Filters cell metadata (obs) dataframe spatially by CCF region labels,
    with an optional buffer region (using stored labels if no buffer).

    Parameters
    ----------
    obs
        dataframe containing the ABC Atlas MERFISH dataset (i.e. adata.obs)
    ccf_regions : list(str)
        list of (abbreviated) CCF region names to select
    buffer, optional
        dilation radius in pixels (1px = 10um), by default 0
    include_children, optional
        include all subregions of the specified regions, by default True
    realigned, optional
        use realigned CCF coordinates and image volume, by default False

    Returns
    -------
    obs
        filtered dataframe
    """    
    # TODO: modify to accept adata or obs
    if include_children:
        regions = get_taxonomy_names(regions)
    if buffer > 0:
        obs, _ = label_ccf_spatial_subset(obs, regions,
                                               distance_px=buffer,
                                               cleanup_mask=True,
                                               filter_cells=True,
                                               realigned=realigned)
    else:
        ccf_label = 'parcellation_structure_realigned' if realigned else 'parcellation_structure'
        obs = obs[obs[ccf_label].isin(regions)]
    return obs

def filter_by_class(adata, exclude=NN_CLASSES, include=None):
    ''' Filters anndata object or dataframe by cell type taxonomy 
    classes. Note that these labels may change across dataset versions!

    Parameters
    ----------
    adata
        anndata object or dataframe containing the ABC Atlas MERFISH dataset
    exclude : list(str), default=NN_CLASSES
        list of classes to filter out
    include : list(str), default=None
        if present, include ONLY cells in this list of classes
    
    Returns 
    -------
    adata
        the anndata object, filtered to only include cells from specific classes
    '''
    if hasattr(adata, 'obs'):
        obs = adata.obs
    else:
        obs = adata
    if include is not None:
        subset = obs['class'].isin(include)
    else:
        subset = ~obs['class'].isin(exclude)
    return adata[subset]


def get_combined_metadata(
    version=CURRENT_VERSION,
    realigned=False,
    drop_unused=True, 
    flip_y=False, 
    round_z=True, 
    cirro_names=False
    ):
    '''Load the cell metadata csv, with memory/speed improvements.
    Selects correct dtypes and optionally renames and drops columns
    
    Parameters
    ----------
    version : str, optional
        version to load, by default=CURRENT_VERSION
    realigned : bool, default=False
        if True, load metadata from realignment results data asset,
        containing 'ccf_realigned' coordinates 
    drop_unused : bool, default=True
        don't load uninformative or unused columns (color etc)
    flip_y : bool, default=False
        flip section and reconstructed y coords so up is positive
    round_z : bool, default=True
        rounds z_section, z_reconstructed coords to nearest 10ths place to
        correct for overprecision in a handful of z coords
    cirro_names : bool, default=False
        rename columns to match older cirro anndata names

    Returns
    -------
        cells_df pandas dataframe
    '''
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

    if realigned:
        # TODO: add version to the data asset mount point to allow multiple
        cells_df = pd.read_parquet("/data/realigned/abc_realigned_metadata_thalamus-boundingbox.parquet")
        if version != CURRENT_VERSION:
            old_df = pd.read_csv(
                ABC_ROOT/f"metadata/MERFISH-C57BL6J-638850-CCF/{version}/views/cell_metadata_with_parcellation_annotation.csv", 
                dtype=dtype, usecols=usecols, index_col='cell_label', engine='c')
            cells_df = old_df.join(cells_df[cells_df.columns.difference(old_df.columns)])
    else:
        cells_df = pd.read_csv(
            ABC_ROOT/f"metadata/MERFISH-C57BL6J-638850-CCF/{version}/views/cell_metadata_with_parcellation_annotation.csv", 
            dtype=dtype, usecols=usecols, index_col='cell_label', engine='c')
    if flip_y:
        cells_df[['y_section', 'y_reconstructed']] *= -1
    if round_z:
        cells_df['z_section'] = cells_df['z_section'].round(1)
        cells_df['z_reconstructed'] = cells_df['z_reconstructed'].round(1)
    if cirro_names:
        cells_df = cells_df.rename(columns=_CIRRO_COLUMNS)
        
    cells_df["left_hemisphere"] = cells_df["z_ccf"] < 5.7
    if realigned:
        cells_df["left_hemisphere_realigned"] = cells_df["z_ccf_realigned"] < 5.7
    cells_df = cells_df.replace("ZI-unassigned", "ZI")
    return cells_df


def get_ccf_labels_image(resampled=True, realigned=False, 
                         subset_to_left_hemi=False):
    '''Loads rasterized image volumes of the CCF parcellation as 3D numpy array.
    
    Voxels are labelled with assigned brain structure parcellation ID #.
    Rasterized voxels are 10 x 10 x 10 micrometers. First (x) axis is 
    anterior-to-posterior, the second (y) axis is superior-to-inferior 
    (dorsal-to-ventral) and third (z) axis is left-to-right.
    
    See ccf_and_parcellation_annotation_tutorial.html and 
    merfish_ccf_registration_tutorial.html in ABC Atlas Data Access JupyterBook
    (https://alleninstitute.github.io/abc_atlas_access/notebooks/) for more
    details on the CCF image volumes.
    
    Parameters
    ----------
    resampled : bool, default=True
        if True, loads the "resampled CCF" labels, which have been aligned into
        the MERFISH space/coordinates
        if False, loads CCF labels that are in AllenCCFv3 average template space
    realigned : bool, default=False
        if resampled and realigned are both True, loads CCF labels from manual realignment, 
        which have been aligned into the MERFISH space/coordinates
        (incompatible with resampled=False as these haven't been calculated)
    subset_to_left_hemi : bool, default=False
        return a trimmed image to use visualizing single-hemisphere results
    
    Returns
    -------
    imdata
        numpy array containing rasterized image volumes of CCF parcellation
    '''
    if resampled and not realigned:
        path = ABC_ROOT/"image_volumes/MERFISH-C57BL6J-638850-CCF/20230630/resampled_annotation.nii.gz"
    elif not resampled and not realigned:
        path = ABC_ROOT/"image_volumes/Allen-CCF-2020/20230630/annotation_10.nii.gz"
    elif resampled and realigned:
        path = "/data/realigned/abc_realigned_ccf_labels.nii.gz"
    else:
        raise UserWarning("This label image is not available")
    img = nibabel.load(path)
    # could maybe keep the lazy dataobj and not convert to numpy?
    imdata = np.array(img.dataobj).astype(int)
    if subset_to_left_hemi:
        # erase right hemisphere (can't drop or indexing won't work correctly)
        imdata[550:,:,:] = 0
        
    return imdata


def label_ccf_spatial_subset(cells_df, ccf_regions,
                             ccf_level='substructure',
                             flip_y=False, distance_px=20, 
                                cleanup_mask=True,
                                filter_cells=False,
                                realigned=False,
                                field_name='spatial_subset'):
    '''Labels cells that are in a spatial subset of the ABC atlas.
    
    Turns a rasterized image volume into a binary mask, then expands the mask
    to ensure coverage of edges despite possible misalignment.
    Adds a boolean column labeling these cells, or returns filtered dataframe.
    
    Parameters
    ----------
    cells_df : pandas dataframe
        dataframe of cell metadata (e.g. adata.obs)
    ccf_regions : list(str)
        list of (abbreviated) CCF region names to select
    distance_px : int, default=20
        dilation radius in pixels (1px = 10um)
    filter_cells : bool, default=False
        filters cells_df to only cells in spatial subset
    flip_y : bool, default=False
        flip y-axis orientation of th_mask so coronal section is right-side up.
        This MUST be set to true if flip_y=True in get_combined_metadata() so
        the cell coordinates and binary mask have the same y-axis orientation
    cleanup_mask : bool, default=True
        removes any regions whose area ratio, as compared to the largest region
        in the binary mask, is lower than 0.1
    realigned : bool, default=False
        use realigned CCF coordinates and image volume
    field_name : bool, default='spatial_subset'
        column to store annotation in (if not filtering)
        
    Returns
    -------
    cells_df 
    '''
    
    # use reconstructed (in MERFISH space) coordinates from cells_df
    if realigned:
        coords = ['x_section','y_section','z_section']
    else:
        coords = ['x_reconstructed','y_reconstructed','z_reconstructed']
    resolutions = np.array([10e-3, 10e-3, 200e-3])
    
    # load 'resampled CCF' (rasterized, in MERFISH space) image volumes from the
    # ABC Atlas dataset (z resolution limited to merscope slices)
    ccf_img = get_ccf_labels_image(resampled=True, realigned=realigned)
    
    # ccf_img voxels are labelled by brain structure parcellation_index, so need
    # to get a list of all indices 
    ccf_index = get_ccf_index(level=ccf_level)
    reverse_lookup = pd.Series(ccf_index.index.values, index=ccf_index)
    index_values = reverse_lookup.loc[ccf_regions]
    
    # generate binary mask
    th_mask = np.isin(ccf_img, index_values) # takes about 5 sec
    # flip y-axis to match flipped cell y-coordinates
    if flip_y:
        th_mask = np.flip(th_mask, axis=1)
    
    mask_img = sectionwise_dilation(th_mask, distance_px, true_radius=False)
    # remove too-small mask regions that are likely mistaken parcellations
    if cleanup_mask:
        mask_img = cleanup_mask_regions(mask_img, area_ratio_thresh=0.1)
    # label cells that fall within dilated TH+ZI mask; by default, 
    cells_df[field_name] = mask_img[
        image_index_from_coords(cells_df[coords], resolutions)
        ]
    if filter_cells:
        return cells_df[cells_df[field_name]].copy().drop(columns=[field_name]), mask_img
    else:
        return cells_df, mask_img

@lru_cache
def _get_ccf_metadata():
    # this metadata hasn't been updated in other versions
    ccf_df = pd.read_csv(
            ABC_ROOT/"metadata/Allen-CCF-2020/20230630/parcellation_to_parcellation_term_membership.csv"
            )
    ccf_df = ccf_df.replace("ZI-unassigned", "ZI")
    return ccf_df

@lru_cache
def _get_devccf_metadata():
    devccf_index = pd.read_csv("/data/KimLabDevCCFv001/KimLabDevCCFv001_MouseOntologyStructure.csv",
                           dtype={'ID':int, 'Parent ID':str})
    # some quotes have both single and double
    for x in ['Acronym','Name']:
        devccf_index[x] = devccf_index[x].str.replace("'","")
    return devccf_index

@lru_cache
def _get_devccf_names(top_nodes):
    devccf_index = _get_devccf_metadata().copy()
    devccf_index['ID'] = devccf_index['ID'].astype(str)
    g = nx.from_pandas_edgelist(devccf_index, source='Parent ID', target='ID', 
                        create_using=nx.DiGraph())
    devccf_index = devccf_index.set_index('Acronym')
    th_ids = list(set.union(*(set(nx.descendants(g, devccf_index.loc[x, 'ID'])) 
                    for x in top_nodes)))
    names = devccf_index.reset_index().set_index('ID').loc[th_ids, 'Acronym']
    return names

@lru_cache
def _get_ccf_names(top_nodes, level=None):
    ccf_df = _get_ccf_metadata()
    th_zi_ind = np.hstack(
            (ccf_df.loc[ccf_df['parcellation_term_acronym']==x, 
                        'parcellation_index'].unique() for x in top_nodes)
    )
    ccf_labels = ccf_df.pivot(index='parcellation_index', values='parcellation_term_acronym', columns='parcellation_term_set_name')
    if level is not None:
        names = ccf_labels.loc[th_zi_ind, level].values
    else:
        names = list(set(ccf_labels.loc[th_zi_ind, :].values.flatten()))
    return names

def get_taxonomy_names(top_nodes, level=None):
    if level=='devccf':
        return _get_devccf_names(top_nodes)
    else:
        return _get_devccf_names(top_nodes, level=level)

def get_ccf_index(level='structure'):
    if level=='devccf':
        ccf_df = _get_devccf_metadata()
        index = ccf_df.set_index('ID')['Acronym']
    else:
        ccf_df = _get_ccf_metadata()
        # parcellation_index to acronym
        index = ccf_df.query(f"parcellation_term_set_name=='{level}'").set_index('parcellation_index')['parcellation_term_acronym']
    return index

@lru_cache
def _get_cluster_annotations(version=CURRENT_VERSION):
    df = pd.read_csv(
        ABC_ROOT/f"metadata/WMB-taxonomy/{version}/cluster_to_cluster_annotation_membership.csv"
    )
    return df

def get_taxonomy_palette(taxonomy_level, version=CURRENT_VERSION):
    df = _get_cluster_annotations(version=version)
    df = df[df["cluster_annotation_term_set_name"]==taxonomy_level]
    palette = df.set_index('cluster_annotation_term_name')['color_hex_triplet'].to_dict()
    return palette
