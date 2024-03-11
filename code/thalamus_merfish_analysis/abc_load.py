from pathlib import Path
from functools import lru_cache
from itertools import chain
import json
import numpy as np
import pandas as pd
import anndata as ad
import scipy.ndimage as ndi
import nibabel

'''
Functions for loading a thalamus + zona incerta (TH+ZI) spatial subset of the
ABC Atlas MERFISH dataset.

functions:
    load_adata:
        loads ABC Atlas MERFISH dataset as an anndata object
    filter_adata_by_class:
        filters anndata obj to only include cells from TH & ZI taxonomy classes
    get_combined_metadata:
        loads the cell metadata csv, with memory/speed improvements
    get_ccf_labels_image:
        loads rasterized image volumes of the CCF parcellation
    label_thalamus_spatial_subset:
        labels cells that are in the thalamus spatial subset of the ABC atlas
    sectionwise_dilation:
        dilates a stack of 2D binary masks by a specified radius (in px)
    cleanup_mask_regions:
        removes too-small, mistaken parcellation regions from TH/ZI binary masks
    convert_taxonomy_labels:
        converts cell type labels between different taxonomy versions
    get_color_dictionary:
        returns dictionary of ABC Atlas hex colors for an input list of cell 
        type labels
'''

ABC_ROOT = Path("/data/abc_atlas/")
CURRENT_VERSION = "20230830"
BRAIN_LABEL = 'C57BL6J-638850'

_CIRRO_COLUMNS = {
    'x':'cirro_x',
    'y':'cirro_y',
    'x_section':'cirro_x',
    'y_section':'cirro_y',
    'brain_section_label':'section',
    'parcellation_substructure':'CCF_acronym'
}
        

def load_adata(version=CURRENT_VERSION, transform='log2', subset_to_TH_ZI=True,
               with_metadata=True, flip_y=True, round_z=True, cirro_names=False, 
               with_colors=False,
               realigned=False,
               loaded_metadata=None):
    '''Load ABC Atlas MERFISH dataset as an anndata object.
    
    Parameters
    ----------
    version : str, default=CURRENT_VERSION
        which release version of the ABC Atlas to load
    transform : {'log2', 'raw', 'both'}, default='log2'
        which transformation of the gene counts to load from the expression matrices.
        if 'both', log2 is stored in X and log2 & raw are stored in adata.layers.
        use both if writing to a permanent file or performing mapping on the 
        output; log2 for a smaller object for plotting & most other analyses.
    subset_to_TH_ZI : bool, default=True
        returns adata that only includes cells in the TH+ZI dataset, as subset
        by label_thalamus_spatial_subset()
    with_metadata : bool, default=True
        include cell metadata in adata
    flip_y : bool, default=True
        flip y-axis coordinates so positive is up (coronal section appears 
        right-side up as expected)
    round_z : bool, default=True
        rounds z_section, z_reconstructed coords to nearest 10ths place to
        correct for overprecision in a handful of z coords
    cirro_names : bool, default=False
        changes metadata field names according to _CIRRO_COLUMNS dictionary
    with_colors : bool, default=False
        imports all colors with the metadata (will take up more space)
    realigned : bool, default=False
        load and use for subsetting the metadata from realignment results data asset,
        containing 'ccf_realigned' coordinates 
    loaded_metadata : DataFrame, default=None
        already loaded metadata DataFrame to merge into AnnData, loading cells in this 
        DataFrame only
        
    Results
    -------
    adata
        anndata object containing the ABC Atlas MERFISH dataset
    '''
    # TODO: add option for true CPM? (vs log2CPV?)
    if transform=='both':
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
        
    if with_metadata or subset_to_TH_ZI or subset_to_left_hemi:
        if loaded_metadata is not None:
            cells_md_df = loaded_metadata
        else:
            cells_md_df = get_combined_metadata(cirro_names=cirro_names, 
                                        flip_y=flip_y,
                                        round_z=round_z,
                                        drop_unused=(not with_colors),
                                        version=version,
                                        realigned=realigned)
        # subset to TH+ZI dataset
        if subset_to_TH_ZI:
            cells_md_df = label_thalamus_spatial_subset(cells_md_df,
                                                        flip_y=flip_y,
                                                        distance_px=20,
                                                        cleanup_mask=True,
                                                        drop_end_sections=True,
                                                        filter_cells=True,
                                                        realigned=realigned)
        cell_labels = adata.obs_names.intersection(cells_md_df.index)
        adata = adata[cell_labels]
        adata = adata.to_memory()
        # add metadata to obs
        adata.obs = adata.obs.join(cells_md_df.loc[cell_labels, cells_md_df.columns.difference(adata.obs.columns)])
    
    if adata.isbacked:
        adata = adata.to_memory()
        
    # access genes by short symbol vs longer names
    adata.var_names = adata.var['gene_symbol']

    # note what counts transform is in X
    adata.uns['counts_transform'] = transform
    
    return adata

def add_tiled_obsm(adata, offset=10, coords_name='section', obsm_field='coords_tiled'):
    """Add obsm coordinates to AnnData object from section coordinates in adata.obs,
    but with sections tiled along the x-axis at a given separation.

    Parameters
    ----------
    adata
        AnnData object
    offset, optional
        separation of sections, by default 10 (5 sufficient for hemi-sections)
    coords_name, optional
        suffix of the coordinate type to use, by default 'section'
    obsm_field, optional
        name for the new obsm entry, by default 'coords_tiled'

    Returns
    -------
        AnnData object, modified in place
    """
    obsm = np.vstack([adata.obs[f"x_{coords_name}"] 
                        + offset*adata.obs[f"z_{coords_name}"].rank(method="dense"),
                        adata.obs[f"y_{coords_name}"]]).T
    adata.obsm[coords_tiled] = obsm
    return adata

def filter_by_thalamus_coords(obs, realigned=False, buffer=0):
    if buffer > 0:
        obs = label_thalamus_spatial_subset(obs,
                                    distance_px=buffer,
                                    cleanup_mask=True,
                                    drop_end_sections=True,
                                    filter_cells=True,
                                    realigned=realigned)
    else:
        ccf_label = 'parcellation_structure_realigned' if realigned else 'parcellation_structure'
        th_names = get_thalamus_names(level='structure')
        obs = obs[obs[ccf_label].isin(th_names)]
    return obs

def filter_adata_by_class(th_zi_adata, filter_nonneuronal=True,
                          filter_midbrain=True, filter_others=True):
    ''' Filters anndata object to only include cells from specific taxonomy 
    classes.

    Parameters
    ----------
    th_zi_adata
        anndata object or dataframe containing the ABC Atlas MERFISH dataset
    filter_nonneuronal : bool, default=True
        filters out non-neuronal classes
    filter_midbrain : bool, default=True
        filters out midbrain classes; may be useful to keep these if interested
        in analyzing midbrain-thalamus boundary in the anterior

    Returns 
    -------
    th_zi_adata
        the anndata object, filtered to only include cells from specific 
        thalamic & zona incerta + optional (midbrain & nonneuronal) classes
    '''
    # hardcoded class categories for v20230830
    th_zi_dataset_classes = ['12 HY GABA', '17 MH-LH Glut', '18 TH Glut']
    midbrain_classes = ['19 MB Glut', '20 MB GABA']
    nonneuronal_classes = ['30 Astro-Epen', '31 OPC-Oligo', '33 Vascular',
                           '34 Immune']

    # always keep th_zi_dataset_classes
    classes_to_keep = th_zi_dataset_classes.copy()

    # optionally include midbrain and/or nonneuronal classes
    if not filter_midbrain:
        classes_to_keep += midbrain_classes
    if not filter_nonneuronal:
        classes_to_keep += nonneuronal_classes
    if filter_others:
        if hasattr(th_zi_adata, 'obs'):
            subset = th_zi_adata.obs['class'].isin(classes_to_keep)
        else:
            subset = th_zi_adata['class'].isin(classes_to_keep)
    else:
        classes_to_exclude = set(midbrain_classes+nonneuronal_classes) - classes_to_keep
        if hasattr(th_zi_adata, 'obs'):
            subset = ~th_zi_adata.obs['class'].isin(classes_to_exclude)
        else:
            subset = ~th_zi_adata['class'].isin(classes_to_exclude)
    return th_zi_adata[subset]


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


def get_ccf_labels_image(resampled=True, realigned=False, subset_to_left_hemi=False):
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


def label_thalamus_spatial_subset(cells_df, flip_y=False, distance_px=20, 
                                  cleanup_mask=True, drop_end_sections=True,
                                  filter_cells=False,
                                  realigned=False):
    '''Labels cells that are in the thalamus spatial subset of the ABC atlas.
    
    Turns a rasterized image volume that includes all thalamus (TH) and zona
    incerta (ZI) CCF structures in a binary mask, then dilates by 200um (20px)
    to ensure inclusion of the vast majority cells in known thalamic subclasses.
    Labels cells that fall in this dilate binary mask as in the 'TH_ZI_dataset' 
    
    Parameters
    ----------
    cells_df : pandas dataframe
        dataframe of cell metadata
    distance_px : int, default=20
        dilation radius in pixels (1px = 10um)
    filter_cells : bool, default=False
        filters cells_df to remove non-TH+ZI cells
    flip_y : bool, default=False
        flip y-axis orientation of th_mask so coronal section is right-side up.
        This MUST be set to true if flip_y=True in get_combined_metadata() so
        the cell coordinates and binary mask have the same y-axis orientation
    cleanup_mask : bool, default=True
        removes any regions whose area ratio, as compared to the largest region
        in the binary mask, is lower than 0.1
        
    Returns
    -------
    cells_df 
        with a new boolean column specifying which cells are in the TH+ZI dataset
    '''
    field_name='TH_ZI_dataset'
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
    # to get a list of all indices that correspond to TH or ZI (sub)structures
    ccf_df = pd.read_csv(
            ABC_ROOT/"metadata/Allen-CCF-2020/20230630/parcellation_to_parcellation_term_membership.csv"
            )
    th_zi_ind = np.hstack(
                    (ccf_df.loc[ccf_df['parcellation_term_acronym']=='TH', 
                                'parcellation_index'].unique(),
                     ccf_df.loc[ccf_df['parcellation_term_acronym']=='ZI', 
                                'parcellation_index'].unique())
                )
    
    # generate binary mask
    th_mask = np.isin(ccf_img, th_zi_ind) # takes about 5 sec
    # flip y-axis to match flipped cell y-coordinates
    if flip_y:
        th_mask = np.flip(th_mask, axis=1)
    # dilate by 200um to try to capture more TH/ZI cells
    mask_img = sectionwise_dilation(th_mask, distance_px, true_radius=False)
    # remove too-small mask regions that are likely mistaken parcellations
    if cleanup_mask:
        mask_img = cleanup_mask_regions(mask_img, area_ratio_thresh=0.1)
    # label cells that fall within dilated TH+ZI mask; by default, 
    cells_df = label_masked_cells(cells_df, mask_img, coords,  
                                           resolutions, field_name=field_name)
    # exclude the 1 anterior-most and 1 posterior-most thalamus sections due to
    # poor overlap between mask & thalamic cells
    if drop_end_sections:
        cells_df[field_name] = (cells_df[field_name] 
                                & (4.81 < cells_df[coords[2]]) 
                                & (cells_df[coords[2]] < 8.39))
    # optionally, remove non-TH+ZI cells from df
    if filter_cells:
        return cells_df[cells_df[field_name]].copy().drop(columns=[field_name])
    else:
        return cells_df


def sectionwise_dilation(mask_img, distance_px, true_radius=False):
    '''Dilates a stack of 2D binary masks by a specified radius (in px).
    
    Parameters
    ----------
    mask_img : array_like
        stack of 2D binary mask, shape (x, y, n_sections)
    distance_px : int
        dilation radius in pixels
    true_radius : bool, default=False
        specifies the method used by ndimage's binary_dilation to dilate
          - False: dilates by 1 px per iteration for iterations=distance_px
          - True: dilates once using a structure of radius=distance_px
        both return similar results but true_radius=False is significantly faster

    Returns
    -------
    dilated_mask_img 
        3D np array, stack of dilated 2D binary masks
    '''
    dilated_mask_img = np.zeros_like(mask_img)
    
    if true_radius:
        # generate a circular structure for dilation
        coords = np.mgrid[-distance_px:distance_px+1, -distance_px:distance_px+1]
        struct = np.linalg.norm(coords, axis=0) <= distance_px
        
    for i in range(mask_img.shape[2]):
        if true_radius:
            dilated_mask_img[:,:,i] = ndi.binary_dilation(mask_img[:,:,i], 
                                                          structure=struct)
        else:
            dilated_mask_img[:,:,i] = ndi.binary_dilation(mask_img[:,:,i], 
                                                           iterations=distance_px)
    return dilated_mask_img


def cleanup_mask_regions(mask_img, area_ratio_thresh=0.1):
    ''' Removes, sectionwise, any binary mask regions whose areas are smaller
    than the specified ratio, as compared to the largest region in the mask.
    
    Parameters
    ----------
    mask_img : array_like
        stack of 2D binary mask, shape (x, y, n_sections)
    area_ratio_thresh : float, default=0.1
        threshold for this_region:largest_region area difference ratio; removes
        any regions smaller than this threshold
        
    Returns
    -------
    new_mask_img
        stack of 2D binary masks with too-small regions removed
    '''
    new_mask_img = np.zeros_like(mask_img)
    for sec in range(mask_img.shape[2]):
        mask_2d = mask_img[:,:,sec]
        labeled_mask, n_regions = ndi.label(mask_2d)

        # calculate the area of the largest region
        largest_region = np.argmax(ndi.sum(mask_2d, labeled_mask, 
                                           range(n_regions+1)))
        largest_area = np.sum(labeled_mask==largest_region)

        # filter out regions with area ratio smaller than the specified threshold
        regions_to_keep = [label for label 
                           in range(1, n_regions+1) 
                           if ( (np.sum(labeled_mask==label) / largest_area) 
                                >= area_ratio_thresh
                              )
                          ]
        # make a new mask with only the remaining objects
        new_mask_img[:,:,sec] = np.isin(labeled_mask, regions_to_keep)

    return new_mask_img


def label_masked_cells(cells_df, mask_img, coords, resolutions,
                                field_name='TH_ZI_dataset'):
    '''Labels cells with coordinates inside a binary masked image region
    
    Parameters
    ----------
    cells_df : pandas dataframe
        dataframe of cell metadata
    mask_img : array_like
        stack of 2D binary masks, shape (x, y, n_sections)
    coords : list
        column names in cells_df that contain the cells xyz coordinates, 
        list of strings of length 3
    resolutions : array
        xyz resolutions used to compare coords to mask_img positions
    field_name : str
        name for column containing the thalamus dataset boolean flag

    Returns
    -------
    cells_df 
        with a new boolean column specifying which cells are in the thalamus dataset
    '''
    coords_index = np.rint(cells_df[coords].values / resolutions).astype(int)
    # tuple() makes this like calling mask_img[coords_index[:,0], coords_index[:,1], coords_index[:,2]]
    cells_df[field_name] = mask_img[tuple(coords_index.T)]
    return cells_df


def convert_taxonomy_labels(input_labels, taxonomy_level, 
                            label_format='id_label',
                            input_version='20230630', output_version='20230830',
                            output_as_dict=False):
    ''' Converts cell type labels between taxonomy versions of the ABC Atlas.
    
    Parameters
    ----------
    labels : list of strings
        list of strings containing the cell type labels to be converted
    taxonomy_level : {'cluster', 'supertype', 'subclass', 'class'}
        specifies the taxonomy level to which 'labels' belong
    label_format : string, {'id_label', 'id', 'label'}, default='id_label'
        indicates format of 'labels' parameter; currently only supports full
        id+label strings, e.g. "1130 TH Prkcd Grin2c Glut_1"
        [TODO: support 'id'-only ("1130") & 
               'label'-only ("TH Prkcd Grin2c Glut_1") user inputs]
    input_version : str, default='20230630'
        ABC Atlas version to which 'labels' belong
    output_version : str, default='20230830'
        ABC Atlas version the labels should be converted to
    output_as_dict : bool, default=False
        specifies whether output is a list (False, default) or dictionary (True)
    
    Results
    -------
    output_labels
        list of converted labels or dictionary mapping from input to converted
        labels, depending 
    '''
    
    # load in the correct cluster annotation membership CSV files
    file = 'cluster_to_cluster_annotation_membership_pivoted.csv'
    in_pivot_df = pd.read_csv(
                        ABC_ROOT/f'metadata/WMB-taxonomy/{input_version}/views/{file}'
                        )
    out_pivot_df = pd.read_csv(
                        ABC_ROOT/f'metadata/WMB-taxonomy/{output_version}/views/{file}'
                        )
    
    # get cluster_alias list, which is stable between ABC atlas taxonomy versions
    in_query_df = in_pivot_df.set_index(taxonomy_level).loc[input_labels].reset_index()
    # clusters are unique, but other taxonomy levels exist in multiple rows
    if taxonomy_level=='cluster':
        cluster_alias_list = in_query_df['cluster_alias'].to_list()
    else:
        # dict() only keeps the last instance of a key due to overwriting,
        # which I'm exploiting to remove duplicates
        cluster_alias_list = list(dict(zip(in_query_df[taxonomy_level],
                                           in_query_df['cluster_alias'])
                                      ).values())
    # use cluster_alias to map to output labels
    out_query_df = out_pivot_df.set_index('cluster_alias').loc[cluster_alias_list].reset_index()
    out_labels_list = out_query_df[taxonomy_level].to_list()
    
    if output_as_dict:
        out_labels_dict = dict(zip(input_labels,out_labels_list))
        return out_labels_dict
    else:
        return out_labels_list

    
def get_taxonomy_label_from_alias(aliases, taxonomy_level, version='20230830',
                                  label_format='id_label',
                                  output_as_dict=False):
    ''' Converts cell type labels between taxonomy versions of the ABC Atlas.
    
    Parameters
    ----------
    aliases : list of strings
        list of strings containing the cluster aliases
    taxonomy_level : {'cluster', 'supertype', 'subclass', 'class'}
        specifies the taxonomy level to retrieve
    label_format : string, {'id_label', 'id', 'label'}, default='id_label'
        indicates format of 'labels' parameter; currently only supports full
        id+label strings, e.g. "1130 TH Prkcd Grin2c Glut_1"
        [TODO: support 'id'-only ("1130") & 
               'label'-only ("TH Prkcd Grin2c Glut_1") user inputs]
    version : str, default='20230830'
        ABC Atlas version the alias should be converted to
    output_as_dict : bool, default=False
        specifies whether output is a list (False, default) or dictionary (True)
    
    Results
    -------
    labels
        list of taxonomy labels or dictionary mapping from alias to taxonomy
        labels
    '''
    
    # load in the specified version of cluster annotation membership CSV files
    file = 'cluster_to_cluster_annotation_membership_pivoted.csv'
    pivot_df = pd.read_csv(
                    ABC_ROOT/f'metadata/WMB-taxonomy/{version}/views/{file}',
                    dtype='str')
    # reindexing ensures that label_list is in same order as input aliases
    query_df = pivot_df.set_index('cluster_alias').loc[aliases].reset_index()
    label_list = query_df[taxonomy_level].to_list()
    if output_as_dict:
        labels_dict = dict(zip(aliases, label_list))
        return labels_dict
    else:
        return label_list
    
def get_color_dictionary(labels, taxonomy_level, label_format='id_label',
                         version='20230830'):
    ''' Returns a color dictionary for the specified cell types labels.
    
    Parameters
    ----------
    labels : list of strings
        list of strings containing the cell type labels to be converted
    taxonomy_level : {'cluster', 'supertype', 'subclass', 'class'}
        specifies the taxonomy level to which 'labels' belong
    label_format : string, {'id_label', 'id', 'label'}, default='id_label'
        indicates format of 'labels' parameter; currently only supports full
        id+label strings, e.g. "1130 TH Prkcd Grin2c Glut_1"
        [TODO: support 'id'-only ("1130") & 
               'label'-only ("TH Prkcd Grin2c Glut_1") user inputs]
    version : str, default='20230830'
        ABC Atlas version of the labels; cannot get colors from a different
        version than your labels; to do that, first use convert_taxonomy_labels())
    
    Results
    -------
    color_dict : dict
        dictionary mapping input 'labels' to their official ABC Atlas hex colors
    '''
    # load metadata csv files
    pivot_file = 'cluster_to_cluster_annotation_membership_pivoted.csv'
    color_file = 'cluster_to_cluster_annotation_membership_color.csv'
    pivot_df = pd.read_csv(
                    ABC_ROOT/f'metadata/WMB-taxonomy/{version}/views/{pivot_file}'
                    )
    color_df = pd.read_csv(
                    ABC_ROOT/f'metadata/WMB-taxonomy/{version}/views/{color_file}'
                    )
    
    # get cluster_alias list, which is stable between ABC atlas taxonomy versions
    pivot_query_df = pivot_df.set_index(taxonomy_level).loc[labels].reset_index()
    # clusters are unique, but other taxonomy levels exist in multiple rows
    if taxonomy_level=='cluster':
        cluster_alias_list = pivot_query_df['cluster_alias'].to_list()
    else:
        # dict() only keeps the last instance of a key due to overwriting,
        # which I'm exploiting to remove duplicates while maintaining order
        cluster_alias_list = list(dict(zip(pivot_query_df[taxonomy_level],
                                           pivot_query_df['cluster_alias'])
                                      ).values())
    # use cluster_alias to map to colors
    color_query_df = color_df.set_index('cluster_alias').loc[cluster_alias_list].reset_index()
    colors_list = color_query_df[taxonomy_level+'_color'].to_list()

    color_dict = dict(zip(labels,colors_list))
    
    return color_dict

@lru_cache
def get_ccf_metadata():
    # this metadata hasn't been updated in other versions
    ccf_df = pd.read_csv(
            ABC_ROOT/"metadata/Allen-CCF-2020/20230630/parcellation_to_parcellation_term_membership.csv"
            )
    ccf_df = ccf_df.replace("ZI-unassigned", "ZI")
    return ccf_df

@lru_cache
def get_devccf_metadata():
    devccf_index = pd.read_csv("/data/KimLabDevCCFv001/KimLabDevCCFv001_MouseOntologyStructure.csv",
                           dtype={'ID':int, 'Parent ID':str})
    # some quotes have both single and double
    for x in ['Acronym','Name']:
        devccf_index[x] = devccf_index[x].str.replace("'","")
    return devccf_index

import networkx as nx
@lru_cache
def get_thalamus_names(level=None):
    if level=='devccf':
        devccf_index = get_devccf_metadata().copy()
        devccf_index['ID'] = devccf_index['ID'].astype(str)
        th_top_level = ['ZIC', 'CZI', 'RtC', 'Th']
        g = nx.from_pandas_edgelist(devccf_index, source='Parent ID', target='ID', 
                            create_using=nx.DiGraph())
        devccf_index = devccf_index.set_index('Acronym')
        th_ids = list(set.union(*(set(nx.descendants(g, devccf_index.loc[x, 'ID'])) 
                        for x in th_top_level)))
        th_names = devccf_index.reset_index().set_index('ID').loc[th_ids, 'Acronym']
    else:
        ccf_df = get_ccf_metadata()
        th_zi_ind = np.hstack(
                (ccf_df.loc[ccf_df['parcellation_term_acronym']=='TH', 
                            'parcellation_index'].unique(),
                    ccf_df.loc[ccf_df['parcellation_term_acronym']=='ZI', 
                            'parcellation_index'].unique())
        )
        ccf_labels = ccf_df.pivot(index='parcellation_index', values='parcellation_term_acronym', columns='parcellation_term_set_name')
        if level is not None:
            th_names = ccf_labels.loc[th_zi_ind, level].values
        else:
            th_names = list(set(ccf_labels.loc[th_zi_ind, :].values.flatten()))
    return th_names

def get_thalamus_substructure_names():
    return get_thalamus_names(level='substructure')

@lru_cache
def get_ccf_index(level='structure'):
    if level=='devccf':
        ccf_df = get_devccf_metadata()
        index = ccf_df.set_index('ID')['Acronym']
    else:
        ccf_df = get_ccf_metadata()
        # parcellation_index to acronym
        index = ccf_df.query(f"parcellation_term_set_name=='{level}'").set_index('parcellation_index')['parcellation_term_acronym']
    return index

@lru_cache
def _get_cluster_annotations(version=CURRENT_VERSION):
    df = pd.read_csv(
        ABC_ROOT/f"metadata/WMB-taxonomy/{version}/cluster_to_cluster_annotation_membership.csv"
    )
    return df

@lru_cache
def get_taxonomy_palette(taxonomy_level, version=CURRENT_VERSION):
    df = _get_cluster_annotations(version=version)
    df = df[df["cluster_annotation_term_set_name"]==taxonomy_level]
    palette = df.set_index('cluster_annotation_term_name')['color_hex_triplet'].to_dict()
    return palette

try:
    nuclei_df_manual = pd.read_csv("/code/resources/prong1_cluster_annotations_by_nucleus.csv", index_col=0)
    nuclei_df_manual = nuclei_df_manual.fillna("")
    nuclei_df_auto = pd.read_csv("/code/resources/annotations_from_eroded_counts.csv",  index_col=0)
    found_annotations = True
except:
    found_annotations = False

def get_obs_from_annotated_clusters(name, obs, by='id', include_shared_clusters=False, manual_annotations=True):
    if not found_annotations:
        raise UserWarning("Can't access annotations sheet from this environment.")
    # if name not in nuclei_df.index:
    #     raise UserWarning("Name not found in annotations sheet")
    nuclei_df = nuclei_df_manual if manual_annotations else nuclei_df_auto
    if include_shared_clusters:
        names = [x for x in nuclei_df.index if any(name in y and not 'pc' in y
                                                    for y in x.split(" "))]
    else: 
        names = [x for x in nuclei_df.index if name in x 
                 and not (' ' in x or 'pc' in x)]
    
    dfs = []
    field = "cluster_alias" if by=='alias' else "cluster_ids_CNN20230720"
    clusters = chain(*[nuclei_df.loc[name, field].split(', ') for name in names])
    if by=='alias':
        obs = obs.loc[lambda df: df['cluster_alias'].isin(clusters)]
    elif by=='id':
        obs = obs.loc[lambda df: df['cluster'].str[:4].isin(clusters)]
    return obs