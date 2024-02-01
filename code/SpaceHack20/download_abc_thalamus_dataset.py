from pathlib import Path
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
    label_thalamus_masked_cells:
        labels cells that are inside the TH+ZI mask from the CCF parcellation
'''

ABC_ROOT = Path('/data/abc_atlas/')
CURRENT_VERSION = '20230830'
BRAIN_LABEL = 'C57BL6J-638850'
WRITE_DIR = Path('my/path/here')

def load_adata(version=CURRENT_VERSION, transform='log2', with_metadata=True):
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
    with_metadata : bool, default=True
        include cell metadata in adata
        
    Results
    -------
    adata
        anndata object containing the ABC Atlas MERFISH dataset
    '''
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
        # takes ~2 min + ~3 GB of memory to load just one set of counts
        adata = ad.read_h5ad(ABC_ROOT/f"expression_matrices/MERFISH-{BRAIN_LABEL}/{version}/{BRAIN_LABEL}-{transform}.h5ad", 
                             backed='r')
   
    # subset to TH+ZI dataset
    flip_y = True
    cells_md_df = get_combined_metadata(flip_y=flip_y,
                                        round_z=True,
                                        drop_unused=True,
                                        version=version)
    cells_md_df = label_thalamus_spatial_subset(cells_md_df,
                                                flip_y=flip_y,
                                                distance_px=20,
                                                cleanup_mask=True,
                                                drop_end_sections=True,
                                                filter_cells=True)
    cell_labels = cells_md_df.index
    adata = adata[adata.obs_names.intersection(cell_labels)]
    
    if transform!='both':
        adata = adata.to_memory()
        
    # access genes by short symbol vs longer names
    adata.var_names = adata.var['gene_symbol']
    
    if with_metadata:
        # add metadata to obs
        adata.obs = adata.obs.join(cells_md_df[cells_md_df.columns.difference(adata.obs.columns)])

    # note what counts transform is in X
    adata.uns['counts_transform'] = transform
    
    return adata


def filter_adata_by_class(th_zi_adata, filter_nonneuronal=True,
                          filter_midbrain=True):
    ''' Filters anndata object to only include cells from specific taxonomy 
    classes.

    Parameters
    ----------
    th_zi_adata
        anndata object containing the ABC Atlas MERFISH dataset
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
    # hardcoded class categories
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

    th_zi_adata = th_zi_adata[th_zi_adata.obs['class'].isin(classes_to_keep)]
    return th_zi_adata


def get_combined_metadata(drop_unused=True, flip_y=False, 
                          round_z=True, version=CURRENT_VERSION):
    '''Load the cell metadata csv, with memory/speed improvements.
    Selects correct dtypes and optionally renames and drops columns
    
    Parameters
    ----------
    drop_unused : bool, default=True
        don't load uninformative or unused columns (color etc)
    flip_y : bool, default=True
        flip section and reconstructed y coords so up is positive
    round_z : bool, default=True
        rounds z_section, z_reconstructed coords to nearest 10ths place to
        correct for overprecision in a handful of z coords
    version : str, optional
        version to load, by default=CURRENT_VERSION

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

    # if as_dask:
    #     cells_df = dd.read_csv(
    #         ABC_ROOT/f"metadata/MERFISH-C57BL6J-638850-CCF/{version}/views/cell_metadata_with_parcellation_annotation.csv", 
    #                         dtype=dtype, usecols=usecols, blocksize=100e6)
    # else:
    cells_df = pd.read_csv(
            ABC_ROOT/f"metadata/MERFISH-C57BL6J-638850-CCF/{version}/views/cell_metadata_with_parcellation_annotation.csv", 
                           dtype=dtype, usecols=usecols, index_col='cell_label', 
                           engine='c')
    if flip_y:
        cells_df[['y_section', 'y_reconstructed']] *= -1
    if round_z:
        cells_df['z_section'] = cells_df['z_section'].round(1)
        cells_df['z_reconstructed'] = cells_df['z_reconstructed'].round(1)

    return cells_df


def get_ccf_labels_image(resampled=True):
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
    
    Returns
    -------
    imdata
        numpy array containing rasterized image volumes of CCF parcellation
    '''
    if resampled:
        path = "image_volumes/MERFISH-C57BL6J-638850-CCF/20230630/resampled_annotation.nii.gz"
    else:
        path = "image_volumes/Allen-CCF-2020/20230630/annotation_10.nii.gz"
    img = nibabel.load(ABC_ROOT/path)
    # could maybe keep the lazy dataobj and not convert to numpy?
    imdata = np.array(img.dataobj)
    return imdata


def label_thalamus_spatial_subset(cells_df, flip_y=False, distance_px=20, 
                                  cleanup_mask=True, drop_end_sections=True,
                                  filter_cells=False):
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
    coords = ['x_reconstructed','y_reconstructed','z_reconstructed']
    resolutions = np.array([10e-3, 10e-3, 200e-3])
    
    # load 'resampled CCF' (rasterized, in MERFISH space) image volumes from the
    # ABC Atlas dataset (z resolution limited to merscope slices)
    ccf_img = get_ccf_labels_image(resampled=True)
    
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
    cells_df = label_thalamus_masked_cells(cells_df, mask_img, coords,  
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

    
def curate_sections()


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


def label_thalamus_masked_cells(cells_df, mask_img, coords, resolutions,
                                field_name='TH_ZI_dataset'):
    '''Labels cells that are inside the TH+ZI mask from the CCF parcellation.
    
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
    drop_end_sections : bool, default=True
        drops the anterior-most section and posterior-most section, which
        contain CCF thalamus parcellation labels but have poor overlap with
        thalamic cell types

    Returns
    -------
    cells_df 
        with a new boolean column specifying which cells are in the thalamus dataset
    '''
    coords_index = np.rint(cells_df[coords].values / resolutions).astype(int)
    # tuple() makes this like calling mask_img[coords_index[:,0], coords_index[:,1], coords_index[:,2]]
    cells_df[field_name] = mask_img[tuple(coords_index.T)]
    return cells_df


if __name__=="__main__":
    th_zi_adata = load_adata(version=CURRENT_VERSION, transform='log2', 
                             with_metadata=True)
    
    th_zi_adata_neurons = filter_adata_by_class(th_zi_adata, 
                                                filter_nonneuronal=True,
                                                filter_midbrain=True)
    
    th_zi_adata_neurons.write_h5ad(Path(WRITE_DIR,
                                        ('abc_atlas_merfish_'+BRAIN_LABEL+'_v'
                                         +CURRENT_VERSION+'_TH_ZI_neurons.h5ad')
                                       ), 
                                   compression='gzip')