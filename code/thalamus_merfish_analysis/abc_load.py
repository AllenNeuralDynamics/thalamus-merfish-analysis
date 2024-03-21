from .abc_load_base import *

_DEVCCF_TOP_NODES_THALAMUS = ['ZIC', 'CZI', 'RtC', 'Th']
_CCF_TOP_NODES_THALAMUS = ['TH', 'ZI']


_CIRRO_COLUMNS = {
    'x':'cirro_x',
    'y':'cirro_y',
    'x_section':'cirro_x',
    'y_section':'cirro_y',
    'brain_section_label':'section',
    'parcellation_substructure':'CCF_acronym'
}

def load_adata_thalamus(subset_to_TH_ZI=True, 
               version=CURRENT_VERSION, transform='log2cpv', 
               with_metadata=True, drop_blanks=True, 
               flip_y=False, realigned=False, 
               **kwargs):
    '''Load ABC Atlas MERFISH dataset as an anndata object.
    
    Parameters
    ----------
    subset_to_TH_ZI : bool, default=True
        returns adata that only includes cells in the TH+ZI dataset, as subset
        by label_thalamus_spatial_subset()
    version : str, default=CURRENT_VERSION
        which release version of the ABC Atlas to load
    transform : {'log2cpm', 'log2cpv', 'raw'}, default='log2cpm'
        which transformation of the gene counts to load from the expression matrices.
    with_metadata : bool, default=True
        include cell metadata in adata
    from_metadata : DataFrame, default=None
        preloaded metadata DataFrame to merge into AnnData, loading cells in this 
        DataFrame only (in this case with_metadata is ignored)
    drop_blanks : bool, default=True
        drop 'blank' gene counts from the dataset
        (blanks are barcodes not actually used in the library, counted for QC purposes)
    flip_y : bool, default=True
        flip y-axis coordinates so positive is up (coronal section appears 
        right-side up as expected)
    realigned : bool, default=False
        load and use for subsetting the metadata from realignment results data asset,
        containing 'ccf_realigned' coordinates 
    **kwargs
        passed to `get_combined_metadata`
        
    Results
    -------
    adata
        anndata object containing the ABC Atlas MERFISH dataset
    '''
    if subset_to_TH_ZI:
        cells_md_df = get_combined_metadata(
            version=version, realigned=realigned, flip_y=flip_y, **kwargs
            )
        cells_md_df = label_thalamus_spatial_subset(cells_md_df,
                                                    flip_y=flip_y,
                                                    realigned=realigned,
                                                    distance_px=20,
                                                    cleanup_mask=True,
                                                    drop_end_sections=True,
                                                    filter_cells=True,)
        adata = load_adata(version=version, transform=transform, drop_blanks=drop_blanks,
                           from_metadata=cells_md_df)
    else:
        adata = load_adata(version=version, transform=transform, drop_blanks=drop_blanks,
                           with_metadata=with_metadata, 
                           flip_y=flip_y, realigned=realigned, **kwargs)
    
    return adata


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

    obs = getattr(th_zi_adata, 'obs', th_zi_adata)
    # optionally include midbrain and/or nonneuronal classes
    if not filter_midbrain:
        classes_to_keep += midbrain_classes
    if not filter_nonneuronal:
        classes_to_keep += nonneuronal_classes
    if filter_others:
        obs = filter_by_class(obs, include=classes_to_keep)
    else:
        classes_to_exclude = set(midbrain_classes+nonneuronal_classes) - classes_to_keep
        obs = filter_by_class(obs, exclude=classes_to_exclude)
    return th_zi_adata[obs.index, :]

def filter_by_thalamus_coords(obs, **kwargs):
    obs = filter_by_ccf_region(obs, ['TH', 'ZI'], **kwargs)
    obs = filter_by_section_range(obs, 5.0, 8.21, 0.2)
    return obs

def label_thalamus_spatial_subset(cells_df, drop_end_sections=True,
                                  field_name='TH_ZI_dataset', **kwargs):
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
    ccf_regions = ['TH','ZI']
    cells_df = label_ccf_spatial_subset(cells_df, ccf_regions, field_name=field_name, **kwargs)
    # exclude the 1 anterior-most and 1 posterior-most thalamus sections due to
    # poor overlap between mask & thalamic cells
    if drop_end_sections:
        cells_df = filter_by_section_range(cells_df, 5.0, 8.21, 0.2)
    return cells_df


@lru_cache
def get_thalamus_names(level=None):
    if level=='devccf':
        return get_ccf_names(_DEVCCF_TOP_NODES_THALAMUS, level=level)
    else:
        return get_ccf_names(_CCF_TOP_NODES_THALAMUS, level=level)
    
def get_thalamus_substructure_names():
    return get_thalamus_names(level='substructure')

# load cluster-nucleus annotations
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

    
def get_color_dictionary(labels, taxonomy_level, label_format='id_label',
                         version='20230830', as_list=False):
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
    
    if as_list:
        return colors_list
    else:
        color_dict = dict(zip(labels,colors_list))
        return color_dict
    
        
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