from functools import lru_cache
from pathlib import Path
from abc_atlas_access.abc_atlas_cache.manifest import Manifest

import anndata as ad
import networkx as nx
import nibabel
import numpy as np
import pandas as pd

from .ccf_images import (
    cleanup_mask_regions,
    image_index_from_coords,
    sectionwise_dilation,
)

"""
Functions for loading (subsets of) the ABC Atlas MERFISH dataset.
"""

ABC_ROOT = Path("/data/abc_atlas/")
CURRENT_VERSION = "20230830"
BRAIN_LABEL = "C57BL6J-638850"

# constants for spatial resolution of 'C57BL6J-638850' dataset
X_RESOLUTION = Y_RESOLUTION = 10e-3
Z_RESOLUTION = 200e-3

# hardcoded non-neuronal class categories for v20230830
# TODO: figure out a way to adapt across versions (by name?)
NN_CLASSES = ["30 Astro-Epen", "31 OPC-Oligo", "33 Vascular", "34 Immune"]


class AtlasFiles:
    def __init__(self, directory, dataset, version):
        # TODO: should manifest just be global?
        with open(directory / "releases" / version / "manifest.json", "r") as file:
            manifest = Manifest(cache_dir=directory, json_input=file)
        _data = f"MERFISH-{dataset}"
        _taxonomy = "WMB-taxonomy"
        _data_ccf = f"MERFISH-{dataset}-CCF"
        _ccf = "Allen-CCF-2020"
        self.manifest = manifest
        # TODO: make explicit method?
        self.adata = lambda transform: manifest.get_file_attributes(directory=_data, file_name=f"{dataset}/{transform}")
        self.gene_metadata = manifest.get_file_attributes(directory=_data, file_name="gene")
        self.cell_metadata = manifest.get_file_attributes(directory=_data_ccf, file_name="cell_metadata_with_parcellation_annotation")
        self.resampled_annotation = manifest.get_file_attributes(directory=_data_ccf, file_name="resampled_annotation")
        self.annotation_10 = manifest.get_file_attributes(directory=_ccf, file_name="annotation_10")
        self.ccf_metadata = manifest.get_file_attributes(directory=_ccf, file_name="parcellation_to_parcellation_term_membership")
        self.cluster_metadata = manifest.get_file_attributes(directory=_taxonomy, file_name="cluster_to_cluster_annotation_membership")
        self.taxonomy_metadata = manifest.get_file_attributes(directory=_taxonomy, file_name="cluster_annotation_term_set")

files = AtlasFiles(ABC_ROOT, BRAIN_LABEL, CURRENT_VERSION)
def get_taxonomy_id(): 
    return pd.read_csv(files.taxonomy_metadata.local_path)["label"].iloc[0].split("_")[0]

def load_adata(
    version=CURRENT_VERSION,
    transform="log2cpm",
    with_metadata=True,
    from_metadata=None,
    drop_blanks=True,
    **kwargs,
):
    """
    Load ABC Atlas MERFISH dataset as an anndata object.

    Parameters
    ----------
    version : str, default=CURRENT_VERSION
        which release version of the ABC Atlas to load
    transform : {'log2cpm', 'log2cpv', 'raw'}, default='log2cpm'
        which transformation of the gene counts to load and/or calculate from
        the expression matrices
    with_metadata : bool, default=True
        include cell metadata in adata
    from_metadata : DataFrame, default=None
        preloaded metadata DataFrame to merge into AnnData, loading cells in this
        DataFrame only (in this case with_metadata is ignored)
    drop_blanks : bool, default=True
        drop 'blank' gene counts from the dataset
        (blanks are barcodes not actually used in the library, counted for QC purposes)
    **kwargs
        passed to `get_combined_metadata`

    Results
    -------
    adata
        anndata object containing the ABC Atlas MERFISH dataset
    """
    # 'log2cpv' is labelled just 'log2' in the ABC atlas; for 'log2cpm', we load
    # 'raw' counts and then do the transform manually later
    transform_load = "log2" if transform == "log2cpv" else "raw"
    adata = ad.read_h5ad(
        files.adata(transform_load).local_path, backed="r",
    )
    genes = adata.var_names
    if drop_blanks:
        genes = [gene for gene in genes if "Blank" not in gene]

    if with_metadata or (from_metadata is not None):
        if from_metadata is not None:
            cells_md_df = from_metadata
        else:
            cells_md_df = get_combined_metadata(version=version, **kwargs)
        cell_labels = adata.obs_names.intersection(cells_md_df.index)
        adata = adata[cell_labels, genes]
        adata = adata.to_memory()
        # add metadata to obs
        adata.obs = adata.obs.join(
            cells_md_df.loc[
                cell_labels, cells_md_df.columns.difference(adata.obs.columns)
            ]
        )
    else:
        adata = adata[:, genes].to_memory()

    if transform == "log2cpm":
        # transform calculation converts sparse 'log2' matrix to dense array
        adata.X = np.asarray(
            np.log2(
                1 + adata.X * 1e6 / np.sum(adata.X.toarray(), axis=1, keepdims=True)
            )
        )
    else:
        # convert sparse matrix (how 'log2' & 'raw' counts are stored in h5ad 
        # file) to dense array to match 'log2cpm' transform option
        adata.X = adata.X.toarray()

    # access genes by short symbol vs longer names
    adata.var_names = adata.var["gene_symbol"]

    # note what counts transform is in X
    adata.uns["counts_transform"] = transform

    return adata


def filter_by_ccf_region(
    obs, regions, buffer=0, realigned=False, include_children=True
):
    """Filters cell metadata (obs) dataframe spatially by CCF region labels,
    with an optional buffer region (using stored labels if no buffer).

    Parameters
    ----------
    obs
        dataframe containing cell metadata (i.e. adata.obs)
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
    if include_children:
        regions = get_ccf_names(regions)
    if buffer > 0:
        obs, _ = label_ccf_spatial_subset(
            obs,
            regions,
            distance_px=buffer,
            cleanup_mask=True,
            filter_cells=True,
            realigned=realigned,
        )
    else:
        ccf_label = (
            "parcellation_structure_realigned"
            if realigned
            else "parcellation_structure"
        )
        obs = obs[obs[ccf_label].isin(regions)]
    return obs


def filter_by_class(obs, exclude=NN_CLASSES, include=None):
    """Filters cell metadata (obs) dataframe by cell type taxonomy
    classes. Note that these labels may change across dataset versions!

    Parameters
    ----------
    obs
        dataframe containing cell metadata (i.e. adata.obs)
    exclude : list(str), default=NN_CLASSES
        list of classes to filter out
    include : list(str), default=None
        if present, include ONLY cells in this list of classes

    Returns
    -------
    adata
        the anndata object, filtered to only include cells from specific classes
    """
    if include is not None:
        obs = obs[obs["class"].isin(include)]
    obs = obs[~obs["class"].isin(exclude)]
    return obs


def filter_by_coordinate_range(obs, coord_col, start=None, stop=None):
    """Filters cell metadata (obs) dataframe by a numeric coordinate column,
    restricting to a range from start to stop (inclusive)
    """
    if start is not None:
        obs = obs[obs[coord_col] >= start]
    if stop is not None:
        obs = obs[obs[coord_col] <= stop]
    return obs


def get_gene_metadata(
    version=CURRENT_VERSION,
    drop_blanks=True,
):
    """ Load the gene metadata csv.
    Optionally drops 'Blank' genes from the dataset.
    """
    df = pd.read_csv(files.gene_metadata.local_path)
    if drop_blanks:
        df = df[~df["gene_symbol"].str.contains("Blank")]
    return df

def get_combined_metadata(
    version=CURRENT_VERSION,
    realigned=False,
    drop_unused=True,
    flip_y=False,
    round_z=True,
):
    """Load the cell metadata csv, with memory/speed improvements.
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

    Returns
    -------
        pandas.DataFrame
    """
    float_columns = [
        "average_correlation_score",
        "x_ccf",
        "y_ccf",
        "z_ccf",
        "x_section",
        "y_section",
        "z_section",
        "x_reconstructed",
        "y_reconstructed",
        "z_reconstructed",
    ]
    cat_columns = [
        "brain_section_label",
        "cluster_alias",
        "neurotransmitter",
        "class",
        "subclass",
        "supertype",
        "cluster",
        "parcellation_index",
        "parcellation_division",
        "parcellation_structure",
        "parcellation_substructure",
        # 'parcellation_organ', 'parcellation_category',
    ]
    dtype = dict(
        cell_label="string",
        **{x: "float" for x in float_columns},
        **{x: "category" for x in cat_columns},
    )
    usecols = list(dtype.keys()) if drop_unused else None

    if realigned:
        # TODO: add version to the data asset mount point to allow multiple
        cells_df = pd.read_parquet(
            "/data/realigned/abc_realigned_metadata_thalamus-boundingbox.parquet"
        )
        if version != CURRENT_VERSION:
            old_df = pd.read_csv(
                files.cell_metadata.local_path,
                dtype=dtype,
                usecols=usecols,
                index_col="cell_label",
                engine="c",
            )
            cells_df = old_df.join(
                cells_df[cells_df.columns.difference(old_df.columns)]
            )
    else:
        cells_df = pd.read_csv(
            files.cell_metadata.local_path,
            dtype=dtype,
            usecols=usecols,
            index_col="cell_label",
            engine="c",
        )
    if flip_y:
        cells_df[["y_section", "y_reconstructed"]] *= -1
    if round_z:
        cells_df["z_section"] = cells_df["z_section"].round(1)
        cells_df["z_reconstructed"] = cells_df["z_reconstructed"].round(1)

    cells_df["left_hemisphere"] = cells_df["z_ccf"] < 5.7
    if realigned:
        cells_df["left_hemisphere_realigned"] = cells_df["z_ccf_realigned"] < 5.7
    cells_df = cells_df.replace("ZI-unassigned", "ZI")
    return cells_df


def get_ccf_labels_image(resampled=True, realigned=False, subset_to_left_hemi=False):
    """Loads rasterized image volumes of the CCF parcellation as 3D numpy array.

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
    """
    if resampled and not realigned:
        path = files.resampled_annotation.local_path
    elif not resampled and not realigned:
        path = files.annotation_10.local_path
    elif resampled and realigned:
        path = "/data/realigned/abc_realigned_ccf_labels.nii.gz"
    else:
        raise UserWarning("This label image is not available")
    img = nibabel.load(path)
    # could maybe keep the lazy dataobj and not convert to numpy?
    imdata = np.array(img.dataobj).astype(int)
    if subset_to_left_hemi:
        # erase right hemisphere (can't drop or indexing won't work correctly)
        imdata[550:, :, :] = 0

    return imdata


def label_outlier_celltypes(
    obs,
    type_col,
    min_group_count=5,
    max_num_groups=None,
    outlier_label="other",
    filter_cells=False,
):
    primary_celltypes = (
        obs[type_col].value_counts().loc[lambda x: x > min_group_count].index
    )
    if max_num_groups is not None and len(primary_celltypes) > max_num_groups:
        primary_celltypes = primary_celltypes[:max_num_groups]
    if filter_cells:
        obs = obs[obs[type_col].isin(primary_celltypes)]
    else:
        if (
            obs[type_col].dtype.name == "category"
            and outlier_label not in obs[type_col].cat.categories
        ):
            obs[type_col] = obs[type_col].cat.add_categories(outlier_label)
        obs.loc[~obs[type_col].isin(primary_celltypes), type_col] = outlier_label
    return obs


def label_ccf_spatial_subset(
    cells_df,
    ccf_regions,
    ccf_level="substructure",
    include_children=True,
    flip_y=False,
    distance_px=20,
    cleanup_mask=True,
    filter_cells=False,
    realigned=False,
    field_name="region_mask",
):
    """Labels cells that are in a spatial subset of the ABC atlas.

    Turns a rasterized image volume into a binary mask, then expands the mask
    to ensure coverage of edges despite possible misalignment.
    Adds a boolean column labeling these cells, or returns filtered dataframe.

    Parameters
    ----------
    cells_df : pandas DataFrame
        dataframe of cell metadata (e.g. adata.obs)
    ccf_regions : list(str)
        list of (abbreviated) CCF region names to select
    ccf_level: {'division', 'structure', 'substructure'}, default='substructure'
        level of the CCF hierarchy to return labels from
    include_children : bool, default=True
        include all subregions of the specified ccf_regions
    flip_y : bool, default=False
        flip y-axis orientation of th_mask so coronal section is right-side up.
        This MUST be set to true if flip_y=True in get_combined_metadata() so
        the cell coordinates and binary mask have the same y-axis orientation
    distance_px : int, default=20
        dilation radius in pixels (1px = 10um)
    cleanup_mask : bool, default=True
        removes any regions whose area ratio, as compared to the largest region
        in the binary mask, is lower than 0.1
    filter_cells : bool, default=False
        filters cells_df to only cells in spatial subset
    realigned : bool, default=False
        use realigned CCF coordinates and image volume
    field_name : bool, default='region_mask'
        column to store annotation in (if not filtering)

    Returns
    -------
    cells_df
    """

    # use reconstructed (in MERFISH space) coordinates from cells_df
    if realigned:
        coords = ["x_section", "y_section", "z_section"]
    else:
        coords = ["x_reconstructed", "y_reconstructed", "z_reconstructed"]
    resolutions = np.array([X_RESOLUTION, Y_RESOLUTION, Z_RESOLUTION])

    # load 'resampled CCF' (rasterized, in MERFISH space) image volumes from the
    # ABC Atlas dataset (z resolution limited to merscope slices)
    ccf_img = get_ccf_labels_image(resampled=True, realigned=realigned)

    # ccf_img voxels are labelled by brain structure parcellation_index, so need
    # to get a list of all indices
    if include_children:
        ccf_regions = get_ccf_names(top_nodes=ccf_regions, level=ccf_level)
    reverse_lookup = get_ccf_index_reverse_lookup(level=ccf_level)
    index_values = reverse_lookup.loc[ccf_regions]

    # generate binary mask
    th_mask = np.isin(ccf_img, index_values)  # takes about 5 sec
    # flip y-axis to match flipped cell y-coordinates
    if flip_y:
        th_mask = np.flip(th_mask, axis=1)

    mask_img = sectionwise_dilation(th_mask, distance_px, true_radius=False)
    # remove too-small mask regions that are likely mistaken parcellations
    if cleanup_mask:
        mask_img = cleanup_mask_regions(mask_img, area_ratio_thresh=0.1)
    # label cells that fall within dilated TH+ZI mask; by default,
    cells_df = _label_masked_cells(
        cells_df, mask_img, coords, resolutions, field_name=field_name
    )
    if filter_cells:
        return cells_df[cells_df[field_name]].copy().drop(columns=[field_name])
    else:
        return cells_df


def _label_masked_cells(
    cells_df, mask_img, coords, resolutions, field_name="region_mask"
):
    cells_df[field_name] = mask_img[
        image_index_from_coords(cells_df[coords], resolutions)
    ]
    return cells_df


@lru_cache
def _get_ccf_metadata():
    # TODO: set categorical dtypes?
    ccf_df = pd.read_csv(files.ccf_metadata.local_path)
    ccf_df = ccf_df.replace("ZI-unassigned", "ZI")
    return ccf_df


@lru_cache
def _get_devccf_metadata():
    devccf_index = pd.read_csv(
        "/data/KimLabDevCCFv001/KimLabDevCCFv001_MouseOntologyStructure.csv",
        dtype={"ID": int, "Parent ID": str},
    )
    # some quotes have both single and double
    for x in ["Acronym", "Name"]:
        devccf_index[x] = devccf_index[x].str.replace("'", "")
    return devccf_index


@lru_cache
def _get_devccf_names(top_nodes):
    devccf_index = _get_devccf_metadata().copy()
    devccf_index["ID"] = devccf_index["ID"].astype(str)
    g = nx.from_pandas_edgelist(
        devccf_index, source="Parent ID", target="ID", create_using=nx.DiGraph()
    )
    devccf_index = devccf_index.set_index("Acronym")
    th_ids = list(
        set.union(
            *(set(nx.descendants(g, devccf_index.loc[x, "ID"])) for x in top_nodes)
        )
    )
    names = devccf_index.reset_index().set_index("ID").loc[th_ids, "Acronym"]
    return names


def _get_ccf_names(top_nodes, level=None):
    ccf_df = _get_ccf_metadata()
    th_zi_ind = np.hstack(
        [
            ccf_df.loc[
                ccf_df["parcellation_term_acronym"] == x, "parcellation_index"
            ].unique()
            for x in top_nodes
        ]
    )
    ccf_labels = ccf_df.pivot(
        index="parcellation_index",
        values="parcellation_term_acronym",
        columns="parcellation_term_set_name",
    )
    if level is not None:
        names = sorted(ccf_labels.loc[th_zi_ind, level].values)
    else:
        names = sorted(list(set(ccf_labels.loc[th_zi_ind, :].values.flatten())))
    return names


def get_ccf_names(top_nodes=None, level=None):
    """Get the names of all CCF regions that are children of the
    specified list of top-level regions

    Parameters
    ----------
    top_nodes
        list of top-level regions
    level, optional : {'division', 'structure', 'substructure', 'devccf'}
        level of the CCF hierarchy to return labels from,
        or None to return CCF labels at all levels,
        or 'devccf' to return labels from Kronman et al. 2023 parcellation,
        by default None

    Returns
    -------
        list of region names
    """
    if top_nodes is None:
        return get_ccf_index(level=level).unique()
    if level == "devccf":
        return _get_devccf_names(top_nodes)
    else:
        return _get_ccf_names(top_nodes, level=level)

def get_ccf_index_reverse_lookup(level="substructure"):
    ccf_index = get_ccf_index(level=level)
    reverse_lookup = pd.Series(ccf_index.index.values, index=ccf_index)
    return reverse_lookup

def get_ccf_index(level="substructure"):
    """Get an index mapping CCF ideas to (abbreviated) names,
    at a given taxonomy level

    Parameters
    ----------
    top_nodes
        list of top-level regions
    level, optional : {'division', 'structure', 'substructure', 'devccf'}
        level of the CCF hierarchy to return labels from,
        or 'devccf' to return labels from Kronman et al. 2023 parcellation,
        by default 'substructure' (lowest level)

    Returns
    -------
        Pandas.Series with index CCF IDs and values CCF acronyms
    """
    if level == "devccf":
        ccf_df = _get_devccf_metadata()
        index = ccf_df.set_index("ID")["Acronym"]
    else:
        ccf_df = _get_ccf_metadata()
        # parcellation_index to acronym
        if level is not None:
            ccf_df = ccf_df.query(f"parcellation_term_set_name=='{level}'")
        index = ccf_df.set_index("parcellation_index")["parcellation_term_acronym"]
    return index


def get_section_index(
    cells_df=None, section_col="brain_section_label", z_col="z_section", z_res=200e-3
    ):
    """Given a cell metadata DataFrame, returns a Series mapping section labels
    to their corresponding section index (integer z-coordinate at specified resolution).
    """
    if cells_df is None:
        cells_df = get_combined_metadata()
    section_index = cells_df.groupby(section_col, observed=True)[z_col].first().dropna().apply(
        lambda x: int(np.rint(x / z_res))
    )
    return section_index

@lru_cache
def _get_cluster_annotations(version=CURRENT_VERSION):
    df = pd.read_csv(files.cluster_metadata.local_path)
    return df


def get_taxonomy_palette(taxonomy_level, version=CURRENT_VERSION):
    """Get the published color dictionary for a given level of
    the ABC cell type taxonomy

    Parameters
    ----------
    taxonomy_level : {'cluster', 'supertype', 'subclass', 'class'}
        specifies the taxonomy level to get labels and colors from
    version : str, default=CURRENT_VERSION
        ABC Atlas version of the labels

    Returns
    -------
    color_dict : dict
        dictionary mapping cell type labels to their official ABC Atlas hex colors
    """
    df = _get_cluster_annotations(version=version)
    df = df[df["cluster_annotation_term_set_name"] == taxonomy_level]
    palette = df.set_index("cluster_annotation_term_name")[
        "color_hex_triplet"
    ].to_dict()
    return palette


def get_taxonomy_label_from_alias(
    aliases,
    taxonomy_level,
    version="20230830",
):
    """Given a list of cluster aliases, returns the corresponding cell type
    labels at a given taxonomy level and version.

    Parameters
    ----------
    aliases : list of strings
        list of strings containing the cluster aliases
    taxonomy_level : {'cluster', 'supertype', 'subclass', 'class'}
        specifies the taxonomy level to retrieve
    version : str, default='20230830'
        ABC Atlas version the alias should be converted to

    Results
    -------
    labels
        list of taxonomy labels
    """

    df = _get_cluster_annotations(version=version)
    df = df[df["cluster_annotation_term_set_name"] == taxonomy_level]
    label_list = (
        df.set_index("cluster_alias")
        .loc[aliases, "cluster_annotation_term_name"]
        .to_list()
    )
    return label_list
