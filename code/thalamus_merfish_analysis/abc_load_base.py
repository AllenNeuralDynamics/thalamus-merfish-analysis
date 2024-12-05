"""
Functions for loading (subsets of) the ABC Atlas MERFISH dataset.
"""

from collections import defaultdict
from functools import cached_property, wraps, lru_cache
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import nibabel
import numpy as np
import pandas as pd
from abc_atlas_access.abc_atlas_cache.manifest import Manifest

from .ccf_images import (
    cleanup_mask_regions,
    image_index_from_coords,
    sectionwise_dilation,
    sectionwise_closing,
    sectionwise_fill_holes,
)
try:
    from importlib.resources import files
except (ImportError, ModuleNotFoundError):
    from importlib_resources import files
ABC_ROOT = Path("/data/abc_atlas/")
CURRENT_VERSION = "20230830"
BRAIN_LABEL = "C57BL6J-638850"


def accept_anndata_input(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        # check first two args for AnnData in case this is an instance or class method
        for i, arg in enumerate(args):
            if i<=1 and isinstance(arg, ad.AnnData):
                adata = arg.copy()
                result = f(*args[:i], adata.obs, *args[i + 1 :], **kwargs)
                result = adata[result.index, :]
                break
        else:
            result = f(*args, **kwargs)
        return result
    return wrapper

class AtlasWrapper:
    """
    Wrapper class for loading and processing ABC Atlas MERFISH dataset files.
    Keeps track of key file paths and versions to use for loading, filtering etc.
    """

    # constants for spatial resolution of 'C57BL6J-638850' dataset
    X_RESOLUTION = Y_RESOLUTION = 10e-3
    Z_RESOLUTION = 200e-3

    def __init__(self, directory=ABC_ROOT, dataset=BRAIN_LABEL, version=CURRENT_VERSION):
        # TODO: should manifest just be global?
        with open(directory / "releases" / version / "manifest.json", "r") as file:
            manifest = Manifest(cache_dir=directory, json_input=file)
        _data = f"MERFISH-{dataset}"
        _taxonomy = "WMB-taxonomy"
        _data_ccf = f"MERFISH-{dataset}-CCF"
        _ccf = "Allen-CCF-2020"
        self.dataset = dataset
        self.version = version
        self.manifest = manifest
        self.files = SimpleNamespace(
            adata_raw=manifest.get_file_attributes(directory=_data, file_name=f"{dataset}/raw"),
            adata_log2=manifest.get_file_attributes(directory=_data, file_name=f"{dataset}/log2"),
            gene_metadata=manifest.get_file_attributes(directory=_data, file_name="gene"),
            cell_metadata=manifest.get_file_attributes(
                directory=_data_ccf, file_name="cell_metadata_with_parcellation_annotation"
            ),
            resampled_annotation=manifest.get_file_attributes(
                directory=_data_ccf, file_name="resampled_annotation"
            ),
            annotation_10=manifest.get_file_attributes(directory=_ccf, file_name="annotation_10"),
            ccf_metadata=manifest.get_file_attributes(
                directory=_ccf, file_name="parcellation_to_parcellation_term_membership"
            ),
            cluster_metadata_full=manifest.get_file_attributes(
                directory=_taxonomy, file_name="cluster_to_cluster_annotation_membership"
            ),
            cluster_metadata=manifest.get_file_attributes(
                directory=_taxonomy, file_name="cluster_to_cluster_annotation_membership_pivoted"
            ),
            taxonomy_metadata=manifest.get_file_attributes(
                directory=_taxonomy, file_name="cluster_annotation_term_set"
            ),
            scrnaseq_metadata=manifest.get_file_attributes(
                directory="WMB-10X", file_name="cell_metadata_with_cluster_annotation"
            ),
        )
    @cached_property
    def NN_CLASSES(self):
        return self._cluster_annotations.loc[
            lambda df: df["subclass"].str.contains("NN"), "class"
        ].unique().tolist()
    
    @cached_property
    def taxonomy_classes(self):
        return self._cluster_annotations["class"].unique()

    def get_taxonomy_class_by_name(self, name):
        classes = [x for x in self.taxonomy_classes if name in x]
        if len(classes) == 1:
            return classes[0]
        elif len(classes) == 0:
            raise ValueError(f"No class found containing '{name}'")
        else:
            raise ValueError(f"Multiple classes found containing '{name}': {classes}")

    @cached_property
    def taxonomy_id(self):
        return pd.read_csv(self.files.taxonomy_metadata.local_path)["label"].iloc[0].split("_")[0]

    def load_adata(
        self,
        transform="log2cpt",
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
        transform : {'log2cpt', 'log2cpm', 'log2cpv', 'raw'}, default='log2cpt'
            which transformation of the gene counts to load and/or calculate from
            the expression matrices
            {cpt: counts per thousand, cpm: per million, cpv: per cell volume}
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
        # 'log2cpv' is labelled 'log2' in the ABC Atlas; for 'log2cpm' or 'log2cpt',
        #  we load 'raw' counts and then do the transform manually later
        adata_file = self.files.adata_log2 if transform == "log2cpv" else self.files.adata_raw
        adata = ad.read_h5ad(adata_file.local_path, backed="r")
        genes = adata.var_names
        if drop_blanks:
            genes = [gene for gene in genes if "Blank" not in gene]

        if with_metadata or (from_metadata is not None):
            if from_metadata is not None:
                cells_md_df = from_metadata
            else:
                cells_md_df = self.get_combined_metadata(**kwargs)
            cell_labels = adata.obs_names.intersection(cells_md_df.index)
            adata = adata[cell_labels, genes]
            adata = adata.to_memory()
            # add metadata to obs
            adata.obs = adata.obs.join(
                cells_md_df.loc[cell_labels, cells_md_df.columns.difference(adata.obs.columns)]
            )
        else:
            adata = adata[:, genes].to_memory()

        scale_factors = {
            "log2cpm": 1e6,
            "log2cpt": 1e3,
            # "log2cph": 1e2  # or other scale_factor options
        }
        # convert sparse matrix (how 'log2' & 'raw' counts are stored in h5ad
        # file) to dense array to match transform output
        adata.X = adata.X.toarray()
        if transform in scale_factors:
            scale_factor = scale_factors[transform]
            adata.X = np.log2(adata.X * scale_factor / np.sum(adata.X, axis=1, keepdims=True) + 1)

        # access genes by short symbol vs longer names
        adata.var_names = adata.var["gene_symbol"]

        # note what counts transform is in X
        adata.uns["counts_transform"] = transform

        return adata

    def filter_by_ccf_labels(self, obs, regions, include_children=True, realigned=False):
        """Filters cell metadata (obs) dataframe by stored CCF region labels.

        Parameters
        ----------
        obs
            dataframe containing cell metadata (i.e. adata.obs)
        regions : list(str)
            list of (abbreviated) CCF region names to select
        include_children, optional
            include all subregions of the specified regions, by default True
        realigned, optional
            use realigned CCF coordinates and image volume, by default False

        Returns
        -------
        obs
            filtered dataframe
        mask_img
            stack of 2D binary masks (x, y, n_sections) used for filtering
        """
        level = "substructure"
        if include_children:
            regions = self.get_ccf_names(regions, level=level)
        ccf_label = f"parcellation_{level}_realigned" if realigned else f"parcellation_{level}"
        obs = obs[obs[ccf_label].isin(regions)]
        return obs

    @accept_anndata_input
    def filter_by_class(self, obs, exclude=None, include=None):
        """Filters cell metadata (obs) dataframe by cell type taxonomy
        classes. Note that these labels may change across dataset versions!

        Parameters
        ----------
        obs
            dataframe containing cell metadata (i.e. adata.obs)
        exclude : list(str), default=self.NN_CLASSES
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

        if exclude is None:
            exclude = self.NN_CLASSES
        obs = obs[~obs["class"].isin(exclude)]
        return obs

    @staticmethod
    def filter_by_coordinate_range(obs, coord_col, start=None, stop=None):
        """Filters cell metadata (obs) dataframe by a numeric coordinate column,
        restricting to a range from start to stop (inclusive)
        """
        if start is not None:
            obs = obs[obs[coord_col] >= start]
        if stop is not None:
            obs = obs[obs[coord_col] <= stop]
        return obs

    def get_gene_metadata(self, drop_blanks=True):
        """Load the gene metadata csv.
        Optionally drops 'Blank' genes from the dataset.
        """
        df = pd.read_csv(self.files.gene_metadata.local_path)
        if drop_blanks:
            df = df[~df["gene_symbol"].str.contains("Blank")]
        return df

    def get_sc_metadata(self, drop_unused=True):
        """Load the cell metadata csv for 10X single-cell data.

        Returns
        -------
            pandas.DataFrame
        """
        cat_columns = [
            "feature_matrix_label",
            "dataset_label",
            "cluster_alias",
            "class",
            "subclass",
            "supertype",
            "cluster",
        ]
        dtype = dict(
            cell_label="string",
            **{x: "category" for x in cat_columns},
        )
        usecols = list(dtype.keys()) if drop_unused else None

        cells_df = pd.read_csv(
            self.files.scrnaseq_metadata.local_path,
            dtype=dtype,
            usecols=usecols,
            index_col="cell_label",
            engine="pyarrow",
        )
        return cells_df

    def get_combined_metadata(
        self,
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
        dtype = defaultdict(
            str,
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
            if self.version != CURRENT_VERSION:
                old_df = pd.read_csv(
                    self.files.cell_metadata.local_path,
                    dtype=dtype,
                    usecols=usecols,
                    index_col="cell_label",
                    engine="pyarrow",
                )
                cells_df = old_df.join(cells_df[cells_df.columns.difference(old_df.columns)])
        else:
            cells_df = pd.read_csv(
                self.files.cell_metadata.local_path,
                dtype=dtype,
                usecols=usecols,
                index_col="cell_label",
                engine="pyarrow",
            )
        if flip_y:
            cells_df[["y_section", "y_reconstructed"]] *= -1
        if round_z:
            cells_df["z_section"] = cells_df["z_section"].round(1)
            cells_df["z_reconstructed"] = cells_df["z_reconstructed"].round(1)

        cells_df["left_hemisphere"] = cells_df["z_ccf"] < 5.7
        if realigned:
            cells_df["left_hemisphere_realigned"] = cells_df["z_ccf_realigned"] < 5.7
        cells_df["parcellation_substructure"] = cells_df["parcellation_substructure"].cat.rename_categories({"ZI-unassigned": "ZI"})
        return cells_df

    def get_ccf_labels_image(self, resampled=True, realigned=False, devccf=False, subset_to_left_hemi=False, img_path=None):
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
            if devccf:
                path = "/data/merscope_638850_devccf_resampled/KimLabDevCCFv001_Annotations_ASL_Oriented_10um_resampled.nii.gz"
            else:
                path = self.files.resampled_annotation.local_path
        elif not resampled and not realigned:
            if devccf:
                path = "/data/KimLabDevCCFv001/KimLabDevCCFv001_Annotations_ASL_Oriented_10um.nii.gz"
            else:
                path = self.files.annotation_10.local_path
        elif resampled and realigned:
            if devccf:
                path = "/data/realigned/abc_realigned_devccf_labels.nii.gz"
            else:
                path = "/data/realigned/abc_realigned_ccf_labels.nii.gz"
        else:
            raise UserWarning("This label image is not available")
        if img_path is not None:
            path = img_path
        img = nibabel.load(path)
        imdata = np.array(img.dataobj).astype(np.int64)
        if subset_to_left_hemi:
            # erase right hemisphere (can't drop or indexing won't work correctly)
            imdata[550:, :, :] = 0

        return imdata

    def label_outlier_celltypes(
        self,
        obs,
        type_col,
        min_group_count=5,
        max_num_groups=None,
        outlier_label="other",
        filter_cells=False,
    ):
        primary_celltypes = obs[type_col].value_counts().loc[lambda x: x > min_group_count].index
        if max_num_groups is not None and len(primary_celltypes) > max_num_groups:
            primary_celltypes = primary_celltypes[:max_num_groups]
        if filter_cells:
            obs = obs[obs[type_col].isin(primary_celltypes)]
        else:
            if (
                obs[type_col].dtype.name == "category"
                and outlier_label not in obs[type_col].cat.categories
            ):
                obs = obs.copy()
                obs[type_col] = obs[type_col].cat.add_categories(outlier_label)
            obs.loc[~obs[type_col].isin(primary_celltypes), type_col] = outlier_label
        return obs

    def label_ccf_spatial_subset(
        self,
        cells_df,
        ccf_regions,
        ccf_level="substructure",
        include_children=True,
        flip_y=False,
        distance_px=20,
        fill_holes_in_mask=False,
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
        fill_holes_in_mask : bool, default=False
            fills internal holes in the mask; usually, internal white matter tracts
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
            dataframe with new column labeling cells in spatial subset;
            if filter_cells=True, returns only cells in subset
        mask_img
            stack of 2D binary masks (x, y, n_sections) used for labeling & filtering
        """
        # use reconstructed (in MERFISH space) coordinates from cells_df
        if realigned:
            coords = ["x_section", "y_section", "z_section"]
        else:
            coords = ["x_reconstructed", "y_reconstructed", "z_reconstructed"]
        resolutions = np.array([self.X_RESOLUTION, self.Y_RESOLUTION, self.Z_RESOLUTION])

        # load 'resampled CCF' (rasterized, in MERFISH space) image volumes from the
        # ABC Atlas dataset (z resolution limited to merscope slices)
        ccf_img = self.get_ccf_labels_image(resampled=True, realigned=realigned)

        # ccf_img voxels are labelled by brain structure parcellation_index, so need
        # to get a list of all indices
        if include_children:
            ccf_regions = self.get_ccf_names(top_nodes=ccf_regions, level=ccf_level)
        reverse_lookup = self.get_ccf_index_reverse_lookup(level=ccf_level)
        index_values = reverse_lookup.loc[ccf_regions]

        ##### Create binary mask to use for filtering cells #####
        # generate binary mask
        th_mask = np.isin(ccf_img, index_values)  # takes about 5 sec
        # flip y-axis to match flipped cell y-coordinates
        if flip_y:
            th_mask = np.flip(th_mask, axis=1)

        # fills internal holes in the mask (usually, internal white matter tracts)
        if fill_holes_in_mask:
            # TODO allow for different distance_px selections; this default has
            # only been tested for the TH+ZI subset
            dist_for_TH = 2
            th_mask = sectionwise_fill_holes(sectionwise_closing(th_mask, distance_px=dist_for_TH))

        # dilate mask, if specified, to ensure inclusion of cells along misaligned edges
        if distance_px == 0:
            mask_img = th_mask
        else:
            mask_img = sectionwise_dilation(th_mask, distance_px, true_radius=False)

        # remove too-small mask regions that are likely mistaken parcellations
        if cleanup_mask:
            mask_img = cleanup_mask_regions(mask_img, area_ratio_thresh=0.1)

        ##### Label & Filter Cells #####
        # label cells that fall within dilated mask
        cells_df = _label_masked_cells(
            cells_df, mask_img, coords, resolutions, field_name=field_name
        )
        if filter_cells:
            filtered_df = cells_df[cells_df[field_name]].copy().drop(columns=[field_name])
            return filtered_df, mask_img
        else:
            return cells_df, mask_img

    @cached_property
    def _ccf_metadata(self):
        # TODO: set categorical dtypes?
        ccf_df = pd.read_csv(self.files.ccf_metadata.local_path)
        ccf_df = ccf_df.replace("ZI-unassigned", "ZI")
        return ccf_df

    def _get_ccf_names(self, top_nodes, level=None):
        ccf_df = self._ccf_metadata
        th_zi_ind = np.hstack(
            [
                ccf_df.loc[ccf_df["parcellation_term_acronym"] == x, "parcellation_index"].unique()
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
        return np.unique(names)

    def get_ccf_names(self, top_nodes=None, level=None, include_unassigned=True):
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
        include_unassigned : bool, default=False
            include 'unassigned' regions in the list of names
            (these are generally substructures of a labelled region without
            a more specific substructure name)

        Returns
        -------
            list of region names
        """
        if top_nodes is None:
            names = self.get_ccf_index(level=level).unique()
        if level == "devccf":
            names = self._get_devccf_names(top_nodes)
        else:
            names = self._get_ccf_names(top_nodes, level=level)
        if not include_unassigned:
            names = [x for x in names if "unassigned" not in x]
        return names

    def get_ccf_index_reverse_lookup(self, level="substructure"):
        ccf_index = self.get_ccf_index(level=level)
        reverse_lookup = pd.Series(ccf_index.index.values, index=ccf_index)
        return reverse_lookup

    def get_ccf_index(self, level="substructure"):
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
            index = self._get_devccf_metadata()["Acronym"]
        else:
            ccf_df = self._ccf_metadata
            # parcellation_index to acronym
            if level is not None:
                ccf_df = ccf_df.query(f"parcellation_term_set_name=='{level}'")
            index = ccf_df.set_index("parcellation_index")["parcellation_term_acronym"]
        return index

    @property
    def _section_metadata_file(self):
        return f"section_metadata_{self.dataset}_{self.version}.csv"

    @cached_property
    def _section_metadata(self):
        path = files("thalamus_merfish_analysis")/"resources" / self._section_metadata_file
        if not path.is_file():
            raise FileNotFoundError(
                f"Section metadata not saved for {self.dataset} version {self.version}"
            )
        df = pd.read_csv(path)
        return df

    def get_section_index(self, section_col="brain_section_label"):
        """Gets a Series mapping a specified section label column
        to the corresponding section index (integer z-coordinate at specified resolution).
        """
        df = self._section_metadata.set_index(section_col)["section_index"]
        return df

    def save_section_index(
        self,
        z_col="z_section",
        section_cols=["brain_section_label", "z_reconstructed"],
        overwrite=False,
    ):
        """Saves a section index mapping to a csv file for faster access."""
        cells_df = self.get_combined_metadata()
        section_index = (
            # TODO: could get section cols as any col constant across section
            cells_df.groupby(z_col, observed=True)[section_cols].first().dropna().reset_index()
        )
        # Note: this transformation may vary for different datasets
        section_index["section_index"] = section_index[z_col].apply(
            lambda x: int(np.rint(x / self.Z_RESOLUTION))
        )
        path = Path("/code/thalamus_merfish_analysis/resources") / self._section_metadata_file
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Section index already saved for {self.dataset} version {self.version}"
            )
        section_index.to_csv(path, header=True, index=True)

    @cached_property
    def _cluster_annotations(self):
        df = pd.read_csv(self.files.cluster_metadata.local_path)
        return df

    @cached_property
    def _cluster_annotations_full(self):
        df = pd.read_csv(self.files.cluster_metadata_full.local_path)
        return df

    def get_taxonomy_palette(self, taxonomy_level):
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
        df = self._cluster_annotations_full
        df = df[df["cluster_annotation_term_set_name"] == taxonomy_level]
        palette = df.set_index("cluster_annotation_term_name")["color_hex_triplet"].to_dict()
        return palette

    def get_taxonomy_label_from_alias(self, aliases, taxonomy_level='cluster'):
        """Given a list of cluster aliases, returns the corresponding cell type
        labels at a given taxonomy level

        Parameters
        ----------
        aliases : list of strings
            list of strings containing the cluster aliases
        taxonomy_level : {'cluster', 'supertype', 'subclass', 'class'}
            specifies the taxonomy level to retrieve

        Results
        -------
        labels
            list of taxonomy labels
        """
        df = self._cluster_annotations
        label_list = df.set_index("cluster_alias").loc[aliases, taxonomy_level].to_list()
        return label_list
    
    def get_alias_from_cluster_label(self, clusters):
        df = self._cluster_annotations
        label_list = df.set_index("cluster").loc[clusters, "cluster_alias"].to_list()
        return label_list

    def get_ccf_palette(self, parcellation_level):
        """Get the published color dictionary for a given level of the Allen
        Reference Atlas (ARA) CCFv3 anatomical parcellation

        Parameters
        ----------
        parcellation_level : {'organ', 'category','division', 'structure',
                            'substructure'}
            specifies the parcellation level to get labels and colors from

        Returns
        -------
        color_dict : dict
            dictionary mapping parcellation labels to their official ARA hex colors
        """
        df = self._ccf_metadata
        df = df[df["parcellation_term_set_name"] == parcellation_level]
        palette = df.set_index("parcellation_term_acronym")["color_hex_triplet"].to_dict()

        return palette

    @staticmethod
    @lru_cache
    def _get_devccf_metadata():
        devccf_index = pd.read_excel("/data/KimLabDevCCFv1/DevCCFv1_OntologyStructure.xlsx", header=[0,1])
        devccf_index = devccf_index["DevCCF"].set_index("ID16")
        return devccf_index


    @classmethod
    def _get_devccf_names(cls, top_nodes, filter=True):
        devccf_index = cls._get_devccf_metadata().reset_index().set_index("Acronym")
        ids = devccf_index.loc[top_nodes, "ID16"].astype(str)
        if filter:
            devccf_index = devccf_index.query("P56")
        names = list(set(chain(
            *(devccf_index.loc[lambda df: df["Structure ID Path16"].str.contains(x)].index for x in ids)
        )))
        return names

DEFAULT_ATLAS_WRAPPER = AtlasWrapper()


def _label_masked_cells(cells_df, mask_img, coords, resolutions, field_name="region_mask"):
    cells_df[field_name] = mask_img[image_index_from_coords(cells_df[coords], resolutions)]
    return cells_df



