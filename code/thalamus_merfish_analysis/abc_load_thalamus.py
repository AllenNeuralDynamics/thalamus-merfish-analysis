from functools import lru_cache
from importlib_resources import files
from itertools import chain

import numpy as np
import pandas as pd

from .abc_load_base import AtlasWrapper, accept_anndata_input

_DEVCCF_TOP_NODES_THALAMUS = ["ZIC", "CZI", "RtC", "Th"]
_CCF_TOP_NODES_THALAMUS = ["TH", "ZI"]


class ThalamusWrapper(AtlasWrapper):
    """Thalamus-specific wrapper class for the ABC Atlas MERFISH dataset,
    containing version/dataset-specific methods for loading, filtering, etc.
    """

    # hardcoded class categories for v20230830
    TH_ZI_CLASSES = ["12 HY GABA", "17 MH-LH Glut", "18 TH Glut"]
    MB_CLASSES = ["19 MB Glut", "20 MB GABA"]  # midbrain
    # cls.NN_CLASSES already defined in abc_load_base.py
    TH_SECTIONS = np.arange(25, 42)

    def load_standard_thalamus(self, data_structure="adata"):
        """Loads a preprocessed, neuronal thalamus subset of the ABC Atlas MERFISH dataset.

        Standardizes parameters for load_adata_thalamus() / get_combined_metadata(),
        filter_by_class_thalamus(), & filter_by_thalamus_coords() for consistency in
        the exact data loaded across capsules & notebooks.
        Also ensures loading full anndata vs just obs metadata returns the same
        subset of cells.

        Parameters
        ----------
        data_structure : {'adata', 'obs'}, default='adata'
            load the full AnnData object with gene expression in .X and cell
            metadata DataFrame in .obs ('adata') OR load just the cell metadata
            DataFrame ('obs') to same time/memory

        Results
        -------
        data_th
            AnnData object or DataFrame containing the ABC Atlas MERFISH dataset

        """
        # load
        if data_structure == "adata":
            data_th = self.load_adata_thalamus()
        elif data_structure == "obs":
            # still contains all cells; filter_by_thalamus_coords() subsets to TH+ZI
            data_th = self.get_combined_metadata(
                drop_unused=True, realigned=False, flip_y=False, round_z=True
            )
            data_th = self.filter_by_thalamus_coords(data_th)
        else:
            raise ValueError("data_structure must be adata or obs.")

        # preprocessing
        # default: include=TH_ZI_CLASSES+MB_CLASSES
        data_th = self.filter_by_class_thalamus(
            data_th,
            display_filtered_classes=False,
        )

        return data_th.copy()

    def load_adata_thalamus(
        self,
        subset_to_TH_ZI=True,
        transform="log2cpt",
        with_metadata=True,
        drop_blanks=True,
        flip_y=False,
        realigned=False,
        include_white_matter=True,
        drop_unused=True,
        **kwargs,
    ):
        """Load ABC Atlas MERFISH dataset as an anndata object.

        Parameters
        ----------
        subset_to_TH_ZI : bool, default=True
            returns adata that only includes cells in the TH+ZI dataset, as subset
            by filter_by_thalamus_coords()
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
        flip_y : bool, default=True
            flip y-axis coordinates so positive is up (coronal section appears
            right-side up as expected)
        realigned : bool, default=False
            load and use for subsetting the metadata from realignment results data asset,
            containing 'ccf_realigned' coordinates
        include_white_matter : bool, default=True
            include cells that fall in white matter tracts within the thalamus
            when subsetting to TH+ZI
        drop_unused : bool, default=True
            drop unused columns from metadata
        **kwargs
            passed to `get_combined_metadata`

        Results
        -------
        adata
            anndata object containing the ABC Atlas MERFISH dataset
        """
        if subset_to_TH_ZI:
            cells_md_df = self.get_combined_metadata(
                realigned=realigned, flip_y=flip_y, drop_unused=drop_unused, **kwargs
            )
            cells_md_df = self.filter_by_thalamus_coords(
                cells_md_df,
                buffer=0,
                include_white_matter=include_white_matter
            )
            adata = self.load_adata(
                transform=transform, drop_blanks=drop_blanks, from_metadata=cells_md_df
            )
        else:
            adata = self.load_adata(
                transform=transform,
                drop_blanks=drop_blanks,
                with_metadata=with_metadata,
                flip_y=flip_y,
                realigned=realigned,
                **kwargs,
            )

        return adata

    @classmethod
    @accept_anndata_input
    def filter_by_class_thalamus(
        cls,
        obs,
        include=TH_ZI_CLASSES + MB_CLASSES,
        exclude=None,
        display_filtered_classes=True,
    ):
        """Filters anndata object to only include cells from specific taxonomy
        classes.

        Parameters
        ----------
        obs
            anndata object or dataframe containing the ABC Atlas MERFISH dataset
        exclude : list of str, default=None
            list of classes to filter out
        include : list of str, default=None
            if present, include ONLY cells in this list of classes (acts prior
            to 'exclude' and thus excludes any class not explicitly in this list)
        display_filtered_classes : bool, default=True
            whether to print the classes filtered out of the input data

        Returns
        -------
        obs
            the dataset, filtered to only include cells from specific
            thalamic & zona incerta + optional (midbrain & nonneuronal) classes
        """
        classes_input = sorted(obs["class"].cat.remove_unused_categories().cat.categories.to_list())
        obs = cls.filter_by_class(obs, exclude=exclude, include=include)
        classes_output = sorted(
            obs["class"].cat.remove_unused_categories().cat.categories.to_list()
        )
        # (optional) print out to make explicit to the user which classes are
        # being excluded from the dataset & which they could choose to include
        if display_filtered_classes:
            print(
                f"Classes present in input data: {classes_input}\n"
                f"Classes present in output data: {classes_output}\n"
                f"Classes filtered out of input data: {sorted(list(set(classes_input) - set(classes_output)))}"
            )

        return obs

    @accept_anndata_input
    def filter_by_thalamus_coords(self, obs, buffer=0, include_white_matter=True, **kwargs):
        """Filters to only include cells within thalamus CCF boundaries +/- a buffer.

        Parameters
        ----------
        obs : AnnData or DataFrame
            object containing the ABC Atlas MERFISH metadata; if AnnData, .obs
            should contain the metadata
        buffer : int, default=0
            buffer in microns to add to the thalamus mask
        include_white_matter : bool, default=True
            whether to include cells that fall in white matter tracts within
            the thalamus when filtering
        **kwargs
            passed to 'label_ccf_spatial_subset'

        Returns
        -------
        obs
            the filtered AnnData or DataFrame object
        """
        obs, _ = self.label_ccf_spatial_subset(
                obs,
                ["TH", "ZI"],
                distance_px=buffer,
                filter_cells=True,
                fill_holes_in_mask=include_white_matter,
                **kwargs
            )
        obs = self.filter_thalamus_sections(obs)
        return obs

    @staticmethod
    def filter_thalamus_sections(obs):
        """Filters anterior-to-posterior coordinates to include only sections containing thalamus.

        Includes filtering for 1 anterior-most & 1 posterior-most section with poor
        alignment between thalamus CCF structure and mapped thalamic cells.
        """
        return obs.query("5.0 <= z_section <= 8.2")

    @lru_cache
    def get_thalamus_names(self, level=None):
        if level == "devccf":
            return self.get_ccf_names(_DEVCCF_TOP_NODES_THALAMUS, level=level)
        else:
            return self.get_ccf_names(_CCF_TOP_NODES_THALAMUS, level=level)

    def get_thalamus_substructure_names(self):
        return self.get_thalamus_names(level="substructure")

    def get_thalamus_ccf_indices(self):
        th_ccf_names = self.get_thalamus_names(level="substructure")
        # convert ccf names to the unique parcellation_index used in the image volume
        ccf_index = self.get_ccf_index(level="substructure")
        reverse_lookup = pd.Series(ccf_index.index.values, index=ccf_index)
        th_index_values = reverse_lookup.loc[th_ccf_names]

        return th_index_values

    # load cluster-nucleus annotations
    try:
        nuclei_df_manual = pd.read_csv(
            files("thalamus_merfish_analysis.resources")
            / "prong1_cluster_annotations_by_nucleus.csv",
            index_col=0,
        )
        nuclei_df_manual = nuclei_df_manual.fillna("")
        nuclei_df_auto = pd.read_csv(
            files("thalamus_merfish_analysis.resources") / "annotations_from_eroded_counts.csv",
            index_col=0,
        )
        found_annotations = True
    except FileNotFoundError:
        found_annotations = False

    @classmethod
    def get_obs_from_annotated_clusters(
        cls,
        nuclei_names,
        obs,
        by="id",
        include_shared_clusters=False,
        manual_annotations=True,
    ):
        """Get cells from specific thalamic nucle(i) based on manual nuclei:cluster
        annotations.

        Parameters
        ----------
        nuclei_names : str or list of str
            name(s) of thalamic nuclei to search for in the manual annotations resource
        obs : DataFrame
            cell metadata DataFrame
        by : {'id', 'alias'}, default='id'
            whether to search for name in cluster_id (4-digits at the start of each
            cluster label, specific to a taxonomy version) or cluster_alias (unique
            ID # for each cluster, consistent across taxonomy versions) column
        include_shared_clusters : bool, default=False
            whether to include clusters that are shared with multiple thalamic nuclei
        manual_annotations : bool, default=True
            whether to use manual annotations or automatic annotations CSV

        Returns
        -------
        obs
            cell metadata DataFrame with only cells from the specified cluster(s)
        """

        if not cls.found_annotations:
            raise UserWarning("Can't access annotations sheet from this environment.")

        nuclei_df = cls.nuclei_df_manual if manual_annotations else cls.nuclei_df_auto

        # if single name, convert to list
        nuclei_names = [nuclei_names] if isinstance(nuclei_names, str) else nuclei_names
        all_names = []
        for name in nuclei_names:
            if include_shared_clusters:
                # 'name in y' condition returns e.g. ['AD','IAD'] if name='AD' but
                # 'name==y' only returns 'AD' (but is now suseptible to typos like
                # extra spaces - maybe there's a better solution?)
                curr_names = [x for x in nuclei_df.index if any(y == name for y in x.split(" "))]
            else:
                curr_names = [
                    x for x in nuclei_df.index if x == name and not (" " in x or "pc" in x)
                ]
            all_names.extend(curr_names)

            if curr_names == []:
                unique_nuclei = sorted(
                    set(nucleus for entry in nuclei_df.index for nucleus in entry.split())
                )
                raise UserWarning(
                    f"Nuclei name(s) not found in annotations sheet. Please select from valid nuclei names below:\n{unique_nuclei=}"
                )

        field = "cluster_alias" if by == "alias" else "cluster_ids_CNN20230720"
        clusters = chain(*[nuclei_df.loc[name, field].split(", ") for name in all_names])
        if by == "alias":
            obs = obs.loc[lambda df: df["cluster_alias"].isin(clusters)]
        elif by == "id":
            # cluster id is the first 4 digits of 'cluster' column entries
            obs = obs.loc[lambda df: df["cluster"].str[:4].isin(clusters)]
        return obs

    # TODO: save this in a less weird format (json?)
    @staticmethod
    @lru_cache
    def get_thalamus_cluster_palette():
        palette_df = pd.read_csv(
            files("thalamus_merfish_analysis.resources") / "cluster_palette_glasbey.csv"
        )
        return dict(zip(palette_df["Unnamed: 0"], palette_df["0"]))


DEFAULT_ATLAS_WRAPPER = ThalamusWrapper()