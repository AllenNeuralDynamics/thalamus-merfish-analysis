import pandas as pd
from .abc_load import ABC_ROOT

def get_color_dictionary(
    labels, taxonomy_level, label_format="id_label", version="20230830", as_list=False
):
    """Returns a color dictionary for the specified cell types labels.

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
    """
    # load metadata csv files
    pivot_file = "cluster_to_cluster_annotation_membership_pivoted.csv"
    color_file = "cluster_to_cluster_annotation_membership_color.csv"
    pivot_df = pd.read_csv(
        ABC_ROOT / f"metadata/WMB-taxonomy/{version}/views/{pivot_file}"
    )
    color_df = pd.read_csv(
        ABC_ROOT / f"metadata/WMB-taxonomy/{version}/views/{color_file}"
    )

    # get cluster_alias list, which is stable between ABC atlas taxonomy versions
    pivot_query_df = pivot_df.set_index(taxonomy_level).loc[labels].reset_index()
    # clusters are unique, but other taxonomy levels exist in multiple rows
    if taxonomy_level == "cluster":
        cluster_alias_list = pivot_query_df["cluster_alias"].to_list()
    else:
        # dict() only keeps the last instance of a key due to overwriting,
        # which I'm exploiting to remove duplicates while maintaining order
        cluster_alias_list = list(
            dict(
                zip(pivot_query_df[taxonomy_level], pivot_query_df["cluster_alias"])
            ).values()
        )
    # use cluster_alias to map to colors
    color_query_df = (
        color_df.set_index("cluster_alias").loc[cluster_alias_list].reset_index()
    )
    colors_list = color_query_df[taxonomy_level + "_color"].to_list()

    if as_list:
        return colors_list
    else:
        color_dict = dict(zip(labels, colors_list))
        return color_dict


def convert_taxonomy_labels(
    input_labels,
    taxonomy_level,
    label_format="id_label",
    input_version="20230630",
    output_version="20230830",
    output_as_dict=False,
):
    """Converts cell type labels between taxonomy versions of the ABC Atlas.

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
    """

    # load in the correct cluster annotation membership CSV files
    file = "cluster_to_cluster_annotation_membership_pivoted.csv"
    in_pivot_df = pd.read_csv(
        ABC_ROOT / f"metadata/WMB-taxonomy/{input_version}/views/{file}"
    )
    out_pivot_df = pd.read_csv(
        ABC_ROOT / f"metadata/WMB-taxonomy/{output_version}/views/{file}"
    )

    # get cluster_alias list, which is stable between ABC atlas taxonomy versions
    in_query_df = in_pivot_df.set_index(taxonomy_level).loc[input_labels].reset_index()
    # clusters are unique, but other taxonomy levels exist in multiple rows
    if taxonomy_level == "cluster":
        cluster_alias_list = in_query_df["cluster_alias"].to_list()
    else:
        # dict() only keeps the last instance of a key due to overwriting,
        # which I'm exploiting to remove duplicates
        cluster_alias_list = list(
            dict(
                zip(in_query_df[taxonomy_level], in_query_df["cluster_alias"])
            ).values()
        )
    # use cluster_alias to map to output labels
    out_query_df = (
        out_pivot_df.set_index("cluster_alias").loc[cluster_alias_list].reset_index()
    )
    out_labels_list = out_query_df[taxonomy_level].to_list()

    if output_as_dict:
        out_labels_dict = dict(zip(input_labels, out_labels_list))
        return out_labels_dict
    else:
        return out_labels_list
