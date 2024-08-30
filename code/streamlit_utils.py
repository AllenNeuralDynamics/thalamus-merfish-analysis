# Utility functions and non-user configurable defaults for the Streamlit app
import streamlit as st

_suffix = "_qp"

def process_query_param(k, v):
    if "list" not in k and len(v) == 1:
        v = v[0]
    if v in ["True", "False"]:
        v = (v == "True")
    return v

def ss_from_qp():
    state = {key: st.query_params.get_all(key) for key in st.query_params}
    state = {k: process_query_param(k, v) for k, v in state.items()}
    st.session_state.update(state)

def ss_to_qp():
    qp_state = {k: v for k, v in st.session_state.items() if _suffix in k}
    qp_state = {k: v for k, v in qp_state.items() if v is not None}
    st.query_params.from_dict(qp_state)

ss = st.session_state
def propagate_value_from_lookup(from_key, to_key, lookup_fcn):
    if len(ss[from_key]) > 0:
        try:
            ss[to_key] = lookup_fcn(ss[from_key])
        except UserWarning as exc:
            st.warning(exc)

import os
from pathlib import Path
from thalamus_merfish_analysis import ccf_plots as cplots
from thalamus_merfish_analysis import ccf_images as cimg
from thalamus_merfish_analysis.abc_load_thalamus import ThalamusWrapper
from anndata import read_h5ad

version = "20230830"
ccf_level = "structure"
lump_structures = False
has_realigned_asset = Path(
    "/data/realigned/abc_realigned_metadata_thalamus-boundingbox.parquet"
).exists()
abc = ThalamusWrapper(version=version)

th_names = [x for x in abc.get_thalamus_names() if "unassigned" not in x]
th_subregion_names = [x for x in abc.get_thalamus_names(level=ccf_level) if "unassigned" not in x]
palettes = {level: abc.get_taxonomy_palette(level) for level in ["subclass", "supertype"]}
palettes["cluster"] = abc.get_thalamus_cluster_palette()
cplots.CCF_REGIONS_DEFAULT = abc.get_thalamus_names()

@st.cache_resource
def get_adata(transform="cpm", version=version, realigned=False):
    obs_th_neurons, _, _ = get_data(realigned, version=version, extend_borders=True)
    return abc.load_adata(transform=transform, from_metadata=obs_th_neurons)


@st.cache_resource
def get_sc_data(
    sc_dataset="WMB-10Xv3",
    transform="log2",
):
    sc_obs = abc.get_sc_metadata()
    sc_obs = sc_obs[sc_obs["dataset_label"] == sc_dataset]
    sc_obs_filtered = abc.filter_by_class(sc_obs)
    obs_join = sc_obs_filtered
    sc_transform = "log2" if "log2" in transform else "raw"
    adata = read_h5ad(
        abc.manifest.get_file_attributes(
            directory=sc_dataset, file_name=f"{sc_dataset}-TH/{sc_transform}"
        ).local_path,
        backed="r",
    )
    adata = adata[obs_join.index.intersection(adata.obs.index)].to_memory()
    adata.var_names = adata.var["gene_symbol"]
    adata.obs = adata.obs.join(obs_join)
    return adata, sc_obs_filtered


@st.cache_data
def get_data(realigned, version=version, extend_borders=False):
    obs = abc.get_combined_metadata(realigned=has_realigned_asset, drop_unused=False)
    # remove non-neuronal and some other outlier non-thalamus types
    obs_neurons = abc.filter_by_class_thalamus(obs, display_filtered_classes=False)
    buffer = 5 if extend_borders else 0
    obs_th_neurons = abc.filter_by_thalamus_coords(obs_neurons, realigned=realigned, buffer=buffer)
    return obs_th_neurons


@st.cache_data
def get_image_volumes(realigned, lump_structures=lump_structures, edge_width=1):
    ccf_images = abc.get_ccf_labels_image(resampled=True, realigned=realigned)

    if lump_structures:
        ccf_index = abc.get_ccf_index(level="structure")
        reverse_index = (
            ccf_index.reset_index()
            .groupby("parcellation_term_acronym")["parcellation_index"]
            .first()
        )
        mapping = ccf_index.map(reverse_index).to_dict()
        ccf_images = cimg.map_label_values(ccf_images, mapping, section_list=abc.TH_SECTIONS)
    ccf_boundaries = None
    ccf_boundaries = cimg.sectionwise_label_erosion(
        ccf_images, edge_width, fill_val=0, return_edges=True, section_list=abc.TH_SECTIONS
    )
    return ccf_images, ccf_boundaries

