import os
import streamlit as st
from streamlit_utils import (
    abc,
    get_ccf_data,
    version,
    has_realigned_asset,
    ss_to_qp,
    ss_from_qp,
)
from abc_merfish_analysis import ccf_plots as cplots

ss_to_qp()
ss_from_qp()
ss = st.session_state
pg = st.navigation([
    st.Page("streamlit_main.py", default=True), 
    # st.Page("streamlit_annotation.py")
])
st.set_page_config(page_title="Thalamus MERFISH explorer", layout="wide")

with st.expander("CCF alignment settings"):
    realigned = st.radio(
        "CCF alignment",
        [False, True],
        index=0,
        key="realigned_qp",
        format_func=lambda realigned: "thalamus-specific section-wise affine alignment"
        if realigned
        else "published nonlinear alignment",
    )
    extend_borders = st.checkbox("Extend CCF borders", key="extend_borders_qp")
    devccf = st.checkbox("Use DevCCF (Paxinos-based) parcellation", key="devccf_qp")

if realigned and not has_realigned_asset:
    realigned = False
    st.warning("Realigned metadata not found, using published alignment")

cplots.CCF_REGIONS_DEFAULT = abc.get_thalamus_names(level='devccf' if devccf else None)
# always use CCFv3 names for user options
ss.th_subregion_names = abc.get_thalamus_names('structure', include_unassigned=False)
ccf_images, ccf_boundaries = get_ccf_data(realigned, devccf=devccf)
coords = "section" if realigned else "reconstructed"
ss.common_args = dict(
    x_col="x_" + coords,
    y_col="y_" + coords,
    boundary_img=ccf_boundaries,
    ccf_images=ccf_images,
    ccf_level='devccf' if devccf else 'structure',
)

with st.sidebar:
    st.markdown("## App Details")
    st.write(f"Version: {version}")
    st.write(f"Gene data version: {abc.files.adata_raw.version}")
    st.write(f"Metadata version: {abc.files.cell_metadata.version}")
    st.write(f"CCF version: {abc.files.ccf_metadata.version}")
    st.write(f"Taxonomy ID: {abc.taxonomy_id}")
    st.markdown("### App direct link")
    url = f"https://codeocean.allenneuraldynamics.org/cw/{os.getenv('CO_COMPUTATION_ID')}/"
    st.markdown(f"{url}")

# extend size of multiselects for full section names
st.markdown(
    """
    <style>
        .stMultiSelect [data-baseweb=select] span{
            max-width: 280px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

pg.run()
