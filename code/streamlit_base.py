import os
import streamlit as st
from streamlit_utils import (
    abc,
    get_image_volumes,
    version,
    has_realigned_asset,
    ccf_level,
    ss_to_qp,
    ss_from_qp,
)

ss_to_qp()
ss_from_qp()
ss = st.session_state
pg = st.navigation([
    st.Page("streamlit_app.py", default=True), 
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

if realigned and not has_realigned_asset:
    realigned = False
    st.warning("Realigned metadata not found, using published alignment")

ccf_images, ccf_boundaries = get_image_volumes(realigned)
coords = "section" if realigned else "reconstructed"
ss.common_args = dict(
    section_col="brain_section_label",
    x_col="x_" + coords,
    y_col="y_" + coords,
    boundary_img=ccf_boundaries,
    ccf_images=ccf_images,
    ccf_level=ccf_level,
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
