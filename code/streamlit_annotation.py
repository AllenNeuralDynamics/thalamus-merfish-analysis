import uuid
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid
from streamlit_utils import abc, get_data, th_subregion_names
from thalamus_merfish_analysis import ccf_plots as cplots

ss = st.session_state
obs_th_neurons = get_data(ss.realigned_qp, extend_borders=ss.extend_borders_qp)

anno_path = "/scratch/annotations_c2n_combined.csv"
ss.anno_df = ss.get("anno_df", pd.read_csv(anno_path, index_col=0))
ag_keys = ["ag_current", "ag_tentative", "ag_unused"]

def reload_aggrids(keys=ag_keys, overwrite=True):
    for key in keys:
        if overwrite or key not in ss:
            ss[key] = str(uuid.uuid4())
    if overwrite:
        st.rerun()
reload_aggrids(overwrite=False)

nucleus = st.selectbox("Select nucleus", th_subregion_names, on_change=reload_aggrids)
include_shared = st.checkbox("Include shared clusters")

palette = abc.get_thalamus_cluster_palette()
kwargs_cluster_annotations = dict(
    bg_cells=obs_th_neurons,
    point_size=3,
    edge_color='silver',
    point_hue='cluster',
    point_palette=palette,
    min_group_count=0,
)
current, tentative, unannotated = st.columns(3)
update_anno = ["cellValueChanged", "selectionChanged"]

with current:
    st.markdown("## Working clusters")
    # this will use ss.anno_df
    clusters = abc.get_annotated_clusters([nucleus], include_shared_clusters=include_shared, annotations_df=ss.anno_df)
    
    anno_current = ss.anno_df.loc[clusters].copy()
    options = {
        "rowSelection": "multiple",
        "columnDefs": [
            {
                "field": "cluster",
                "headerCheckboxSelection": True,
                "headerCheckboxSelectionFilteredOnly": True,
                "checkboxSelection": True,
            },
            {"field": "nuclei", "editable": True},
        ],
        "defaultColDef": {"filter": True},
    }
    with st.empty():
        ag_current = AgGrid(anno_current.reset_index(), gridOptions=options, key=ss["ag_current"], update_on=update_anno)
    if st.button("Apply changes (to memory)"):
        ss.anno_df.update(ag_current["data"].set_index("cluster"))
    if st.button("Save changes (to disk)"):
        ss.anno_df.to_csv(anno_path)
    if st.button("Copy selected to working list"):
        ss.tentative = pd.concat([
            ss.tentative, 
            ag_current["selected_data"].set_index("cluster")
        ])
        reload_aggrids(["ag_tentative"])

    # in case of clusters not in saved palette...
    plots = []
    if st.button("Plot", key="plot_current"):
        obs_annot = obs_th_neurons.loc[obs_th_neurons["cluster"].isin(clusters)].copy()
        plots = cplots.plot_ccf_overlay(
            obs_annot, 
            ccf_highlight=[nucleus],
            **kwargs_cluster_annotations,
            **ss.common_args,
        )
    for plot in plots:
        st.pyplot(plot)

with tentative:
    st.markdown("## Tentative clusters")
    ss.tentative = ss.get("tentative", ss.anno_df.iloc[:0])
    options = {
        "rowSelection": "multiple",
        "columnDefs": [
            {
                "field": "cluster",
                "headerCheckboxSelection": True,
                "headerCheckboxSelectionFilteredOnly": True,
                "checkboxSelection": True,
            },
            {"field": "nuclei", "editable": True},
        ],
        "defaultColDef": {"filter": True},
    }
    with st.empty():
        out = AgGrid(ss.tentative.reset_index(), options, key="test", update_on=update_anno)
    if st.button("Apply changes (to annotations)", key="apply_tentative"):
        data = out["data"].set_index("cluster")
        ss.anno_df = ss.anno_df.reindex(ss.anno_df.index.union(data.index))
        ss.anno_df.update(data)
        ss.tentative = ss.anno_df.iloc[:0]
        reload_aggrids()
    if st.button("Remove selected", key="remove_tentative"):
        ss.tentative = ss.tentative.drop(index=out["selected_data"]["cluster"])
        reload_aggrids()

    plots_2 = []
    if st.button("Plot", key="plot_tentative"):
        obs_annot = obs_th_neurons.loc[obs_th_neurons["cluster"].isin(ss.tentative.index)].copy()
        plots_2 = cplots.plot_ccf_overlay(
            obs_annot, 
            ccf_highlight=[nucleus],
            **kwargs_cluster_annotations,
            **ss.common_args,
        )
    for plot in plots_2:
        st.pyplot(plot)

with unannotated:
    st.markdown("## Available clusters")
    # Filter out tentative too or just annotated?
    if st.checkbox("Exclude annotated clusters"):
        unannotated_df = obs_th_neurons.loc[~obs_th_neurons["cluster"].isin(ss.anno_df.index)]
    else:
        unannotated_df = obs_th_neurons.loc[~obs_th_neurons["cluster"].isin(clusters)]
    ccf_col = "parcellation_structure_realigned" if ss.realigned_qp else "parcellation_structure"
    if st.checkbox("Filter to cells in selected nucleus"):
        unannotated_df = unannotated_df.loc[lambda df: df[ccf_col]==nucleus]
        
    unannotated_clusters = unannotated_df.groupby("cluster", observed=True)[["subclass", "supertype"]].first()
    unannotated_clusters = unannotated_clusters.assign(
        nuclei= lambda df: ss.anno_df.reindex(df.index)["nuclei"]
    )
    options = {
        "rowSelection": "multiple",
        "columnDefs": [
            {
                "field": "cluster",
                "headerCheckboxSelection": True,
                "headerCheckboxSelectionFilteredOnly": True,
                "checkboxSelection": True,
            },
            {"field": "supertype"},
            {"field": "subclass"},
            {"field": "nuclei"},
        ],
        "defaultColDef": {"filter": True},
    }
    with st.empty():
        selected = AgGrid(unannotated_clusters.reset_index(), options, update_on=["cellValueChanged"], key=ss["ag_unused"])["selected_data"]
    if st.button("Move selected to working list"):
        # rerun to retrieve aggrid data?
        # st.rerun()
        df = selected.set_index("cluster")[["nuclei"]]
        df["nuclei"] = df["nuclei"].fillna("").apply(lambda x: nucleus if x=="" else x+" "+nucleus)
        ss.tentative = pd.concat([
            ss.tentative, 
            df.assign(cluster_alias=abc.get_alias_from_cluster_label(df.index))
        ])
        reload_aggrids(["ag_tentative"])
