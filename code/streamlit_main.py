import matplotlib.pyplot as plt
import streamlit as st
from thalamus_merfish_analysis import ccf_plots as cplots
from thalamus_merfish_analysis import de_genes as deg


import streamlit_utils as stu
from streamlit_utils import abc, get_data, get_adata, get_sc_data, propagate_value_from_lookup, palettes, th_sections


ss = st.session_state
obs_th_neurons = get_data(ss.realigned_qp, extend_borders=ss.extend_borders_qp)

def annotation_details_input(context=st, prefix=""):
    expander = context.expander("Cluster-to-region annotation settings")
    manual_annotations = (
        expander.radio("Annotation source", ["automated", "manual"], key=f"{prefix}_anno_qp") == "manual"
    )
    expander.write("Manually-reviewed annotations are higher accuracy but don't cover all nuclei.")
    include_shared_clusters = expander.checkbox("Include shared clusters", key=f"{prefix}_shared_qp")
    expander.write("Shared clusters are found across multiple nuclei, not focused on the selected nuclei.")
    return manual_annotations, include_shared_clusters, expander

pane1, pane2 = st.columns(2)
with pane2:
    st.header("Gene expression plots")
    transform = st.radio(
        "Transform", ["log2cpt", "log2cpm", "log2cpv", "raw"], index=0, key="transform_qp"
    )
    merfish_genes = abc.get_gene_metadata()["gene_symbol"].values
    single_gene, multi_gene, de_genes = st.tabs(
        ["Single gene plots", "Multiple gene overlay", "Differential expression"]
    )
    with single_gene:
        st_gp = st.form("gene_plot")
        gene = st_gp.selectbox("Select gene", merfish_genes, index=None, key="gp_gene_qp")
        nuclei = st_gp.multiselect(
            "Nuclei to highlight", ss.th_subregion_names, key="gp_regionlist_qp"
        )
        if ss.devccf_qp:
            nuclei = stu.get_devccf_matched_regions(nuclei)
        sections = st_gp.multiselect("Sections", th_sections, key="gp_sectionlist_qp")
        if len(sections) == 0:
            sections = None
        focus_plot = (
            st_gp.checkbox("Focus on selected nuclei", key="gp_focus_qp") and len(nuclei) > 0
        )
        if st_gp.form_submit_button("Plot gene expression", on_click=stu.ss_to_qp):
            adata = get_adata(transform=transform)
            plots = cplots.plot_expression_ccf(
                adata,
                gene,
                nuclei=cplots.CCF_REGIONS_DEFAULT,
                highlight=nuclei,
                sections=sections,
                zoom_to_highlighted=focus_plot,
                **ss.common_args,
            )
            for plot in plots:
                st.pyplot(plot)
    with multi_gene:
        st_mg = st.form("multigene_plot")
        gene_set = st_mg.multiselect("Select genes", merfish_genes, key="mg_genelist_qp")
        sections = st_mg.multiselect("Sections", th_sections, key="mg_sectionlist_qp")
        dark_background = st_mg.checkbox("Dark background", key="mg_dark_qp")
        if st_mg.form_submit_button("Plot multi-gene expression", on_click=stu.ss_to_qp):
            adata = get_adata(transform=transform)
            plots = cplots.plot_hcr(
                adata,
                gene_set,
                sections=sections,
                dark_background=dark_background,
                **ss.common_args,
            )
            for plot in plots:
                st.pyplot(plot)
    with de_genes:
        with st.form("load_genes"):
            dataset = st.radio(
                "Choose dataset", ["MERFISH", "WMB-10Xv3", "WMB-10Xv2"], index=1, key="de_data_qp"
            )
            sc_data = dataset != "MERFISH"
            load_genes = st.form_submit_button("Load gene data", on_click=stu.ss_to_qp)

        taxonomy_level = st.selectbox(
            "Taxonomy level", ["cluster", "supertype", "subclass"], key="de_tax_qp"
        )
        manual_annotations, include_shared_clusters, _ = annotation_details_input(prefix="de")
        de_form = st.form("plot_de_genes")
        groups = [
            de_form.expander("Select group 1", expanded=True),
            de_form.expander("Select reference, empty to compare to rest of thalamus (slow)", expanded=False),
        ]
        grouped_types = [0, 0]
        for i, box in enumerate(groups):
            regions = box.multiselect(
                "By nucleus", ss.th_subregion_names, key=f"de_regionlist{i}_qp"
            )
            types_by_annotation = abc.get_obs_from_annotations(
                regions,
                obs_th_neurons,
                include_shared_clusters=include_shared_clusters,
                manual_annotations=manual_annotations,
            )[taxonomy_level].unique()
            box.write("OR")
            types_by_name = box.multiselect(
                "By name", obs_th_neurons[taxonomy_level].unique(), key=f"de_typelist{i}_qp"
            )
            # TODO: allow typing list of names?
            grouped_types[i] = list(set(types_by_annotation) | set(types_by_name))
        group, reference = grouped_types
        plot_de_genes = de_form.form_submit_button("Plot DE genes", on_click=stu.ss_to_qp, disabled="adata" in globals())
        if load_genes or plot_de_genes:
            if sc_data:
                adata, sc_obs_filtered = get_sc_data(sc_dataset=dataset, transform=transform)
            else:
                adata = get_adata(transform=transform)
        if plot_de_genes and len(group) > 0:
            with st.spinner("Calculating DE genes"):
                if len(reference) == 0:
                    reference = "rest"
                else:
                    intersection = set(group) & set(reference)
                    if len(intersection) > 0:
                        st.warning(f"Groups share cell types: {intersection}")
                highlight = merfish_genes if sc_data else None
                deg.run_sc_deg_analysis(
                    adata, taxonomy_level, group, reference=reference, highlight_genes=highlight
                )
                # TODO: add regions to plot title (restrict to either by nucleus or by taxonomy?)
            st.pyplot(plt.gcf())
            if sc_data:
                st.write("Highlighted genes overlap with MERFISH gene panel.")
        else:
            st.write("Select groups of cell types to compare, then click 'Plot")


with pane1:
    st.header("Cell type taxonomy spatial plots")
    by_nucleus, by_section = st.tabs(["by region or cell type", "by section"])

    with by_section:
        sections = st.multiselect(
            "Section",
            th_sections,
            key="bs_sectionlist_qp",
        )
        st.button(
            "Show all sections",
            on_click=lambda: setattr(ss, "bs_sectionlist_qp", th_sections),
        )
        celltype_label = st.selectbox(
            "Level of celltype hierarcy",
            ["subclass", "supertype", "cluster"],
            index=0,
            key="bs_tax_qp",
        )
        show_legend = st.checkbox("Show legend", key="bs_leg_qp")
        if len(sections) > 0:
            plots = cplots.plot_ccf_overlay(
                obs_th_neurons,
                ccf_names=None,
                point_hue=celltype_label,
                sections=sections,
                point_palette=palettes[celltype_label],
                legend="cells" if show_legend else False,
                **ss.common_args,
            )
            for plot in plots:
                st.pyplot(plot)

    with by_nucleus:
        nucleus_groups = {
            "None": [],
            "Motor": ["VAL", "VM"],
            "Vision": ["LGd", "LP"],
            "Somatosensory": ["VPM", "VPL", "PO"],
            "Limbic/Anterior": ["AD", "AV", "AM"],
            "Auditory": ["MG", "MG"],
        }
        anno_details = st.container()
        manual_annotations, include_shared_clusters, expander = annotation_details_input(anno_details, prefix="bn")
        show_borders = st.checkbox("Show all boundaries", key="bn_borders_qp")

        celltype_label = st.selectbox(
            "Level of celltype hierarcy",
            ["subclass", "supertype", "cluster"],
            index=2,
            key="bn_tax_qp",
        )
        def celltype_lookup(nuclei):
            return abc.get_annotated_cell_types(
                nuclei,
                taxonomy_level=celltype_label,
                include_shared_clusters=include_shared_clusters,
                manual_annotations=manual_annotations,
            )
        expander.button("Apply", on_click=propagate_value_from_lookup,
            args=("bn_regionlist_qp", "bn_typelist_qp", celltype_lookup),
        )

        st.selectbox(
            "Select nucleus group",
            nucleus_groups.keys(),
            key="bn_group_qp",
            on_change=lambda: (
                propagate_value_from_lookup("bn_group_qp", "bn_regionlist_qp", nucleus_groups.get),
                propagate_value_from_lookup("bn_regionlist_qp", "bn_typelist_qp", celltype_lookup),
            )
        )
        nuclei = st.multiselect(
            "Select individual nuclei", 
            ss.th_subregion_names, 
            key="bn_regionlist_qp",
            on_change=propagate_value_from_lookup,
            args=("bn_regionlist_qp", "bn_typelist_qp", celltype_lookup),
        )
        if ss.devccf_qp:
            nuclei = stu.get_devccf_matched_regions(nuclei)
        celltypes = st.multiselect(
            "Select cell types",
            obs_th_neurons[celltype_label].unique(),
            key="bn_typelist_qp",
        )

        if st.button("Plot"):
            obs2 = obs_th_neurons.loc[
                obs_th_neurons[celltype_label].isin(celltypes)
            ]
            plots = cplots.plot_ccf_overlay(
                obs2,
                ccf_names=None if show_borders else nuclei,
                ccf_highlight=nuclei,
                point_hue=celltype_label,
                sections=None,
                min_group_count=0,
                point_palette=palettes[celltype_label],
                bg_cells=obs_th_neurons,
                **ss.common_args,
            )
            for plot in plots:
                st.pyplot(plot)

stu.ss_to_qp()
