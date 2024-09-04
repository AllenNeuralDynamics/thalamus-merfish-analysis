from itertools import chain
import os
from pathlib import Path
import matplotlib.pyplot as plt
import streamlit as st
import streamlit_utils as stu
from thalamus_merfish_analysis import ccf_plots as cplots
from thalamus_merfish_analysis import ccf_images as cimg
from thalamus_merfish_analysis import de_genes as deg

# from thalamus_merfish_analysis import abc_load as abc
from thalamus_merfish_analysis.abc_load_thalamus import ThalamusWrapper
from anndata import read_h5ad

ss = st.session_state
stu.ss_to_qp()
stu.ss_from_qp()
st.set_page_config(page_title="Thalamus MERFISH explorer", layout="wide")
version = "20230830"
section_col = "brain_section_label"
ccf_level = "structure"
lump_structures = False

abc = ThalamusWrapper(version=version)

with st.expander("CCF alignment settings"):
    realigned = st.radio(
        "CCF alignment",
        [False, True],
        index=0,
        format_func=lambda realigned: "thalamus-specific section-wise affine alignment"
        if realigned
        else "published nonlinear alignment",
    )
    extend_borders = st.checkbox("Extend CCF borders")
if realigned:
    ccf_label = f"parcellation_{ccf_level}_realigned"
    coords = "section"
else:
    ccf_label = f"parcellation_{ccf_level}"
    coords = "reconstructed"

th_names = [x for x in abc.get_thalamus_names() if "unassigned" not in x]
th_subregion_names = [x for x in abc.get_thalamus_names(level=ccf_level) if "unassigned" not in x]
palettes = {level: abc.get_taxonomy_palette(level) for level in ["subclass", "supertype"]}
palettes["cluster"] = abc.get_thalamus_cluster_palette()
cplots.CCF_REGIONS_DEFAULT = abc.get_thalamus_names()

has_realigned_asset = Path(
    "/data/realigned/abc_realigned_metadata_thalamus-boundingbox.parquet"
).exists()
if realigned and not has_realigned_asset:
    realigned = False
    st.warning("Realigned metadata not found, using published alignment")


def propagate_value_from_lookup(from_key, to_key, lookup_fcn):
    if len(ss[from_key]) > 0:
        try:
            ss[to_key] = lookup_fcn(ss[from_key])
        except UserWarning as exc:
            st.warning(exc)

@st.cache_data
def get_data(version, ccf_label, extend_borders=False):
    obs = abc.get_combined_metadata(realigned=has_realigned_asset, drop_unused=False)
    # remove non-neuronal and some other outlier non-thalamus types
    obs_neurons = abc.filter_by_class_thalamus(obs)
    buffer = 5 if extend_borders else 0
    obs_th_neurons = abc.filter_by_thalamus_coords(obs_neurons, buffer=buffer)
    sections_all = sorted(obs_th_neurons[section_col].unique())
    subclasses_all = obs_th_neurons["subclass"].value_counts().loc[lambda x: x > 100].index
    return obs_th_neurons, sections_all, subclasses_all


@st.cache_data
def get_image_volumes(realigned, sections_all, lump_structures=False, edge_width=1):
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


obs_th_neurons, sections_all, subclasses_all = get_data(version, ccf_label)
ccf_images, ccf_boundaries = get_image_volumes(
    realigned, sections_all, lump_structures=lump_structures
)


@st.cache_resource
def get_adata(transform="cpm"):
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


common_args = dict(
    section_col=section_col,
    x_col="x_" + coords,
    y_col="y_" + coords,
    boundary_img=ccf_boundaries,
    ccf_images=ccf_images,
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

def annotation_details_input(context=st, prefix=""):
    with context.expander("Cluster-to-region annotation settings"):
        manual_annotations = (
            st.radio("Annotation source", ["automated", "manual"], key=f"{prefix}_anno_qp") == "manual"
        )
        st.text("Manually-reviewed annotations are higher accuracy but don't cover all nuclei.")
        include_shared_clusters = st.checkbox("Include shared clusters", key=f"{prefix}_shared_qp")
        st.text("Shared clusters are found across multiple nuclei, not focused on the selected nuclei.")
    return manual_annotations, include_shared_clusters

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
            "Nuclei to highlight", th_subregion_names, key="gp_regionlist_qp"
        )
        sections = st_gp.multiselect("Sections", sections_all, key="gp_sectionlist_qp")
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
                **common_args,
            )
            for plot in plots:
                st.pyplot(plot)
    with multi_gene:
        st_mg = st.form("multigene_plot")
        gene_set = st_mg.multiselect("Select genes", merfish_genes, key="mg_genelist_qp")
        sections = st_mg.multiselect("Sections", sections_all, key="mg_sectionlist_qp")
        dark_background = st_mg.checkbox("Dark background", key="mg_dark_qp")
        if st_mg.form_submit_button("Plot multi-gene expression", on_click=stu.ss_to_qp):
            adata = get_adata(transform=transform)
            plots = cplots.plot_hcr(
                adata,
                gene_set,
                sections=sections,
                dark_background=dark_background,
                **common_args,
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

        de_form = st.form("plot_de_genes")
        taxonomy_level = de_form.selectbox(
            "Taxonomy level", ["cluster", "supertype", "subclass"], key="de_tax_qp"
        )
        manual_annotations, include_shared_clusters = annotation_details_input(context=de_form, prefix="de")
        groups = [
            de_form.expander("Select group 1", expanded=True),
            de_form.expander("Select reference, empty to compare to rest of thalamus (slow)", expanded=False),
        ]
        grouped_types = [0, 0]
        for i, box in enumerate(groups):
            regions = box.multiselect(
                "By nucleus", th_subregion_names, key=f"de_regionlist{i}_qp"
            )
            types_by_annotation = abc.get_obs_from_annotations(
                regions,
                obs_th_neurons,
                taxonomy_level='cluster',
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
            sections_all,
            key="bs_sectionlist_qp",
        )
        st.button(
            "Show all sections",
            on_click=lambda: setattr(ss, "bs_sectionlist_qp", sections_all),
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
                ccf_level=ccf_level,
                point_hue=celltype_label,
                sections=sections,
                point_palette=palettes[celltype_label],
                legend="cells" if show_legend else False,
                **common_args,
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
        manual_annotations, include_shared_clusters = annotation_details_input(prefix="bn")
        show_borders = st.checkbox("Show all boundaries", key="bn_borders_qp")

        celltype_label = st.selectbox(
            "Level of celltype hierarcy",
            ["subclass", "supertype", "cluster"],
            index=2,
            key="bn_tax_qp",
        )
        def celltype_lookup(nuclei):
            return abc.get_obs_from_annotations(
                nuclei,
                obs_th_neurons,
                taxonomy_level='cluster',
                include_shared_clusters=include_shared_clusters,
                manual_annotations=manual_annotations,
            )[celltype_label].unique()

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
            th_subregion_names, 
            key="bn_regionlist_qp",
            on_change=propagate_value_from_lookup,
            args=("bn_regionlist_qp", "bn_typelist_qp", celltype_lookup),
        )
        celltypes = st.multiselect(
            "Select cell types",
            obs_th_neurons[taxonomy_level].unique(),
            key="bn_typelist_qp",
        )

        if st.button("Plot"):
            obs2 = obs_th_neurons.loc[
                obs_th_neurons[celltype_label].isin(celltypes)
            ]
            plots = cplots.plot_ccf_overlay(
                obs2,
                ccf_names=None if show_borders else nuclei,
                ccf_level=ccf_level,
                ccf_highlight=nuclei,
                point_hue=celltype_label,
                sections=None,
                min_group_count=0,
                point_palette=palettes[celltype_label],
                bg_cells=obs_th_neurons,
                **common_args,
            )
            for plot in plots:
                st.pyplot(plot)

stu.ss_to_qp()
