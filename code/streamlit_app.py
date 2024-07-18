from pathlib import Path
import streamlit as st
import pandas as pd
from thalamus_merfish_analysis import ccf_plots as cplots
from thalamus_merfish_analysis import ccf_images as cimg
# from thalamus_merfish_analysis import abc_load as abc
from thalamus_merfish_analysis.abc_load_thalamus import ThalamusWrapper


st.set_page_config(page_title="Thalamus MERFISH explorer", layout="wide")
version = "20230830"
section_col = "brain_section_label"
ccf_level = "substructure"
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
th_subregion_names = [
    x for x in abc.get_thalamus_names(level=ccf_level) if "unassigned" not in x
]
palettes = {
    level: abc.get_taxonomy_palette(level) for level in ["subclass", "supertype"]
}
palettes["cluster"] = abc.get_thalamus_cluster_palette()
cplots.CCF_REGIONS_DEFAULT = th_subregion_names

has_realigned_asset = Path(
    "/data/realigned/abc_realigned_metadata_thalamus-boundingbox.parquet"
).exists()
if realigned and not has_realigned_asset:
    realigned = False
    st.warning("Realigned metadata not found, using published alignment")


@st.cache_data
def get_data(version, ccf_label, extend_borders=False):
    obs = abc.get_combined_metadata(
        realigned=has_realigned_asset, drop_unused=False
    )
    # remove non-neuronal and some other outlier non-thalamus types
    obs_neurons = abc.filter_by_class_thalamus(obs, filter_midbrain=False)
    buffer = 5 if extend_borders else 0
    obs_th_neurons = abc.filter_by_thalamus_coords(obs_neurons, buffer=buffer)
    sections_all = sorted(obs_th_neurons[section_col].unique())
    subclasses_all = (
        obs_th_neurons["subclass"].value_counts().loc[lambda x: x > 100].index
    )
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
        ccf_images = cimg.map_label_values(
            ccf_images, mapping, section_list=abc.TH_SECTIONS
        )
    ccf_boundaries = None
    # ccf_boundaries = cimg.sectionwise_label_erosion(
    #     ccf_images, edge_width, fill_val=0, return_edges=True, section_list=section_list
    # )
    return ccf_images, ccf_boundaries


obs_th_neurons, sections_all, subclasses_all = get_data(version, ccf_label)
ccf_images, ccf_boundaries = get_image_volumes(
    realigned, sections_all, lump_structures=lump_structures
)

@st.cache_resource
def get_adata(transform="cpm"):
    return abc.load_adata(
        transform=transform, from_metadata=obs_th_neurons
    )


common_args = dict(
    section_col=section_col,
    x_col="x_" + coords,
    y_col="y_" + coords,
    boundary_img=ccf_boundaries,
    ccf_images=ccf_images,
)

with st.sidebar:
    st.write(f"Version: {version}")
    st.write(f"Gene data version: {abc.files.adata('raw').version}")
    st.write(f"Metadata version: {abc.files.cell_metadata.version}")
    st.write(f"CCF version: {abc.files.ccf_metadata.version}")
    st.write(f"Taxonomy ID: {abc.get_taxonomy_id()}")

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
pane1, pane2 = st.columns(2)
with pane2:
    st.header("Gene expression")
    transform = st.radio(
        "Gene expression units", ["log2cpt", "log2cpm", "log2cpv", "raw"], index=0
    )
    genes = abc.get_gene_metadata()["gene_symbol"].values
    single_gene, multi_gene = st.tabs(["Single gene plots", "Multiple gene overlay"])
    with single_gene:
        with st.form("gene_plot"):
            gene = st.selectbox("Select gene", genes, index=None)
            nuclei = st.multiselect("Nuclei to highlight", th_subregion_names)
            sections = st.multiselect("Sections", sections_all, key="gene_sections")
            if len(sections) == 0:
                sections = None
            focus_plot = st.checkbox("Focus on selected nuclei") and len(nuclei) > 0
            plot_single_gene = st.form_submit_button("Plot gene expression")
        if plot_single_gene:
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
        with st.form("multigene_plot"):
            gene_set = st.multiselect("Select genes", genes)
            sections = st.multiselect("Sections", sections_all)
            dark_background = st.checkbox("Dark background")
            plot_multi_gene = st.form_submit_button("Plot multi-gene expression")
        if plot_multi_gene:
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

with pane1:
    st.header("Cell type taxonomy annotations")

    kwargs = dict(bg_cells=None, point_size=3, **common_args)

    def plot(obs, sections, regions=None, point_hue="subclass"):
        return cplots.plot_ccf_overlay(
            obs,
            ccf_names=regions,
            point_hue=celltype_label,
            sections=sections,
            point_palette=palettes[celltype_label],
            **kwargs,
        )

    types_by_nucleus, types_by_section = st.tabs(
        [
            "by thalamic nucleus",
            "by section",
        ]
    )
    with types_by_section:
        sections = st.multiselect(
            "Section",
            sections_all,
            key="sections",
        )
        st.button(
            "Show all sections",
            on_click=lambda: setattr(st.session_state, "sections", sections_all),
        )
        celltype_label = st.selectbox(
            "Level of celltype hierarcy",
            ["subclass", "supertype", "cluster"],
            index=0,
        )
        show_legend = st.checkbox("Show legend")
        if len(sections) > 0:
            plots = cplots.plot_ccf_overlay(
                obs_th_neurons,
                ccf_names=None,
                ccf_level=ccf_level,
                point_hue=celltype_label,
                sections=sections,
                point_palette=palettes[celltype_label],
                legend="cells" if show_legend else False,
                **kwargs,
            )
            for plot in plots:
                st.pyplot(plot)

    nucleus_groups = {
        "Motor": ["VAL", "VM"],
        "Vision": ["LGd", "LP"],
        "Somatosensory": ["VPM", "VPL", "PO"],
        "Limbic/Anterior": ["AD", "AV", "AM"],
        "Auditory": ["MG"],
    }
    with types_by_nucleus:
        manual_annotations = st.radio(
            "Nucleus vs cluster annotations",
            [True, False],
            index=0,
            format_func=lambda manual_annotations: "manual"
            if manual_annotations
            else "automated",
        )
        groups = st.multiselect(
            "Select nucleus groups",
            nucleus_groups.keys(),
        )
        with st.empty():
            if len(groups) > 0:
                preselect = set.union(*[set(nucleus_groups[group]) for group in groups])
                nuclei = st.multiselect(
                    "Select individual nuclei", th_subregion_names, default=preselect
                )
            else:
                nuclei = st.multiselect("Select individual nuclei", th_subregion_names)

        celltype_label = st.selectbox(
            "Level of celltype hierarcy",
            ["subclass", "supertype", "cluster"],
            index=2,
            key=2,
        )
        include_shared_clusters = st.checkbox("Include shared clusters")

        try:
            if len(nuclei) > 0:
                obs2 = abc.get_obs_from_annotated_clusters(
                    nuclei,
                    obs_th_neurons,
                    include_shared_clusters=include_shared_clusters,
                    manual_annotations=manual_annotations,
                )
                if len(obs2) > 0:
                    # TODO: expand names to include subregions?
                    # regions = [
                    #     x
                    #     for x in th_subregion_names
                    #     if any(
                    #         (name in x and "pc" not in x) or (name == x)
                    #         for name in nuclei
                    #     )
                    # ]
                    plots = cplots.plot_ccf_overlay(
                        obs2,
                        ccf_names=nuclei,
                        ccf_level=ccf_level,
                        # highlight=nuclei, TODO: fix highlight for raster plots
                        point_hue=celltype_label,
                        sections=None,
                        min_group_count=0,
                        point_palette=None
                        if celltype_label == "cluster"
                        else palettes[celltype_label],
                        **kwargs,
                    )
                    for plot in plots:
                        st.pyplot(plot)
                else:
                    st.write("No annotations found for nuclei")
        except UserWarning as exc:
            str(exc)
