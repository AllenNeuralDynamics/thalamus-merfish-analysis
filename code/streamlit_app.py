from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from thalamus_merfish_analysis import ccf_plots as cplots
from thalamus_merfish_analysis import ccf_images as cimg
from thalamus_merfish_analysis import abc_load as abc
# from abc_load import _get_ccf_metadata
# _get_ccf_metadata = st.cache_data(_get_ccf_metadata)


# TODO: cache growing when shouldn't be? or just in debug?
st.set_page_config(
    page_title="Thalamus MERFISH explorer",
    layout="wide"
)
version = "20230830"
section_col = 'z_section'
ccf_level = 'substructure'
lump_structures = False
with st.expander("CCF alignment settings"):
    realigned = st.radio("CCF alignment", [False, True], index=0,
                     format_func = lambda realigned: 
                      'thalamus-specific section-wise affine alignment' if realigned 
                      else 'published nonlinear alignment'
                      )
if realigned:
    ccf_label = f'parcellation_{ccf_level}_realigned'
    coords = 'section'
else:
    ccf_label = f'parcellation_{ccf_level}'
    coords = 'reconstructed'
    
th_names = [x for x in abc.get_thalamus_names() if not 'unassigned' in x]
th_subregion_names = [x for x in abc.get_thalamus_names(level=ccf_level) if not 'unassigned' in x]
palettes = {level: abc.get_taxonomy_palette(level) for level in 
            ['subclass','supertype']}
palettes['cluster'] = abc.get_thalamus_cluster_palette()
cplots.CCF_REGIONS_DEFAULT = th_subregion_names

has_realigned_asset = Path("/data/realigned/abc_realigned_metadata_thalamus-boundingbox.parquet").exists()
if realigned and not has_realigned_asset:
    realigned = False
    st.warning("Realigned metadata not found, using published alignment")

@st.cache_data
def get_data(version, ccf_label):
    obs = abc.get_combined_metadata(realigned=has_realigned_asset, version=version, drop_unused=False)
    # remove non-neuronal and some other outlier non-thalamus types
    obs_neurons = abc.filter_by_class_thalamus(obs, filter_midbrain=False)
    obs_th = obs[obs[ccf_label].isin(th_names)]
    obs_th_neurons = obs.loc[obs_neurons.index.intersection(obs_th.index)]
    sections_all = sorted(obs_th_neurons[section_col].unique())
    subclasses_all = obs_th_neurons['subclass'].value_counts().loc[lambda x: x>100].index
    return obs_th_neurons, sections_all, subclasses_all


@st.cache_data
def get_image_volumes(realigned, sections_all, lump_structures=False, edge_width=1):
    section_list = np.rint(np.array(sections_all)/0.2).astype(int)
    ccf_images = abc.get_ccf_labels_image(resampled=True, realigned=realigned)
    
    if lump_structures:
        ccf_index = abc.get_ccf_index(level='structure')
        reverse_index = ccf_index.reset_index().groupby('parcellation_term_acronym')['parcellation_index'].first()
        mapping = ccf_index.map(reverse_index).to_dict()
        ccf_images = cimg.map_label_values(ccf_images, mapping, section_list=section_list)
        
    ccf_boundaries = cimg.sectionwise_label_erosion(
        ccf_images, edge_width, fill_val=0, return_edges=True,
        section_list=section_list
        )
    return ccf_images, ccf_boundaries

obs_th_neurons, sections_all, subclasses_all = get_data(version, ccf_label)
ccf_polygons, ccf_boundaries = get_image_volumes(realigned, sections_all, lump_structures=lump_structures)

@st.cache_resource
def get_adata(transform="cpm"):
    return abc.load_adata(version=version, transform=transform, from_metadata=obs_th_neurons)

with st.sidebar:
    st.write(f"Version: {version}")
    st.write(f"Gene data version: {abc.files.adata('raw').version}")
    st.write(f"Metadata version: {abc.files.cell_metadata.version}")
    st.write(f"CCF version: {abc.files.ccf_metadata.version}")
    st.write(f"Taxonomy ID: {abc.get_taxonomy_id()}")
    
pane1, pane2 = st.columns(2)
common_args = dict(
    section_col=section_col,
    x_col = 'x_'+coords,
    y_col = 'y_'+coords,
    boundary_img=ccf_boundaries
)
def full_section_name(section_z):
    return f"Section {np.rint(section_z/200e-3)} (z={section_z})"

with pane2:
    st.header("Gene expression")
    transform = st.radio("Gene expression units", ['log2cpm', 'log2cpv', 'raw'], index=0)
    genes = abc.get_gene_metadata()["gene_symbol"].values
    gene = st.selectbox("Select gene", genes, index=None)
    nuclei = st.multiselect("Nuclei to highlight", th_subregion_names)
    sections = st.multiselect("Sections", sections_all, format_func=full_section_name)
    if len(sections)==0: sections = None
    focus_plot = st.checkbox("Focus on selected nuclei") and len(nuclei)>0
    if gene is not None:
        adata = get_adata(transform=transform)
        plots = cplots.plot_expression_ccf(
            adata, gene, ccf_polygons,
            nuclei=cplots.CCF_REGIONS_DEFAULT, 
            highlight=nuclei,
            sections=sections,
            zoom_to_highlighted=focus_plot,
            **common_args
        )
        for plot in plots:
            st.pyplot(plot)

with pane1:
    st.header("Cell type taxonomy annotations")

    kwargs = dict(
        bg_cells=obs_th_neurons,
        s=3, 
        **common_args
    )
    
    def plot(obs, sections, regions=None, point_hue='subclass'):
        return cplots.plot_ccf_overlay(obs, ccf_images, 
                                    ccf_names=regions,
                                        point_hue=celltype_label, 
                                        sections=sections,
                                        point_palette=palettes[celltype_label],
                                        **kwargs)
    tab1, tab2 = st.tabs([
        "by thalamic nucleus",
        "by section",
    ])
    with tab2:
        sections = st.multiselect(
            "Section", sections_all, format_func=full_section_name,
        )
        celltype_label = st.selectbox(
            "Level of celltype hierarcy",
            ['subclass','supertype','cluster'], index=0,
        )
        show_legend = st.checkbox(
            "Show legend"
        )
        obs = obs_th_neurons.loc[lambda df: df['subclass'].isin(subclasses_all)]
        if len(sections)>0 and len(obs)>0:
            plots = cplots.plot_ccf_overlay(obs, ccf_polygons, 
                                        ccf_names=None,
                                            ccf_level=ccf_level,
                                            point_hue=celltype_label, 
                                            sections=sections,
                                            point_palette=palettes[celltype_label],
                                            legend='cells' if show_legend else False,
                                            **kwargs)
            for plot in plots:
                st.pyplot(plot)


    nucleus_groups = {
        "Motor": ['VAL','VM'],
        "Vision": ['LGd','LP'],
        "Somatosensory": ['VPM','VPL','PO'],
        "Limbic/Anterior": ['AD','AV','AM'],
        "Auditory": ['MG']
    }
    with tab1:
        manual_annotations = st.radio("Nucleus vs cluster annotations", [True, False], index=0,
                        format_func = lambda manual_annotations: 
                        'manual' if manual_annotations 
                        else 'automated'
                        )
        groups = st.multiselect(
            "Select nucleus groups", nucleus_groups.keys(),
        )
        with st.empty():
            if len(groups)>0:
                preselect = set.union(*[set(nucleus_groups[group]) for group in groups])
                nuclei = st.multiselect(
                    "Select individual nuclei", th_subregion_names,
                    default=preselect
                )
            else:
                nuclei = st.multiselect(
                    "Select individual nuclei", th_subregion_names,
                )
        
        celltype_label = st.selectbox(
            "Level of celltype hierarcy",
            ['subclass','supertype','cluster'], index=2,
            key=2
        )
        include_shared_clusters = st.checkbox(
            "Include shared clusters"
        )
        
        try:
            if len(nuclei)>0:
                obs2 = pd.concat([
                    abc.get_obs_from_annotated_clusters(nucleus, obs_th_neurons, 
                                                        include_shared_clusters=include_shared_clusters,
                                                        manual_annotations=manual_annotations)
                    for nucleus in nuclei
                ])
                if len(obs2)>0:
                    regions = [x for x in th_subregion_names if any((name in x and not 'pc' in x) or (name==x) 
                                                                    for name in nuclei)]
                    plots = cplots.plot_ccf_overlay(obs2, ccf_polygons, 
                                                ccf_names=regions,
                                                ccf_level=ccf_level,
                                                # highlight=nuclei, TODO: fix highlight for raster plots
                                                point_hue=celltype_label, 
                                                sections=None,
                                                min_group_count=0,
                                                point_palette=None if celltype_label=='cluster' else palettes[celltype_label],
                                                **kwargs)
                    for plot in plots:
                        st.pyplot(plot)
                else:
                    st.write("No annotations found for nuclei")
        except UserWarning as exc:
            str(exc)
