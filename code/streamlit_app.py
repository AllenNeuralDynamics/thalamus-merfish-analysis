import streamlit as st
import numpy as np
from thalamus_merfish_analysis import ccf_plots as cplots
from thalamus_merfish_analysis import ccf_images as cimg
from thalamus_merfish_analysis import abc_load as abc
# from abc_load import get_ccf_metadata
# get_ccf_metadata = st.cache_data(get_ccf_metadata)

version = "20230830"
realigned = st.radio("CCF alignment", [False, True], index=0,
                     format_func = lambda realigned: 
                      'thalamus-specific section-wise affine alignment' if realigned 
                      else 'published nonlinear alignment'
                      )
section_col = 'z_section'
if realigned:
    ccf_label = 'parcellation_substructure_realigned'
    coords = 'section'
else:
    ccf_label = 'parcellation_substructure'
    coords = 'reconstructed'
    
th_names = abc.get_thalamus_substructure_names()
th_subregion_names = list(set(th_names).difference(['TH-unassigned']))
palettes = {level: abc.get_taxonomy_palette(level) for level in 
            ['subclass','supertype','cluster']}

@st.cache_data
def get_data(version, ccf_label):
    obs = abc.get_combined_metadata(realigned=True, version=version, drop_unused=False)
    # remove non-neuronal and some other outlier non-thalamus types
    obs_neurons = abc.filter_adata_by_class(obs, filter_midbrain=False)
    obs_th = obs[obs[ccf_label].isin(th_names)]
    obs_th_neurons = obs.loc[obs_neurons.index.intersection(obs_th.index)]
    sections_all = sorted(obs_th_neurons[section_col].unique())
    subclasses_all = obs_th_neurons['subclass'].value_counts().loc[lambda x: x>100].index
    return obs_th_neurons, sections_all, subclasses_all

@st.cache_data
def get_image_volumes(realigned, sections_all, edge_width=2):
    ccf_polygons = abc.get_ccf_labels_image(resampled=True, realigned=realigned)
    ccf_boundaries = cimg.sectionwise_label_erosion(
        ccf_polygons, edge_width, fill_val=0, return_edges=True,
        section_list=np.rint(np.array(sections_all)/0.2).astype(int)
        )
    return ccf_polygons, ccf_boundaries

obs_th_neurons, sections_all, subclasses_all = get_data(version, ccf_label)
ccf_polygons, ccf_boundaries = get_image_volumes(realigned, sections_all)

kwargs = dict(
    bg_cells=obs_th_neurons,
    section_col=section_col,
    x_col = 'x_'+coords,
    y_col = 'y_'+coords,
    s=3, 
    shape_palette='dark_outline',
    boundary_img=ccf_boundaries
)


# @st.cache_resource
def plot(obs, sections, regions=None, point_hue='subclass'):
    return cplots.plot_ccf_overlay(obs, ccf_polygons, 
                                   ccf_names=regions,
                                    point_hue=celltype_label, 
                                    sections=sections,
                                    point_palette=palettes[celltype_label],
                                    **kwargs)

tab1, tab2 = st.tabs([
    "by section",
    "by CCF region"
])
with tab1:
    sections = st.multiselect(
        "Section z coordinate", sections_all, [7.2]
    )
    celltype_label = st.selectbox(
        "Level of celltype hierarcy",
        ['subclass','supertype','cluster'], index=0,
    )
    if celltype_label=='cluster':
        legend = False
        palette = None
    else:
        legend = 'cells'
        palette = palettes[celltype_label]
    obs = obs_th_neurons.loc[lambda df: df['subclass'].isin(subclasses_all)]
    plots = cplots.plot_ccf_overlay(obs, ccf_polygons, 
                                   ccf_names=None,
                                    point_hue=celltype_label, 
                                    sections=sections,
                                    point_palette=palette,
                                    legend=legend,
                                    **kwargs)
    for plot in plots:
        st.pyplot(plot)


with tab2:
    nucleus = st.selectbox(
        "CCF region", th_subregion_names, index=None,
    )
    celltype_label = st.selectbox(
        "Level of celltype hierarcy",
        ['subclass','supertype','cluster'], index=2,
        key=2
    )
    try:
        obs2 = abc.get_obs_from_annotated_clusters(nucleus, obs_th_neurons)
        regions = [x for x in th_subregion_names if nucleus in x]
        plots = cplots.plot_ccf_overlay(obs2, ccf_polygons, 
                                    ccf_names=regions,
                                        point_hue=celltype_label, 
                                        sections=None,
                                        min_group_count=0,
                                        point_palette=palettes[celltype_label],
                                        **kwargs)
        for plot in plots:
            st.pyplot(plot)
    except UserWarning as exc:
        str(exc)
