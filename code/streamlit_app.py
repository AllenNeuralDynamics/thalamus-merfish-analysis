import streamlit as st
import numpy as np
import pandas as pd
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
ccf_level = 'substructure'
lump_structures = False
if realigned:
    ccf_label = f'parcellation_{ccf_level}_realigned'
    coords = 'section'
else:
    ccf_label = f'parcellation_{ccf_level}'
    coords = 'reconstructed'
    
th_names = [x for x in abc.get_thalamus_names() if not 'unassigned' in x]
th_subregion_names = [x for x in abc.get_thalamus_names(level=ccf_level) if not 'unassigned' in x]
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
def get_image_volumes(realigned, sections_all, lump_structures=True, edge_width=2):
    section_list = np.rint(np.array(sections_all)/0.2).astype(int)
    ccf_polygons = abc.get_ccf_labels_image(resampled=True, realigned=realigned)
    
    if lump_structures:
        ccf_index = abc.get_ccf_index(level='structure')
        reverse_index = ccf_index.reset_index().groupby('parcellation_term_acronym')['parcellation_index'].first()
        mapping = ccf_index.map(reverse_index).to_dict()
        ccf_polygons = cimg.map_label_values(ccf_polygons, mapping, section_list=section_list)
        
    ccf_boundaries = cimg.sectionwise_label_erosion(
        ccf_polygons, edge_width, fill_val=0, return_edges=True,
        section_list=section_list
        )
    return ccf_polygons, ccf_boundaries

obs_th_neurons, sections_all, subclasses_all = get_data(version, ccf_label)
ccf_polygons, ccf_boundaries = get_image_volumes(realigned, sections_all, lump_structures=False)

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
    "by thalamic nucleus",
    "by section",
])
with tab2:
    sections = st.multiselect(
        "Section z coordinate", sections_all, 
        default=[7.2]
    )
    celltype_label = st.selectbox(
        "Level of celltype hierarcy",
        ['subclass','supertype','cluster'], index=0,
    )
    show_legend = st.checkbox(
        "Show legend"
    )
    if celltype_label=='cluster':
        palette = None
    else:
        palette = palettes[celltype_label]
    obs = obs_th_neurons.loc[lambda df: df['subclass'].isin(subclasses_all)]
    if len(sections)>0 and len(obs)>0:
        plots = cplots.plot_ccf_overlay(obs, ccf_polygons, 
                                    ccf_names=None,
                                        ccf_level=ccf_level,
                                        point_hue=celltype_label, 
                                        sections=sections,
                                        point_palette=palette,
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
                "Select individual nuclei", th_names,
                default=preselect
            )
        else:
            nuclei = st.multiselect(
                "Select individual nuclei", th_names,
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
