import numpy as np

from .abc_load import X_RESOLUTION, Y_RESOLUTION, Z_RESOLUTION, get_ccf_index
from .abc_load_base import _label_masked_cells
from .ccf_images import sectionwise_label_erosion, map_label_values

# column names to access in notebooks as well
ERODED_CCF_INDEX_COL = 'parcellation_index_eroded'
ERODED_CCF_STRUCTURE_COL = 'parcellation_structure_eroded'


def label_cells_by_eroded_ccf(obs, ccf_img, distance_px=5, realigned=False):
    ''' Labels cells in obs by their eroded CCF parcellation index and structure.

    Eroding of CCF structures can help to minimize the impacts of imprecise or  
    averaged CCF boundaries on downstream analyses, such as per-nucleus metrics.

    Parameters
    ----------
    obs : DataFrame
        cell metadata df
    ccf_img : array of shape (x, y, n_sections)
        image volume of CCF labels, loaded via abc_load_base.get_ccf_labels_image(...)
    distance_px : int, default=5
        number of pixels (1px=1um) by which to erode each CCF structure
    realigned : bool, default=False
        whether to use manually realigned coordinates & structures

    Returns
    -------
    obs : pandas DataFrame
        cell metadata df with new columns [ERODED_CCF_INDEX_COL, ERODED_CCF_STRUCTURE_COL]
    '''
    # determine which sections are present in obs
    coords = 'section' if realigned else 'reconstructed'
    sections = sorted(obs[f'z_{coords}'].unique())
    # some functions requires sections passed as ccf_img indices
    sections_int = np.rint(np.array(sections)/0.2).astype(int) if sections is not None else None

    # merge CCF substructures into their parcellation_structure before eroding 
    ccf_img_merge = merge_substructures(ccf_img, sections=sections)

    # erode CCF structures 
    ccf_img_erode = sectionwise_label_erosion(ccf_img, 
                                              distance_px=distance_px, 
                                              section_list=sections_int)

    # label cells in obs by their new, eroded CCF parcellation                                                          
    coord_names = [f"{x}_{coords}" for x in 'xyz'] 
    resolutions = np.array([X_RESOLUTION, Y_RESOLUTION, Z_RESOLUTION])
    obs = _label_masked_cells(obs, ccf_img_erode, coord_names, 
                              resolutions, field_name=ERODED_CCF_INDEX_COL)

    # add parcellation_strcuture labels to obs
    ccf_index = get_ccf_index(level='structure')
    obs[ERODED_CCF_STRUCTURE_COL] = obs[ERODED_CCF_INDEX_COL].map(ccf_index)

    return obs


def merge_substructures(ccf_img, sections=None):
    ''' Merges CCF parcellation_substructures into their parcellation_structure.

    Parameters
    ----------
    ccf_img : array of shape (x, y, n_sections)
        image volume of CCF labels, loaded via abc_load_base.get_ccf_labels_image(...)
    sections : list of float, default=None
        list of section numbers to process, default=None processes all sections.
        Use subset of sections when possible as it drastically decreases runtime.

    Returns
    -------
    ccf_img : array 
        image volume, of same shape as ccf_images input parameter, with 
        substructures merged into their parcellation_structure
    '''

    # group substructures by structure (e.g. ['AMv', 'AMd'] -> 'AM')
    ccf_index = get_ccf_index(level='structure')
    reverse_index = ccf_index.reset_index().groupby('parcellation_term_acronym')['parcellation_index'].first()
    mapping = ccf_index.map(reverse_index).to_dict()

    # map substructure to structure indices
    sec_int = np.rint(np.array(sections)/0.2).astype(int) if sections is not None else None
    ccf_img = map_label_values(ccf_img, mapping, section_list=sec_int)

    return ccf_img