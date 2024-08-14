import numpy as np

from . import abc_load as abc
from .abc_load_base import _label_masked_cells
from .ccf_images import sectionwise_label_erosion, map_label_values

# column names to access in notebooks as well
_ERODED_CCF_INDEX_COL = 'parcellation_index_eroded'


def label_cells_by_eroded_ccf(obs, ccf_img, distance_px=5, ccf_level='substructure', realigned=False):
    ''' Labels cells in obs based on eroded CCF parcellations.

    Eroding of CCF parcellations can help to minimize the impacts of imprecise or  
    averaged CCF boundaries on downstream analyses, such as per-nucleus metrics.

    Parameters
    ----------
    obs : DataFrame
        cell metadata df
    ccf_img : array of shape (x, y, n_sections)
        image volume of CCF labels, loaded via abc_load_base.get_ccf_labels_image(...)
    ccf_level : str, default='substructure'
        Level of CCF parcellation to use
    distance_px : int, default=5
        number of pixels (1px=1um) by which to erode each CCF structure
    realigned : bool, default=False
        whether to use manually realigned coordinates & labels

    Returns
    -------
    obs : pandas DataFrame
        cell metadata df with new column containing labels from eroded CCF parcellations
    ccf_label_eroded
        name of new column
    '''
    # determine which sections are present in obs
    coords = 'section' if realigned else 'reconstructed'
    sections = sorted(obs[f'z_{coords}'].unique())
    # some functions requires sections passed as ccf_img indices
    sections_int = np.rint(np.array(sections)/0.2).astype(int) if sections is not None else None

    # merge CCF substructures into their parcellation_structure before eroding 
    if ccf_level != 'substructure':
        ccf_img_merge = merge_substructures(ccf_img, ccf_level=ccf_level, sections=sections)
    else:
        ccf_img_merge = ccf_img

    # erode CCF structures 
    ccf_img_erode = sectionwise_label_erosion(ccf_img_merge, 
                                              distance_px=distance_px, 
                                              section_list=sections_int)

    # label cells in obs by their new, eroded CCF parcellation
    coord_names = [f"{x}_{coords}" for x in 'xyz'] 
    resolutions = np.array([abc.X_RESOLUTION, abc.Y_RESOLUTION, abc.Z_RESOLUTION])
    # TODO: no need to actually add this to obs
    obs = _label_masked_cells(obs, ccf_img_erode, coord_names, 
                              resolutions, field_name=_ERODED_CCF_INDEX_COL)

    # add parcellation_strcuture labels to obs
    ccf_index = abc.get_ccf_index(level='structure')
    ccf_label_eroded = f'parcellation_{ccf_level}_eroded'
    obs[ccf_label_eroded] = obs[_ERODED_CCF_INDEX_COL].map(ccf_index)

    return obs, ccf_label_eroded


def merge_substructures(ccf_img, ccf_level='structure', sections=None):
    ''' Merges CCF parcellation image values to match a higher level of the 
    parcellation hierarchy

    Parameters
    ----------
    ccf_img : array of shape (x, y, n_sections)
        image volume of CCF labels, loaded via abc_load_base.get_ccf_labels_image(...)
    ccf_level : str, default='structure'
        Level of CCF parcellation to use
    sections : list of float, default=None
        list of section numbers to process, default=None processes all sections.
        Use subset of sections when possible as it drastically decreases runtime.

    Returns
    -------
    ccf_img : array 
        image volume, of same shape as ccf_images input parameter, with 
        substructures merged into their parcellation_structure
    '''

    # create mapping from lower to higher level labels
    ccf_index = abc.get_ccf_index(level=ccf_level)
    reverse_index = ccf_index.reset_index().groupby('parcellation_term_acronym')['parcellation_index'].first()
    mapping = ccf_index.map(reverse_index).to_dict()

    # map image values
    sec_int = np.rint(np.array(sections)/0.2).astype(int) if sections is not None else None
    ccf_img = map_label_values(ccf_img, mapping, section_list=sec_int)

    return ccf_img