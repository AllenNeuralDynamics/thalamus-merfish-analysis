
import scipy.ndimage as ndi
import numpy as np

def sectionwise_label_erosion(label_img, distance_px, fill_val=0, return_edges=False, section_list=None):
    '''Erodes a stack of 2D label images by a specified radius (in px).
    
    Parameters
    ----------
    label_img : array_like
        stack of 2D label images, shape (x, y, n_sections)
    distance_px : int
        dilation radius in pixels
    Returns
    -------
    result_img 
        3D np array, like label_img but with each label eroded by distance_px
        and the eroded area filled with fill_val
    '''
    result_img = np.zeros_like(label_img)
        
    if section_list is None:
        section_list = range(label_img.shape[2])
    for i in section_list:
        # TODO: could restrict range to thal indices
        result_img[:,:,i] = label_erosion(label_img[:,:,i], distance_px, 
                                          fill_val=fill_val,
                                          return_edges=return_edges)
    return result_img

def label_erosion(label_img, distance_px, fill_val=0, return_edges=False):
    label_vals = np.unique(label_img)
    result_img = np.zeros_like(label_img)
    for i in label_vals:
        mask = (label_img==i)
        eroded = ndi.binary_erosion(mask, iterations=distance_px)
        if return_edges:
            eroded = mask & ~eroded
        result_img += i*eroded
    result_img = np.where(label_img==0, 0, 
                          np.where(result_img==0, fill_val, 
                                   result_img))
    return result_img


def map_label_values(label_img, mapping, section_list=None):
    if section_list is None:
        section_list = range(label_img.shape[2])
    subset = label_img[:,:,section_list]
    label_vals = np.unique(subset)
    result_img = np.zeros_like(subset)
    for i in label_vals:
        mask = (label_img[:,:,section_list]==i)
        result_img += mapping[i]*mask
    full_result = np.zeros_like(label_img)
    full_result[:,:,section_list] = result_img
    return full_result