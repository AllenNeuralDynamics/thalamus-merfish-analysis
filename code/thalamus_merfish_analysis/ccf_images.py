import scipy.ndimage as ndi
import numpy as np


def sectionwise_dilation(mask_img, distance_px, true_radius=False):
    '''Dilates a stack of 2D binary masks by a specified radius (in px).
    
    Parameters
    ----------
    mask_img : array_like
        stack of 2D binary mask, shape (x, y, n_sections)
    distance_px : int
        dilation radius in pixels
    true_radius : bool, default=False
        specifies the method used by ndimage's binary_dilation to dilate
          - False: dilates by 1 px per iteration for iterations=distance_px
          - True: dilates once using a structure of radius=distance_px
        both return similar results but true_radius=False is significantly faster

    Returns
    -------
    dilated_mask_img 
        3D np array, stack of dilated 2D binary masks
    '''
    dilated_mask_img = np.zeros_like(mask_img)
    
    if true_radius:
        # generate a circular structure for dilation
        coords = np.mgrid[-distance_px:distance_px+1, -distance_px:distance_px+1]
        struct = np.linalg.norm(coords, axis=0) <= distance_px
        
    for i in range(mask_img.shape[2]):
        if true_radius:
            dilated_mask_img[:,:,i] = ndi.binary_dilation(mask_img[:,:,i], 
                                                          structure=struct)
        else:
            dilated_mask_img[:,:,i] = ndi.binary_dilation(mask_img[:,:,i], 
                                                           iterations=distance_px)
    return dilated_mask_img


def sectionwise_closing(mask_img, distance_px):
    """ Closes a stack of 2D binary masks by a specified radius (in px).
    
    Parameters
    ----------
    mask_img : array_like
        stack of 2D binary mask, shape (x, y, n_sections)
    distance_px : int
        closing radius in pixels
        
    Returns
    -------
    closed_mask_img
        3D np array, stack of closed 2D binary masks
    """
    closed_mask_img = np.zeros_like(mask_img)
    
    for i in range(mask_img.shape[2]):
        closed_mask_img[:,:,i] = ndi.binary_closing(mask_img[:,:,i], 
                                                    iterations=distance_px)
        
    return closed_mask_img


def sectionwise_fill_holes(mask_img):
    '''Fills holes in a stack of 2D binary masks.
    
    Parameters
    ----------
    mask_img : array_like
        stack of 2D binary mask, shape (x, y, n_sections)
        
    Returns
    -------
    filled_mask_img
        stack of 2D binary masks with holes filled
    '''
    filled_mask_img = np.zeros_like(mask_img)
    
    for i in range(mask_img.shape[2]):
        filled_mask_img[:,:,i] = ndi.binary_fill_holes(mask_img[:,:,i])
        
    return filled_mask_img


def cleanup_mask_regions(mask_img, area_ratio_thresh=0.1):
    ''' Removes, sectionwise, any binary mask regions whose areas are smaller
    than the specified ratio, as compared to the largest region in the mask.
    
    Parameters
    ----------
    mask_img : array_like
        stack of 2D binary mask, shape (x, y, n_sections)
    area_ratio_thresh : float, default=0.1
        threshold for this_region:largest_region area difference ratio; removes
        any regions smaller than this threshold
        
    Returns
    -------
    new_mask_img
        stack of 2D binary masks with too-small regions removed
    '''
    new_mask_img = np.zeros_like(mask_img)
    for sec in range(mask_img.shape[2]):
        mask_2d = mask_img[:,:,sec]
        labeled_mask, n_regions = ndi.label(mask_2d)

        # calculate the area of the largest region
        largest_region = np.argmax(ndi.sum(mask_2d, labeled_mask, 
                                           range(n_regions+1)))
        largest_area = np.sum(labeled_mask==largest_region)

        # filter out regions with area ratio smaller than the specified threshold
        regions_to_keep = [label for label 
                           in range(1, n_regions+1) 
                           if ( (np.sum(labeled_mask==label) / largest_area) 
                                >= area_ratio_thresh
                              )
                          ]
        # make a new mask with only the remaining objects
        new_mask_img[:,:,sec] = np.isin(labeled_mask, regions_to_keep)

    return new_mask_img

def sectionwise_label_erosion(label_img, distance_px, fill_val=0, 
                              return_edges=False, section_list=None):
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
    ''' Maps label values in a 3D label image to new values.

    Parameters
    ----------
    label_img : array_like
        3D label image, shape (x, y, n_sections)
    mapping : dict
        mapping of old label values to new label values
    section_list : list of float, default=None
        list of sections to process, default=None processes all sections in label_img

    Returns
    -------
    result_img : array_like
        3D image, same shape as label_img, with values mapped according to the  
        mapping dict. If only a subset of sections were mapped, values in all 
        non-mapped sections are set to zero.
    '''
    # get sections to be mapped
    if section_list is None:
        section_list = range(label_img.shape[2])
    subset = label_img[:,:,section_list].copy()
    
    # create an element-wise mapping function; .get(x, x) returns x if key x is
    # not in the mapping dict
    mapping_function = np.vectorize(lambda x: mapping.get(x, x))
    subset_mapped = mapping_function(subset)
    
    # set labels in any section not mapped to zero
    result_img = np.zeros_like(label_img)
    # place the mapped subset back into an array of the original shape
    result_img[:,:,section_list] = subset_mapped
    
    return result_img


def image_index_from_coords(coord_values, res=10e-3):
    coords_index = np.rint(np.array(coord_values) / res).astype(int)
    return tuple(coords_index.T)