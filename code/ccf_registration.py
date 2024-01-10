from scipy.ndimage import grey_dilation
from scipy.linalg import block_diag
import numpy as np
import json
from pathlib import Path
from colorcet import glasbey, glasbey_light, glasbey_dark, glasbey_warm, glasbey_cool
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb

CMAP = ListedColormap(['white'] + glasbey)

def lighten(hex):
    rgb = to_rgb(hex)
    rgb = 1 - 0.25*(1 - np.array(rgb))
    return tuple(rgb)

def cmap_with_emphasis(n_focus=0, n_secondary=0, dark_background=False):
    secondary = list(np.outer(np.linspace(0.25, 0.75, n_secondary), np.ones(3)))
    colors = glasbey_warm[:n_focus] + secondary + (glasbey_cool if n_focus>0 else glasbey)
    if not dark_background:
        cmap = ['white'] + colors
    else:
        cmap = ['black'] + colors
    return [to_rgb(x) for x in cmap]

def subset_to_ref_bounds(df, coords, ref_subset):
    ref_subset = df.index.intersection(ref_subset)
    df = df.loc[lambda df: 
        (df[coords] <= df.loc[ref_subset, coords].max()).all(axis=1) &
        (df[coords] >= df.loc[ref_subset, coords].min()).all(axis=1)]
    return df.copy()

def add_rescaled_coords(df, coords_xy, xy_res=0.0025):
    # in mm
    coords_scaled = [x+"_scaled" for x in coords_xy]
    coords_int = [x+"_int" for x in coords_xy]
    df[coords_scaled] = (df[coords_xy] - df[coords_xy].min())/xy_res
    # probably not important to calc bounds pre-rounding here
    # nx, ny = np.ceil(df[['x','y']].max().values).astype(int)
    df[coords_int] = df[coords_scaled].round().astype(int)
    df[coords_scaled] = df[coords_scaled] / df[coords_scaled].max()
    return coords_scaled, coords_int

def rasterize_by_dilation(df, coords, label, shape=None, dilate=1):
    if shape is None:
        shape = df[coords[::-1]].max().values + 1
    img = np.zeros(shape, dtype=int)
    for i, group in df.groupby(label):
        img[tuple(group[coords[::-1]].values.T)] = i
    for _ in range(dilate):
        img = grey_dilation(img, size=(3,3))
    return np.flipud(img)

# to and from homogeneous coords (added dim for affine transform)
# representing points as rows (transform by right-multiplication)
def to_hom(x):
    if len(x.shape)==1:
        x = x[np.newaxis, :]
    return np.hstack([x, np.ones((x.shape[0], 1))])
    
def from_hom(x):
    return x[:,:-1]

def apply_affine_left(M, x):
    return from_hom(to_hom(x.T) @ M.T).T

def image_index_from_coords(coord_values, res=10e-3):
    coords_index = np.rint(np.array(coord_values) / res).astype(int)
    return tuple(coords_index.T)

def calculate_quicknii_transform(scale=25):
    """Get affine matrix for transformation from quicknii pixel coordinates
    to CCF (in mm).
    scale: mm resolution of quicknii CCF template
    """
    s = scale/1000
    qn_to_ccf = np.array([[0,0,s,0], [-s,0,0,0], [0,-s,0,0], [13.175,7.975,0,1]])
    ccf_to_qn = np.linalg.inv(qn_to_ccf)
    return ccf_to_qn

def calculate_affine_transform(df, coords_from, coords_to, n_sample=1000):
    samples = np.random.randint(0, len(df), n_sample)
    # intent is for x, y here to be normalized coords from add_rescaled_coords
    # coords_from = ['x', 'y', 'z_reconstructed']
    affine = np.matmul(np.linalg.pinv(to_hom(df[coords_from].iloc[samples].values)), to_hom(df[coords_to].iloc[samples].values))
    # these entries must be zero for a valid affine transform in homogeneous coordinates
    assert all(np.isclose(0, affine[:3,-1]))
    affine[:3,-1] = 0
    return affine

def calculate_anchor_entry(z, to_ccf, ccf_to_qn):
    xyo_base = to_hom(np.array([[1, 1, z], [0, 0, z], [0, 1, z]]))
    xyo_qn = from_hom(xyo_base @ to_ccf @ ccf_to_qn)
    ouv_qn = np.array([[0, 0, 1], [1, 0, -1], [0, 1, -1]]) @ xyo_qn
    anchor_list = list(ouv_qn.flatten())
    return anchor_list

def process_anchor_entry(anchor_list, z, ccf_to_qn):
    ouv_qn = np.array(anchor_list).reshape((3, 3))
    xyo_qn = np.linalg.inv(np.array([[0, 0, 1], [1, 0, -1], [0, 1, -1]])) @ ouv_qn
    xyo_base = np.array([[1, 1, z], [0, 0, z], [0, 1, z]])
    # like to_hom() but for matrix (ie only adding one on diag)
    to_ccf = block_diag(np.linalg.inv(xyo_base) @ xyo_qn, [1]) @ np.linalg.inv(ccf_to_qn)
    return to_ccf

def read_quicknii_file(path, scale=25):
    z_from_name = lambda x: float(x)/10
    ccf_to_qn = calculate_quicknii_transform(scale)
    transforms = dict()
    with open(path, 'r') as f:
        alignment = json.load(f)
    for record in alignment['slices']:
        slice_name = record["nr"]
        transforms[slice_name] = process_anchor_entry(record['anchoring'], 
                                                      z_from_name(slice_name), 
                                                      ccf_to_qn)
    return transforms
            
def export_to_quicknii(df, base_filename, img_label, img_coords,
                       coords_from=None, cmap=CMAP, slice_label=None, scale=25,
                       path='.', save_json=True, save_images=True, format='jpg'):
    coords_scaled, coords_int = add_rescaled_coords(
        df, img_coords[:2], xy_res=0.0025)
    nx, ny = df[coords_int].max() + 1
    
    if coords_from is not None:
        coords_from = coords_from.copy()
        coords_from[:2], _ = add_rescaled_coords(df, coords_from[:2])
    else:
        coords_from = coords_scaled + [img_coords[2]]
    coords_to = ['x_ccf', 'y_ccf', 'z_ccf']
    ccf_to_qn = calculate_quicknii_transform(scale)
    to_ccf = calculate_affine_transform(df, coords_from, coords_to)
    
    path = Path(path)
    if save_images or save_json:
        path.mkdir(parents=True, exist_ok=True)
        
    slices = []
    for n, (zval, df_slice) in enumerate(df.groupby(img_coords[2])):
        if slice_label is not None:
            n = int(df_slice[slice_label].values[0])
        
        filename = f"{base_filename}_{n:03}.{format}"
        if save_images:
            img = rasterize_by_dilation(df_slice, coords_int, img_label, shape=(ny, nx))
            # plt.imsave(path/filename, img, cmap=cmap)
            plt.imsave(path/filename, np.array(cmap)[img])
        
        anchoring = calculate_anchor_entry(zval, to_ccf, ccf_to_qn)
        slices.append(dict(anchoring=anchoring, 
                     filename=filename, 
                     height=ny,
                     width=nx,
                     nr=n))
    data = dict(name=base_filename, slices=slices)
    if save_json:
        with open(path/f'{base_filename}.json', 'w') as f:
            json.dump(data, f, indent=4)
    return data

def preprocess_for_qn_export(df, nn_classes, img_coords, spatial_ref_index, 
                             subset_for_fine_labels=None,
                             slice_label='slice_int',
                             img_label='subclass_int'):
    """Process cell dataframe for export by subsetting to spatial bounds
    of reference dataset
    """
    df = subset_to_ref_bounds(df, img_coords, spatial_ref_index)
    subset_for_fine_labels = df.index.intersection(subset_for_fine_labels)
    
    working_label = "thal_class"
    df[slice_label] = df['z_section'].apply(lambda x: str(int(x*10)))

    df[working_label] = df["class"].astype('string')
    df.loc[subset_for_fine_labels, working_label] = df.loc[subset_for_fine_labels, "subclass"].astype('string')

    all_labels = df[working_label].unique()
    # ordering by counts prioritizes separation of larger groups
    focus_labels = df.loc[subset_for_fine_labels, working_label].value_counts().index
    other_labels = df.loc[df.index.difference(subset_for_fine_labels)].loc[
        lambda df: ~df['class'].isin(nn_classes), working_label].value_counts().index

    label_order = ["background"] + list(focus_labels) + nn_classes + list(other_labels)
    label_map = {x: label_order.index(x) for x in all_labels}
    df[img_label] = df[working_label].map(label_map)
    cmap = cmap_with_emphasis(len(focus_labels), n_secondary=len(nn_classes))
    return df, cmap