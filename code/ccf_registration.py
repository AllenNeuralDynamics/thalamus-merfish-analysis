from scipy.ndimage import grey_dilation
import numpy as np
import json
from pathlib import Path
from colorcet import glasbey_light, glasbey_dark, glasbey_warm, glasbey_cool
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb

CMAP = ListedColormap(['black'] + glasbey_light)

def lighten(hex):
    rgb = to_rgb(hex)
    rgb = 1 - 0.25*(1 - np.array(rgb))
    return tuple(rgb)

def cmap_with_emphasis(n_focus, n_secondary=0, dark_background=False):
    secondary = list(np.outer(np.linspace(0.25, 0.75, n_secondary), np.ones(3)))
    glasbey = glasbey_warm[:n_focus] + secondary + glasbey_cool
    if not dark_background:
        cmap = ['white'] + glasbey
    else:
        cmap = ['black'] + glasbey
    return [to_rgb(x) for x in cmap]

def add_rescaled_coords(df, coords_xy, xy_res=0.0025):
    # in mm
    coords_scaled = [x+"_scaled" for x in coords_xy]
    coords_int = [x+"_int" for x in coords_xy]
    rescale = lambda df: (df - df.min())/xy_res
    df[coords_scaled] = df[coords_xy].apply(rescale)
    # probably not important to calc bounds pre-rounding here
    # nx, ny = np.ceil(df[['x','y']].max().values).astype(int)
    df[coords_int] = df[coords_scaled].round().astype(int)
    df[coords_scaled] = df[coords_scaled] / df[coords_scaled].max()
    return coords_scaled, coords_int

def subset_to_ref_bounds(df, coords, ref_subset):
    df = df.loc[lambda df: 
        (df[coords] <= df.loc[ref_subset, coords].max()).all(axis=1) & 
        (df[coords] >= df.loc[ref_subset, coords].min()).all(axis=1)]
    return df.copy()
    
def rasterized_image_stack(data, x, y, z, label, dilate=1):
    shape = data[[y,x]].max().values + 1
    images = []
    for _, df in data.groupby(z):
        img = rasterize_by_dilation(data, x, y, label, shape=shape, dilate=dilate)
        images.append(img)
    stack = np.stack(images)
    return stack

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
    return np.hstack([x, np.ones((x.shape[0], 1))])
    
def from_hom(x):
    return x[:,:-1]

# to mm not um
def get_qn_transform(scale=25):
    s = scale/1000
    qn_to_ccf = np.array([[0,0,s,0], [-s,0,0,0], [0,-s,0,0], [13.175,7.975,0,1]])
    ccf_to_qn = np.linalg.inv(qn_to_ccf)
    return ccf_to_qn

def get_ccf_transform(df, coords_from, coords_to):
    samples = np.random.randint(0, len(df), 4)
    # samples = range(4)
    # intent is for x, y here to be normalized coords from add_rescaled_coords
    # coords_from = ['x', 'y', 'z_reconstructed']
    coords_to = ['x_ccf', 'y_ccf', 'z_ccf']
    to_ccf = np.matmul(np.linalg.inv(to_hom(df[list(coords_from)].iloc[samples].values)), to_hom(df[coords_to].iloc[samples].values))
    assert all(np.isclose(0, to_ccf[:3,-1]))
    return to_ccf


def get_anchor_entry(z, to_ccf, ccf_to_qn):
    # xyo = to_hom(np.array([[1, 0, z], [0, 1, z], [0, 0, z]]))
    xyo = to_hom(np.array([[1, 1, z], [0, 0, z], [0, 1, z]]))
    xyo_qn = from_hom(np.matmul(np.matmul(xyo, to_ccf), ccf_to_qn))
    ouv_qn = np.matmul(np.array([[0, 0, 1], [1, 0, -1], [0, 1, -1]]), xyo_qn)
    anchoring = list(ouv_qn.flatten())
    return anchoring

def export_to_quicknii(df, base_filename, img_label, img_coords,
                       coords_from=None, cmap=CMAP, slice_label=None, scale=25,
                       path='.', save_json=True, save_images=True, format='jpg'):
    coords_scaled, coords_int = add_rescaled_coords(
        df, img_coords[:2], xy_res=0.0025)
    nx, ny = df[coords_int].max() + 1
    
    if coords_from is not None:
        coords_from[:2], _ = add_rescaled_coords(df, coords_from[:2])
    else:
        coords_from = coords_scaled + [img_coords[2]]
    coords_to = ['x_ccf', 'y_ccf', 'z_ccf']
    ccf_to_qn = get_qn_transform(scale)
    to_ccf = get_ccf_transform(df, coords_from, coords_to)
    
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
        
        anchoring = get_anchor_entry(zval, to_ccf, ccf_to_qn)
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