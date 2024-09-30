from importlib_resources import files
import sys

sys.path.append("/code/")
import pandas as pd
from thalamus_merfish_analysis import abc_load as abc
from thalamus_merfish_analysis import ccf_registration as ccf


df_full = abc.get_combined_metadata()
minmax = pd.read_csv(
    files("thalamus_merfish_analysis.resources") / "brain3_thalamus_coordinate_bounds.csv",
    index_col=0,
)
xx = minmax["x_section"].values
yy = minmax["y_section"].values
df = df_full.query(f"{yy[0]} <= y_section <= {yy[1]} and {xx[0]} <= x_section <= {xx[1]}").copy()
df = abc.filter_thalamus_sections(df)
df[["y_section", "y_reconstructed"]] *= -1

slice_label = 'slice_int'
img_label = 'subclass_int'

img_coords = ['x_section', 'y_section', 'z_section']
coords_from = ['x_reconstructed', 'y_reconstructed', 'z_reconstructed']

nn_classes = ["31 OPC-Oligo",
              "30 Astro-Epen",
              "33 Vascular",
              "34 Immune"]
th_subclasses = ['168 SPA-SPFm-SPFp-POL-PIL-PoT Sp9 Glut', '101 ZI Pax6 Gaba',
                  '109 LGv-ZI Otx2 Gaba', '093 RT-ZI Gnb3 Gaba',
                  '149 PVT-PT Ntrk1 Glut', '151 TH Prkcd Grin2c Glut',
                  '152 RE-Xi Nox4 Glut', '154 PF Fzd5 Glut',
                  '203 LGv-SPFp-SPFm Nkx2-2 Tcf7l2 Gaba',
                  '148 AV Col27a1 Glut', '146 LH Pou4f1 Sox1 Glut',
                  '147 AD Serpinb7 Glut', '153 MG-POL-SGN Nts Glut',
                  '110 BST-po Iigp1 Glut', '150 CM-IAD-CL-PCN Sema5b Glut',
                  '145 MH Tac2 Glut']

ref_subclasses = df_full.loc[lambda df: df['subclass'].isin(th_subclasses)].index
df, cmap = ccf.preprocess_for_qn_export(df, nn_classes, img_coords,
                                        subset_for_fine_labels=ref_subclasses,
                                        slice_label=slice_label,
                                        img_label=img_label)


out = ccf.export_to_quicknii(df, 'brain3_20220830', cmap=cmap,
                       img_coords=img_coords, coords_from=coords_from,
                       img_label=img_label, slice_label=slice_label, scale=25,
                   path='/results/qn_25_affine_thal_subclasses_20220630', 
                   save_json=True, save_images=True)