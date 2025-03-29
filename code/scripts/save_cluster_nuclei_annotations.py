impot pandas as pd
from abc_merfish_analysis import abc_load as abc
from abc_merfish_analysis import ccf_erode as cerd
from collections import defaultdict
from itertools import chain

# load just the obs
obs = abc.load_standard_thalamus(data_structure='obs')
# load CCF image volumes
realigned=False
ccf_images = abc.get_ccf_labels_image(resampled=True, realigned=realigned)
if realigned:
    ccf_label = 'parcellation_structure_realigned'
    coords = '_section'
else:
    ccf_label = 'parcellation_structure'
    coords = '_reconstructed'
    

x_col = 'x'+coords
y_col = 'y'+coords
section_col = z_col = 'z'+coords

## Define eroded CCF regions
ccf_metrics_level = "structure"
obs_erode, ccf_label_eroded = cerd.label_cells_by_eroded_ccf(obs, ccf_images, ccf_level=ccf_metrics_level, distance_px=5) # default is erosion by 5px (50um)
# So, we'll set all cells in section 6.6 to 'unassigned' CCF structure
obs_erode.loc[lambda df: df['z_section']==6.6, ccf_label_eroded] = 'unassigned'


min_count=0.1
nuclei_to_cluster = (obs.groupby('parcellation_structure_eroded')['cluster'].apply(lambda x:
                     x.str[:4].value_counts(normalize=True).loc[lambda x: x>min_count].index.to_list()))

cluster_to_nuclei = defaultdict(list)
for nucleus, clusters in nuclei_to_cluster.items():
    if 'unassigned' in nucleus:
        continue
    for cluster in clusters:
        cluster_to_nuclei[cluster].append(nucleus)
cluster_to_nuclei = pd.Series(cluster_to_nuclei)
annotations = (cluster_to_nuclei.rename('nuclei').apply(lambda x: " ".join(x)).reset_index()
    .groupby('nuclei')['index'].agg(lambda x: ", ".join(x))
    .rename('cluster_ids_CNN20230720'))
annotations.to_csv("resources/annotations_from_eroded_counts.csv")