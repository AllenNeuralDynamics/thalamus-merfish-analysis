# %%
import sys

sys.path.append("/code/")
from collections import defaultdict
import pandas as pd
from thalamus_merfish_analysis import abc_load as abc
from thalamus_merfish_analysis.ccf_erode import label_cells_by_eroded_ccf


obs = abc.get_combined_metadata(realigned=True)
# obs = abc.filter_by_ccf_labels(obs, ["TH","ZI"], realigned=realigned)
ccf_level = "structure"
ccf_label = f"parcellation_{ccf_level}"
ccf_label_aligned = f"parcellation_{ccf_level}_realigned"
# subset just the neurons
obs_neurons = abc.filter_by_class(obs).copy()

# %%
th_names = abc.get_thalamus_names(level=ccf_level)
metric_names = list(set(th_names).difference(["TH-unassigned"]))


# %%
# define celltype lists based on strict spatial subset
# obs_th_neurons = obs_neurons[obs_neurons[ccf_label+'_realigned'].isin(th_names) |
#                              obs_neurons[ccf_label].isin(th_names)]
# th_celltypes = dict()
# th_celltypes['subclass'] = obs_th_neurons['subclass'].value_counts().loc[lambda x: x>100].index
# print(f"{len(th_celltypes['subclass'])=}")

# th_celltypes['supertype'] = obs_th_neurons['supertype'].value_counts().loc[lambda x: x>20].index
# print(f"{len(th_celltypes['supertype'])=}")

# th_celltypes['cluster'] = obs_th_neurons['cluster'].value_counts().loc[lambda x: x>10].index
# print(f"{len(th_celltypes['cluster'])=}")

# %%
clusters_old = (
    obs_neurons.loc[obs_neurons[ccf_label].isin(th_names), "cluster"]
    .value_counts()
    .loc[lambda x: x > 10]
    .index
)
clusters_new = (
    obs_neurons.loc[obs_neurons[ccf_label + "_realigned"].isin(th_names), "cluster"]
    .value_counts()
    .loc[lambda x: x > 10]
    .index
)
len(clusters_new), len(clusters_old)

# %%
f"union: {len(clusters_new.union(clusters_old))}, intersection: {len(clusters_new.intersection(clusters_old))}"


clusters = clusters_new.intersection(clusters_old)
obs_th_neurons = obs[obs["cluster"].isin(clusters)]
# %%
df_cells = obs_neurons
ccf_img_realigned = abc.get_ccf_labels_image(realigned=True)
df_cells, ccf_label_eroded_aligned = label_cells_by_eroded_ccf(
    df_cells, ccf_img_realigned, ccf_level=ccf_level, realigned=True
)
# %%
ccf_img = abc.get_ccf_labels_image(realigned=False)
df_cells, ccf_label_eroded = label_cells_by_eroded_ccf(
    df_cells, ccf_img, ccf_level=ccf_level, realigned=False
)


# %%
def get_nucleus_celltype_metrics(
    obs, ccf_label, celltype_label, ccf_list=None, celltype_list=None
):
    if celltype_list is None:
        celltype_list = obs[celltype_label].unique()
    # else: obs = obs[obs[celltype_label].isin(celltype_list)]
    if ccf_list is None:
        ccf_list = obs[ccf_label].unique()
    # else: obs = obs[obs[ccf_label].isin(ccf_list)]
    # could subset like this, but need to track the negatives...
    beta = 0.5
    records = []
    for celltype_name in celltype_list:
        celltype = obs[celltype_label] == celltype_name
        for ccf_name in ccf_list:
            nucleus = obs[ccf_label] == ccf_name
            tp = (nucleus & celltype).sum()
            fp = (~nucleus & celltype).sum()
            fn = (nucleus & ~celltype).sum()
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            jaccard = tp / (tp + fp + fn)
            f1 = 2 * recall * precision / (recall + precision)
            fbeta = (1 + beta**2) * recall * precision / (recall + beta**2 * precision)
            # if precision>0.5 or recall>0.5 or f1>0.4:
            record = {
                "nucleus": ccf_name,
                "celltype": celltype_name,
                "nucleus_precision": precision,
                "nucleus_recall": recall,
                "nucleus_f1": f1,
                "nucleus_fbeta": fbeta,
                "jaccard": jaccard,
            }
            records.append(record)
    result = pd.DataFrame.from_records(records).set_index(["nucleus", "celltype"])
    return result


# %%
# : run 4 times (original/realigned and eroded/non) and take max for each nucleus/type pair.

# from multiprocessing import Pool
# import functools
# with Pool(processes=4) as p:
#     results = p.map(
#         functools.partial(
#             get_nucleus_celltype_metrics, df_cells, 'cluster',
#                                            ccf_list=th_names,
#                                            celltype_list=clusters[:10]),
#         [ccf_label, ccf_label_aligned, ccf_label_eroded, ccf_label_eroded_aligned]
#         )
results = [
    get_nucleus_celltype_metrics(
        df_cells, label, "cluster", ccf_list=th_names, celltype_list=clusters[:1]
    )
    for label in [
        ccf_label,
        ccf_label_aligned,
        ccf_label_eroded,
        ccf_label_eroded_aligned,
    ]
]
# %%
# stack dataframes across new column level and aggregate
df = pd.concat(results, keys=["original", "realigned", "eroded", "eroded_realigned"])
df_max = df.groupby(level=[1,2]).mean().reset_index()
#
# %%

# df_max[df_max["nucleus_fbeta"] > 0.05].to_csv("/results/nuclei_cluster_metrics.csv")
# nuclei_to_cluster = df_match.groupby("nucleus")["celltype"].apply(
#     lambda x: x.sort_values(ascending=False).str[:4].to_list()
# )
# nuclei_to_cluster.to_csv("/results/annotations_from_fbeta_n2c_all.csv")

df_taxonomy = abc._cluster_annotations.loc[lambda df: df["cluster_annotation_term_set_name"] == "cluster"].copy()
df_taxonomy["short_name"] = df_taxonomy["cluster_annotation_term_name"].str[:4]
def get_alias(clusters):
    return df_taxonomy.set_index("cluster_annotation_term_name").loc[clusters, "cluster_alias"].to_list()

top_by_cluster = df_max.groupby("celltype")["nucleus_f1"].idxmax().to_list()
top_by_region = df_max.groupby("nucleus")["nucleus_f1"].idxmax().to_list()

# looser threshold for top match, stricter for additional nuclei
(df_max[lambda df: (df["nucleus_f1"] > 0.1) |
((df["nucleus_f1"] > 0.05) & df.index.isin(top_by_cluster+top_by_region))]
.sort_values(ascending=False, by="nucleus_f1")
# group by celltype and concatenate nucleus names
.groupby("celltype")["nucleus"]
.apply(lambda x: ' '.join(x))
.rename("nuclei")
.reset_index()
# get cluster aliases
.assign(cluster_alias=lambda df: get_alias(df["celltype"]))
.to_csv("/results/annotations_c2n_auto.csv"))
# %%
# Manual annotations
df_manual = pd.read_csv("/root/capsule/code/thalamus_merfish_analysis/resources/prong1_cluster_annotations_by_nucleus.csv")
cluster = "cluster_ids_CNN20230720"
nucleus = "nuclei"

def get_name(clusters):
    return df_taxonomy.set_index("short_name").loc[clusters, "cluster_annotation_term_name"].to_list()

(df_manual
# .loc[lambda df: df["checked"]==1]
.set_index("nuclei")[cluster].astype(str)
.apply(lambda x: x.split(", "))
.explode()
.reset_index()
.assign(cluster=lambda df: get_name(df[cluster]))
.groupby("cluster")[nucleus]
.apply(lambda x: ' '.join(x))
.reset_index()
# get cluster aliases
.assign(cluster_alias=lambda df: get_alias(df["cluster"]))
.to_csv("/results/annotations_c2n_manual.csv"))
# .to_csv("/results/annotations_c2n_manual_checked.csv"))

# %%
