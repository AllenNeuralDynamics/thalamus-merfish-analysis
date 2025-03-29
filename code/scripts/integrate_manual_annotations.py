import pandas as pd
from abc_merfish_analysis import abc_load as abc

df_taxonomy = abc._cluster_annotations.copy()
df_taxonomy["short_name"] = df_taxonomy["cluster"].str[:4]
def get_alias(clusters):
    return df_taxonomy.set_index("cluster").loc[clusters, "cluster_alias"].to_list()
# Manual annotations

df_manual = pd.read_csv("/root/capsule/code/abc_merfish_analysis/resources/prong1_cluster_annotations_by_nucleus.csv")
cluster = "cluster_ids_CNN20230720"
nucleus = "nuclei"

def get_name(clusters):
    return df_taxonomy.set_index("short_name").loc[clusters, "cluster"].to_list()

def flatten_annotations(df):
    return (df
        .set_index(nucleus)[cluster].astype(str)
        .apply(lambda x: x.split(", "))
        .explode()
        .reset_index()
        .assign(cluster=lambda df: get_name(df[cluster]))
        .groupby("cluster")[nucleus]
        .apply(lambda x: ' '.join(x))
        .reset_index()
        # get cluster aliases
        .assign(cluster_alias=lambda df: get_alias(df["cluster"])))
anno_manual = pd.concat(
    [
        flatten_annotations(df_manual.loc[lambda df: df["checked"]!=1]).assign(checked=0), 
        flatten_annotations(df_manual.loc[lambda df: df["checked"]==1]).assign(checked=1)
    ],
).set_index("cluster")
anno_manual.to_csv("/scratch/annotations_c2n_manual.csv")

anno_auto = pd.read_csv("/code/abc_merfish_analysis/resources/annotations_c2n_auto.csv", index_col=0).set_index("cluster")

# remove complete nuclei from auto annotations
complete_nuclei = df_manual.loc[lambda df: (df["checked"]==1) & ~df["nuclei"].str.contains(" "), "nuclei"]
anno_auto = anno_auto.loc[lambda df: ~df["nuclei"].isin(complete_nuclei)]

anno_combined = anno_auto.reindex(anno_auto.index.union(anno_manual.index))
anno_combined.update(anno_manual)
anno_combined.cluster_alias = anno_combined.cluster_alias.astype(int)
anno_combined.to_csv("/scratch/annotations_c2n_combined.csv")