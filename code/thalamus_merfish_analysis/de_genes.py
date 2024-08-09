from __future__ import annotations
from collections.abc import Sequence, Set
import numpy as np
from matplotlib import rcParams, gridspec
from matplotlib import pyplot as plt

from matplotlib.axes import Axes
from anndata import AnnData
import scanpy.tl


def combine_groups(obs, taxonomy_level, group_var, group):
    group_new = join_short_taxonomy_names(group)
    obs.loc[lambda df: df[taxonomy_level].isin(group), group_var] = group_new
    return obs, group_new


def join_short_taxonomy_names(names):
    return "/".join([name.split(" ")[0] for name in names])


def run_sc_deg_analysis(
    adata,
    taxonomy_level,
    group,
    reference="rest",
    n_genes_display=20,
    rankby_abs=True,
    method="wilcoxon",
    key="rank_genes_groups",
    highlight_genes=None,
):
    """
    Run differential expression analysis on a Scanpy AnnData object.
    Reference should be a list of groups, rather than a single group (or 'rest').
    """

    combined_group = len(group) > 1
    combined_ref = (reference != "rest") and len(reference) > 1
    if combined_group or combined_ref:
        group_var = f"combined_{taxonomy_level}"
        adata.obs[group_var] = adata.obs[taxonomy_level].astype(str)
        if combined_group:
            adata.obs, group = combine_groups(adata.obs, taxonomy_level, group_var, group)
        if combined_ref:
            adata.obs, reference = combine_groups(adata.obs, taxonomy_level, group_var, reference)
    else:
        group_var = taxonomy_level
        group = group[0]
        reference = reference[0]
    # rank genes
    scanpy.tl.rank_genes_groups(
        adata,
        group_var,
        groups=[group],
        reference=reference,
        n_genes=None,
        rankby_abs=True,
        method=method,
        key_added=key,
    )

    # plot top n_genes_display
    plot_ranked_genes(
        adata,
        n_genes=n_genes_display,
        key=key,
        highlight_genes=highlight_genes,
        group_type_label=taxonomy_level,
    )
    return adata


def plot_ranked_genes(
    adata: AnnData,
    groups: str | Sequence[str] | None = None,
    n_genes: int = 20,
    gene_symbols: str | None = None,
    key: str | None = "rank_genes_groups",
    fontsize: int = 8,
    ncols: int = 4,
    sharey: bool = True,
    highlight_genes: Set | None = None,
    group_type_label: str | None = None,
    ax: Axes | None = None,
    **kwds,
):
    """
    Plot ranking of genes.
    (From scanpy.pl.rank_genes_groups, with points added)

    Parameters
    ----------
    adata
        Annotated data matrix.
    groups
        The groups for which to show the gene ranking.
    gene_symbols
        Key for field in `.var` that stores gene symbols if you do not want to
        use `.var_names`.
    n_genes
        Number of genes to show.
    fontsize
        Fontsize for gene names.
    ncols
        Number of panels shown per row.
    sharey
        Controls if the y-axis of each panels should be shared. But passing
        `sharey=False`, each panel has its own y-axis range.

    """
    if "n_panels_per_row" in kwds:
        n_panels_per_row = kwds["n_panels_per_row"]
    else:
        n_panels_per_row = ncols
    if n_genes < 1:
        raise NotImplementedError(
            "Specifying a negative number for n_genes has not been implemented for "
            f"this plot. Received n_genes={n_genes}."
        )

    reference = str(adata.uns[key]["params"]["reference"])
    group_names = adata.uns[key]["names"].dtype.names if groups is None else groups
    # one panel for each group
    # set up the figure
    n_panels_x = min(n_panels_per_row, len(group_names))
    n_panels_y = np.ceil(len(group_names) / n_panels_x).astype(int)

    fig = plt.figure(
        figsize=(
            n_panels_x * rcParams["figure.figsize"][0],
            n_panels_y * rcParams["figure.figsize"][1],
        )
    )
    gs = gridspec.GridSpec(nrows=n_panels_y, ncols=n_panels_x, wspace=0.22, hspace=0.3)

    ax0 = None
    ymin = np.Inf
    ymax = -np.Inf
    for count, group_name in enumerate(group_names):
        gene_names = adata.uns[key]["names"][group_name][:n_genes]
        scores = adata.uns[key]["scores"][group_name][:n_genes]

        # Setting up axis, calculating y bounds
        if sharey:
            ymin = min(ymin, np.min(scores))
            ymax = max(ymax, np.max(scores))

            if ax0 is None:
                ax = fig.add_subplot(gs[count])
                ax0 = ax
            else:
                ax = fig.add_subplot(gs[count], sharey=ax0)
        else:
            ymin = np.min(scores)
            ymax = np.max(scores)
            buffer = 0.1 * (ymax - ymin)
            ymax += buffer

            ax = fig.add_subplot(gs[count])
            ax.set_ylim(ymin, ymax)

        ax.set_xlim(-0.9, n_genes - 0.1)

        # Mapping to gene_symbols
        if gene_symbols is not None:
            if adata.raw is not None and adata.uns[key]["params"]["use_raw"]:
                gene_names = adata.raw.var[gene_symbols][gene_names]
            else:
                gene_names = adata.var[gene_symbols][gene_names]

        s = 4
        ax.plot(scores, "o", color="black", markersize=s)
        if highlight_genes is not None:
            for ig, gene_name in enumerate(gene_names):
                if gene_name in highlight_genes:
                    ax.plot(ig, scores[ig], "o", color="red", markersize=s)
        # Making labels
        for ig, gene_name in enumerate(gene_names):
            ax.text(
                ig,
                scores[ig] + 1,
                gene_name,
                rotation=45,
                verticalalignment="bottom",
                horizontalalignment="left",
                rotation_mode="anchor",
                fontsize=fontsize,
            )

        title = f"{group_name} vs. {reference}"
        if group_type_label is not None:
            title += f" ({group_type_label})"
        ax.set_title(title)
        ax.set_xticks(range(0, n_genes, 5))
        if count >= n_panels_x * (n_panels_y - 1):
            ax.set_xlabel("ranking")

        # print the 'score' label only on the first panel per row.
        if count % n_panels_x == 0:
            ax.set_ylabel("score")

    if sharey is True:
        ymax += 0.3 * (ymax - ymin)
        ax.set_ylim(ymin, ymax)
