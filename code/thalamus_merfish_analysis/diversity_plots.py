def barplot_counts_fractions(th_metrics, level, thresholded=True):
    fig, ax1 = plt.subplots(figsize=(8,4))
    if thresholded:
        count = f'count_gt5_{level}'
        ntot = obs_neurons_ccf[level].value_counts().loc[lambda x: x>5].shape[0]
    else:
        count = f'n_{level}'
        ntot = obs_neurons_ccf[level].value_counts().loc[lambda x: x>0].shape[0]
    th_metrics_sorted = th_metrics.sort_values(by=count, ascending=True)
    # Plot the absolute values on the left y-axis
    ax1.scatter(th_metrics_sorted.index, th_metrics_sorted[count], color='#5DA7E5', alpha=0)
    ax1.set_ylabel(f'unique {level} count', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.set_xticks(th_metrics_sorted.index)
    ax1.set_xticklabels(th_metrics_sorted.index, rotation=90)
    ax1.set_ylim(0, th_metrics_sorted[count].max()*1.05)
    plt.grid(visible=True, axis='y')

    # Create a secondary y-axis on the right
    ax2 = ax1.twinx()

    # Plot the fraction values on the right y-axis
    frac = th_metrics_sorted[count].astype(float)/ntot
    ax2.bar(th_metrics_sorted.index, frac, color='#5DA7E5', label=level)
    ax2.set_ylabel(f'fraction of total count (n={ntot})', color='k', rotation=270, labelpad=15)
    ax2.set_ylim(0, frac.max()*1.05)
    ax2.tick_params(axis='y', labelcolor='k')
    # ax2.legend()

    # plt.xticks(rotation=90)
    plt.xlabel('CCF thalamic subregions')
    plt.title(f'{level} count per thalamic CCF structure')
    return fig

fig = barplot_counts_fractions(th_metrics.loc[regions_subset], 'cluster', thresholded=True)
fig.savefig("/results/nuclei_cluster_counts_barplot.pdf", transparent=True)

def plot_metrics_multiple_levels(th_metrics, metric, levels=['cluster','subclass']):
    fig, ax1 = plt.subplots(figsize=(8,4))

    th_metrics_sorted = th_metrics.sort_values(by="_".join([metric, levels[0]]), ascending=True)
    for level in levels[::-1]:
        ax1.scatter(th_metrics_sorted.index, th_metrics_sorted["_".join([metric, level])], 
                     label=level, zorder=2) 
    ax1.set_xticks(th_metrics_sorted.index)
    ax1.set_xticklabels(th_metrics_sorted.index, rotation=90)
    ax1.set_ylabel(metric)
    ax1.legend()
    plt.grid(visible=True, axis='both', zorder=0, color='whitesmoke')


# display as a stacked bar graph
from colorcet import glasbey_light

def plot_stacked_barplot(df, taxonomy_level, legend=True, palette=None):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
    if palette is None:
        palette = abc.get_taxonomy_palette(taxonomy_level)
        # palette = sns.color_palette(glasbey_light, n_colors=len(df.columns))
    palette['other'] = 'grey'
    df.plot(kind='bar', stacked=True, ax=axes, legend=legend, color=palette)
    if legend:
        axes.legend(loc='upper left', bbox_to_anchor=(0.05, -0.3), ncol=4)
    axes.set_xticklabels(df.index, rotation=90)
    axes.set_xlabel('CCF structure')
    # axes.set_yticks([])
    axes.set_ylabel('proportion of cells in unique '+taxonomy_level)

    fig.subplots_adjust(hspace=0.1)
    
    # add text
    for i, subregion in enumerate(df.index):
        n_nonzero = (df.loc[subregion, df.columns!='other']>0).sum()
        axes.text(i, 1.01, n_nonzero, horizontalalignment='center')