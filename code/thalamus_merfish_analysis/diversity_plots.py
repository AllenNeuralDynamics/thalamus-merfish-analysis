import matplotlib.pyplot as plt


TH_DIVERSITY_REGIONS = ['AD', 'AV', 'AM', 'CL', 'CM', 'IAD', 'IMD',
                        'LD', 'LGd', 'LH', 'LP', 'MD', 'MH', 
                        'PCN', 'PF', 'PO', 'PVT', 'RE', 'RT', 'SPA',
                        'VAL', 'VM', 'VPL', 'VPM', 'VPMpc'
                        ]


def barplot_dual_y_count_frac(th_metrics, level, gt5_only=True):
    ''' Plot a barplot with both count and fraction of cells in each thalamic region.

    Parameters
    ----------
    th_metrics : pd.DataFrame
        DataFrame made with diversity_metrics.calculate_diversity_metrics()
    level : str, {'cluster', 'supertype', 'subclass'}
        ABC Atlas taxonomy level to plot
    gt5_only : bool
        If True, use the _gt5 columns for count and fraction
    '''
    # set th_metrics col based on thresholding to >5 cells or not
    count_col = f'count_gt5_{level}' if gt5_only else f'count_{level}'
    frac_col = f'frac_gt5_{level}' if gt5_only else f'frac_{level}'
    
    # sort regions so they're displayed low to high count, L to R
    th_metrics_sorted = th_metrics.sort_values(by=count_col, ascending=True)
    
    fig, ax1 = plt.subplots(figsize=(8,4))
    # Plot the absolute counts on the left y-axis
    ax1.scatter(th_metrics_sorted.index, th_metrics_sorted[count_col], 
                color='#5DA7E5', alpha=0)
    ax1.set_ylabel(f'unique {level} count', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.set_xticks(th_metrics_sorted.index)
    ax1.set_xticklabels(th_metrics_sorted.index, rotation=90)
    ax1.set_xlabel('CCF thalamic subregions')
    ax1.set_ylim(0, th_metrics_sorted[count_col].max()*1.05)
    plt.grid(visible=True, axis='y')

    # Plot the fraction values on the right y-axis
    ax2 = ax1.twinx()
    ax2.bar(th_metrics_sorted.index, th_metrics_sorted[frac_col], 
            color='#5DA7E5', label=level)
    # ntot = obs_neurons_ccf[level].value_counts().loc[lambda x: x>5].shape[0]
    ax2.set_ylabel(f'fraction of total {level} count', color='k', rotation=270, labelpad=15)
    ax2.set_ylim(0, th_metrics_sorted[frac_col].max()*1.05)
    ax2.tick_params(axis='y', labelcolor='k')
    

    plt.title(f'{level} count per thalamic CCF structure')
    return fig

def plot_metric_multiple_levels(th_metrics, 
                                metric, 
                                levels=['cluster','supertype','subclass'],
                                ylabel=None):

    if ylabel is None:
        ylabel = metric
    
    fig, ax1 = plt.subplots(figsize=(8,4))
    
    if levels==None:
        # enable plotting of a single metric
        th_metrics_sorted = th_metrics.sort_values(by=metric, ascending=True)
        ax1.scatter(th_metrics_sorted.index, th_metrics_sorted[metric], zorder=2)
    else:
        # sort by the metric of the first item in levels list
        th_metrics_sorted = th_metrics.sort_values(by="_".join([metric, levels[0]]), 
                                                   ascending=True)
        for level in levels[::-1]:
            ax1.scatter(th_metrics_sorted.index, 
                        th_metrics_sorted["_".join([metric, level])], 
                        label=level, zorder=2) 
        ax1.legend()

    ax1.set_xticks(th_metrics_sorted.index)
    ax1.set_xticklabels(th_metrics_sorted.index, rotation=90)
    ax1.set_xlabel('CCF structures')
    ax1.set_ylabel(ylabel)
    plt.grid(visible=True, axis='both', zorder=0, color='whitesmoke')
    
    return fig

# # display as a stacked bar graph
# from colorcet import glasbey_light

# def plot_stacked_barplot(df, taxonomy_level, legend=True, palette=None):
#     fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
#     if palette is None:
#         palette = abc.get_taxonomy_palette(taxonomy_level)
#         # palette = sns.color_palette(glasbey_light, n_colors=len(df.columns))
#     palette['other'] = 'grey'
#     df.plot(kind='bar', stacked=True, ax=axes, legend=legend, color=palette)
#     if legend:
#         axes.legend(loc='upper left', bbox_to_anchor=(0.05, -0.3), ncol=4)
#     axes.set_xticklabels(df.index, rotation=90)
#     axes.set_xlabel('CCF structure')
#     # axes.set_yticks([])
#     axes.set_ylabel('proportion of cells in unique '+taxonomy_level)

#     fig.subplots_adjust(hspace=0.1)
    
#     # add text
#     for i, subregion in enumerate(df.index):
#         n_nonzero = (df.loc[subregion, df.columns!='other']>0).sum()
#         axes.text(i, 1.01, n_nonzero, horizontalalignment='center')