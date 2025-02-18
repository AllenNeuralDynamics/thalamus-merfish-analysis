import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from colorcet import glasbey_light
import plotly.graph_objects as go

from . import abc_load as abc


plt.rcParams.update({'font.size': 7})


def barplot_dual_y_count_frac(metrics_df, taxonomy_level, gt5_only=True):
    ''' Plot a barplot with both count and fraction of cells in each region.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame made with diversity_metrics.calculate_diversity_metrics()
    taxonomy_level : str, {'cluster', 'supertype', 'subclass'}
        ABC Atlas taxonomy level to plot
    gt5_only : bool
        If True, use the _gt5 columns for count and fraction
    '''
    # set metrics_df col based on thresholding to >5 cells or not
    count_col = f'count_gt5_{taxonomy_level}' if gt5_only else f'count_{taxonomy_level}'
    frac_col = f'frac_gt5_{taxonomy_level}' if gt5_only else f'frac_{taxonomy_level}'
    
    # sort regions so they're displayed low to high count, L to R
    metrics_df = metrics_df.sort_values(by=count_col, ascending=True)
    
    fig, ax1 = plt.subplots(figsize=(8,4))
    # Plot the absolute counts on the left y-axis
    ax1.scatter(metrics_df.index, metrics_df[count_col], 
                color='#5DA7E5', alpha=0)
    ax1.set_ylabel(f'unique {taxonomy_level} count', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.set_xticks(metrics_df.index)
    ax1.set_xticklabels(metrics_df.index, rotation=90)
    ax1.set_xlabel('CCF subregions')
    ax1.set_ylim(0, metrics_df[count_col].max()*1.05)
    plt.grid(visible=True, axis='y')

    # Plot the fraction values on the right y-axis
    ax2 = ax1.twinx()
    ax2.bar(metrics_df.index, metrics_df[frac_col], 
            color='#5DA7E5', label=taxonomy_level)
    # ntot = obs_neurons_ccf[taxonomy_level].value_counts().loc[lambda x: x>5].shape[0]
    ax2.set_ylabel(f'fraction of total {taxonomy_level} count', color='k', rotation=270, labelpad=15)
    ax2.set_ylim(0, metrics_df[frac_col].max()*1.05)
    ax2.tick_params(axis='y', labelcolor='k')
    

    plt.title(f'{taxonomy_level} count per CCF structure')
    return fig

def plot_metric_multiple_levels(metrics_df, 
                                metric, 
                                taxonomy_levels=['cluster','supertype','subclass'],
                                ylabel=None):

    if ylabel is None:
        ylabel = metric
    
    fig, ax1 = plt.subplots(figsize=(8,4))
    
    if taxonomy_levels==None:
        # enable plotting of a single metric
        metrics_df = metrics_df.sort_values(by=metric, ascending=True)
        ax1.scatter(metrics_df.index, metrics_df[metric], zorder=2)
    else:
        # sort by the metric of the first item in taxonomy_levels list
        metrics_df = metrics_df.sort_values(by="_".join([metric, 
                                                                taxonomy_levels[0]]), 
                                                   ascending=True)
        for level in taxonomy_levels[::-1]:
            ax1.scatter(metrics_df.index, 
                        metrics_df["_".join([metric, level])], 
                        label=level, zorder=2) 
        ax1.legend()

    ax1.set_xticks(metrics_df.index)
    ax1.set_xticklabels(metrics_df.index, rotation=90)
    ax1.set_xlabel('CCF structures')
    ax1.set_ylabel(ylabel)
    plt.grid(visible=True, axis='both', zorder=0, color='whitesmoke')
    
    return fig


def barplot_stacked_proportions(obs, taxonomy_level, ccf_metrics,
                                ccf_regions=None,
                                legend=True, palette=None,
                                min_cell_frac=0.01,
                                min_cell_count=None,
                                ordered_regions=None,
                                orientation='vertical'):
    """ Generate a stacked barplot showing the proportion of each taxonomy level
    category in each CCF region.
    
    Parameters
    ----------
    obs : pd.DataFrame
        dataframe of cells with CCF annotations & mapped taxonomy levels
    taxonomy_level : str, {'cluster', 'supertype', 'subclass'}
        ABC Atlas taxonomy level to plot
    ccf_metrics : pd.DataFrame
        DataFrame made with diversity_metrics.calculate_diversity_metrics()
    ccf_regions : list of str, default=None
        list of CCF regions to restrict the plot to
    legend : bool, default=True
        whether to display the legend
    palette : dict, default=None
        dictionary mapping taxonomy level categories to colors
    min_cell_frac : float, in range [0,1], default=0.01
        sets minimum fraction of cells required for a category to be included;
        categories <= this threshold are aggregated into an 'other' category
    min_cell_count : int, default=None, suggested=5
        sets minimum number of cells required for a category to be included; 
        categories <= this threshold are aggregated into an 'other' category.
        If set, it supercedes min_cell_frac
    ordered_regions : list of str, default=None
        list of CCF regions to plot, in the order they should be displayed
    orientation : str, {'vertical', 'horizontal'}, default='vertical'
        orientation of the barplot; 'vertical' displays regions on x-axis using
        'bar', 'horizontal' displays regions on the y-axis using 'barh'
        
    """
    # Set the palette
    if palette is None:
        try:
            palette = abc.get_taxonomy_palette(taxonomy_level)
        except ValueError:
            # if the level is not in the atlas, use glasbey_light
            palette = glasbey_light
    # add 'other' to the palette
    palette['other'] = 'lightgrey'

    # Calculate the proportion of each taxonomy level category per region
    proportions_df = calculate_level_proportions(obs, 
                                                 taxonomy_level, 
                                                 min_count=min_cell_count, 
                                                 min_frac=min_cell_frac)
    if ccf_regions is None:
        ccf_regions = proportions_df.index
    else:
        ccf_regions = list(set(ccf_regions) & set(proportions_df.index))                                                
    # filter to only the regions of interest
    proportions_df = proportions_df.loc[ccf_regions]
    # clean up category columns that now are all zeros post-filtering
    proportions_df = proportions_df.loc[:,(proportions_df!=0).any(axis=0)]

    # reorder the proportions df
    if ordered_regions is None:
        # Sort ccf_regions by # of non-zero categories & Inverse Simpson's Index
        nonzero_counts = (proportions_df.drop(columns=['other'])!=0).sum(axis=1)
        nonzero_counts.name = 'nonzero_counts'
        inverse_simpsons = ccf_metrics.loc[
                                ccf_regions,
                                f'inverse_simpsons_{taxonomy_level}']
        # combine two metrics into a df that we can sort by
        metrics_to_sort_by = pd.concat([nonzero_counts, inverse_simpsons], axis=1)
        sorted_regions = metrics_to_sort_by.sort_values(
                                by=['nonzero_counts', 
                                    f'inverse_simpsons_{taxonomy_level}'], 
                                ascending=[True, True]
                                ).index
    else:
        sorted_regions = ordered_regions
    proportions_df = proportions_df.loc[sorted_regions]

    # Plot stacked barplot, using barh or bar 
    if orientation=='horizontal':
        fig, ax = plt.subplots(1,1, figsize=(5,12))
        proportions_df.plot(kind='barh', stacked=True, ax=ax, legend=legend, 
                            color=palette)
        # axis formatting for horizontal barplot
        ax.set_yticklabels(proportions_df.index)
        ax.set_ylabel('CCF structure')
        ax.set_ylim(ax.get_ylim()[0]-0.3, ax.get_ylim()[1]+0.2) # make room for ax.text() annotations
        ax.invert_yaxis()  # put lowest-diversity region at the top
        ax.set_xlim(0,1.11)  # make room for ax.text() annotations
        ax.set_xlabel('proportion of cells in unique '+taxonomy_level)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0', '', '0.5', '', '1'])
        ax.tick_params(which='both', direction='in', top=True, labeltop=False)
    else:
        fig, ax = plt.subplots(1,1, figsize=(12,5))
        proportions_df.plot(kind='bar', stacked=True, ax=ax, legend=legend, 
                            color=palette)
        # axis formatting for vertical barplot
        ax.set_xticklabels(proportions_df.index, rotation=90)
        ax.set_xlabel('CCF structure')
        ax.set_xlim(ax.get_xlim()[0]-0.1, ax.get_xlim()[1]+0.1) # make room for ticks & text annotations
        ax.set_ylim(0,1.09)  # make room for ax.text() annotations
        ax.set_ylabel('proportion of cells in unique '+taxonomy_level)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ['0', '', '0.5', '', '1'])
        ax.tick_params(which='both', direction='in', right=True, labelright=False)
    
    # format legend
    if legend:
        # Reorder the legend labels alphabetically
        handles, labels = ax.get_legend_handles_labels()
        order = sorted(range(len(labels)), key=lambda k: labels[k])
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                  loc='upper left', bbox_to_anchor=(0, -0.3), ncol=2)

    # display the number of non-zero, non-other categories above each region's bar
    for i, region in enumerate(proportions_df.index):
        n_all = ccf_metrics.loc[region, f'count_{taxonomy_level}']
        n_nonzero = (proportions_df.loc[region, proportions_df.columns!='other']>0).sum()
        if orientation=='horizontal':
            ax.text(1.02, i, f'{n_nonzero}', verticalalignment='center',
                    horizontalalignment='left')
        else:
            ax.text(i, 1.02, f'{n_all}/n({n_nonzero})', horizontalalignment='center')

    return fig


def calculate_level_proportions(obs, 
                                taxonomy_level,
                                ccf_label='parcellation_structure_eroded',
                                min_frac=0.01,
                                min_count=None):
    ''' Calculate the proportion of each level in each CCF region for a stacked 
    barplot.

    Parameters
    ----------
    obs : pd.DataFrame
        dataframe of cells with CCF annotations & mapped taxonomy levels
    taxonomy_level : str, {'cluster', 'supertype', 'subclass'}
        ABC Atlas taxonomy level 
    ccf_label : str, default='parcellation_structure_eroded'
        column name in obs_ccf where the CCF annotations can be found
    min_frac : float, in range [0,1], default=0.01
        sets minimum fraction of cells required for a category to be included;
        categories <= this threshold are aggregated into an 'other' category
    min_count : int, default=None, suggested=5
        sets minimum number of cells required for a category to be included; 
        categories <= this threshold are aggregated into an 'other' category
        If set, it supercedes min_frac

    Returns
    -------
    proportions_df : pd.DataFrame
        df with the proportion of each taxonomy_level in each CCF region, where
        index=obs_ccf[ccf_label].unique(), columns=obs_ccf[taxonomy_level].unique()
    '''
    # count the number of cells in each (structure, taxonomy_level) pair & save as a df  
    # where index=ccf_label & columns=taxonomy_level
    counts_df = obs.groupby([ccf_label, taxonomy_level], observed=True
                            ).size().unstack(fill_value=0)
    
    # if min_count is set, it supercedes min_frac
    if min_count is not None:
        to_other = (counts_df <= min_count)
    else:
        # Calculate the fraction of each category, per region
        total_counts = counts_df.sum(axis=1)
        fraction_df = counts_df.div(total_counts, axis=0)
        # Get categories to move to 'other'
        to_other = (fraction_df <= min_frac)
    
    # Aggregate & move counts below threshold to 'other' column
    other_col_df = counts_df[to_other].sum(axis=1)
    counts_df = counts_df[~to_other]  # replace counts below threshold with NaN
    counts_df = counts_df.join(other_col_df.rename('other')).fillna(0)
    # clean up columns that are not empty
    counts_df = counts_df.loc[:,(counts_df!=0).any(axis=0)]

    # calculate proportions from counts
    proportions_df = counts_df.div(counts_df.sum(axis=1), axis=0)

    return proportions_df


def sankey_diagram(
    obs, 
    source_col, 
    target_col,
    source_cats_to_plot=None, 
    target_cats_to_plot=None,
    source_color_dict=None,
    target_color_dict=None,
):
    '''Plot a Sankey diagram for two cell metadata columns.
    
    Parameters
    ----------
    obs : pd.DataFrame
        DataFrame or AnnData.obs containing the source and target columns
    source_col : str
        Column name for source nodes
    target_col : str
        Column name for target nodes
    source_cats_to_plot : list of str, optional
        List of source categories to plot. If None, all used categories are plotted.
    target_cats_to_plot : list of str, optional
        List of target categories to plot. If None, all used categories are plotted.
    source_color_dict : dict, optional
        Dictionary mapping source categories to colors. If None, glasbey_light colormap is used.
    target_color_dict : dict, optional
        Dictionary mapping target categories to colors. If None, all target nodes are set to a neutral gray.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
    
    '''
    # ----- DATA CLEANING ------------------------------------------------------
    # Extract the two columns & set to categorical
    obs = obs[[source_col, target_col]].copy().dropna()
    obs[source_col] = obs[source_col].astype('category')
    obs[target_col] = obs[target_col].astype('category')
    
    # Filter the dataframe to include only the specified regions
    if source_cats_to_plot is not None:
        obs = obs[obs[source_col].isin(source_cats_to_plot)]
    if target_cats_to_plot is not None:
        obs = obs[obs[target_col].isin(target_cats_to_plot)]
        
    # Remove unused categories from the columns
    obs[source_col] = obs[source_col].cat.remove_unused_categories()
    obs[target_col] = obs[target_col].cat.remove_unused_categories()


    # ----- PREPARE DATA FOR SANKEY DIAGRAM ------------------------------------
    # Count occurrences of pair-wise transitions (aka links) from source node to target node
    transition_counts = obs.groupby([source_col, target_col]).size().reset_index(name='count')

    # Generate a unique identifier index for each label (node) 
    # - source and target nodes indexed together
    unique_node_labels = pd.concat([obs[source_col], obs[target_col]]).unique()
    label_to_index_map = {label: i for i, label in enumerate(unique_node_labels)}

    # Create source, target, and value lists for Sankey link dict
    # Sankey expects:
    # - values = count for each pairwise transition (link)
    # - sources = source index for each pairwise transition (link)
    # - targets = target index for each pairwise transition (link)
    # - len(sources)==len(targets)==len(values)
    values = transition_counts['count'].tolist()
    sources = transition_counts[source_col].map(label_to_index_map).tolist()
    targets = transition_counts[target_col].map(label_to_index_map).tolist()
    
    # ----- CREATE COLOR DICTIONARIES ------------------------------------------
    # Generate source_colors list 
    if source_color_dict is None:
        # use default Glasbey color palette as default
        source_color_dict = {label: glasbey_light[i] for i, label in enumerate(obs[source_col].cat.categories)}
    else:
        # check that we have hex colors in the provided dict
        if not all(isinstance(v, str) and v.startswith('#') for v in source_color_dict.values()):
            raise ValueError("source_color_dict values must be hex color strings.")
    
    # Create source color list from the dict
    source_colors_hex = [source_color_dict[x] for x in unique_node_labels if x in source_color_dict]  
      
    # Convert the dict of hex colors to rgba str colors to later generate transition/link colors
    source_color_dict_rgba_tuple = {k: mcolors.to_rgba(v) for k, v in source_color_dict.items()}
    source_color_dict_rbga_str = {label: f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.8)"
                                    for label, (r, g, b, a) in source_color_dict_rgba_tuple.items()}
    
    # Generate target_colors from the provided dict or default to a neutral gray
    if target_color_dict is not None:
        if all(isinstance(v, str) and v.startswith('#') for v in target_color_dict.values()):
            target_colors_hex = [target_color_dict[x] for x in unique_node_labels if x in target_color_dict]
        else:
            raise ValueError("target_color_dict values must be hex color strings.")
    else:
        target_colors_hex = ['#AAAAAA' for _ in obs[target_col].cat.categories]
        
    # Generate link colors to match source node colors
    link_colors_rgba = [
        source_color_dict_rbga_str[src_label].replace("0.8)", "0.4)") 
        for src_label in transition_counts[source_col]
    ]
    
    # ----- PLOT SANKEY DIAGRAM ------------------------------------------------
    # Build Sankey diagram using plotly.graph_objects
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=list(unique_node_labels),
            color=source_colors_hex + target_colors_hex
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            # color="rgba(150, 150, 150, 0.3)",  # Neutral link color
            color=link_colors_rgba,
        )
    ))

    # Adjust figure size based on number of unique labels
    width = max(300, len(unique_node_labels) * 10)
    height = max(200, len(unique_node_labels) * 20)
    fig.update_layout(
        title_text=f"Sankey Diagram: {source_col} → {target_col}",
        font_size=10,
        width=width,
        height=height
    )
    fig.show()
    
    return fig