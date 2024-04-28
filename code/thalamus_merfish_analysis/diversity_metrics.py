import numpy as np
import pandas as pd


def calculate_diversity_metrics(obs_ccf, 
                                ccf_label='parcellation_structure_eroded'):
    ''' Calculate a set of diversity metrics for each region in the thalamus.

    Parameters
    ----------
    obs_ccf : pd.DataFrame
        dataframe of observations with CCF annotations
    ccf_label : str, default='parcellation_structure_eroded'
        column name in obs_ccf where the CCF annotations can be found

    Returns
    -------
    th_metrics_df : pd.DataFrame
        dataframe with diversity metrics for each region in the thalamus
    '''

    # calculate a pre-selected set of diversity metrics for each CCF region
    th_metrics_df = pd.concat([
        # number of unique categories (e.g. clusters) found in each region
        get_region_metric(obs_ccf, count_unique, "count", ccf_label=ccf_label),
        # fraction of all possible categories (e.g. clusters) uniquely found in each region
        get_region_metric(obs_ccf, count_unique, "frac", ccf_label=ccf_label, 
                          norm_fcn=count_unique),
        # number of unique categories found in each region, normalized by the 
        # total number of cells present in that region
        # controls for differences in cell density, region size
        get_region_metric(obs_ccf, count_unique_norm, "count_norm2cells", ccf_label=ccf_label),
        # number of unique categories with more than 5 cells
        get_region_metric(obs_ccf, count_gt5, "count_gt5", ccf_label=ccf_label),
        # fraction of all possible categories (e.g. clusters) with >5 cells found in each region
        get_region_metric(obs_ccf, count_gt5, "frac_gt5", ccf_label=ccf_label, 
                          norm_fcn=count_unique),
        # inverse simpsons diversity index
        get_region_metric(obs_ccf, inverse_simpsons_index, "inverse_simpsons", ccf_label=ccf_label),
        # shannon diversity index
        get_region_metric(obs_ccf, shannon_index, "shannon_index", ccf_label),
        # get_region_metric(obs_ccf, lambda x: 2**calc_shannon_index(x), "shannon_diversity", ccf_label),
        # get_region_metric(obs_ccf, calc_shannon_index, "shannon_index_norm", ccf_label,
        #                   norm_fcn=lambda x: np.log2(len(x.unique()))
        ], axis=1)

    th_metrics_df['count_cells'] = obs_ccf[ccf_label].value_counts()

    return th_metrics_df


def get_region_metric(obs_ccf,
                      function, 
                      metric_name,
                      ccf_label='parcellation_structure_eroded',
                      exclude=['unassigned','TH-unassigned'],
                      norm_fcn=None,
                      levels=['cluster','supertype','subclass']):
    ''' Calculate a metric for each region in the thalamus.

    Parameters
    ----------
    obs_ccf : pd.DataFrame
        dataframe of observations with CCF annotations
    function : function
        function to apply to each region
    metric_name : str
        name of the metric, used for the column name
    ccf_label : str, default='parcellation_structure_eroded'
        column name in obs_ccf where the CCF annotations can be found
    exclude : list of str, default=['unassigned','TH-unassigned']
        list of categories in ccf_label to exclude from the analysis
    norm_fcn : function, optional, default=None
        function to use to normalize the metric
    levels : 
        levels to group by, by default ['cluster','supertype','subclass']

    '''
    # calculate metric per region, for each level
    th_metrics_df = (obs_ccf.loc[lambda df: ~df[ccf_label].isin(exclude)]  # filter out unassigned regions
                        .groupby(ccf_label, observed=True)[levels]  # groupby ccf_label col, keep only levels columns
                        .aggregate(function)  # apply function to each (region, level) pair
                        .rename(columns=lambda x: "_".join([metric_name, x])))  # rename columns to '[metric_name]_[level]'
    
    # normalize metric if norm_fcn is provided
    if norm_fcn is not None:
        th_metrics_df = th_metrics_df / obs_ccf[levels].apply(norm_fcn).values[None,:]
    
    return th_metrics_df


def count_unique(x):
    ''' Count the number of unique elements in x (a pd.Series 
    or 1D np.array).
    '''
    return len(x.unique())

def count_unique_norm(x):
    ''' Count the number of unique elements in x (a pd.Series 
    or 1D np.array), normalized by the total number of items.'''
    return len(x.unique())/len(x)

def count_gt5(x):
    '''Count the number of unique elements in a series that occur more than 5 times.'''
    return len(x.value_counts().loc[lambda c: c>5])

def inverse_simpsons_index(x):
    '''Calculate the Inverse Simpson's Index for a series.
    
    The Inverse Simpson's Index (ISI) ranges from 1 to inf, where 1 
    represents no diversity and inf represents infinite diversity.
    '''
    return 1/np.sum((x.value_counts().loc[lambda c: c>0]/len(x))**2)

def shannon_index(x):
    '''Calculate the Shannon Diversity Index for a pandas Series.
    
    Shannon diversity index, normalized to (0,1)
    1 = high diversity (clusters found in equal proportions)
    0 = low diversity (some clusters found in higher proportions in subregion)
    '''
    # calculate proportion of each category
    cateogory_counts = x.value_counts()
    cateogory_counts = cateogory_counts[cateogory_counts>0] # cleanup zeros so log() doesn't throw warnings
    p = cateogory_counts / cateogory_counts.sum()
    
    # calculate shannon diversity index
    shannon_ind = (-1)*((p * np.log2(p)).sum())
    
    return shannon_ind