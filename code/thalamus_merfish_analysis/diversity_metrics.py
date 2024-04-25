import numpy as np
import pandas as pd

TH_DIVERSITY_REGIONS = ['AD', 'AV', 'AM', 'CL', 'CM', 'IAD', 'IMD',
                        'LD', 'LGd', 'LH', 'LP', 'MD', 'MH', 
                        'PCN', 'PF', 'PO', 'PVT', 'RE', 'RT', 'SPA',
                        'VAL', 'VM', 'VPL', 'VPM', 'VPMpc'
                        ]


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

    # calculate a pre-selected set of diversity metrics for each region
    th_metrics_df = pd.concat([
        get_region_metric(obs_ccf, n_unique, "n", ccf_label=ccf_label),
        get_region_metric(obs_ccf, n_unique, "frac", ccf_label=ccf_label, 
                          norm_fcn=n_unique),
        get_region_metric(obs_ccf, unique_norm, "n_norm", ccf_label=ccf_label),
        get_region_metric(obs_ccf, count_gt5, "count_gt5", ccf_label=ccf_label),
        get_region_metric(obs_ccf, inverse_simpsons, "inverse_simpsons", ccf_label=ccf_label),
        ], axis=1)

    th_metrics_df['n_cell'] = obs_ccf[ccf_label].value_counts()

    return th_metrics_df


def get_region_metric(obs_ccf,
                      function, 
                      metric_name,
                      ccf_label='parcellation_structure_eroded',
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
    norm_fcn : function, optional, default=None
        function to use to normalize the metric
    levels : 
        levels to group by, by default ['cluster','supertype','subclass']

    '''
    ccf_to_exclude = ['unassigned','TH-unassigned']

    # calculate metric for each (region, level) pair
    th_metrics_df = (obs_ccf.loc[lambda df: ~df[ccf_label].isin(ccf_to_exclude)]  # filter out unassigned regions
                        .groupby(ccf_label, observed=True)[levels]  # groupby ccf_label col, keep only levels columns
                        .aggregate(function)  # apply function to each (region, level) pair
                        .rename(columns=lambda x: "_".join([metric_name, x])))  # rename columns to '[metric_name]_[level]'
    
    # normalize metric if norm_fcn is provided
    if norm_fcn is not None:
        th_metrics_df = th_metrics_df / obs_ccf[levels].apply(norm_fcn).values[None,:]
    
    return th_metrics_df


def n_unique(x):
    ''' Count the number of unique elements in a series.'''
    return len(x.unique())

def unique_norm(x):
    ''' Count the number of unique elements in a series, normalized by the 
    total number of elements.'''
    return len(x.unique())/len(x)

def count_gt5(x):
    '''Count the number of unique elements in a series that occur more than 5 times.'''
    return len(x.value_counts().loc[lambda c: c>5])

def inverse_simpsons(x):
    '''Calculate the Inverse Simpson's Index for a series.
    
    The Inverse Simpson's Index (ISI) ranges from 1 to inf, where 1 
    represents no diversity and inf represents infinite diversity.
    '''
    return 1/np.sum((x.value_counts().loc[lambda c: c>0]/len(x))**2)


def calc_shannon_index(obs_col):
    # calculate proportion of each category
    cateogory_counts = obs_col.value_counts()
    cateogory_counts = cateogory_counts[cateogory_counts>0] # cleanup zeros so log() doesn't throw warnings
    p = cateogory_counts / cateogory_counts.sum()
    
    # calculate shannon diversity index
    shannon_ind = (-1)*((p * np.log2(p)).sum())
    
    # normalized separately
    # normalize by log2 of number of categories to limit index to range (0, 1)
    # shannon_norm = shannon_ind / np.log2(n_total_categories)
    return shannon_ind