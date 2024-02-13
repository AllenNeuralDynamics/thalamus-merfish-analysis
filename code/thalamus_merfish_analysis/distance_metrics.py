from sklearn.metrics import pairwise_distances
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
import seaborn as sns
import numpy as np

def cluster_distances_from_labels(obs_df, y_col, x_col=None,
                                  y_names=None, x_names=None,
                                  metric='braycurtis'):
    """Generate distances between clusters, for cluster pairs within
    one clustering scheme if only y is specified, or for all pairs between
    two clustering schemes if x and y are specified.
    Cluster labels within a single column are converted to binary indicator
    vectors before calculating distances.

    Parameters
    ----------
    obs_df
        dataframe of clustered observations (cells)
    y_col
        column containing cluster labels
    x_col, optional
        column containing second cluster scheme labels,
        or optionally a list of columns for a soft (probabilistic) cluster assignment
    y_names, optional
        subset of clusters to use from first column
    x_names, optional
        subset of clusters to use from second column
    metric, optional
        distance metric to use (from sklearn.metrics or scipy.spatial.distance), 
        by default 'braycurtis'

    Returns
    -------
    dist
        distance matrix (as np.ndarray)
    y_names
        names of y clusters in order
    x_names
        names of x clusters in order
    """    
    y_names = obs_df[y_col].unique() if y_names is None else y_names
    Y = np.vstack([obs_df[y_col]==name for name in y_names])
    if x_col is None:
        X = None
        x_names = y_names
    elif type(x_col) is str:
        x_names = obs_df[x_col].unique() if x_names is None else x_names
        X = np.vstack([obs_df[x_col]==name for name in x_names])
    # multiple x columns
    else:
        x_names = x_col if x_names is None else x_names
        X = obs_df[x_col].values.T
    dist = pairwise_distances(Y, X, metric=metric)
    return dist, y_names, x_names
    
def order_distances_by_clustering(dist, cluster_method='complete'):
    y = ssd.squareform(dist)

    link = sch.linkage(y, method=cluster_method, optimal_ordering=True)
    dend = sch.dendrogram(link, no_plot=True)
    order = dend["leaves"]
    return np.array(order)
    
def order_distances_x_to_y(D, reorder_y=True, min_similarity_x=0.1):
    argmin = D.argmin(axis=0)
    min_dist = D.min(axis=0)
    x_order = np.argsort(argmin + min_dist)
    x_order = [i for i in x_order if 1-min_dist[i] > min_similarity_x]
    # y_show = np.unique(argmin) if show_matched_y_only else slice(None)
    
    y_order = list(range(D.shape[0]))
    if reorder_y:
        argmin = argmin[1-min_dist > min_similarity_x]
    # reorder rows without a match, grouping with similar matched rows
        y_old = [i for i in range(D.shape[0]) if i not in np.unique(argmin) 
                 ]
        y_new = [D[:, D[y, :].argmin()].argmin() + 1 for y in y_old]
        for i in np.argsort(y_new)[::-1]:
            inew = y_new[i]
            iold = y_old[i]
            y_order[y_order.index(iold)] = -1
            y_order = y_order[:inew] + [iold] + y_order[inew:]
        y_order = np.array(y_order)
        y_order = y_order[y_order != -1]
    return y_order, x_order

def plot_ordered_similarity_heatmap(D, y_order=None, x_order=None, y_names=None, x_names=None,
                           label="Dice coefficient (similarity)", cmap='rocket_r'):
    if y_order is None: 
        y_order = list(range(D.shape[0]))
    if x_order is None: 
        x_order = y_order
    
    yticklabels = np.array(y_names)[y_order] if y_names is not None else y_order
    xticklabels = np.array(x_names)[x_order] if x_names is not None else []
    sns.heatmap(1-D[np.ix_(y_order,x_order)], yticklabels=yticklabels, xticklabels=xticklabels, 
                cbar_kws=dict(label=label), cmap=cmap, vmin=0, vmax=1)