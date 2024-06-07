from sklearn.metrics import pairwise_distances
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

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
        distance metric to use (from sklearn.metrics.pairwise.distance_metrics
        or scipy.spatial.distance, see sklearn.metrics.pairwise_distances for 
        options), by default 'braycurtis'

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
    
def order_distances_by_clustering(D, cluster_method='complete'):
    """ Order rows and columns of a distance matrix by hierarchical clustering.
    
    Parameters
    ----------
    D: np.ndarray
        distance matrix (e.g. distance between clusters using braycurtis metric from sklearn)
    cluster_method: str, optional
        method for hierarchical clustering, by default 'complete'
        
    Returns
    -------
    order: np.ndarray
        order of rows/columns
    """
    # convert to squareform for hierarchical clustering
    y = ssd.squareform(D)
    # perform hierarchical clustering
    link = sch.linkage(y, method=cluster_method, optimal_ordering=True)
    # get order of leaves
    dend = sch.dendrogram(link, no_plot=True)
    order = dend["leaves"]
    
    return np.array(order)
    
def order_distances_x_to_y(D, 
                           reorder_y=True, 
                           min_similarity_x=0.1):
    """ Order rows or columns of a distance matrix by similarity to each other.
    
    Parameters
    ----------
    D: np.ndarray
        distance matrix (e.g. distance between clusters using braycurtis metric from sklearn)
    reorder_y: bool, optional
        whether to reorder the rows (y) of the matrix, by default True
    min_similarity_x: float, optional
        minimum similarity to keep a row in the matrix, by default 0.1
        
    Returns
    -------
    y_order, x_order: np.ndarray
        order of rows, columns
    """
    # rank columns (x)
    argmin = D.argmin(axis=0)
    min_dist = D.min(axis=0) # if distances are <1, this serves as a tiebreaker
    x_order = np.argsort(argmin + min_dist)
    x_order = [i for i in x_order if 1-min_dist[i] > min_similarity_x]
    
    # rank rows (y)
    y_order = list(range(D.shape[0]))
    if reorder_y:
        argmin = argmin[1-min_dist > min_similarity_x]
        # reorder rows without a match, grouping with similar matched rows
        y_old = [i for i in range(D.shape[0]) if i not in np.unique(argmin) 
                 ]
        # sorts to a row with a matching argmin
        # alternatively, could choose by overall pattern match (correlation etc)
        y_new = [D[:, D[y, :].argmin()].argmin() + 1 for y in y_old]
        for i in np.argsort(y_new)[::-1]:
            inew = y_new[i]
            iold = y_old[i]
            y_order[y_order.index(iold)] = -1
            y_order = y_order[:inew] + [iold] + y_order[inew:]
        y_order = np.array(y_order)
        y_order = y_order[y_order != -1]
    return y_order, x_order


def plot_ordered_similarity_heatmap(D,
                                    y_names=None, 
                                    x_names=None,
                                    y_order=None, 
                                    x_order=None,
                                    triangular=False, 
                                    vmin=0, 
                                    vmax=1,
                                    label="similarity (Bray-Curtis)", 
                                    cmap='rocket_r', 
                                    **kwargs):
    """ Plot a 2D heatmap of a distance matrix, with rows and columns ordered by 
    nearest distances.
    
    Parameters
    ----------
    D: np.ndarray
        distance matrix
        e.g. as output by cluster_distances_from_labels()
    y_names, x_names: list, default=None
        names of rows (y), columns (x) of the matrix.
        e.g. as output by cluster_distances_from_labels()
    y_order, x_order : np.ndarray, default=None
        indices of elements of x_names to plot, in desired order.
        e.g. as output by order_distances_x_to_y()
    triangular: bool, default=False
        whether to plot only the lower triangle of the matrix
    vmin, vmax: float, optional, default={0, 1}
        min & max values for color scale
    label: str, default="similarity (Bray-Curtis)"
        label for the colorbar
    cmap: str, default='rocket_r'
        colormap for the heatmap
    **kwargs: dict
        additional keyword arguments for seaborn.heatmap()
        
    Returns
    -------
    fig: matplotlib.figure.Figure
    """
    # set order if not provided
    if y_order is None: 
        y_order = list(range(D.shape[0]))
    if x_order is None: 
        x_order = y_order
    # subset & reorder [x/y]_names according to [x/y]_order
    yticklabels = np.array(y_names)[y_order] if y_names is not None else y_order
    xticklabels = np.array(x_names)[x_order] if x_names is not None else x_order
    
    # reorder distance matrix   
    Dplot = D[np.ix_(y_order,x_order)]
    # mask upper triangle of distance matrix, if specified
    if triangular:
        Dplot[np.triu_indices_from(Dplot, k=1)] = np.nan
    
    # plot heatmap
    fig, ax = plt.subplots()
    hm = sns.heatmap(1-Dplot, 
                     yticklabels=yticklabels, xticklabels=xticklabels, 
                     cbar_kws=dict(label=label, pad=0.01), 
                     cmap=cmap, vmin=vmin, vmax=vmax,
                     ax=ax, **kwargs)
    # axis formatting
    ax.axis('equal')
    ax.tick_params(axis='both', which='both', direction='out')
    ax.tick_params(axis='x', labelrotation=90)
    # colorbar formatting
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(['0', '', '0.5', '', '1'])
    cbar.ax.tick_params(axis='y', direction='in')
    
    return fig


def plot_heatmap_xlabel_colors(x_names, x_order, color_dict, 
                               xticklabel_is_id=True):
    """ Plot a single-row heatmap with colors for each x label.
    
    Parameters
    ----------
    x_names: list
        list of all possible x-labels; e.g. as output by cluster_distances_from_labels()
    x_order: np.ndarray
        indices of the items in x_names to plot, in desired order; e.g. as 
        output by order_distances_x_to_y()
    color_dict: dict
        dictionary mapping x-labels to colors
        
    Returns
    -------
    fig: matplotlib.figure.Figure
    """
    # reorder & subset x_names according to x_order
    x_labels = np.array(x_names)[x_order]
    # make list of colors for each x-label
    color_list = [color_dict[x] for x in x_labels if x in color_dict]
    
    # HARDCODED: use only the taxonomy category ID # for the xticklabels
    if xticklabel_is_id:
        xticklabels = [label.split(" ")[0] for label in x_labels] 
    else:
        xticklabels = x_labels
    
    fig, ax = plt.subplots()
    # plot 1D heatmap
    hm = sns.heatmap(np.arange(len(xticklabels)).reshape(1, -1),
                               ax=ax,
                               square=True, 
                               cmap=color_list, 
                               cbar=False,
                               xticklabels=xticklabels)
    hm.set_xticklabels(xticklabels, rotation=90)
    
    return fig