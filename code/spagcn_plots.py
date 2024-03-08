import pathlib
import numpy as np
import matplotlib.colors as clr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.lines import Line2D
import seaborn as sns
from colorcet import glasbey

# RESULTS_DIR = pathlib.Path('/results/figures/')
# RESULTS_DIR.mkdir(exist_ok=True)

# source: https://sashamaps.net/docs/resources/20-colors/
sasha_trubetskoy_colors_21 = ['#e6194B', '#3cb44b', '#ffe119',  # red, green, yellow,
                              '#4363d8', '#f58231', '#911eb4',  # blue, orange, purple,
                              '#42d4f4', '#f032e6', '#bfef45',  # cyan, magenta, lime,
                              '#fabed4', '#469990', '#dcbeff',  # pink, teal, lavender,
                              '#9A6324', '#aaffc3', '#a9a9a9',  # brown, mint, grey,
                              '#000000',                        # black,
                              '#808000', '#800000', '#000075',  # olive, maroon, navy (dark and harder to tell apart f/ each other)
                              '#ffd8b1', '#fffac8']  # apricot, beige (light, hardest to see in scatter plots w/ white background)

# hard-coding colors that I find to be easily distinguishable
# source: https://sashamaps.net/docs/resources/20-colors/
DOMAIN_COLORS_27 = ['#e6194B', '#3cb44b', '#ffe119',  # red, green, yellow,
                     '#4363d8', '#f58231', '#911eb4',  # blue, orange, purple,
                     '#42d4f4', '#f032e6', '#bfef45',  # cyan, magenta, lime,
                     '#fabed4', '#469990', '#dcbeff',  # pink, teal, lavender,
                     '#9A6324', '#aaffc3', '#a9a9a9',  # brown, mint, grey,
                     '#000000',                        # black,
                     '#808000', '#800000', '#000075',  # olive, maroon, navy (dark and harder to tell apart f/ each other)
                     '#ffd8b1', '#fffac8',             # apricot, beige (light, hardest to see in scatter plots w/ white background)
                     '#915282', '#ff7266',             # extra colors from colorcet.glaseby (dusty purple, salmon, 
                     '#8287ff', '#9ae4ff', '#eb0077',  # periwinkle, light blue, dark pink,
                     '#ff4600']                        # red-orange,


def plot_spaGCN_domains(ad, pred_col, plot_title=None):
    # set default plot title
    if plot_title==None:
        plot_title = 'SpaGCN MERFISH domains\n'+pred_col

    fig = plt.figure(figsize=(16,8))
    ax = plt.gca()
    marker_size = 15000/ad.shape[0]
    
    # set domain:color mapping dict & save to ad.uns so future plots look the same
    obsm_key = 'spaGCN_predicted_domains'
    spg_domain_ids = sorted(ad.obsm[obsm_key][pred_col].unique().tolist())
    palette_dict = dict(zip(spg_domain_ids, DOMAIN_COLORS_27))
    ad.uns[pred_col+"_colors"] = palette_dict

    # scatter plot iteratively (best way to use color palette dictionary)
    for domain in spg_domain_ids:
        curr_domain_ad = ad[ad.obsm[obsm_key][pred_col]==domain]
        s = ax.scatter(curr_domain_ad.obs['x_reconstructed'], 
                       curr_domain_ad.obs['y_reconstructed'],
                       c=palette_dict[domain], 
                       label=domain,
                       s=marker_size)
    # legend properties
#     lgd = ax.legend(loc="center left", bbox_to_anchor=(1.,0.5), frameon=False,
#                     markerscale=7, fontsize=14)#, scatterpoints=1)
    lgd = ax.legend(loc="upper center", bbox_to_anchor=(0.5,0.), ncol=4, 
                    frameon=False, markerscale=7, fontsize=14)#, scatterpoints=1)
    lgd.set_title('SpaGCN domains',prop={'size':14})  # set legend title fontsize
    # hide all x-y axes/ticks/labels
    ax.spines[['top', 'bottom', 'right', 'left']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xlim(2.5, 5.5)
    ax.set_ylim(-7.5, -4)
    # other plot properties
    ax.set_aspect('equal', 'box')
    plt.title(plot_title, fontsize=14)

    # Save
    # fig.savefig(RESULTS_DIR/'brain3_{sec_name}_SpaGCN_spatial_domains_{pred_col}.png', bbox_inches='tight')
    
    return ad, fig


def plot_domains_multisection(data, pred_col, obsm_key='spaGCN_predicted_domains',
                              sections=None, n_rows=2,
                              plot_title=None):
    
    # allow to input adata or just obs
    if hasattr(data, 'obs'):
        obs = ad.obs
        ad = data.copy()
    else:
        obs = data.copy()
    
    # set default plot title
    if plot_title==None:
        plot_title = 'SpaGCN MERFISH domains\n'+pred_col
        
    s = 10 #15000/ad.shape[0]
        
    fig = plt.figure(figsize=(30, 10))

    # set grid geometry based on the number of sections
    if sections==None:
        section_ids = obs['brain_section_label'].cat.categories.tolist()
    else:
        section_ids = sections
    n_rows = n_rows
    n_cols = int(np.ceil(len(section_ids)/n_rows))

    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(n_rows, n_cols),  # creates 2x2 grid of axes
                     axes_pad=(0.1,0.6),  # pad between axes in inch.
                     share_all=True
                     )
    
    # set domain:color mapping dict
    if obsm_key==None:
        hue_col = pred_col
    else:
        hue_col = 'spaGCN_domain'
        obs[hue_col] = ad.obsm[obsm_key][pred_col]
        
    spg_domain_ids = sorted(obs[hue_col].unique().tolist())
    sns_palette = sns.color_palette(glasbey, n_colors=len(spg_domain_ids))
    palette_dict = dict(zip(spg_domain_ids, sns_palette))
    # ad.uns[pred_col+"_colors"] = palette_dict
        
    # plot each domain separately
    for i, sec in enumerate(section_ids):
        ax = grid[i]
        obs_sec = obs[obs['z_reconstructed']==sec]
        
        sns.scatterplot(obs_sec, ax=ax, x='x_reconstructed', y='y_reconstructed', 
                        hue=hue_col, s=s, palette=palette_dict, 
                        linewidth=0, legend=False)
        
        # lgd = ax.legend(loc="lower center", bbox_to_anchor=(0.5,0.95), 
        #                 frameon=False, markerscale=7, fontsize=16)
        ax.set_aspect('equal', 'box')
        # hide all x-y axes, ticks, labels, etc.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.spines[['top', 'bottom', 'right', 'left']].set_visible(False)
        
        n_plots = i  # counter so we can hide extra grids at the end

    # # turn off axes for the empty subplots, which is just a hacky fix because
    # # ngrids=len(svg) doesn't work in this version of matplotlib
    # for j in np.arange(n_plots+1,((n_cols*n_rows))):
    #     ax = grid[j]
    #     ax.set_visible(False)
    legend_handles = [Line2D([0], [0], linestyle='None', marker='o', 
                             markersize=14, color=color, label=label) 
                      for label, color in palette_dict.items()]

    
    legend = fig.legend(handles=legend_handles, loc='upper center', 
                        bbox_to_anchor=(0.5, 0), ncol=n_cols, frameon=False, 
                        fontsize=20)
    
    plt.show()
    
    # Save
    # fig.savefig(RESULTS_DIR/'brain3_all_sec_SpaGCN_spatial_domains_{pred_col}.png', bbox_inches='tight')
    
    return fig


def plot_spaGCN_domains_separately(ad, pred_col, n_subplot_col=4):
    
    fig = plt.figure(figsize=(30, 30))

    # set grid geometry based on the number of domains
    obsm_key = 'spaGCN_predicted_domains'
    spg_domain_ids = sorted(ad.obsm[obsm_key][pred_col].unique().tolist())
    n_cols = n_subplot_col
    n_rows = int(np.ceil(len(spg_domain_ids)/n_cols))

    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(n_rows, n_cols),  # creates 2x2 grid of axes
                     axes_pad=(0.1,0.6),  # pad between axes in inch.
                     share_all=True
                     )

    # set marker size
    marker_size = 5 #20000/ad.shape[0] 
    color_palette = ad.uns[pred_col+"_colors"]

    # plot each domain separately
    for i, domain in enumerate(spg_domain_ids):
        ax = grid[i]

        # split off current domain from all others
        curr_domain_ad = ad[ad.obsm[obsm_key][pred_col]==domain]
        other_domains_ad = ad[ad.obsm[obsm_key][pred_col]!=domain]

        # plot all other domains in grey
        s0 = ax.scatter(other_domains_ad.obs['x_reconstructed'],
                        other_domains_ad.obs['y_reconstructed'],
                        c='silver', s=marker_size/2, alpha=0.1)

        # plot current domain in it's assigned color
        curr_domain_color = color_palette[domain]
        s0 = ax.scatter(curr_domain_ad.obs['x_reconstructed'],
                        curr_domain_ad.obs['y_reconstructed'],
                        c=curr_domain_color, s=marker_size,
                        label='SpaGCN domain '+str(domain))
        lgd = ax.legend(loc="lower center", bbox_to_anchor=(0.5,0.95), 
                        frameon=False, markerscale=7, fontsize=16)
        ax.set_aspect('equal', 'box')
        # hide all x-y axes, ticks, labels, etc.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.spines[['top', 'bottom', 'right', 'left']].set_visible(False)
        
        n_plots = i  # counter so we can hide extra grids at the end

    # turn off axes for the empty subplots, which is just a hacky fix because
    # ngrids=len(svg) doesn't work in this version of matplotlib
    for j in np.arange(n_plots+1,((n_cols*n_rows))):
        ax = grid[j]
        ax.set_visible(False)
        
    # # Save
    # fig.savefig(RESULTS_DIR+'brain3_'+sec_name+'_SpaGCN_spatial_domains_'+pred_col+'_indiv.png', bbox_inches='tight')
        
    return fig