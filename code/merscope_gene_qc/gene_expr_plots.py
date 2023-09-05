import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse


# list of 9 AP positions that have matching brain1 & brain3 sections
section_pairs_brain1_brain3 = np.array([
                                        ['1198980077','1199651060'],
                                        ['1198980089','1199651054'],
                                        ['1198980101','1199651048'],
                                        ['1198980108','1199651045'],
                                        ['1198980114','1199651042'],
                                        ['1198980120','1199651039'],
                                        ['1198980134','1199651033'],
                                        ['1198980146','1199651027'],
                                        ['1198980152','1199651024']
                                       ])

def plot_brain1_vs_brain3_expression(ad, gene,
                                     match_colorbar_ranges=True,
                                     display_fig=True,
                                     **kwargs):
    '''
    Displays a grid of 6 paired brain1 and brain3 sections for a single gene.
    
    Parameters
    ----------
    ad:
        cell-by-gene anndata object containing both brain1 and brain3 data.
        Expects to find:  spatial coordinates for 
        each cell stored in adata.obsm['spatial_cirro'], x:[:,0], y:[:,1]
    
    genes: list of strings
        gene names to plot in columns
    
    sections: list of strings
        section IDs to plot in rows
    
    match_gene_colorbars:
        for each gene, sets the vmin & vmax to the same values across all 
        sections
        
    display_fig:
        sets whether or not the figure is displayed. When generating many plots
        in a row, set to False to speed up runtime.
        
    figsize: tuple
        sets figsize in matplotlib.pyplot.subplots()
        
    title: string
        overall title for the figure, e.g. 'brain1'
        
    base_fontsize: int
        sets the fontsize for the section & gene name labels; all other fonts
        (title font, colorbar label, etc.) are set as fractions or multiples of
        this base fontsize.
        
    **kwargs:
        passed through to matplotlib.pyplot.scatter
    
    Returns
    -------
        Matplotlib figure
    
    '''

    # six manually selected, representative brain1-brain3 AP position section pairs 
    # (out of 11 total matching AP postitions) 
    sec_pairs_b1_b3 = np.array([
                                ['1198980077','1199651060'],  # more anterior
                                ['1198980089','1199651054'],
                                ['1198980101','1199651048'],
                                ['1198980108','1199651045'],
                                ['1198980114','1199651042'],
                                # ['1198980120','1199651039'], # despite being paired in cirrocumulus, 
                                                               # these do not appear to be the same AP position
                                ['1198980134','1199651033']  # more posterior
                               ])

    n_brains = sec_pairs_b1_b3.shape[1]
    n_sections = sec_pairs_b1_b3.shape[0]

    fig, axs = plt.subplots(n_brains, n_sections, figsize=(24,7))
    
    big_font = 24
    small_font = 16

    for i, ap_pos in enumerate(sec_pairs_b1_b3):

        # Handle colorbar range input
        match_colorbar_ranges = True
        if match_colorbar_ranges:
            # set colorbar range based on gene expression for both brains
            gene_expr = ad[ad.obs['section'].isin(ap_pos)][:,gene].X.A.flatten()
            vmax = np.round(np.max(gene_expr)*0.95)  # *1.0 is often too dim
            vmin = 0
        else:
            vmax, vmin = None

        for brain, sec_id in enumerate(ap_pos):
            # print('brain:',brain,', sec:',str(i))
            curr_data = ad[ad.obs['section']==sec_id]
            ax = axs[brain,i]

            sc = ax.scatter(curr_data.obsm["spatial_cirro"][:,0], 
                            curr_data.obsm["spatial_cirro"][:,1],
                            c=curr_data[:,gene].X.A, 
                            cmap='Blues', vmax=vmax, vmin=vmin,
                            s=35000/ad.shape[0])
            ax.set_aspect('equal', 'box')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(None)
            ax.spines[['top', 'bottom', 'right', 'left']].set_visible(False)
            # ax.set_facecolor('silver')

            # label sections
            if (brain==0) | (brain==1):
                ax.set_title('sec '+sec_id, fontsize=12)

            # label leftmost section with brain
            if i==0:
                if brain==0:
                    ax.set_ylabel('brain1\n(mouse 609882)', fontsize=small_font+4)
                if brain==1:
                    ax.set_ylabel('WMB - brain3\n(mouse 638850)', fontsize=small_font+4)
            else:
                ax.set_ylabel(None)

            # only display colorbar on rightmost sections
            if i==(n_sections-1):
                cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
                cbar.ax.tick_params(labelsize=small_font)
                cbar.set_label('log2(CPV+1)', fontsize=small_font)

    fig.suptitle(gene, fontsize=big_font+4, y=1.0)
    fig.tight_layout()
    
    if not display_fig:
        plt.close()
    
    return fig



def plot_expr_section_by_gene(ad, genes, sections,
                              match_gene_colorbars=True, display_fig=True, 
                              figsize=(24,7), title=None, 
                              base_fontsize=16,
                              **kwargs):
    '''
    Displays an n_sections x n_genes grid of gene expression images.

    Parameters
    ----------
    ad:
        cell-by-gene anndata object; expects to find spatial coordinates for 
        each cell stored in adata.obsm['spatial_cirro'], x:[:,0], y:[:,1]
    
    genes: list of strings
        gene names to plot in columns
    
    sections: list of strings
        section IDs to plot in rows
    
    match_gene_colorbars:
        for each gene, sets the vmin & vmax to the same values across all 
        sections
        
    display_fig:
        sets whether or not the figure is displayed. When generating many plots
        in a row, set to False to speed up runtime.
        
    figsize: tuple
        sets figsize in matplotlib.pyplot.subplots()
        
    title: string
        overall title for the figure, e.g. 'brain1'
        
    base_fontsize: int
        sets the fontsize for the section & gene name labels; all other fonts
        (title font, colorbar label, etc.) are set as fractions or multiples of
        this base fontsize.
        
    **kwargs:
        passed through to matplotlib.pyplot.scatter
    
    Returns
    -------
        Matplotlib figure
    
    '''
    
    # check if ad.X is stored as a sparse or dense matrix (need to access 
    # differently, depending) - brain1 asset is dense, brain3 is sparse
    if sparse.issparse(ad.X):
        x_is_sparse = True
    else:
        x_is_sparse = False
    
    # set fontsize options
    big_fontsize = int(np.round(base_fontsize*1.5))
    small_fontsize = int(np.round(base_fontsize*0.75))
    
    # make section-by-gene grid
    n_rows= len(sections)
    n_cols = len(genes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # scatter plot of expression of each (section, gene) pair
    for gene_counter, gene in enumerate(genes):

        # Handle colorbar range input
        if match_gene_colorbars:
            # set colorbar range to (0, max of all sections) per gene
            if x_is_sparse:
                gene_expr = ad[ad.obs['section'].isin(sections)][:,gene].X.A.flatten()
            else:
                gene_expr = ad[ad.obs['section'].isin(sections)][:,gene].X.flatten()
            vmax = np.round(np.max(gene_expr)*0.95) # *1.0 is often too dim to see faint spatial patterns
            vmin = 0
        else:
            vmax, vmin = None, None
            

        for sec_counter, sec in enumerate(sections):
            
            curr_data = ad[ad.obs['section']==sec]
            
            if x_is_sparse:
                colors_expr = curr_data[:,gene].X.A
            else:
                colors_expr = curr_data[:,gene].X
            
            ax = axes[sec_counter, gene_counter]
            
            # expression displayed as color in scatter plot
            sc = ax.scatter(curr_data.obsm["spatial_cirro"][:,0], 
                            curr_data.obsm["spatial_cirro"][:,1],
                            c=colors_expr, cmap='Blues', vmax=vmax, vmin=vmin,
                            s=35000/ad.shape[0], **kwargs)
            cbar = plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.01)
            cbar.ax.tick_params(labelsize=12)
            ax.set_aspect('equal', 'box')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(None)
            ax.spines[['top', 'bottom', 'right', 'left']].set_visible(False)

            # label top row with genes
            if (sec_counter==0):
                ax.set_title(gene, fontsize=base_fontsize)

            # label leftmost section with section IDs
            if gene_counter==0:
                ax.set_ylabel('sec '+sec, fontsize=base_fontsize)
            else:
                ax.set_ylabel(None)

            # only display colorbar label on rightmost sections
            if gene_counter==(n_cols-1):
                cbar.set_label('log2(CPV+1)', fontsize=base_fontsize)

    fig.suptitle(title, fontsize=big_fontsize, y=1.0)
    fig.tight_layout()
    
    if not display_fig:
        plt.close()
    
    return fig