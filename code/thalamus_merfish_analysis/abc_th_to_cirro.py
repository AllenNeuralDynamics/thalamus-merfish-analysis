import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import seaborn as sns
import colorcet as cc

from .abc_load import (load_adata_thalamus, 
                       filter_by_class_thalamus,
                       filter_by_thalamus_coords,
                       get_taxonomy_palette)

def export_h5ad_for_cirro(generate_umap_tsne=True):
    ''' Load thalamus subset of ABC Atlas, modify for Cirrocumulus compatibility,
    and save out as h5ad file.

    Returns
    -------
    adata : Cirrocumulus-compatible AnnData object
    /results/abc_atlas_mouse_thalamus_cirro.h5ad : h5ad file containing adata
    '''
    print('Loading thalamus subset from ABC Atlas ...')
    adata = load_adata_for_cirro()

    print('Generating spatial coordinates for Cirrocumulus...')
    adata = add_montage_coords(adata)
    adata = add_coords_to_obsm(adata)

    print('Adding color palettes for Cirrocumulus...')
    adata = add_colors_to_uns(adata)

    if generate_umap_tsne:
        print('Calculating low-dimensional embeddings ... This may take 10-15 minutes.')
        adata = add_umap_tsne_to_obsm(adata)

    # add unsupervised machine learning results
    print('Adding unsupervised machine learning results...')
    adata = add_spagcn_to_adata(adata)
    adata = add_nsf_to_obs(adata)

    # save as h5ad
    adata.write('/results/abc_atlas_mouse_thalamus_cirro.h5ad',
                compression="gzip")

    return adata


def load_adata_for_cirro():
    ''' Load thalamus subset of ABC Atlas as AnnData object.

    Cannot use `load_standard_thalamus()` because y coords need to be inverted  
    to display the brain sections in their proper orientation in Cirrocumulus.
    '''
    # load with inverted y coord so sections appear in correct coronal orientation
    adata = load_adata_thalamus(flip_y=True)

    # filter by class
    adata = filter_by_class_thalamus(adata,
                                         filter_nonneuronal=True,
                                         filter_midbrain=False,
                                         filter_other_nonTH=True)
    
    # filter by thalamus coordinates
    adata = filter_by_thalamus_coords(adata, 
                                          buffer=0)

    return adata


def add_montage_coords(adata, 
                       section_col='z_reconstructed',
                       x_col='x_reconstructed',
                       y_col='y_reconstructed',
                       new_coord_suffix='_cirro',
                       n_cols=3):
    ''' Creates 2D montage spatial coordinates from 2D+section coordinates and 
    adds them to adata.obs.
    
    Montage spatial coordinates are used for simultaneous display of multiple 
    sections on a 2D screen, e.g. in cirrocumulus.

    Parameters
    ----------
    adata : AnnData object
        AnnData object with spatial data.
    section_col : str
        Column in adata.obs with section numbers.
    x_col, y_col : str
        Columns in adata.obs with x,y coordinates.
    new_coord_suffix : str
        Suffix to add to new x,y columns
    n_cols : int, default=4
        Number of columns to display sections in

    Returns
    -------
    adata : AnnData object
        with x & y montage spatial coordinates added to adata.obs
    '''
    obs = adata.obs.copy()

    # copy over original xy coords to new columns             
    obs['x'+new_coord_suffix] = obs[x_col]
    obs['y'+new_coord_suffix] = obs[y_col]

    # Dynamically set x_shift, y_shift based on current x_col, y_col units
    width_max = abs(obs[x_col].max() - obs[x_col].min())
    height_max = abs(obs[y_col].max() - obs[y_col].min())
    # set x_shift and y_shift as a ratio of max width/height
    x_shift = width_max*1.2
    y_shift = -(height_max*1.5) # neg to start each new row below previous row

    sections = sorted(obs[section_col].unique())
    count = 0
    # start with anterior-most section in top left and end with posterior-most 
    # section in bottom right
    for i, sec in enumerate(reversed(sections)):
        # increment x_shift each col, reset after each row; in 1st col x_shift=0
        curr_x_shift = x_shift * (count % n_cols)
        # increment y_shift after completing each row; in 1st row y_shift=0
        curr_y_shift = y_shift * (i // n_cols)
        
        # apply x,y shifts to current section
        mask = obs[section_col]==sec
        obs.loc[mask, 'x'+new_coord_suffix] += curr_x_shift
        obs.loc[mask, 'y'+new_coord_suffix] += curr_y_shift
        
        count+=1

    adata.obs = obs
    
    return adata


def add_coords_to_obsm(adata):
    ''' Copy cirro and CCF spatial coordinates into adata.obsm, where 
    cirrocumulus expects to find them. 
    
    3D CCF coords should be in .obs already from loading the ABC Atlas data.
    2D cirro coords should have been added to .obs with add_cirro_coords().
    '''

    if {'x_cirro','y_cirro'}.issubset(adata.obs.columns):
        adata.obsm['cirro_spatial'] = adata.obs[['x_cirro','y_cirro']].to_numpy()
    else:
        UserWarning("No cirrocumulus spatial coordinates, ['x_cirro','y_cirro'], found in adata.obs. Run add_cirro_coords() first.")


    if {'x_ccf','y_ccf','z_ccf'}.issubset(adata.obs.columns):
        adata.obsm['ccf_spatial_3d'] = adata.obs[['x_ccf','y_ccf','z_ccf']].to_numpy()
        # flip y coords so data displays in correct orientation
        adata.obsm['ccf_spatial_3d'][:,1] = -adata.obsm['ccf_spatial_3d'][:,1]
    else:
        UserWarning("No CCF spatial coordinates, ['x_ccf','y_ccf','z_ccf'], found in adata.obs.")

    return adata


def add_colors_to_uns(adata):
    ''' Add colors to adata.uns for each taxonomy level in adata.obs.

    Clusters use custom color palette; all others use official ABC Atlas palettes.

    Cirrocumulus expects to find the colors for a given 'field' in .obs in the 
    key 'field_color' in adata.uns (e.g. adata.obs['cluster'] -> 
    adata.uns['cluster_colors']), stored as a numpy array of objects containing 
    colors as hex strings. It assumes the colors are in corresponding order to
    the order returned by adata.obs.my_column.cat.categories.
    '''
    # check that all taxonomy levels are present
    taxonomy_levels = ['neurotransmitter', 'class', 'subclass', 'supertype', 'cluster']
    assert set(taxonomy_levels).issubset(adata.obs.columns), f"adata.obs.columns is missing at least one of: {taxonomy_levels}"

    for level in taxonomy_levels:
        if level=='cluster':
            # use custom color palette for clusters
            palette_df = pd.read_csv('/code/resources/cluster_palette_glasbey.csv')
            abc_color_dict = dict(zip(palette_df['Unnamed: 0'], palette_df['0']))
        else:
            # get the official ABC color palette for all other taxonomy levels
            abc_color_dict = get_taxonomy_palette(level)

        # get the categories that exist in this dataset
        # MUST be kept in the order returned by .cat.categories
        curr_cats = adata.obs[level].cat.categories

        # make new color dict for only the categories that exist in this dataset
        # cat_color_dict = dict((cat, abc_color_dict[cat]) for cat in curr_cats if cat in abc_color_dict)
        cat_colors_array = [abc_color_dict[cat] if cat in abc_color_dict else '#ffffff' 
                            for cat in curr_cats]

        # add colors as an array to adata.uns
        adata.uns[level+'_colors'] = np.array(cat_colors_array, dtype='object')

    return adata


def add_umap_tsne_to_obsm(adata):
    ''' Generate low dimensional embeddings via scanpy's UMAP and tSNE and add
    coordinates to adata.obsm for viewing in Cirrocumulus.

     Returns
    -------
    adata : AnnData object
        with UMAP, tSNE coordinates added to adata.obsm keys 'X_umap', 'X_tsne'
    '''
    # pre-processing
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)

    # generate UMAP (saved to adata.obsm['X_umap'])
    print('Generating UMAP embedding. This may take 1-2 minutes ...')
    sc.tl.umap(adata)

    # calculate tSNE (saved to adata.obsm['X_tsne'])
    print('Generating tSNE embedding. This may take approx. 10 minutes ...')
    sc.tl.tsne(adata)

    # cleanup unneeded data generated by pca, neighbors, tsne
    adata.obsm.pop('X_pca')
    adata.uns.pop('pca')
    adata.uns.pop('neighbors')
    adata.obsp.pop('connectivities')
    adata.obsp.pop('distances')
    adata.varm.pop('PCs')

    return adata

def add_spagcn_to_adata(adata, domains_to_add='res1pt4', spagcn_col='spagcn'):
    ''' Add result domains from SpaGCN package to adata.obs & custom color 
    palette to adata.uns.

    Parameters
    ----------
    adata : AnnData object
    domains_to_add : str
        column name in spagcn_df with SpaGCN domains (just one for now)
    spagcn_col : str
        name of new column to be added to adata.obs; must be a single word, no underscores

    Returns
    -------
    adata: with spagcn domains in adata.obs and corresponding colors in adata.uns
    '''

    # load in SpaGCN domain results
    # temporarily a static file in '../code/resources' until I get a reproducible run setup for the spagcn capsule
    spagcn_df = pd.read_parquet('/code/resources/spagcn_predicated_domains.parquet')

    for col in spagcn_df.columns:
        # preserve numerical order when converting values to str, then column to categorical
        cats_str = [str(x) for x in sorted(spagcn_df[col].unique())]
        spagcn_df[col] = pd.Categorical(spagcn_df[col].astype(str), 
                                        categories=cats_str)

    # for now, just grab one domain result
    spagcn_df.rename(columns={domains_to_add:spagcn_col}, inplace=True)

    # add spagcn domains to copy of adata.obs
    obs_spagcn = adata.obs.join(spagcn_df[spagcn_col], on='cell_label')

    # fill NaNs from .join with new 'no data' category (added at the end of .cat.categories)
    obs_spagcn[spagcn_col] = obs_spagcn[spagcn_col].cat.add_categories('no data').fillna('no data')

    # generate color array for spagcn domains
    spagcn_cats = obs_spagcn[spagcn_col].cat.categories
    spg_colors = np.array(cc.glasbey[:len(spagcn_cats)], dtype='object')
    # set the 'no data' category (last category) color to white so it doesn't show up in cirro
    spg_colors[-1] = '#ffffff'

    # add spagcn domain colors to adata.uns
    # cirrocumulus expects the key to be in the key: {.obs column name}+'_colors'
    if '_' in spagcn_col:
        raise ValueError("spagcn_col must be a single word, with no underscores")
    adata.uns[spagcn_col+'_colors'] = spg_colors

    # add updated obs back into adata
    adata.obs = obs_spagcn

    return adata


def add_nsf_to_obs(adata):
    '''Add all calculated NSF patterns to adata.obs. '''

    # load NSF results
    adata_nsf = ad.read_zarr("/root/capsule/data/nsf_2000_adata/nsf_2000_adata.zarr")

    # get NSF pattern columns from obs
    nsf_cols_bool = adata_nsf.obs.columns.str.startswith('nsf')
    nsf_cols_list = adata_nsf.obs.columns[nsf_cols_bool]
    nsf_patterns_df = adata_nsf.obs.loc[:, nsf_cols_bool]
    
    # add NSF patterns to adata.obs
    obs_with_nsf = adata.obs.join(nsf_patterns_df, on='cell_label')

    # NaNs resulting from .join must be filled with 0s in order to load into cirro
    obs_with_nsf[nsf_cols_list] = obs_with_nsf[nsf_cols_list].fillna(0)

    adata.obs = obs_with_nsf

    return adata


if __name__ == '__main__':
    export_h5ad_for_cirro(generate_umap_tsne=True)