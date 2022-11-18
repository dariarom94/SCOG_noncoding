#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np

import scanpy as sc
import episcanpy as epi

import matplotlib.pyplot as plt
import seaborn as sns

import anndata as ad

import os
import argparse
import sys

from sklearn.metrics import silhouette_samples, silhouette_score

def arg_parser(parser):
    """
    Global options for pipeline.
    """
    parser.add_argument(
        "-b", "--whichbed",
        dest = "which_bed",
        help = "which bed file to use for grouping",
        type = str, 
        required = True)
    
    parser.add_argument(
        "-t", "--thr",
        dest = "thr",
        help = "which thr  to use",
        type = str, 
        required = True)
    
    args = parser.parse_args()
    
    return args


def main():
    #paths setup

    data_path = "/nobackup/lab_bock/users/dromanovskaia/projects/SCOG_noncoding/data/"

    ## which bed file
    parser = argparse.ArgumentParser(
        description = "bla")

    args = arg_parser(parser)
    regions_id = args.which_bed
    thr_nf = args.thr
    
    results_path = os.path.join("/nobackup/lab_bock/users/dromanovskaia/projects/SCOG_noncoding/results_analysis/", regions_id, "n_features_" + thr_nf)
    print("working with " + regions_id + "and nfeature threshold of " + thr_nf)
    thr_nf=int(thr_nf)
    
    ##technical setup
    sc.settings.figdir = results_path
    os.makedirs(results_path)

## reading in annotation data
 #   anno = ad.read(data_path + "10x-ATAC-Brain5k.h5ad")
  #  anno_per_bc = {bc: anno for bc, anno in zip(anno.obs.index, anno.obs.cell_type)}
    anno_new = pd.read_csv(os.path.join(data_path, "annotation.csv"), index_col = 0)
    anno_per_bc = {bc: anno for bc, anno in zip(anno_new.index, anno_new.anno)}

    fragments_file=os.path.join(data_path, "atac_v1_adult_brain_fresh_5k_fragments.tsv.gz")

    bed_file=os.path.join(data_path, "noncoding_bed_files", regions_id+".bed")


## barcode info

    barcode_info = pd.read_csv(data_path + "atac_v1_adult_brain_fresh_5k_singlecell.csv")

    valid_barcodes = barcode_info[barcode_info.is__cell_barcode == 1].barcode.tolist()
    print("calculating region matrix from fragment file....")
## adding the matrix
    adata = epi.ct.peak_mtx(
        fragments_file,
        bed_file, 
        valid_barcodes,
        normalized_peak_size=None,
        fast=False
    )

    adata.obs["cell_type"] = [anno_per_bc[bc] if bc in anno_per_bc else "No Annotation" for bc in adata.obs.index.tolist()]

## binorize the matrix

    print("Max before:\t{}".format(np.max(adata.X)))
    epi.pp.binarize(adata)
    print("Max after:\t{}".format(np.max(adata.X)))

    epi.pp.qc_stats(adata, verbose=True)

    min_features = thr_nf
    max_features = None
## plotQC
    epi.pl.violin(adata, "n_features", min_threshold=min_features, max_threshold=max_features, 
                  show_log=True, show_mean=True, show_median=True, print_statistics=False, save = results_path+"/features_violin.png")
    epi.pl.histogram(adata, "n_features", bins=40, min_threshold=min_features, max_threshold=max_features, 
                     show_log=True, show_mean=True, show_median=True, print_statistics=True, save=results_path+"/features_hist.png")

## calculate     
    epi.pp.nucleosome_signal(adata, fragments_file, n=10000)


    min_cells = 2
    max_cells = None

    epi.pl.violin(adata, "n_cells", min_threshold=min_cells, max_threshold=max_cells, show_log=True, show_mean=True, show_median=True, print_statistics=False, save=results_path+"/ncells_violin.png")
    epi.pl.histogram(adata, "n_cells", bins=40, min_threshold=min_cells, max_threshold=max_cells, show_log=True, show_mean=True, show_median=True, print_statistics=True, save=results_path+"/ncells_hist.png")

### setting up filters:
    epi.pp.set_filter(adata, "n_features", min_threshold=min_features, max_threshold=max_features)
    epi.pp.set_filter(adata, "n_cells", min_threshold=min_cells)
    adata = epi.pp.apply_filters(adata, verbose=True)

    print("normalizing data....")
# ## normalize
    epi.pp.normalize_total(adata)
    epi.pp.log1p(adata)


# ## PCA
    print("dimentionality reduction....")
    epi.pp.pca(adata, n_comps=30)
    n_comps = epi.pp.find_elbow(adata, use_log=True, show_anno=False)
    epi.pp.neighbors(adata, n_pcs=n_comps, method="umap")
    epi.tl.umap(adata)
    sc.tl.louvain(adata, resolution=1.5, key_added='louvain_r1.5')
    sc.tl.louvain(adata, resolution=0.5, key_added='louvain_r0.5')

    
    plt.rcParams["figure.figsize"]= (4, 4)
    epi.pl.umap(adata, color=[None,  'louvain_r1.5', 'louvain_r0.5', "cell_type", 'n_features', 'nucleosome_signal'], save=regions_id+"_results.png")
    
    ## extracting the data to save 
    umapdf = pd.DataFrame(adata.obsm['X_umap'], index= adata.obs.index)
    umapdf.columns = ["UMAP1", "UMAP2"]
    adata.obs['region'] = regions_id
    ##silhouette score
    adata.obs['silhouette_cellType'] = silhouette_samples(adata.obsm['X_umap'],  adata.obs['cell_type'])
    umapdf.join(adata.obs[["cell_type", "louvain_r1.5", "louvain_r0.5", "region"]]).to_csv(os.path.join(results_path, regions_id+"results_summary.csv"))
   
    adata.write(os.path.join(results_path, regions_id+'_data_processed.h5ad'))
    ## sil. scores
    sil_scores={}
    for tocalc in ["cell_type", "louvain_r1.5", "louvain_r0.5"]:
        sil_scores[tocalc] = [silhouette_score(adata.obsm['X_umap'],  adata.obs[tocalc])]
    pd.DataFrame(sil_scores, index=[regions_id]).to_csv(os.path.join(results_path, 'silhouette_scores.csv'))
    print("Done!")

if __name__ ==  '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Program canceled by user!")
        sys.exit(1)



