'''
Author: Xinye Zhao xzhao429@gatech.edu
Date: 2022-10-21 17:50:55
LastEditors: Xinye Zhao xzhao429@gatech.edu
LastEditTime: 2022-10-21 22:48:09
FilePath: /scMODD/scripts/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
Author: Xinye Zhao xzhao429@gatech.edu
Date: 2022-10-21 17:50:55
LastEditors: Xinye Zhao xzhao429@gatech.edu
LastEditTime: 2022-10-21 22:44:41
FilePath: /scMODD/scripts/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import warnings
import pandas as pd
import os
warnings.filterwarnings("ignore")
plt.style.use('seaborn-white')

def roc_pr_plot(y_label, y_pre ,path=None, title=None):
    precision, recall, thresholds_pr = precision_recall_curve(y_label, y_pre)
    fpr, tpr, thersholds_roc = roc_curve(y_label, y_pre, pos_label=1)
    pr_auc = auc(recall, precision)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(fpr, tpr, 'r--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
     
    ax.set_xlim([0, 1.05])  
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - {}'.format(title))
    ax.legend(loc="center right")
    ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='-',c='b', lw=0.5)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(recall, precision, 'r--', label='PR (area = {0:.2f})'.format(pr_auc), lw=2)
     
    ax2.set_xlim([0, 1.05]) 
    ax2.set_ylim([0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')  
    ax2.set_title('PR Curve - {}'.format(title))
    ax2.legend(loc="center right")
    ax2.plot((0, 1), (1, 0), transform=ax2.transAxes, ls='-',c='b', lw=0.5)
    if path:
        fig.savefig(path)
def umap_preproc(matrix):
    adata = ad.AnnData(matrix)
    adata_filter_gene = ad.AnnData(matrix)
    print('Before filter genes: ', adata.shape)
    sc.pp.filter_genes(adata_filter_gene, min_cells=3)

    adata.raw = adata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    print(adata.var.highly_variable.shape)
    idxs = adata.var.highly_variable
    adata = adata[:, adata.var.highly_variable]
    print('After filter genes: ', adata.shape)

    adata_hvg = adata.raw.to_adata()[:, idxs]
    sc.tl.pca(adata, svd_solver='arpack')

    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    print(adata.X.shape, adata_hvg.X.shape, adata_filter_gene.X.shape)
    return adata, adata_hvg, adata_filter_gene
#! Use raw count data to build simulated doublets
def generate_dbl_vector(ct1_idx, ct2_idx, num, mtx):
    a = np.random.choice(ct1_idx, num, replace=True)
    b = np.random.choice(ct2_idx, num, replace=True)
    rand_mtx1 = mtx[a]
    rand_mtx2 = mtx[b]
    sampled_dbl_mtx = (rand_mtx1+rand_mtx2)
    rand = np.random.rand(sampled_dbl_mtx.shape[0], sampled_dbl_mtx.shape[1]) - 0.5
    sampled_dbl_mtx = np.round(sampled_dbl_mtx + rand)
    return sampled_dbl_mtx

def evaluate_scMODD(ground_truth, root_dir, scMODD_results, PLOT=False):
    y_pre = scMODD_results['score'].values
    meta_name = 'meta_sc_data_10.csv'
    ground_truth = pd.read_csv(os.path.join(root_dir, meta_name)).reset_index(drop=True)

    y_label = ground_truth['0'].copy()
    y_label[y_label == 'doublet'] = 1
    y_label[y_label == 'singlet'] = 0
    y_label = [int(i) for i in y_label]

    precision, recall, thresholds_pr = precision_recall_curve(y_label, y_pre)
    fpr, tpr, thersholds_roc = roc_curve(y_label, y_pre, pos_label=1)
    pr_auc = auc(recall, precision)
    roc_auc = auc(fpr, tpr)
    print('AUROC: {}', 'AUPRC: {}'.format(roc_auc, pr_auc))
    if PLOT:
        roc_pr_plot(y_label, y_pre)
