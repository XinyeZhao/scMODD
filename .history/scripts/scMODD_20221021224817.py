'''
Author: Xinye Zhao xzhao429@gatech.edu
Date: 2022-10-21 17:57:41
LastEditors: Xinye Zhao xzhao429@gatech.edu
LastEditTime: 2022-10-21 22:48:17
FilePath: /scMODD/scripts/scMODD.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import scanpy as sc
from scipy import sparse
import itertools
import warnings
warnings.filterwarnings("ignore")
plt.style.use('seaborn-white')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from .utils import *
from .model import *

def scMODD(file_name, root_dir, save_path, this_resolution=0.8, model = 'NB', ):
  print(file_name)
  simulated_data = sparse.load_npz(os.path.join(root_dir, file_name.split('.')[0]+'.npz')).todense()
  filtered, adata_hvg, adata_filter_gene = umap_preproc(simulated_data)
  sc.tl.leiden(filtered, resolution=this_resolution)
  leiden_lst = filtered.obs['leiden']
  print(np.unique(leiden_lst).shape)

  for name, this_mtx in {'hvg_gene': adata_hvg}.items():
    np_array_transform = np.array(this_mtx.X)
    print('Now Running: ', np_array_transform.shape)
    search_lst = leiden_lst

    matrix = np_array_transform
    ct_idx = {}
    # create index dict for each cell type
    for ct in np.unique(search_lst):
        ct_idx[ct] = search_lst[search_lst == ct].index

    # pre_compute sgl theta and miu
    nb_sgl_param_dict = {}
    for this_ct, this_idx in ct_idx.items():
        this_mtx = np.array(matrix)[[int(i) for i in this_idx]]
        all_miu, all_var, all_theta = NB_pre(this_mtx)
        nb_sgl_param_dict[this_ct] = [all_miu, all_var, all_theta]

    # pre_compute dbl theta and miu
    doublet_ct_lst = None
    nb_dbl_param_dict = {}
    artificial_dbl_clt = np.array([])
    doublet_ct_lst = list(itertools.combinations(np.unique(search_lst), 2))
    for ct1, ct2 in doublet_ct_lst:
        ct1_dx, ct2_idx = [int(i) for i in ct_idx[ct1]], [int(i) for i in ct_idx[ct2]]
        artifitial_cmb_mtx = generate_dbl_vector(ct1_dx, ct2_idx, num=500, mtx = matrix)
        if artificial_dbl_clt.shape[0] == 0:
            artificial_dbl_clt = artifitial_cmb_mtx
        else:
            artificial_dbl_clt = np.vstack((artificial_dbl_clt, artifitial_cmb_mtx))
        cmb_miu, cmb_var, cmb_theta = NB_pre(artifitial_cmb_mtx)
        dbl_type = (ct1+'+'+ct2) if ((ct1) < (ct2)) else (ct2+'+'+ct1)
        nb_dbl_param_dict[dbl_type] = [cmb_miu, cmb_var, cmb_theta]

    scores_df_fast = pd.DataFrame(index=range(matrix.shape[0]))
    scores_artificial_dbl = pd.DataFrame(index=range(artificial_dbl_clt.shape[0]))

    for sgl, sgl_param in nb_sgl_param_dict.items():
        miu, var, theta = sgl_param
        this_score_all = NB_model(matrix, theta, miu)
        this_score =  np.mean(this_score_all, axis = 1)
        scores_df_fast[sgl] = this_score

        this_score_all_artificial = NB_model(artificial_dbl_clt, theta, miu)
        this_score_artificial =  np.mean(this_score_all_artificial, axis = 1)
        scores_artificial_dbl[sgl] = this_score_artificial

    for dbl, dbl_param in nb_dbl_param_dict.items():
        dbl_miu, dbl_var, dbl_theta = dbl_param
        this_score_all = (NB_model(matrix, dbl_theta, dbl_miu))
        this_score = np.mean(this_score_all, axis = 1)
        scores_df_fast[dbl] = this_score

        this_score_all_artificial = (NB_model(artificial_dbl_clt, dbl_theta, dbl_miu))
        this_score_artificial = np.mean(this_score_all_artificial, axis = 1)
        scores_artificial_dbl[dbl] = this_score_artificial

    combined_mtx = pd.concat((scores_df_fast, scores_artificial_dbl), axis=0)
    y_lables = [0]*scores_df_fast.shape[0]+[1]*scores_artificial_dbl.shape[0]
    clf= MLPClassifier(random_state=1, max_iter=300).fit(combined_mtx, y_lables)

    rltss = clf.predict_proba(scores_df_fast)
    y_pre = rltss[:, 1]
    print('Results is saved to :{}'.format(os.path.join(save_path, 'scMODD_predicted_score.csv')))
    pd.DataFrame({"score": y_pre}).to_csv(os.path.join(save_path, 'scMODD_predicted_score.csv'))