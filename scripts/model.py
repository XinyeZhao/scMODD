'''
Author: Xinye Zhao xzhao429@gatech.edu
Date: 2022-10-21 17:47:49
LastEditors: Xinye Zhao xzhao429@gatech.edu
LastEditTime: 2022-10-21 17:48:26
FilePath: /scMODD/scripts/model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma, gammaln
import warnings
from scipy.optimize import curve_fit
import numpy.matlib
warnings.filterwarnings("ignore")
plt.style.use('seaborn-white')

# NB model
def func_fit(x, a):
    return a* x
def NB_pre(this_mtx, CURVE_FIT=True):
    all_miu = np.mean(this_mtx, axis=0)
    all_var = np.var(this_mtx, axis=0)
    # curve fitting
    X1 = (all_var-all_miu)
    Y1 = np.power(all_miu, 2)
    popt, _ = curve_fit(func_fit, X1, Y1)
    all_theta = np.repeat(popt, this_mtx.shape[1])
    return all_miu, all_var, all_theta

def NB_model(y, theta, miu):
    eps = 1e-10
    #! log(0) -> nan
    log_t1 = theta * (np.log(theta+eps) - np.log(theta+miu+eps))
    log_t2 = y * (np.log(miu+eps) - np.log(miu+theta+eps))
    log_t3 = gammaln(y+theta+eps) - gammaln(y+1+eps) - gammaln(theta+eps)
    #! log(0) -> nan
    results =log_t1 + log_t2 + log_t3
    return results

def generate_classify_param(score_lst, rank_lst=[1, 2]):
    rank_a, rank_b = rank_lst
    max_idxa, max_idxb = np.argsort(score_lst)[-rank_a], np.argsort(score_lst)[-rank_b]
    max_a, max_b = np.sort(score_lst)[-rank_a], np.sort(score_lst)[-rank_b]
    return max_a, max_b, max_idxa, max_idxb

# ZINB model
def func_fit(x, a):
    return a* x
def ZINB_pre(matrix):
    this_pi = np.mean(matrix == 0, axis=0)
    this_pi[this_pi == 0] = 1e-10
    this_pi[this_pi == 1] = 1 - 1e-10
    matrix = np.where(matrix, matrix, np.nan)
    this_mean = np.nanmean(matrix, axis=0)
    this_var = np.nanvar(matrix, axis=0)
    filter_flag = ~np.isnan(this_mean) & ~np.isnan(this_var)
    Y1 = np.power(this_mean[filter_flag], 2)
    X1 = this_var[filter_flag] - this_mean[filter_flag]
    popt, _ = curve_fit(func_fit, X1, Y1)
    this_theta = np.repeat(popt, matrix.shape[1])
    this_mean[np.isnan(this_mean)] = 1e-10
    return this_mean, this_theta, this_pi

def ZINB_model(y, theta, miu, pi):
    this_nb = NB_model(y, theta, miu)
    pi = numpy.matlib.repmat(pi, y.shape[0], 1)
    this_nb[y==0] = np.log(np.exp(this_nb[y==0]) * (1-pi[y==0])+pi[y==0])
    this_nb[y!=0] = np.log(1-pi[y!=0]) + (this_nb[y!=0])

    return this_nb