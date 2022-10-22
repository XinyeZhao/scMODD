# In[]
from sympy import root
from scripts.scMODD import scMODD
import pandas as pd
from scripts.utils import *
# In[] 
root_dir = './demo_data/'
file_name = 'example_data.csv'

scMODD(file_name, root_dir=root_dir, save_path='./demo_data/', model = 'NB')

# In[]
# If the ground truth is available, then the following scripts can be used to evaluate scMODD performance
ground_truth = pd.read_csv('./demo_data/ground_truth_for_example_data.csv')
scMODD_results = pd.read_csv('./demo_data/scMODD_predicted_score.csv')
PLOT = True
# evaluate_scMODD(ground_truth=ground_truth, scMODD_results=scMODD_results)
y_pre = scMODD_results['score'].values

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
# %%
