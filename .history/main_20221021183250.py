'''
Author: Xinye Zhao xzhao429@gatech.edu
Date: 2022-10-21 17:49:29
LastEditors: Xinye Zhao xzhao429@gatech.edu
LastEditTime: 2022-10-21 18:32:50
FilePath: /scMODD/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# In[]
from scripts.scMODD import scMODD

# In[] 
root_dir = './demo_data/'
file_name = 'sc_data_10.csv'

scMODD(file_name, root_dir=root_dir, save_path='./demo_data/', model = 'NB', )

# In[]
# Evaluate results
meta_name = 'meta_sc_data_10.csv'
meta_data = pd.read_csv(os.path.join(root_dir, meta_name)).reset_index(drop=True)

y_label = meta_data['x'].copy()
y_label[y_label == 'doublet'] = 1
y_label[y_label == 'singlet'] = 0
y_label = [int(i) for i in y_label]

precision, recall, thresholds_pr = precision_recall_curve(y_label, y_pre)
fpr, tpr, thersholds_roc = roc_curve(y_label, y_pre, pos_label=1)
pr_auc = auc(recall, precision)
roc_auc = auc(fpr, tpr)
# %%
