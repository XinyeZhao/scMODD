'''
Author: Xinye Zhao xzhao429@gatech.edu
Date: 2022-10-21 17:49:29
LastEditors: Xinye Zhao xzhao429@gatech.edu
LastEditTime: 2022-10-21 18:16:55
FilePath: /scMODD/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# In[]
from scripts.scMODD import scMODD

# In[] 
# Read data
root_dir = './demo_data/'
file_name = 'sc_data_10.csv'
scMODD(file_name, root_dir=root_dir, save_path='./demo_data/', model = 'NB', )

# %%
