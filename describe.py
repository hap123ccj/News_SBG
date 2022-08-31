import os
import pandas as pd
from tqdm import tqdm
import random

# df = pd.DataFrame()
path = 'THUCNews/'
folder_list = os.listdir(path)
for folder in folder_list:
    file_list = os.listdir(path+folder)
    print(len(file_list))
#     file_list = random.sample(file_list,1000)
#     for file in tqdm(file_list):
#         file_path = path+folder+'/'+file
#         f = open(file_path)
#         n = len(f.readlines())
#         f.close()
#         df.loc[file_path,'folder'] = folder
#         df.loc[file_path,'lines'] = n
# df.to_csv('file_lines.csv')
df = pd.read_csv('file_lines.csv')
for folder in folder_list:
    df_folder = df.loc[df['folder']==folder,'lines']
    print(folder+': '+str(df_folder.mean()))


