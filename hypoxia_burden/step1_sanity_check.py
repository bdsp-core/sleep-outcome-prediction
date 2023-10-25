"""
check if in MGH dataset, the resp annotation is consistent with AHI?
i.e., no missing resp annotation?
"""
from collections import defaultdict
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import sys
sys.path.insert(0, '/sbgenomics/workspace/sleep_general')
from mgh_sleeplab import get_path_from_bdsp

suffix = '_good'

df_resp_label = pd.read_csv(f'../apnea_detection/all_resp_labels{suffix}.zip', compression='zip')
sid_dov2ids = defaultdict(list)
for i in range(len(df_resp_label)):
    sid_dov2ids[(df_resp_label.HashID.iloc[i], df_resp_label.DOVshifted.iloc[i])].append(i)

df = pd.read_excel('../mastersheet_outcome_deid.xlsx')
df['AHI_annot'] = np.nan

"""
base_folder = '/sbgenomics/project-files/bdsp-opendata-repository/PSG/data/S0001'
data_folders = os.listdir(base_folder)
signal_paths = []
annot_paths = []
for i in tqdm(range(len(df))):
    sid  = df.HashID.iloc[i]
    dov  = df.DOVshifted.iloc[i]
    signal_path, annot_path = get_path_from_bdsp(sid, dov, base_folder=base_folder, data_folders=data_folders, raise_error=False)
    signal_paths.append(signal_path)
    annot_paths.append(annot_path)
df_path = pd.DataFrame(data={'SignalPath':signal_paths, 'AnnotPath':annot_paths})
df_path.to_csv('paths.csv', index=False)
"""
df_path = pd.read_csv('paths.csv')

for i in tqdm(range(len(df))):
    sid  = df.HashID.iloc[i]
    dov  = df.DOVshifted.iloc[i]
    dov2 = dov.strftime('%Y-%m-%d')
    ahi = df.AHI.iloc[i]

    #signal_path, annot_path = get_path_from_bdsp(sid, dov, base_folder=base_folder, data_folders=data_folders, raise_error=False)
    signal_path = df_path.SignalPath.iloc[i]
    annot_path = df_path.AnnotPath.iloc[i]
    if pd.isna(signal_path) or pd.isna(annot_path):
        continue
    annot = pd.read_csv(annot_path)
    annot['event'] = annot.event.str.lower()

    sleep_ids = annot.event.str.startswith('sleep_stage_n')|annot.event.str.startswith('sleep_stage_r')
    tst = np.sum(sleep_ids)*30/3600
    if tst==0:
        continue

    # AHI from original labels
    ahi_annot1 = len(df_resp_label.iloc[sid_dov2ids[(sid, dov2)]])/tst
    """
    # AHI from original labels
    ahi_annot2 = np.sum(annot.event.str.contains('pnea'))/tst

    if not (ahi_annot1==0 and ahi_annot2==0 and ahi>0):
        if abs(ahi_annot1-ahi)<abs(ahi_annot2-ahi):
            df.loc[i, 'AHI_annot'] = ahi_annot1
        else:
            df.loc[i, 'AHI_annot'] = ahi_annot2
    """
    df.loc[i, 'AHI_annot'] = ahi_annot1

cols = ['HashID', 'DOVshifted', 'AHI', 'AHI_annot']
df = df[cols]
df = df.dropna().reset_index(drop=True)
df = df[((df.AHI>0)&(df.AHI_annot>0))|((df.AHI==0)&(df.AHI_annot==0))].reset_index(drop=True)
df = df[np.abs(df.AHI-df.AHI_annot)<5].reset_index(drop=True)
df.to_csv(f'dataset_AHI_AHI_annot{suffix}.csv', index=False)
print(pearsonr(df.AHI, df.AHI_annot))

import matplotlib.pyplot as plt
plt.close()
plt.scatter(df.AHI, df.AHI_annot, s=5, c='k')
plt.plot([0,100], [0,100], c='r')
plt.xlabel('AHI')
plt.ylabel('AHI annot')
plt.tight_layout()
plt.savefig(f'AHI_vs_AHI_annot{suffix}.png')

