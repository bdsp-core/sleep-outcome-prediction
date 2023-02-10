"""
check if in MGH dataset, the resp annotation is consistent with AHI?
i.e., no missing resp annotation?
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.insert(0, '/sbgenomics/workspace/sleep_general')
from mgh_sleeplab import get_path_from_bdsp

base_folder = '/sbgenomics/project-files/bdsp-opendata-repository/PSG/data/S0001'
data_folders = os.listdir(base_folder)

df = pd.read_excel('../mastersheet_outcome_deid.xlsx')
df['AHI_annot'] = np.nan

for i in tqdm(range(len(df))):
    sid = df.HashID.iloc[i]
    dov = df.DOVshifted.iloc[i]

    signal_path, annot_path = get_path_from_bdsp(sid, dov, base_folder=base_folder, data_folders=data_folders, raise_error=False)
    if annot_path is None:
        continue
    annot = pd.read_csv(annot_path)
    annot['event'] = annot.event.str.lower()
    if annot.event.str.contains('redact').any():
        continue

    resp_ids = annot.event.str.contains('resp')&annot.event.str.contains('event')&annot.event.str.contains('pnea')
    sleep_ids = annot.event.str.startswith('sleep_stage_n')|annot.event.str.startswith('sleep_stage_r')
    tst = np.sum(sleep_ids)*30/3600

    if tst>0:
        df.loc[i, 'AHI_annot'] = resp_ids.sum() / tst

df[['HashID', 'DOVshifted', 'AHI', 'AHI_annot']].to_excel('dataset_AHI_AHI_annot.xlsx', index=False)

import matplotlib.pyplot as plt
plt.close()
plt.scatter(df.AHI, df.AHI_annot, s=5, c='k')
plt.plot([0,100], [0,100], c='r')
plt.xlabel('AHI')
plt.ylabel('AHI annot')
plt.tight_layout()
plt.savefig('AHI_vs_AHI_annot.png')

