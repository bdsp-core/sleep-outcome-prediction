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
df['RED'] = np.nan

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
    if 'duration' not in annot.columns:
        continue

    resp_ids = annot.event.str.contains('resp')&annot.event.str.contains('event')&annot.event.str.contains('pnea')
    df.loc[i, 'RED'] = annot.duration[resp_ids].mean()

df.loc[df.RED<5, 'RED'] = np.nan
df[['HashID', 'DOVshifted', 'RED']].to_excel('dataset_RED.xlsx', index=False)

