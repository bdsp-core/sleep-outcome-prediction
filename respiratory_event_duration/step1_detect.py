from collections import defaultdict
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


keys = ['HashID', 'DOVshifted']

df = pd.read_excel('../mastersheet_outcome_deid.xlsx')
df_path = pd.read_csv('../hypoxia_burden/paths.csv')
df = pd.concat([df, df_path], axis=1)
df = df.dropna(subset=keys+['SignalPath', 'AnnotPath']).reset_index(drop=True)

df_sanity_check = pd.read_csv('../hypoxia_burden/dataset_AHI_AHI_annot_good.csv')
df_sanity_check['DOVshifted'] = pd.to_datetime(df_sanity_check.DOVshifted)
df = df.merge(df_sanity_check[keys], on=keys, how='inner', validate='1:1')

df_resp_label = pd.read_csv('../apnea_detection/all_resp_labels_good.zip', compression='zip')
sid_dov2respids = defaultdict(list)
for i in range(len(df_resp_label)):
    sid_dov2respids[(df_resp_label.HashID.iloc[i], df_resp_label.DOVshifted.iloc[i])].append(i)

df.loc[:, 'RED'] = 0
for i in tqdm(range(len(df))):
    hashid = df.HashID.iloc[i]
    dov = df.DOVshifted.iloc[i]
    dov2 = dov.strftime('%Y-%m-%d')
        
    df_ = df_resp_label.iloc[sid_dov2respids[(hashid,dov2)]]
    if len(df_)>0:
        df.loc[i, 'RED'] = df_.duration.mean()

df[keys+['RED']].to_csv('outcomes_RED.csv', index=False)
