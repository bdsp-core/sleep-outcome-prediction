from collections import defaultdict
import datetime
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.insert(0, '../../sleep_general')
from mgh_sleeplab import *
from sleep_analysis_functions import compute_spo2_clean, compute_hypoxia_burden


base_folder = '/sbgenomics/project-files/bdsp-opendata-repository/PSG/data/S0001'
data_folders = os.listdir(base_folder)

"""
df_resp_label = pd.read_csv('all_resp_labels.zip', compression='zip')
ids = pd.isna(df_resp_label.duration)&df_resp_label.event.str.startswith('respiratory event - dur:')
df_resp_label.loc[ids, 'duration'] = df_resp_label.event[ids].str.extract(r'dur:([\s\d.]+)sec').astype(float).values
ids = pd.isna(df_resp_label.duration)&df_resp_label.event.str.startswith('* respiratory event - dur:')
df_resp_label.loc[ids, 'duration'] = df_resp_label.event[ids].str.extract(r'dur:([\s\d.]+)sec').astype(float).values
ids = (df_resp_label.duration<5)|(df_resp_label.duration>30)
df_resp_label.loc[ids, 'duration'] = np.nan

sid_dov2ids = defaultdict(list)
for i in tqdm(range(len(df_resp_label))):
    sid_dov2ids[(df_resp_label.HashID.iloc[i], df_resp_label.DOVshifted.iloc[i])].append(i)
"""

df = pd.read_excel('../mastersheet_outcome_deid.xlsx')
df['hypoxia_burden'] = np.nan
df['hypoxia_note'] = np.nan
save_cols = ['HashID', 'DOVshifted', 'hypoxia_burden', 'hypoxia_note']

p_spo2 = re.compile('s[pa]o2', re.IGNORECASE)
p_abd = re.compile('abd', re.IGNORECASE)
p_chest = re.compile('(?:chest|tho)', re.IGNORECASE)
import pdb;pdb.set_trace()
for i in tqdm(range(len(df))):
    try:
        sid = df.HashID.iloc[i]
        dov = df.DOVshifted.iloc[i]
        dov2 = dov.strftime('%Y-%m-%d')

        # load and prepare data
        signal_path, annot_path = get_path_from_bdsp(sid, dov, base_folder=base_folder, data_folders=data_folders, raise_error=False)
        
        with h5py.File(signal_path, 'r') as ff:
            fs = ff['recording']['samplingrate'][()].item()
            signal_labels = ff['hdr']['signal_labels'][()]
            channel_names = [''.join(map(chr, ff[signal_labels[j,0]][()].flatten())) for j in range(len(signal_labels))]
            
            channel_idx = [j for j in range(len(channel_names)) if re.match(p_spo2, channel_names[j])]
            assert len(channel_idx)==1, f'no or multiple SpO2 channel(s): {channel_names}'
            spo2 = ff['s'][:,channel_idx[0]]
            channel_idx = [j for j in range(len(channel_names)) if re.match(p_abd, channel_names[j])]
            assert len(channel_idx)==1, f'no or multiple ABD channel(s): {channel_names}'
            abd = ff['s'][:,channel_idx[0]]
            channel_idx = [j for j in range(len(channel_names)) if re.match(p_chest, channel_names[j])]
            assert len(channel_idx)==1, f'no or multiple CHEST channel(s): {channel_names}'
            chest = ff['s'][:,channel_idx[0]]
            
        import pdb;pdb.set_trace()
                    
        annot = pd.read_csv(annot_path)
        annot['event'] = annot.event.str.lower()
        sleep_ids = annot.event.str.startswith('sleep_stage_n')|annot.event.str.startswith('sleep_stage_r')
        hours_sleep = np.sum(sleep_ids)*30/3600
        
        # detect apnea
        resp = 
        data = pd.DataFrame(data=np.c_[spo2, resp], columns=['spo2', 'apnea'])

        dt_start = pd.Timestamp('2000-01-01 00:00:00')
        dt_end = dt_start + datetime.timedelta(seconds=(data.shape[0]-1) / fs)
        data.index = pd.date_range(start=dt_start, end=dt_end, periods=len(data))

        # compute hypoxia variables
        data = compute_spo2_clean(data, fs=int(round(fs)))
        data['spo2'] = data['spo2_clean']
        data['apnea_binary'] = np.isin(data['apnea'],[1,2,3,4]).astype(int)
        data['apnea_end'] = data['apnea_binary'].diff()==-1
        data, hypoxia_burden = compute_hypoxia_burden(data, fs, hours_sleep=hours_sleep, apnea_name='apnea')

        df.loc[i, 'hypoxia_burden'] = hypoxia_burden
        df.loc[i, 'hypoxia_note'] = 'all good'
        
        if i%100==0:
            df[save_cols].to_csv('outcomes_hypoxia2.csv', index=False)
        
    except Exception as e:
        print(i, sid, str(e))
        df.loc[i, 'hypoxia_note'] = str(e)
        continue
        
df[save_cols].to_csv('outcomes_hypoxia2.csv', index=False)
