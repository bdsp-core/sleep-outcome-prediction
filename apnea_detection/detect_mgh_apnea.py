import datetime
from collections import defaultdict
from itertools import groupby
import os
import re
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from apnea_detection import load_apnea_binary_model, detect_apnea
import sys
sys.path.insert(0, '/sbgenomics/workspace/sleep_general')
from mgh_sleeplab import *


use_gpu = True
apnea_model = load_apnea_binary_model('apnea_detection_model_binary', use_gpu=use_gpu, n_gpu=int(use_gpu))

df = pd.read_excel('../mastersheet_outcome_deid.xlsx')
df['DOVshifted'] = df.DOVshifted.dt.strftime('%Y-%m-%d')

df_resp_label = pd.read_csv('all_resp_labels.zip', compression='zip')
cols = list(df_resp_label.columns)
df_resp_label = df_resp_label.merge(df, on=['HashID', 'DOVshifted'], how='left')
df_resp_label = df_resp_label[~pd.isna(df_resp_label.BDSPPatientID)].reset_index(drop=True)[cols]

remove_ids = pd.isna(df_resp_label.duration)|(df_resp_label.duration<5)|(df_resp_label.duration>60)
sid_dov_to_remove = df_resp_label.loc[remove_ids, ['HashID', 'DOVshifted']].drop_duplicates(ignore_index=True)
sid_dov_to_remove['remove'] = 1
df_resp_label_to_keep = df_resp_label.merge(sid_dov_to_remove, on=['HashID', 'DOVshifted'], how='left')
df_resp_label_to_keep = df_resp_label_to_keep[pd.isna(df_resp_label_to_keep.remove)].reset_index(drop=True)[df_resp_label.columns]

df_res = pd.read_csv('resp_labels_WaveNet_old.csv')
df_res['DOVshifted'] = df_res['DOVshifted'].str.replace('−','-')
df_res['remove'] = 1
sid_dov_to_remove = sid_dov_to_remove[['HashID', 'DOVshifted']].merge(df_res[['HashID', 'DOVshifted', 'remove']], on=['HashID', 'DOVshifted'], how='left')
sid_dov_to_remove = sid_dov_to_remove[pd.isna(sid_dov_to_remove.remove)].reset_index(drop=True)

#"""
base_folder = '/sbgenomics/project-files/bdsp-opendata-repository/PSG/data/S0001'
data_folders = os.listdir(base_folder)
n_err = 0
newFs = 10
df_res = defaultdict(list)
p_abd = re.compile('abd', re.IGNORECASE)
for i in tqdm(range(100)):#len(sid_dov_to_remove))):
    try:
        sid = sid_dov_to_remove.HashID.iloc[i]
        dov = datetime.datetime.strptime(sid_dov_to_remove.DOVshifted.iloc[i], '%Y-%m-%d')

        # load and prepare data
        signal_path, annot_path = get_path_from_bdsp(sid, dov, base_folder=base_folder, data_folders=data_folders, raise_error=False)

        with h5py.File(signal_path, 'r') as ff:
            Fs = ff['recording']['samplingrate'][()].item()
            signal_labels = ff['hdr']['signal_labels'][()]
            channel_names = [''.join(map(chr, ff[signal_labels[j,0]][()].flatten())) for j in range(len(signal_labels))]

            year = int(ff['recording']['year'][()])
            month = int(ff['recording']['month'][()])
            day = int(ff['recording']['day'][()])
            hour = int(ff['recording']['hour'][()])
            minute = int(ff['recording']['minute'][()])
            second = int(ff['recording']['second'][()])
            t0 = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
            channel_idx = [j for j in range(len(channel_names)) if re.match(p_abd, channel_names[j])]
            assert len(channel_idx)==1, f'no or multiple ABD channel(s): {channel_names}'
            abd = ff['s'][:,channel_idx[0]]

        annot = annotations_preprocess(pd.read_csv(annot_path), Fs, t0=t0)
        sleep_stages = vectorize_sleep_stages(annot, len(abd))

        apnea, apnea_prob = detect_apnea(apnea_model, abd, sleep_stages, Fs, use_gpu=use_gpu, newFs=newFs)
        apnea[np.isnan(apnea)] = -1

        df_res['HashID'].append(sid)
        df_res['DOVshifted'].append(dov.strftime('%Y−%m−%d'))
        df_res['epoch'].append(1)
        df_res['time'].append(t0.strftime('%H:%M:%S'))
        df_res['duration'].append(1/newFs)
        df_res['event'].append('placeholder - WaveNet')
        cc = 0
        for k,l in groupby(apnea):
            ll = len(list(l))
            if k==1:
                df_res['HashID'].append(sid)
                df_res['DOVshifted'].append(dov.strftime('%Y−%m−%d'))
                df_res['epoch'].append(int(cc/newFs/30)+1)
                df_res['time'].append((t0+datetime.timedelta(seconds=cc/newFs)).strftime('%H:%M:%S'))
                df_res['duration'].append(ll/newFs)
                df_res['event'].append('respevent - obstructiveapnea - 1 - WaveNet predicted')
            cc += ll
        if i%10==0:
            df_res_ = pd.DataFrame(data=df_res)
            df_res_.to_csv('resp_labels_WaveNet0-100.csv', index=False)
    except Exception as ee:
        n_err += 1
        print(f'[{n_err}] {sid}, {dov}: {str(ee)}')
import pdb;pdb.set_trace()
df_res = pd.DataFrame(data=df_res)
df_res.to_csv('resp_labels_WaveNet.csv', index=False)
#"""
#df_res = pd.read_csv('resp_labels_WaveNet.csv')
df_res = pd.concat([df_resp_label_to_keep, df_res], axis=0, ignore_index=True)
df_res.to_csv('all_resp_labels_with_WaveNet.zip', index=False, compression='zip')

