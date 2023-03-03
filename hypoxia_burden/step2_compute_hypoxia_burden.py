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


#base_folder = '/sbgenomics/project-files/bdsp-opendata-repository/PSG/data/S0001'
#data_folders = os.listdir(base_folder)
keys = ['HashID', 'DOVshifted']

df = pd.read_excel('../mastersheet_outcome_deid.xlsx')
df_path = pd.read_csv('paths.csv')
df = pd.concat([df, df_path], axis=1)
df = df.dropna(subset=keys+['SignalPath', 'AnnotPath']).reset_index(drop=True)

df_sanity_check = pd.read_csv('dataset_AHI_AHI_annot_good.csv')
df_sanity_check['DOVshifted'] = pd.to_datetime(df_sanity_check.DOVshifted)
df = df.merge(df_sanity_check[keys], on=keys, how='inner', validate='1:1')

df_resp_label = pd.read_csv('../apnea_detection/all_resp_labels_good.zip', compression='zip')
sid_dov2respids = defaultdict(list)
for i in range(len(df_resp_label)):
    sid_dov2respids[(df_resp_label.HashID.iloc[i], df_resp_label.DOVshifted.iloc[i])].append(i)

#df2 = pd.read_csv('outcomes_hypoxia2.csv')
#df2 = df2[df2.hypoxia_note!='all good'].iloc[4:].reset_index(drop=True)
#df = df[np.in1d(df.HashID, df2.HashID)].reset_index(drop=True)

df['hypoxia_burden'] = np.nan
df['hypoxia_note'] = np.nan
save_cols = ['HashID', 'DOVshifted', 'hypoxia_burden', 'hypoxia_note']

p_spo2 = re.compile('s[pa]o2', re.IGNORECASE)
for i in tqdm(range(len(df))):
    #try:
    sid = df.HashID.iloc[i]
    dov = df.DOVshifted.iloc[i]
    dov2 = dov.strftime('%Y-%m-%d')
    ahi = df.AHI.iloc[i]

    # load and prepare data
    #signal_path, annot_path = get_path_from_bdsp(sid, dov, base_folder=base_folder, data_folders=data_folders, raise_error=False)
    signal_path = df.SignalPath.iloc[i]
    annot_path = df.AnnotPath.iloc[i]

    with h5py.File(signal_path, 'r') as ff:
        fs = ff['recording']['samplingrate'][()].item()
        signal_labels = ff['hdr']['signal_labels'][()]
        channel_names = [''.join(map(chr, ff[signal_labels[j,0]][()].flatten())) for j in range(len(signal_labels))]

        channel_idx = [j for j in range(len(channel_names)) if re.match(p_spo2, channel_names[j])]
        assert len(channel_idx)==1, f'no or multiple SpO2 channel(s): {channel_names}'
        spo2 = ff['s'][:,channel_idx[0]]

        year = int(ff['recording']['year'][()])
        month = int(ff['recording']['month'][()])
        day = int(ff['recording']['day'][()])
        hour = int(ff['recording']['hour'][()])
        minute = int(ff['recording']['minute'][()])
        second = int(ff['recording']['second'][()])
        t0 = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)

    annot = pd.read_csv(annot_path)
    annot['event'] = annot.event.str.lower()
    sleep_ids = annot.event.str.contains('stage_n')|annot.event.str.contains('stage_r')
    tst = np.sum(sleep_ids)*30/3600

    df_apnea_label = df_resp_label.iloc[sid_dov2respids[(sid, dov2)]]

    resp = vectorize_respiratory_events(annotations_preprocess(df_apnea_label, fs, t0=t0), len(spo2))
    data = pd.DataFrame(data=np.c_[spo2, resp], columns=['spo2', 'apnea'])

    dt_start = pd.Timestamp('2000-01-01 00:00:00')
    dt_end = dt_start + datetime.timedelta(seconds=(data.shape[0]-1) / fs)
    data.index = pd.date_range(start=dt_start, end=dt_end, periods=len(data))

    # compute hypoxia variables
    data = compute_spo2_clean(data, fs=int(round(fs)))
    data['spo2'] = data['spo2_clean']
    data['apnea_binary'] = np.isin(data['apnea'],[1,2,3,4]).astype(int)
    data['apnea_end'] = data['apnea_binary'].diff()==-1
    data, hypoxia_burden = compute_hypoxia_burden(data, fs, hours_sleep=tst, apnea_name='apnea')

    df.loc[i, 'hypoxia_burden'] = hypoxia_burden
    df.loc[i, 'hypoxia_note'] = 'all good'

    if i%10==0:
        df[save_cols].to_csv('outcomes_hypoxia3.csv', index=False)
        
    #except Exception as e:
    #    print(i, sid, str(e))
    #    df.loc[i, 'hypoxia_note'] = str(e)
        
df[save_cols].to_csv('outcomes_hypoxia3.csv', index=False)
