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


use_gpu = False
apnea_model = load_apnea_binary_model('apnea_detection_model_binary', use_gpu=use_gpu, n_gpu=int(use_gpu))

df = pd.read_excel('/sbgenomics/workspace/sleep-outcome-prediction/mastersheet_outcome_deid.xlsx')

base_folder = '/sbgenomics/project-files/bdsp-opendata-repository/PSG/data/S0001'
data_folders = os.listdir(base_folder)

p_abd = re.compile('abd', re.IGNORECASE)
import pdb;pdb.set_trace()
for i in tqdm(range(len(df))):
    #try:
    sid = df.HashID.iloc[i]
    dov = df.DOVshifted.iloc[i]

    # load and prepare data
    signal_path, annot_path = get_path_from_bdsp(sid, dov, base_folder=base_folder, data_folders=data_folders, raise_error=False)

    with h5py.File(signal_path, 'r') as ff:
        Fs = ff['recording']['samplingrate'][()].item()
        signal_labels = ff['hdr']['signal_labels'][()]
        channel_names = [''.join(map(chr, ff[signal_labels[j,0]][()].flatten())) for j in range(len(signal_labels))]

        channel_idx = [j for j in range(len(channel_names)) if re.match(p_abd, channel_names[j])]
        assert len(channel_idx)==1, f'no or multiple ABD channel(s): {channel_names}'
        abd = ff['s'][:,channel_idx[0]]
        
    annot = annotations_preprocess(pd.read_csv(annot_path), Fs)
    sleep_stages = vectorize_sleep_stages(annot, len(abd))
        
    newFs = 10
    apnea, apnea_prob = detect_apnea(apnea_model, abd, sleep_stages, Fs, use_gpu=use_gpu, newFs=newFs)