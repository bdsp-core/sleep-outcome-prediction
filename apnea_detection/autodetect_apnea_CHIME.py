from itertools import groupby
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch as th
import torch.nn as nn
import sys
model_dir = '/data/Dropbox (Partners HealthCare)/SleepCAISRNetwork/step2_detect/apnea_detect_models'
sys.path.insert(0, model_dir)
from perf_functions import smooth
from segment_signal_Update import clip_z_normalize
sys.path.insert(0, os.path.join(model_dir, 'wavenet_vocoder'))
from wavenet import WaveNet
from read_dataset import load_CHIMES_dataset


def load_model(model_type, use_gpu=False, n_gpu=0):
    all_model_types = ['Binary', 'Multiclass']
    assert model_type in all_model_types, f'model_type can only be one of {all_model_types}'
    model_path = os.path.join(model_dir, model_type)
    if model_type=='Multiclass':
        class_tags = ['no event', 'event']
    elif model_type=='Binary':
        class_tags = ['NoEvent', 'Event']
    n_cl = len(class_tags)
    
    # load all models
    names = ['Boost1', 'Boost2', 'Boost3', 'Boost4', 'Boost5', model_type]
    models = []
    for name in names:
        # load model and reset cpu module
        model = th.load(os.path.join(model_dir, model_type, f'CVall_{name}.pth')).module.cpu()
        if use_gpu:
            model = model.cuda()
            if n_gpu > 1:
                model = nn.DataParallel(model, device_ids=list(range(n_gpu)))
        # set model to evaluation mode
        model.eval()
        models.append(model)
    boost_models = models[:-1]
    final_model = models[-1]
    
    # retrieve thresholds
    thres_path = os.path.join(model_dir, f'{model_type}_thresholds_CVall.csv')
    thresholds = pd.read_csv(thres_path, sep='\t').values[0]
    
    return {'boost':boost_models, 'final':final_model, 'thres':thresholds, 'class_tags':class_tags, 'n_class':n_cl}
    

def compute_ahi(apnea, sleep_stage, Fs):
    apnea2 = np.array(apnea)
    apnea2[np.isnan(apnea)] = -999
    cc = 0
    apnea_count = 0
    for k,l in groupby(apnea2):
        ll = len(list(l))
        if k==1 and np.any(sleep_stage[cc:cc+ll]<5):
            apnea_count += 1
        cc += ll
    tst = np.sum(sleep_stage<5)/Fs/3600
    ahi = apnea_count / tst
    return ahi            
    
    
if __name__=='__main__':
    # load mastersheet
    df = pd.read_csv('/data/brain_age_descriptive/mycode/data/CHIMES_data_list.txt', sep='\t')
    # load model
    use_gpu = True
    model = load_model('Binary', use_gpu=use_gpu)
    
    #df = df.iloc[[15, 103, 278, 305]].reset_index(drop=True)
    # for each record, detect
    df_ahi = []
    err_msgs = {}
    for i in tqdm(range(len(df))):
        try:
            signal, sleep_stage, params = load_CHIMES_dataset(df.signal_file.iloc[i], df.label_file.iloc[i], channels=['Abd'])
            signal = signal[0]
            Fs = params['Fs']
            
            apnea, apnea_prob = detect_apnea(model, signal, sleep_stage, Fs, use_gpu=use_gpu)
            #TODO
            signal = signal[::10]
            sleep_stage = sleep_stage[::10]
            Fs = 10
            
            ahi = compute_ahi(apnea, sleep_stage, Fs)
            df_ahi.append(ahi)
        except Exception as ee:
            df_ahi.append(np.nan)
            err_msgs[i] = str(ee)
            print(err_msgs[i])
    
    df['AHI'] = df_ahi
    df.to_csv('autodetect_apnea_CHIME_result.csv', index=False)
    
