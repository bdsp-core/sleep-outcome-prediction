import pickle
import numpy as np
import scipy.io as sio
import pandas as pd
from tqdm import tqdm
import torch as th
import torch.nn as nn
import sys
sys.path.insert(0, '../mycode')
from read_dataset import *
sys.path.insert(0, '../mycode-figure1')
from detect_apnea import detect_apnea
sys.path.insert(0, 'wavenet_vocoder')
from wavenet import WaveNet


def get_artifact_indicator_lda(model, thres, specs, specs_db, freqs, spec_db_avg_low_thres):
    """
    model:      the trained LDA model
    thres:      this is a dictionary {'1%': a number, '5%': a number, '10%': a number}
    specs:      spectrogram in power, numpy array, shape = (#epochs, #channel, #freq)
    specsi_db:  spectrogram in decibel, numpy array, shape = (#epochs, #channel, #freq)
    freqs:      frequency values, numpy array, shape = (#freq,)

    This function first compute the two features: total power, and 2nd order diff (squareness);
    then feed the features into the model,
    and then it outputs a dictionary {'1%': a boolean array, '5%': a boolean array, '10%': a boolean array}.
    Each boolean array has shape=(#epoch, #channel), where True means artifact.
    """
    # compute features
    
    # feature: total power
    shape = specs.shape
    specs = specs.reshape(-1, shape[-1])
    tp = specs.sum(axis=1) * (freqs[1] - freqs[0])
    tp_db = 10 * np.log10(tp)

    # feature: 2nd order diff for measuring the squareness of spectrum
    specs_db = specs_db.reshape(-1, shape[-1])
    specs_db_n = specs_db / specs_db.std(axis=1, keepdims=True)
    diff2 = np.abs(np.diff(np.diff(specs_db_n, axis=1), axis=1)).max(axis=1)
    diff2_log = np.log(diff2)

    X = np.c_[tp_db, diff2_log]
    yp = model.decision_function(X)

    res = {}
    for k, v in thres.items():
        res[k] = (yp >= v)|(specs_db.mean(axis=1)<=spec_db_avg_low_thres)
        res[k] = res[k].reshape(shape[:2])

    return res
    
    
def load_apnea_binary_model(model_path, use_gpu=False, n_gpu=0):
    class_tags = ['NoEvent', 'Event']
    n_cl = len(class_tags)
    
    # load all models
    names = ['Boost1', 'Boost2', 'Boost3', 'Boost4', 'Boost5', 'Binary']
    models = []
    for name in names:
        # load model and reset cpu module
        model = th.load(os.path.join(model_path, f'CVall_{name}.pth')).module.cpu()
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
    thres_path = os.path.join(model_path, f'Binary_thresholds_CVall.csv')
    thresholds = pd.read_csv(thres_path, sep='\t').values[0]
    
    return {'boost':boost_models, 'final':final_model, 'thres':thresholds, 'class_tags':class_tags, 'n_class':n_cl}
    
    
def main():
    use_gpu = True
    
    df = pd.read_excel('../mycode/all_data_paths.xlsx')
    df = df[df.Dataset=='CHIMES'].reset_index(drop=True)

    # load artifact LDA model
    with open('artifact_model_LDA.pickle', 'rb') as ff:
        res = pickle.load(ff)
    artifact_model = res['model']
    artifact_thres = {'10%':res['thres_FNR10%'], 'balanced':res['best_thres']}
    spec_db_avg_low_thres = res['spec_db_avg_low_thres']

    apnea_model = load_apnea_binary_model('apnea_detection_model_binary', use_gpu=use_gpu, n_gpu=int(use_gpu))
    
    for i in tqdm(range(len(df))):
        feature_path = df.feature_file.iloc[i]
        if not os.path.exists(feature_path):
            continue
        keys = sio.whosmat(feature_path)
        keys = [x[0] for x in keys]
        if 'artifact_indicator' in keys and 'apnea_indicator' in keys:
            continue
        feat_mat = sio.loadmat(feature_path)
        dataset = df.Dataset.iloc[i]
        
        if 'predicted_sleep_stages_smoothed' in feat_mat:
            sleep_stages = np.argmax(feat_mat['predicted_sleep_stages_smoothed'], axis=1)+1
        else:
            sleep_stages = feat_mat['sleep_stages'].flatten()
        freqs = feat_mat['EEG_frequency'].flatten()
        specs = feat_mat['EEG_specs']
        specs = specs.transpose(0,2,1)
        specs_db = 10*np.log10(specs)
        artifact_indicator = get_artifact_indicator_lda(artifact_model, artifact_thres, specs, specs_db, freqs, spec_db_avg_low_thres)
        artifact_indicator = np.any(artifact_indicator['balanced'], axis=1).astype(int)
        #plt.close()
        #fig=plt.figure()
        #ax1=fig.add_subplot(311)
        #ax1.plot(sleep_stages)
        #ax2=fig.add_subplot(312,sharex=ax1)
        #ax2.imshow(specs_db.mean(axis=1).T,aspect='auto',origin='lower',cmap='turbo',vmin=-10,vmax=20)
        #ax3=fig.add_subplot(313,sharex=ax1)
        #ax3.plot(artifact_indicator)
        #plt.show()
        
        if dataset=='MGH':
            channels = ['ABD']
            abd, sleep_stages, params = load_MGH_dataset(df.signal_file.iloc[i], df.label_file.iloc[i], channels=channels)
        elif dataset=='CHIMES':
            channels = ['ABD']
            # need to adjust /home/sunhaoqi/.local/lib/python3.7/site-packages/mne/io/edf/edf.py
            # Line 633
            # day, month, year ----> year, month, day
            abd, sleep_stage, params = load_CHIMES_dataset(df.signal_file.iloc[i], df.label_file.iloc[i], channels=channels)
        elif dataset=='ChicagoPediatric':
            channels = ['ABD']
            abd, sleep_stage, params = load_ChicagoPediatric_dataset(df.signal_file.iloc[i], df.label_file.iloc[i], channels=channels)
            
        Fs = params['Fs']
        abd = abd[0]
        newFs = 10
        
        try:
            apnea, apnea_prob = detect_apnea(apnea_model, abd, sleep_stages, Fs, use_gpu=use_gpu, newFs=newFs)
            
            # convert into which 30s-epoch has apnea
            assert int(Fs)//int(newFs) == Fs/newFs
            apnea = np.repeat(apnea, Fs//newFs)
            apnea = np.r_[apnea, np.zeros(len(abd)-len(apnea))]
            seg_start_ids = feat_mat['seg_times'].flatten().astype(int)
            window_size = int(round(30*Fs))
            apnea_segs = apnea.reshape(1,-1)[:,list(map(lambda x:np.arange(x,x+window_size), seg_start_ids))][0]
            apnea_indicator = ((apnea_segs==1).sum(axis=1)>1*Fs).astype(int)
        except Exception as ee:
            print(str(ee))
            apnea_indicator = np.zeros_like(artifact_indicator)
        
        # report
        Nseg = len(artifact_indicator)
        Nartifact = (artifact_indicator==1).sum()
        Napnea = (apnea_indicator==1).sum()
        Nor = ((artifact_indicator==1)|(apnea_indicator==1)).sum()
        print(f'{Nartifact}/{Nseg} ({Nartifact/Nseg*100:.1f}%) is artifact')
        print(f'{Napnea}/{Nseg} ({Napnea/Nseg*100:.1f}%) is apnea')
        print(f'{Nor}/{Nseg} ({Nor/Nseg*100:.1f}%) is artifact or apnea')
        
        # save
        feat_mat['artifact_indicator'] = artifact_indicator
        feat_mat['apnea_indicator'] = apnea_indicator # later, when remove epochs, can keep W+"is apnea"
        sio.savemat(feature_path, feat_mat)
        

if __name__=='__main__':
    main()
    
