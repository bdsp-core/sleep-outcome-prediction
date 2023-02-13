import os
import sys
import numpy as np
import pandas as pd
from scipy.special import softmax
from mne.filter import notch_filter, filter_data
import torch as th
import torch.nn as nn
from perf_functions import smooth
from segment_signal_Update import clip_z_normalize


def load_apnea_binary_model(model_path, use_gpu=False, n_gpu=0):
    class_tags = ['NoEvent', 'Event']
    n_cl = len(class_tags)
    
    sys.path.insert(0, os.path.join(model_path, 'wavenet_vocoder'))
    #from wavenet import WaveNet

    # load all models
    names = ['Boost1', 'Boost2', 'Boost3', 'Boost4', 'Boost5', 'Binary']
    models = []
    for name in names:
        # load model and reset cpu module
        model = th.load(os.path.join(model_path, f'CVall_{name}.pth'), map_location=th.device('cpu')).module.cpu()
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


def detect_apnea(model, signal, sleep_stage, Fs, use_gpu=False, newFs=10):
    # preprocess
    n_jobs = 1
    line_freq = [50., 60.]  # [Hz]  # mutli-center study, not sure 50 or 60?
    chest_bandpass_freq = [None, 10]  # [Hz]
    window_size = 4100  # samples, as model is trained on.
    step_time = 1       # seconds   #min stepsize = 1/Fs.
    output_node = -2048 # <---  define the position <output_node>, either "2050" for the middle node or "-1" for the last node
    
    # filter signal
    line_freq = [x for x in line_freq if x<Fs/2]
    if len(line_freq)>0: 
        signal = notch_filter(signal, Fs, line_freq, n_jobs=n_jobs, verbose='ERROR')  # (#window, #ch, window_size+2padding)
    signal = filter_data(signal, Fs, chest_bandpass_freq[0], chest_bandpass_freq[1], n_jobs=n_jobs, verbose='ERROR') 
    
    # resample data 
    if Fs!=newFs:
        #if signal.ndim!=2:
        #    signal = signal.reshape(1,1,-1)
        if int(Fs)//int(newFs) == Fs/newFs: 
            signal = signal[...,::(int(Fs)//int(newFs))]
            sleep_stage = sleep_stage[::(int(Fs)//int(newFs))]
        else:
            raise NotImplementedError
    Fs = newFs

    # clip signal:
    signal = clip_z_normalize(signal, sleep_stage)
    
    # segment all signal
    start_ids = np.arange(0, signal.shape[0]-window_size+1, step_time*Fs)
    signal_segs = signal[[np.arange(x,x+window_size) for x in start_ids]]  # (#window, #ch, window_size+2padding)
    
    # deal with flat signals (no signal)
    nonflat_ids = np.nanstd(signal_segs, axis=1)>1e-2
    start_ids_all = np.array(start_ids)
    start_ids = start_ids[nonflat_ids]
    signal_segs = signal_segs[nonflat_ids]
    
    boost_models = model['boost']
    final_model = model['final']
    thresholds = model['thres']
    n_cl = model['n_class']
    batch_size = 1000
    overall_ind = np.arange(signal_segs.shape[0])
    overall_probs = np.zeros([signal_segs.shape[0], n_cl])
    for bb in range(len(boost_models)):
        #if bb == 0:
        #    print('Boost: 1', end="")
        #else:
        #    print(', %s'%(bb+1), end="")
        if signal_segs.shape[0]==0:
            continue
        yprob = []
        # run model 1 over all segments
        for i_segs in range(0,signal_segs.shape[0], batch_size):
            signal_segs_tmp = signal_segs[i_segs:i_segs+batch_size, :]
            import pdb;pdb.set_trace()
            signal_segs_tmp = th.Tensor(np.resize(signal_segs_tmp,(signal_segs_tmp.shape[0],1,signal_segs_tmp.shape[1])))
            if use_gpu:
                signal_segs_tmp = signal_segs_tmp.cuda()

            # predict for each segment, i.e. each row of signal_segs part:
            with th.no_grad():
                yprob_tmp = boost_models[bb](signal_segs_tmp)
            yprob.append(yprob_tmp.detach().to('cpu'))

        # concatinate all predections
        yprob = np.concatenate(yprob, axis=0)
        
        # define probability and prediction arrays
        yprob = yprob[:,:,output_node]
        yprob = softmax(yprob,axis=1)
        ypred = yprob[:,1] >= thresholds[bb]

        # Pass data to next model
        # store old probabilities
        if bb == 0:
            overall_probs[:,:2] = np.array(yprob)
        else:
            overall_probs[overall_ind,:2] = np.array(yprob)

        # determine data to pass to 2nd model
        pass_ind = np.where(ypred)[0]
        reject_ind = np.where(~ypred)[0]
        signal_segs = signal_segs[pass_ind,:]
        # save the indices
        overall_ind = overall_ind[pass_ind]

    #print('\n--> Final model')
    # run multiclass model over passed on segments
    yprob = []
    for i_segs in range(0,signal_segs.shape[0], batch_size):
        signal_segs_tmp = signal_segs[i_segs:i_segs+batch_size, :]
        signal_segs_tmp = th.Tensor(np.resize(signal_segs_tmp,(signal_segs_tmp.shape[0],1,signal_segs_tmp.shape[1])))
        if use_gpu:
            signal_segs_tmp = signal_segs_tmp.cuda()

        # predict for each segment, i.e. each row of signal_segs part:
        with th.no_grad():
            yprob_tmp = final_model(signal_segs_tmp)
        yprob.append(yprob_tmp.detach().to('cpu'))

    # concatenate all predections
    yprob = np.concatenate(yprob, axis=0)
    yprob = yprob[:,:,output_node].astype(float)
    
    # define probability and prediction arrays
    yprob = softmax(yprob,axis=1)
    overall_probs[overall_ind,:] = np.array(yprob)
    
    ypred = np.argmax(yprob, axis=1)

    # merch new prediction from multiclass model with rejected predictions from boosting
    predictions_overall_ = np.zeros([len(start_ids)],dtype=int)
    predictions_overall_[overall_ind] = ypred
    
    # put back the non-flat region
    predictions_overall = np.zeros([len(start_ids_all)], dtype=int)
    predictions_overall[nonflat_ids] = predictions_overall_

    # apply smoothning for windows of length 10
    predictions_smooth = smooth(predictions_overall,win=10)

    # Convert predictions to sampling rate ####

    # resample the predictions into 10Hz 
    ypred_in_samplingrate = (np.resize(predictions_overall,(predictions_overall.shape[0],1)) * np.ones([int(1),step_time*Fs])).flatten()
    ysmooth_in_samplingrate = (np.resize(predictions_smooth,(predictions_smooth.shape[0],1)) * np.ones([int(1),step_time*Fs])).flatten()
    # resample probs
    yprob_in_samplingrate = overall_probs.repeat(step_time*Fs, axis=0)

    # specify padding for beginning/end of the prediction array
    nans_for_padding = np.empty((int(window_size/2)-int(step_time*Fs/2),))
    nans_for_padding[:] = np.nan
    apnea_prediction = np.concatenate([nans_for_padding, ypred_in_samplingrate, nans_for_padding]) 
    apnea_prediction_smooth = np.concatenate([nans_for_padding, ysmooth_in_samplingrate, nans_for_padding]) 
    yprob_pad = np.tile(nans_for_padding,(n_cl,1))
    yprob_pad = yprob_pad.reshape(yprob_pad.shape[1], yprob_pad.shape[0])
    apnea_probability = np.concatenate([yprob_pad, yprob_in_samplingrate, yprob_pad])
    
    return apnea_prediction_smooth, apnea_probability
    