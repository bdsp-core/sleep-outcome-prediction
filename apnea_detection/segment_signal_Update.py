#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
from scipy.signal import detrend, resample_poly
import matplotlib.pyplot as plt
#from joblib import Parallel, delayed
import mne
#mne.set_log_level(verbose='WARNING')
from mne.filter import filter_data, notch_filter
#from mne.time_frequency import tfr_array_morlet
#from extract_features_parallel import *
#from biosppy.signals import ecg
import matplotlib.pyplot as plt
import pdb

seg_mask_explanation = [
    'normal',
    'around sleep stage change point',
    'NaN in sleep stage',
    'NaN in signal',
    'overly high/low amplitude',
    'flat signal',
    'NaN in feature',
    'NaN in spectrum',
    'muscle artifact',
    'spurious spectrum',
    'missing or additional R peak']
 

def clip_z_normalize(signal, sleep_stages_):
    signal = signal.flatten()
    sleep_stages = np.array(sleep_stages_)
    # determine all sleep indices
    if sleep_stages is not None:
        sleep_stages[np.isnan(sleep_stages)] = 6
        sleep = np.where(sleep_stages<5)[0]
    else:
        sleep = np.array(range(len(signal)))
    #clip signal based on sleep
    signal_clipped = np.clip(signal, np.percentile(signal[sleep],1), np.percentile(signal[sleep],99))
    signal = (signal - np.mean(signal_clipped)) / np.std(signal_clipped)
    
    return signal


def segment_chest_signal(signal, stages, apneas, window_time, step_time, Fs, newFs=None, notch_freq=None, bandpass_freq=None, start_end_remove_window_num=1, amplitude_thres=None, n_jobs=-1, to_remove_mean=False):#, BW=4
    """Segment CHEST signals.

    Arguments:
    signal -- np.ndarray, size=(channel_num, sample_num)
    stages -- np.ndarray, size=(sample_num,)
    apneas -- np.ndarray, size=(sample_num,)
    window_time -- in seconds
    Fs -- in Hz

    Keyword arguments:
    notch_freq
    bandpass_freq
    start_end_remove_window_num -- default 0, number of windows removed at the beginning and the end of the signal
    amplitude_thres -- default 1000, mark all segments with np.any(signal_seg>=amplitude_thres)=True
    to_remove_mean -- default False, whether to remove the mean of signal from each channel

    Outputs:    
    signal segments -- a list of np.ndarray, each has size=(window_size, channel_num)
    stages --  a list of stages of each window
    apneas --  a list of respiratory events of each window
    segment start ids -- a list of starting ids of each window in (sample_num,)
    segment masks --
    """
    # std_thres1 = 0.01  # changed from Haoqis original values to include more signals that may have flatter areas.
    # std_thres2 = 0.015
    
    window_size = int(round(window_time*Fs))
    step_size = int(round(step_time*Fs))
    
    start_ids = np.arange(0, signal.shape[1]-window_size+1, step_size)

    # # segment apnea signal (T,) to apnea segs (#window, window_size)
    # apnea_segs = np.array([apneas[x:x+window_size] for x in start_ids])
    # notzeroids = ~np.all(apnea_segs==0, axis=1) # remove all segments not including any event

    # nosuddenids = np.sum(np.diff(apnea_segs[:,int(-5*Fs):],axis=1),axis=1)==0 # remove all segments with a class change in the last 5 seconds

    #start_ids = start_ids[np.asarray(notzeroids) & np.asarray(nosuddenids)]
    
    
    # filter signal
    signal = notch_filter(signal, Fs, notch_freq, n_jobs=n_jobs, verbose='ERROR')  # (#window, #ch, window_size+2padding)
    signal = filter_data(signal, Fs, bandpass_freq[0], bandpass_freq[1], n_jobs=n_jobs, verbose='ERROR')  # take the value starting from *padding*, (#window, #ch, window_size+2padding)
    
    # normalize signal:
    signal = clip_z_normalize(signal, stages)
    signal = signal.reshape(1,len(signal))
    
    # determine apnea and stage labels
    if len(start_ids)!=len(stages):
        stages = stages[start_ids+window_size//2]
        apneas = apneas[start_ids+window_size//2]
        
    if start_end_remove_window_num>0:
        start_ids = start_ids[start_end_remove_window_num:-start_end_remove_window_num]
        stages = stages[start_end_remove_window_num:-start_end_remove_window_num]
        apneas = apneas[start_end_remove_window_num:-start_end_remove_window_num]
    
    assert len(start_ids)==len(stages)

    seg_masks = [seg_mask_explanation[0]]*len(start_ids)

    if np.any(np.isnan(stages)):
        ids = np.where(np.isnan(stages))[0]
        for i in ids:
            seg_masks[i] = seg_mask_explanation[2]

    assert signal.shape[0]==1


    # Signal Segments:
    signal_segs = signal[:,[np.arange(x,x+window_size) for x in start_ids]].transpose(1,0,2)  # (#window, #ch, window_size+2padding)

    if (newFs is not None) and not (Fs == newFs):

        if int(Fs)//int(newFs)==Fs/newFs: # original version of Haoqi.
            mne_epochs = mne.EpochsArray(signal_segs, mne.create_info(ch_names=['aa'], sfreq=Fs, ch_types='eeg'), verbose=False)
            mne_epochs.decimate(int(Fs)//int(newFs))
            signal_segs = mne_epochs.get_data()
            Fs = newFs

        # for natus, this requirement is not true, i.e. sampling freq of 512Hz with target frequ of 10 --> no int factor.
        else:
            # downsample full signal, up by 10, down by 512
            signal = resample_poly(signal, newFs, Fs, axis = 1)
            window_sizeResampled = int(round(window_time*newFs))
            step_sizeResampled = int(round(step_time*newFs))
            start_idsResampled = np.arange(0, signal.shape[1]-window_sizeResampled+1, step_sizeResampled)
            if start_end_remove_window_num>0:
                start_idsResampled = start_idsResampled[start_end_remove_window_num:-start_end_remove_window_num]

            signal_segs = signal[:,[np.arange(x,x+window_sizeResampled) for x in start_idsResampled]].transpose(1,0,2)                      
            Fs = newFs

    ## find nan in signal
    nan2d = np.any(np.isnan(signal_segs), axis=2)
    nan1d = np.where(np.any(nan2d, axis=1))[0]

    for i in nan1d:
        seg_masks[i] = '%s_%s'%(seg_mask_explanation[3], np.where(nan2d[i])[0])
        
    ## find large amplitude in signal
    if amplitude_thres is not None:
             
        amplitude_large2d = np.any(np.abs(signal_segs)>amplitude_thres, axis=2)
        amplitude_large1d = np.where(np.any(amplitude_large2d, axis=1))[0]
        for i in amplitude_large1d:
            seg_masks[i] = '%s_%s'%(seg_mask_explanation[4], np.where(amplitude_large2d[i])[0])
            
    # ## find flat signal
    # segment_center_idx = [int((window_time)*Fs//2 - step_time//2*Fs) , int((window_time)*Fs//2 + step_time//2*Fs)]

    # flat_seconds = 10
    # flat_length = int(round(flat_seconds*Fs))
    # assert signal_segs.shape[2]//flat_length*flat_length==signal_segs.shape[2]
    # short_segs = signal_segs.reshape(signal_segs.shape[0], signal_segs.shape[1], signal_segs.shape[2]//flat_length, flat_length)
    # # check if any of the short segments (except first and last) have last std:
    # flat2d = np.any(detrend(short_segs[:,:,1:-1,:], axis=3).std(axis=3)<=std_thres1, axis=2)
    # # check if center has low std:
    # flat2d = np.logical_or(flat2d,np.std(signal_segs[:,:,segment_center_idx[0]:segment_center_idx[1]], axis=2)<=std_thres2)
    # # check if whole segment has low std
    # flat2d = np.logical_or(flat2d, np.std(signal_segs,axis=2)<=std_thres2)
    # flat1d = np.where(np.any(flat2d, axis=1))[0]
        

    
    # for i in flat1d:
    #     # seg_masks[i] = '%s_%s'%(seg_mask_explanation[5], np.where(flat2d[i])[0])
    #     # only mark as flat if sleep stage is Not N2 and N3, for those it might be possible that signal is very flat.
    #     if stages[i] > 2:
    #         seg_masks[i] = '%s'%(seg_mask_explanation[5])

    return signal_segs, stages, apneas, start_ids, seg_masks#, specs, freq

