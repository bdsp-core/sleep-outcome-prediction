from itertools import groupby
import datetime
import sys
import numpy as np
import pandas as pd
from mne.filter import filter_data, notch_filter
from mne.time_frequency import psd_array_multitaper
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn
seaborn.set_style('ticks')
sys.path.insert(0, r'D:\projects\sleep_general')
from mgh_sleeplab import *


if __name__=='__main__':
    if len(sys.argv)>=2:
        if 'pdf' in sys.argv[1].lower():
            display_type = 'pdf'
        elif 'png' in sys.argv[1].lower():
            display_type = 'png'
        else:
            display_type = 'show'
    else:
        raise SystemExit('python %s show/png/pdf'%__file__)
    
    # from manual selection
    outcome = 'Dementia'
    txts = ['62y female, 10-year risk of Dementia is 4.3%', '57y female, 10-year risk of Dementia is 0.9%']
    ids = [2769, 5002]
    
    notch_freq = 60.  # [Hz]
    bandpass_freq = [0.3, 35]  # [Hz]
    show_freq = [0.3, 20]  # [Hz]
    epoch_time = 30  # [s]
    epoch_step_time = 30  # [s]
    std_thres = 0.2  # [uV]
    std_thres2 = 1.  # [uV]
    flat_seconds = 5  # [s]
    amplitude_thres = 500  # [uV]
    figsize = (13,7)
    vis_channel_names = ['Central']
    spec_db_vmax = 20
    spec_db_vmin = 0
    panel_xoffset = -0.04
    panel_yoffset = 1.2
        
    dfX = pd.read_csv('to_be_used_features_NREM.csv')
    dfy = pd.read_excel(f'time2event_{outcome}.xlsx')
    assert np.all(dfX.MRN==dfy.MRN)
    assert np.all(dfX.DateOfVisit==dfy.DateOfVisit)
    df = pd.concat([dfX, dfy.drop(columns=['PatientID', 'MRN', 'DateOfVisit'])], axis=1)
    df = df[(~pd.isna(df.cens_death))&(df.time_death>0)&(~pd.isna(df.cens_outcome))&(df.time_outcome>0)].reset_index(drop=True)

    # find top features that contribute the most to the prediction
    dfX = pd.read_csv(f'survival_results_NREM_bt1000/df_{outcome}_CoxPH_CompetingRisk.csv')
    df_coef = pd.read_csv(f'survival_results_NREM_bt1000/coef_{outcome}_CoxPH_CompetingRisk.csv')
    df_coef = df_coef.rename(columns={'Unnamed: 0':'Name'})
    df_coef = df_coef[df_coef.Name.str.endswith('_1:2')].reset_index(drop=True)
    df_coef['Name'] = df_coef.Name.str.replace('_1:2','')

    print(df_coef.iloc[np.argsort((dfX.loc[ids[0],df_coef.Name.values]-dfX.loc[ids[1],df_coef.Name.values]).values*df_coef.coef.values)[::-1]])
    disp_features = ['delta_alpha_mean_F_NREM', 'delta_alpha_mean_C_NREM']
    vals = df.loc[ids, disp_features].values.T.flatten()
    val_pos = [0,1,2.5,3.5]
    val_names = ['A','B', 'A','B']
    barplot_ylabel = 'delta-to-alpha ratio at NREM'
    
    plt.close()
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows=len(ids), ncols=2, width_ratios=(6,1))
    
    ax = fig.add_subplot(gs[:,1])
    ax.bar(val_pos, vals, tick_label=val_names, ec='k', color=['xkcd:red','xkcd:blue','xkcd:red','xkcd:blue'], alpha=0.8)
    ax.text(np.mean(val_pos[:2]), vals[:2].max(), 'frontal\nchannel', ha='center', va='bottom')
    ax.text(np.mean(val_pos[2:]), vals[2:].max(), 'central\nchannel', ha='center', va='bottom')
    ax.set_ylabel(barplot_ylabel)
    ax.yaxis.grid(True)
    ax.text(-0.3, 1.02, 'C', ha='right', va='top', transform=ax.transAxes, fontweight='bold')
    seaborn.despine()

    for axi, idx in enumerate(ids):
        txt = txts[axi]
        
        patient_folder = df.Path.iloc[idx].replace('\\','/').replace('M:/Datasets_ConvertedData/sleeplab', r'\\mgberisisilon1-mgmt.partners.org\PHS-RISC-LM4\ConvertedData\sleep\sleeplab_mad3')
        print(patient_folder)
        signal_path = [x for x in os.listdir(patient_folder) if x.startswith('Signal_')][0]
        label_path  = [x for x in os.listdir(patient_folder) if x=='annotations.csv'][0]
        
        eeg, params = load_mgh_signal(os.path.join(patient_folder, signal_path), channels=['F3-M.', 'F4-M.', 'C3-M.', 'C4-M.', 'O1-M.', 'O2-M.'], return_signal_dtype='numpy')
        #eeg = eeg.T
        Fs = params['Fs']
        start_time = params['start_time']
        df_annot = pd.read_csv(os.path.join(patient_folder, label_path))
        df_annot = annotations_preprocess(df_annot, Fs)
        sleep_stages = vectorize_sleep_stages(df_annot, eeg.shape[1])

        # filter EEG
        if notch_freq is not None:
            eeg = notch_filter(eeg, Fs, notch_freq, verbose=False)
        if bandpass_freq is not None:
            eeg = filter_data(eeg, Fs, bandpass_freq[0], bandpass_freq[1], verbose=False)
        
        # segment
        epoch_size = int(round(epoch_time*Fs))
        epoch_step = int(round(epoch_step_time*Fs))
        start_ids = np.arange(0, eeg.shape[1]-epoch_size+1, epoch_step)
        seg_ids = list(map(lambda x:np.arange(x,x+epoch_size), start_ids))
        eeg_segs = eeg[:,seg_ids].transpose(1,0,2)  # eeg_segs.shape=(#epoch, #channel, Tepoch)
        sleep_stages = sleep_stages[start_ids]
        
        # compute spectrogram
        NW = 10.
        BW = NW*2./epoch_time
        specs, freq = psd_array_multitaper(eeg_segs, Fs, fmin=show_freq[0], fmax=show_freq[1], adaptive=False, low_bias=True, verbose='ERROR', bandwidth=BW, normalization='full')
        # specs.shape = (#epoch, #channel, #freq)
        # freq.shape = (#freq,)
        specs_db = 10*np.log10(specs)
        specs_db[np.isinf(specs_db)] = np.nan
        specs_db = np.nanmean(np.array([specs_db[:,::2],specs_db[:,1::2]]), axis=0)

        # plot spectrogram
        
        gss = GridSpecFromSubplotSpec(nrows=1+len(vis_channel_names), ncols=2,
                width_ratios=[100,1], height_ratios=[1]+[2.5]*len(vis_channel_names),
                hspace=0.06, wspace=0.01,
                subplot_spec=gs[axi,0])
                
        tt = np.arange(len(specs_db))*epoch_step_time/3600
        xticks = np.arange(0, np.floor(tt.max())+1) 
        xticklabels = []
        for j, x in enumerate(xticks):
            dt = start_time+datetime.timedelta(hours=x)
            #if j==0:
            xx = datetime.datetime.strftime(dt, '%H:%M:%S')#\n%m/%d/%Y')
            xticklabels.append(xx)
        
        # hypogram
        ax = fig.add_subplot(gss[0,0])
        ax.step(tt, sleep_stages, color='k', lw=2)
        cc = 0
        sleep_stages[np.isnan(sleep_stages)] = -1
        for k,l in groupby(sleep_stages):
            ll = len(list(l))
            if k==4:
                ax.plot([tt[cc], tt[cc+ll-1]], [4,4], c='r', lw=3)
            cc += ll
        ax.text(0, 1.03, txts[axi], ha='left', va='bottom', transform=ax.transAxes)
        ax.text(panel_xoffset, panel_yoffset, chr(ord('A')+axi), ha='right', va='top', transform=ax.transAxes, fontweight='bold')
        ax.set_yticks([1,2,3,4,5])
        ax.set_yticklabels(['N3', 'N2', 'N1', 'R', 'W'])
        ax.yaxis.grid(True)
        ax.set_ylim(0.8, 5.2)
        ax.set_xlim(tt.min(), tt.max())
        ax.set_xticks(xticks)
        ax.set_xticklabels([])
        seaborn.despine()
        
        # spectrogram
        for chi in range(len(vis_channel_names)):
            ax = fig.add_subplot(gss[1+chi,0])
            
            specs_db_ch = specs_db[:,chi]#(specs_db[:,chi*2]+specs_db[:,chi*2+1])/2
            #spec_db_vmax, spec_db_vmin = np.nanpercentile(specs_db_ch, (95,10))
            #print(spec_db_vmax, spec_db_vmin)
            im = ax.imshow(specs_db_ch.T, cmap='turbo', origin='lower', aspect='auto',
                    vmax=spec_db_vmax ,vmin=spec_db_vmin,
                    extent=(tt.min(),tt.max(),freq.min(),freq.max()))
            #ax.text(-0.09, 0.5, 'Avg '+vis_channel_names[chi],
            #            ha='left', va='center', transform=ax.transAxes)
            ax.set_ylabel(f'Avg {vis_channel_names[chi]}: Hz')
            ax.set_xticks(xticks)
            if chi==0:
                ax.set_yticks([5,10,15])
            else:
                ax.set_yticks([5,10,15,20])
            if chi==len(vis_channel_names)-1:
                #ax.set_xlabel('time (h)')
                ax.set_xticklabels(xticklabels)
            else:
                ax.set_xticklabels([])
            #if chi==len(vis_channel_names)-2:
            #    ax_cb = fig.add_subplot(gss[1+chi,1])
            #    ax_cb.text(0, 0.02, 'PSD (dB)', ha='left', va='bottom', rotation=90, transform=ax_cb.transAxes)
            #    ax_cb.axis('off')
            if axi==len(ids)-1 and chi==len(vis_channel_names)-1:
                ax_cb = fig.add_subplot(gss[1+chi,1])
                fig.colorbar(im, cax=ax_cb, label='PSD (dB)')
            ax.set_xlim(tt.min(), tt.max())
                
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.18, hspace=0.29, top=0.92)
    
    save_path = 'example_spectrograms'
    if display_type=='pdf':
        plt.savefig(save_path+'.pdf', dpi=600, bbox_inches='tight', pad_inches=0.03)
    elif display_type=='png':
        plt.savefig(save_path+'.png', bbox_inches='tight', pad_inches=0.03)
    else:
        plt.show()
        
