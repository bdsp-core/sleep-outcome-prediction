from itertools import product
import os
import sys
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 12})
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')

def enumerated_product(*args):
    yield from zip(product(*(range(len(x)) for x in args)), product(*args))


if __name__=='__main__':
    if len(sys.argv)>=2:
        if 'pdf' in sys.argv[1].lower():
            display_type = 'pdf'
        elif 'png' in sys.argv[1].lower():
            display_type = 'png'
        elif 'svg' in sys.argv[1].lower():
            display_type = 'svg'
        else:
            display_type = 'show'
    else:
        raise SystemExit('python %s show/png/pdf'%__file__)
            
    sexs = [0,1]
    sexs_txt = ['Female', 'Male']
    curve_names = ['<Q1,sex=0', '[Q1--Q3],sex=0', '>Q3,sex=0', '<Q1,sex=1', '[Q1--Q3],sex=1', '>Q3,sex=1']

    outcomes = [
        'IntracranialHemorrhage', 'IschemicStroke', 'Dementia', 'MCI+Dementia',
        'Atrial_Fibrillation', 'Myocardial_Infarction', 'DiabetesII', 'Hypertension',
        'Bipolar_Disorder', 'Depression',
        'Death'
    ]
    outcomes_txt = [
        'ICH', 'IS', 'Dem', 'MCI/Dem',
        'AFib', 'MI', 'T2D', 'HTN',
        'BD', 'Dep',
        'Death'
    ]
    data_types = ['NREM', 'AHI']
    
    rr1_cox = {}
    rr2_cox = {}
    rr1_aj = {}
    rr2_aj = {}
    val1_cox = {}
    val2_cox = {}
    val3_cox = {}
    val1_aj = {}
    val2_aj = {}
    val3_aj = {}
    for (di, si), (data_type, sex) in enumerated_product(data_types, sexs):
        print(data_type, sexs_txt[si])
        folder = f'/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction (1)/code-haoqi/survival_results_{data_type}_bt1000'
    
        q1q3_idx = curve_names.index(f'[Q1--Q3],sex={sex}')
        q1_idx = curve_names.index(f'<Q1,sex={sex}')
        q3_idx = curve_names.index(f'>Q3,sex={sex}')
        
        rr1_cox[(data_type, sex)] = []
        rr2_cox[(data_type, sex)] = []
        rr1_aj[(data_type, sex)] = []
        rr2_aj[(data_type, sex)] = []
        val1_cox[(data_type, sex)] = []
        val2_cox[(data_type, sex)] = []
        val3_cox[(data_type, sex)] = []
        val1_aj[(data_type, sex)] = []
        val2_aj[(data_type, sex)] = []
        val3_aj[(data_type, sex)] = []
        for ii, outcome in enumerate(outcomes):
            print(outcome)
            
            if outcome in ['Death']:
                model_type = 'CoxPH'
            else:
                model_type = 'CoxPH_CompetingRisk'
            mat_path = os.path.join(folder, f'results_{outcome}_{model_type}.mat')
            if not os.path.exists(mat_path):
                continue
            mat = sio.loadmat(mat_path)
            
            ## AJ
            
            vals_bt = np.array([
                mat['AJ_curve_tes_val'],
                mat['AJ_curve_tes_lower'],
                mat['AJ_curve_tes_upper'],
                ])*100
            if vals_bt.ndim==4:
                vals_bt = vals_bt.transpose(1,0,2,3)
            else:
                vals_bt = vals_bt.transpose(1,0,2)
                vals_bt = vals_bt[..., np.newaxis]
            outcome_idx = [np.array(mat['AJ_curve_tes_states'].flatten()[i]).item() for i in range(len(mat['AJ_curve_tes_states']))].index('event1')
            vals_bt = vals_bt[..., outcome_idx]  # shape = (#T, 3)
            
            survtime = mat['AJ_curve_tes_time'].flatten()
            time_idx = np.argmin(np.abs(survtime-10))
            
            val1_aj_mean = vals_bt[time_idx,0,q1_idx]
            val1_aj_lb = vals_bt[time_idx,1,q1_idx]
            val1_aj_ub = vals_bt[time_idx,2,q1_idx]
            val1_aj[(data_type, sex)].append([val1_aj_mean, val1_aj_lb, val1_aj_ub])
            
            val2_aj_mean = vals_bt[time_idx,0,q1q3_idx]
            val2_aj_lb = vals_bt[time_idx,1,q1q3_idx]
            val2_aj_ub = vals_bt[time_idx,2,q1q3_idx]
            val2_aj[(data_type, sex)].append([val2_aj_mean, val2_aj_lb, val2_aj_ub])
            
            val3_aj_mean = vals_bt[time_idx,0,q3_idx]
            val3_aj_lb = vals_bt[time_idx,1,q3_idx]
            val3_aj_ub = vals_bt[time_idx,2,q3_idx]
            val3_aj[(data_type, sex)].append([val3_aj_mean, val3_aj_lb, val3_aj_ub])
            
            rr1_mean = vals_bt[time_idx,0,q3_idx] / vals_bt[time_idx,0,q1q3_idx]
            rr1_lb = vals_bt[time_idx,1,q3_idx] / vals_bt[time_idx,2,q1q3_idx]
            rr1_ub = vals_bt[time_idx,2,q3_idx] / vals_bt[time_idx,1,q1q3_idx]
            rr1_aj[(data_type, sex)].append([rr1_mean, rr1_lb, rr1_ub])
            
            rr2_mean = vals_bt[time_idx,0,q1q3_idx] / vals_bt[time_idx,0,q1_idx]
            rr2_lb = vals_bt[time_idx,1,q1q3_idx] / vals_bt[time_idx,2,q1_idx]
            rr2_ub = vals_bt[time_idx,2,q1q3_idx] / vals_bt[time_idx,1,q1_idx]
            rr2_aj[(data_type, sex)].append([rr2_mean, rr2_lb, rr2_ub])
            
            ## Cox
            
            vals_bt = mat['cox_curve_tes_bt_val']*100
            if vals_bt.ndim==3:
                vals_bt = vals_bt[..., np.newaxis]
            outcome_idx = [np.array(mat['cox_curve_tes_bt_states'].flatten()[i]).item() for i in range(len(mat['cox_curve_tes_bt_states']))].index('event1')
            vals_bt = vals_bt[..., outcome_idx]  # shape = (#T, #bt, 3)
            vals_bt[vals_bt<0.001] = np.nan
            
            survtime = mat['cox_curve_tes_bt_time'].flatten()
            time_idx = np.argmin(np.abs(survtime-10))
            
            rr1_mean = vals_bt[time_idx,0,q3_idx] / vals_bt[time_idx,0,q1q3_idx]
            rr1_ci = vals_bt[time_idx,1:,q3_idx] / vals_bt[time_idx,1:,q1q3_idx]
            rr1_lb, rr1_ub = np.nanpercentile(rr1_ci, (2.5, 97.5))
            rr1_cox[(data_type, sex)].append([rr1_mean, rr1_lb, rr1_ub])
            rr2_mean = vals_bt[time_idx,0,q1q3_idx] / vals_bt[time_idx,0,q1_idx]
            rr2_ci = vals_bt[time_idx,1:,q1q3_idx] / vals_bt[time_idx,1:,q1_idx]
            rr2_lb, rr2_ub = np.nanpercentile(rr2_ci, (2.5, 97.5))
            rr2_cox[(data_type, sex)].append([rr2_mean, rr2_lb, rr2_ub])
            
            val1_cox_mean = vals_bt[time_idx,0,q1_idx]
            val1_cox_lb, val1_cox_ub = np.nanpercentile(vals_bt[time_idx,1:,q1_idx], (2.5, 97.5))
            val1_cox[(data_type, sex)].append([val1_cox_mean, val1_cox_lb, val1_cox_ub])
            
            val2_cox_mean = vals_bt[time_idx,0,q1q3_idx]
            val2_cox_lb, val2_cox_ub = np.nanpercentile(vals_bt[time_idx,1:,q1q3_idx], (2.5, 97.5))
            val2_cox[(data_type, sex)].append([val2_cox_mean, val2_cox_lb, val2_cox_ub])
            
            val3_cox_mean = vals_bt[time_idx,0,q3_idx]
            val3_cox_lb, val3_cox_ub = np.nanpercentile(vals_bt[time_idx,1:,q3_idx], (2.5, 97.5))
            val3_cox[(data_type, sex)].append([val3_cox_mean, val3_cox_lb, val3_cox_ub])
        rr1_cox[(data_type,sex)] = np.array(rr1_cox[(data_type,sex)])
        rr2_cox[(data_type,sex)] = np.array(rr2_cox[(data_type,sex)])
        rr1_aj[(data_type,sex)] = np.array(rr1_aj[(data_type,sex)])
        rr2_aj[(data_type,sex)] = np.array(rr2_aj[(data_type,sex)])
        val1_cox[(data_type,sex)] = np.array(val1_cox[(data_type,sex)])
        val2_cox[(data_type,sex)] = np.array(val2_cox[(data_type,sex)])
        val3_cox[(data_type,sex)] = np.array(val3_cox[(data_type,sex)])
        val1_aj[(data_type,sex)] = np.array(val1_aj[(data_type,sex)])
        val2_aj[(data_type,sex)] = np.array(val2_aj[(data_type,sex)])
        val3_aj[(data_type,sex)] = np.array(val3_aj[(data_type,sex)])

    """
    # plot 10-year risk ratio


    plt.close()
    fig = plt.figure(figsize=(9,6))
    #val_max = max([max(val2_aj[('NREM',sex)].max(), val2_cox[('NREM',sex)].max()) for sex in sexs])
    rr_types = ['Good-to-average sleep ratio', 'Average-to-poor sleep ratio']

    cc = 0
    for (ri, si), (rr_type, sex) in enumerated_product(rr_types, sexs):
        if cc==0:
            ax = fig.add_subplot(len(sexs)*len(rr_types),1,cc+1)
            ax0 = ax
        else:
            ax = fig.add_subplot(len(sexs)*len(rr_types),1,cc+1,sharex=ax0)#,sharey=ax0)
        cc += 1
        
        xx = np.arange(len(outcomes))
        ww = 0.35
        alpha = 0.45
        panel_xoffset = -0.05
        panel_yoffset = 1
        
        if rr_type=='Good-to-average sleep ratio':
            val1 = rr1_cox[('NREM', sex)]
        elif rr_type=='Average-to-poor sleep ratio':
            val1 = rr2_cox[('NREM', sex)]
        ax.bar(xx, val1[:,0], width=ww, color='r', edgecolor='k', alpha=alpha, capsize=3, yerr=np.array([val1[:,0]-val1[:,1],val1[:,2]-val1[:,0]]), label='sleep EEG', bottom=1)
        for xi, x in enumerate(xx):
            ax.text(x, val1[xi,2], f'{val1[xi,0]:.1f}', ha='center', va='bottom', clip_on=False, fontsize=11, color='r')
            
        if rr_type=='Good-to-average sleep ratio':
            val2 = rr1_cox[('AHI', sex)]
        elif rr_type=='Average-to-poor sleep ratio':
            val2 = rr2_cox[('AHI', sex)]
        ax.bar(xx+ww, val2[:,0], width=ww, color='b', edgecolor='k', alpha=alpha, capsize=3, yerr=np.array([val2[:,0]-val2[:,1],val2[:,2]-val2[:,0]]), label='AHI', bottom=1)
        for xi, x in enumerate(xx+ww):
            ax.text(x, val2[xi,2], f'{val2[xi,0]:.1f}', ha='center', va='bottom', clip_on=False, fontsize=11, color='b')
        
        ax.text(0.01, 0.955, sexs_txt[si], ha='left', va='top', transform=ax.transAxes)
        ax.text(panel_xoffset, panel_yoffset, chr(ord('A')+si), ha='right', va='top', transform=ax.transAxes, fontweight='bold')
        if si==0:
            ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0,0.755))
            
        ax.set_xlim(-0.3,len(outcomes)-1+ww+0.3)
        ax.set_xticks(np.arange(len(outcomes))+ww/2)
        ax.set_xticklabels(outcomes_txt)#, rotation=-20, ha='left')
        if si==0:
            plt.setp(ax.get_xticklabels(), visible=False)
        ax.yaxis.grid(True)
        ax.set_ylabel('10-Year\nRisk Ratio')
        #ax.set_ylim(0,val_max+2)
        #ax.set_yticks([0,5,10,15,20,25,30])
        sns.despine()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.07)
    save_name = 'risk_ratios'
    if display_type=='pdf':
        plt.savefig(save_name+'.pdf', dpi=600, bbox_inches='tight', pad_inches=0.02)
    elif display_type=='png':
        plt.savefig(save_name+'.png', bbox_inches='tight', pad_inches=0.02)
    elif display_type=='svg':
        plt.savefig(save_name+'.svg', bbox_inches='tight', pad_inches=0.02)
    else:
        plt.show()
    """


    # plot 10-year risk

    plt.close()
    fig = plt.figure(figsize=(9,5))
    val_max = max([max(val2_aj[('NREM',sex)].max(), val2_cox[('NREM',sex)].max()) for sex in sexs])

    for si, sex in enumerate(sexs):
        if si==0:
            ax = fig.add_subplot(len(sexs),1,si+1)
            ax0 = ax
        else:
            ax = fig.add_subplot(len(sexs),1,si+1,sharex=ax0,sharey=ax0)
        
        xx = np.arange(len(outcomes))
        ww = 0.35
        alpha = 0.45
        panel_xoffset = -0.05
        panel_yoffset = 1
        
        val1 = val2_aj[('NREM',sex)]
        ax.bar(xx, val1[:,0], width=ww, color='k', edgecolor='k', alpha=alpha, capsize=3, yerr=np.array([val1[:,0]-val1[:,1],val1[:,2]-val1[:,0]]), label='Ground truth')
        for xi, x in enumerate(xx):
            ax.text(x, val1[xi,2], f'{val1[xi,0]:.1f}', ha='center', va='bottom', clip_on=False, fontsize=11, color='k')
            
        val2 = val2_cox[('NREM',sex)]
        ax.bar(xx+ww, val2[:,0], width=ww, color='r', edgecolor='k', alpha=alpha, capsize=3, yerr=np.array([val2[:,0]-val2[:,1],val2[:,2]-val2[:,0]]), label='Model prediction')
        for xi, x in enumerate(xx+ww):
            ax.text(x, val2[xi,2], f'{val2[xi,0]:.1f}', ha='center', va='bottom', clip_on=False, fontsize=11, color='r')
        
        ax.text(0.01, 0.955, sexs_txt[si], ha='left', va='top', transform=ax.transAxes)
        ax.text(panel_xoffset, panel_yoffset, chr(ord('A')+si), ha='right', va='top', transform=ax.transAxes, fontweight='bold')
        if si==0:
            ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0,0.755))
            
        ax.set_xlim(-0.3,len(outcomes)-1+ww+0.3)
        ax.set_xticks(np.arange(len(outcomes))+ww/2)
        ax.set_xticklabels(outcomes_txt)#, rotation=-20, ha='left')
        if si==0:
            plt.setp(ax.get_xticklabels(), visible=False)
        ax.yaxis.grid(True)
        ax.set_ylabel('10-Year Risk (%)')
        ax.set_ylim(0,val_max+2)
        ax.set_yticks([0,5,10,15,20,25,30])
        sns.despine()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.07)
    save_name = 'risks'
    if display_type=='pdf':
        plt.savefig(save_name+'.pdf', dpi=600, bbox_inches='tight', pad_inches=0.02)
    elif display_type=='png':
        plt.savefig(save_name+'.png', bbox_inches='tight', pad_inches=0.02)
    elif display_type=='svg':
        plt.savefig(save_name+'.svg', bbox_inches='tight', pad_inches=0.02)
    else:
        plt.show()

