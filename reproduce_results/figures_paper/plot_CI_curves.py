import os
import sys
import subprocess
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns
sns.set_style('ticks')


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
    
    #outcome_groups = ['neuro', 'cardio', 'psych', 'mortality']
    #outcome_groups_txt = ['Neurological outcomes', 'Cardiovascular outcomes', 'Psychiatric outcomes', 'Mortality']
    
    outcomes = [
        'IntracranialHemorrhage', 'IschemicStroke', 'Dementia', 'MCI+Dementia',
        'Atrial_Fibrillation', 'Myocardial_Infarction', 'DiabetesII', 'Hypertension',
        'Bipolar_Disorder', 'Depression',
        'Death']
    outcomes_txt = [
        'Intracranial hemorrhage', 'Ischemic stroke', 'Dementia', 'MCI or Dementia',
        'Atrial fibrillation', 'Myocardial infarction', 'Type 2 diabetes', 'Hypertension',
        'Bipolar disorder', 'Depression',
        'Death']
    
    suffix = ''#_sens1'
    folder = f'/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction (1)/code-haoqi/survival_results_NREM{suffix}'
    sexs = [0,1]
    sexs_txt = ['Female', 'Male']
    curve_names = ['<Q1,sex=0', '[Q1--Q3],sex=0', '>Q3,sex=0', '<Q1,sex=1', '[Q1--Q3],sex=1', '>Q3,sex=1']
    
    colors = ['k','b','r']
    panel_xoffset = -0.26
    panel_yoffset = 1.03
    
    plt.close()
    fig = plt.figure(figsize=(13.7,6.3))
    gs = GridSpec(3,4, figure=fig)
    
    save_name = f'CI_curve'#_{sexs_txt[si]}'
    for si, sex in enumerate(sexs):
        print(sexs_txt[si])
        q1q3_idx = curve_names.index(f'[Q1--Q3],sex={sex}')
        q1_idx = curve_names.index(f'<Q1,sex={sex}')
        q3_idx = curve_names.index(f'>Q3,sex={sex}')
        
        for axi, outcome in enumerate(outcomes):
            print(outcome)
            outcome_txt = outcomes_txt[axi]
            
            if outcome in ['Death'] or 'sens' in folder:
                model_type = 'CoxPH'
            else:
                model_type = 'CoxPH_CompetingRisk'

            mat_path = os.path.join(folder, f'results_{outcome}_{model_type}.mat')
            if not os.path.exists(mat_path):
                continue
            mat = sio.loadmat(mat_path)
            
            # AJ
            aj_outcome_idx = [np.array(mat['AJ_curve_tes_states'].flatten()[i]).item() for i in range(len(mat['AJ_curve_tes_states']))].index('event1')
            aj_time = mat['AJ_curve_tes_time'].flatten()
            if mat['AJ_curve_tes_val'].ndim==2:
                mat['AJ_curve_tes_val'] = mat['AJ_curve_tes_val'][..., np.newaxis]
                mat['AJ_curve_tes_lower'] = mat['AJ_curve_tes_lower'][..., np.newaxis]
                mat['AJ_curve_tes_upper'] = mat['AJ_curve_tes_upper'][..., np.newaxis]
            aj_val_1 = mat['AJ_curve_tes_val'][:,q1q3_idx,aj_outcome_idx]
            aj_val_2 = mat['AJ_curve_tes_val'][:,q1_idx,aj_outcome_idx]
            aj_val_3 = mat['AJ_curve_tes_val'][:,q3_idx,aj_outcome_idx]

            aj_CI_1 = np.array([
                mat['AJ_curve_tes_lower'][:,q1q3_idx,aj_outcome_idx],
                mat['AJ_curve_tes_upper'][:,q1q3_idx,aj_outcome_idx],])
            aj_CI_2 = np.array([
                mat['AJ_curve_tes_lower'][:,q1_idx,aj_outcome_idx],
                mat['AJ_curve_tes_upper'][:,q1_idx,aj_outcome_idx],])
            aj_CI_3 = np.array([
                mat['AJ_curve_tes_lower'][:,q3_idx,aj_outcome_idx],
                mat['AJ_curve_tes_upper'][:,q3_idx,aj_outcome_idx],])
            
            # Cox
            
            cox_time = mat['cox_curve_tes_bt_time'].flatten()
            cox_outcome_idx = [np.array(mat['cox_curve_tes_bt_states'].flatten()[i]).item() for i in range(len(mat['cox_curve_tes_bt_states']))].index('event1')
            if mat['cox_curve_tes_bt_val'].ndim==3:
                mat['cox_curve_tes_bt_val'] = mat['cox_curve_tes_bt_val'][..., np.newaxis]
            cox_val_1 = mat['cox_curve_tes_bt_val'][:,0,q1q3_idx,cox_outcome_idx]
            cox_val_2 = mat['cox_curve_tes_bt_val'][:,0,q1_idx,cox_outcome_idx]
            cox_val_3 = mat['cox_curve_tes_bt_val'][:,0,q3_idx,cox_outcome_idx]
            
            cox_CI_1 = np.percentile(mat['cox_curve_tes_bt_val'][:,1:,q1q3_idx,cox_outcome_idx], (2.5, 97.5), axis=1) 
            cox_CI_2 = np.percentile(mat['cox_curve_tes_bt_val'][:,1:,q1_idx,cox_outcome_idx], (2.5, 97.5), axis=1) 
            cox_CI_3 = np.percentile(mat['cox_curve_tes_bt_val'][:,1:,q3_idx,cox_outcome_idx], (2.5, 97.5), axis=1)
                
            gss = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[axi//4, axi%4], wspace=0)
            ax = fig.add_subplot(gss[0,si])
            #aj_mat = sio.loadmat(f'../{result_folder}/AJ_output_{outcome}_{model_type}.mat')
            #aj_outcome_idx = [aj_mat['states'][i,0][0] for i in range(len(aj_mat['states']))].index('event1')
            #ax.fill_between(aj_mat['time'], aj_mat['lower'][:,aj_outcome_idx], aj_mat['upper'][:,aj_outcome_idx], step='pre', color='k', alpha=0.2)
            #ax.step(aj_mat['time'], aj_mat['val'][:,aj_outcome_idx], ls='--', c='k', label='Aalen-Johansen estimator (ground-truth)')
            for ii in [0,1,2]:
                #aj_CI = eval(f'aj_CI_{ii+1}')
                #ax.fill_between(aj_time, aj_CI[0]*100, aj_CI[1]*100, step='pre', color=colors[ii], alpha=0.1)
                cox_CI = eval(f'cox_CI_{ii+1}')
                ax.fill_between(cox_time, cox_CI[0]*100, cox_CI[1]*100, step='pre', color=colors[ii], alpha=0.2)
            
            for ii in [0,1,2]:
                aj_val  = eval(f'aj_val_{ii+1}')
                ax.step(aj_time, aj_val*100, c=colors[ii], ls='--', lw=1)
            
            for ii in [0,1,2]:
                cox_val    = eval(f'cox_val_{ii+1}')
                ax.step(cox_time, cox_val*100, c=colors[ii], lw=2)
            
            if si==1:
                tt = ax.text(-0.94, 0.95, outcome_txt, ha='left', va='top', transform=ax.transAxes, clip_on=False)#, zorder=1000)
                tt.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='none'))
                
            ax.set_xlim([0, 10])
            if si==0:
                ax.set_xticks([0,2,4,6,8,10])
            else:
                ax.set_xticks([2,4,6,8,10])
            if outcome in ['IntracranialHemorrhage', 'IschemicStroke', 'Bipolar_Disorder', 'Dementia', 'Atrial_Fibrillation', 'Myocardial_Infarction']:
                ax.set_ylim([0, 15])
                ax.set_yticks([0,3,6,9,12])
            elif outcome in ['MCI+Dementia', 'DiabetesII']:
                ax.set_ylim([0, 45])
                ax.set_yticks([0,10,20,30,40])
            elif outcome in ['Hypertension', 'Depression', 'Death']:
                ax.set_ylim([0, 55])
                ax.set_yticks([0,10,20,30,40,50])
            if si==1:
                ax.set_yticklabels([])
            #ax.set_ylim([-0.01, max(survprob_1.max(), survprob_2.max(), survprob_3.max())*100+0.1])
            #ax.legend(frameon=False, loc='upper left')
            ax.yaxis.grid(True)
            if si==0:
                if axi//4==2:
                    ax.set_xlabel('Time since PSG (year)')
                    ax.xaxis.set_label_coords(1, -.2)
                else:
                    if axi!=7:
                        ax.set_xticklabels([])
                if axi%4==0:
                    ax.set_ylabel('P(outcome) (%)')
                ax.text(panel_xoffset-0.25*(axi%4==0), panel_yoffset, chr(ord('A')+axi), ha='right', va='top', transform=ax.transAxes, fontweight='bold')
            else:
                if axi//4!=2:
                    if axi!=7:
                        ax.set_xticklabels([])
            ax.text(0.05, 0.77, sexs_txt[si], ha='left', va='top', transform=ax.transAxes)
            sns.despine()

    # legend panel
    ax = fig.add_subplot(gs[2,3])
    ax.plot([0,1],[0,0], c='r', lw=2, label='Poor sleep')#f'median(z)+{dz}')
    ax.plot([0,1],[0,0], c='k', lw=2, label='Average sleep')
    ax.plot([0,1],[0,0], c='b', lw=2, label='Good sleep')
    ax.plot([0,1],[0,0], c='k', lw=1, ls='--', label='Ground truth')
    ax.fill_between([0,1],[0,0],[1,1], color='k', alpha=0.2, label='95% CI')
    ax.set_xlim(100,101)
    ax.legend(frameon=False, loc='center', bbox_to_anchor=(0.47,0.3))
    ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.24)
    if display_type=='pdf':
        plt.savefig(save_name+'.pdf', dpi=600, bbox_inches='tight', pad_inches=0.02)
    elif display_type=='png':
        plt.savefig(save_name+'.png', bbox_inches='tight', pad_inches=0.02)
    elif display_type=='svg':
        plt.savefig(save_name+'.svg', bbox_inches='tight', pad_inches=0.02)
    else:
        plt.show()
        

raise SystemExit
base_path = os.getcwd()
## get AJ estimate of CIF

rcode = f"""library(survival)
library(R.matlab)
library(readxl)

# get data
df <- read_excel(file.path('{base_path}', 'data.xlsx'))

# inverse cens to occur
names(df)[names(df)=="time_death"] <- "event2_time"
names(df)[names(df)=="time_outcome"] <- "event1_time"
names(df)[names(df)=="cens_death"] <- "event2_occur"
names(df)[names(df)=="cens_outcome"] <- "event1_occur"
df$event2_occur = 1-df$event2_occur
df$event1_occur = 1-df$event1_occur
df$id = 1:nrow(df)
rownames(df) <- NULL

# AJ estimate of cumulative incidence for competing risk
etime <- with(df, ifelse(event1_occur==0, event2_time, event1_time))
event <- with(df, ifelse(event1_occur==0, 2*event2_occur, 1))
event <- factor(event, 0:2, labels=c("censor", "event1", "event2"))
AJ_fit <- survfit(Surv(etime, event) ~ 1, data=df, id=id)
writeMat(file.path('{base_path}', 'AJ_output.mat'), time=AJ_fit$time, val=AJ_fit$pstate, lower=AJ_fit$lower, upper=AJ_fit$upper, states=AJ_fit$states)
"""
with open('Rcode.R', 'w') as ff:
    ff.write(rcode)
data_types = ['mean(z)', 'mean(z)-stdev(z)', 'mean(z)+stdev(z)']
z_range = np.std(zp)/3
AJ_mats = {}
for data_type in data_types:
    if data_type=='mean(z)':
        target_z = np.mean(zp)
    elif data_type=='mean(z)-stdev(z)':
        target_z = np.mean(zp) - np.std(zp)
    elif data_type=='mean(z)+stdev(z)':
        target_z = np.mean(zp) + np.std(zp)
    df2 = df[(zp>=target_z-z_range)&(zp<=target_z+z_range)].reset_index(drop=True)
    df2.to_excel('data.xlsx', index=False)
    if os.path.exists('AJ_output.mat'):
        os.remove('AJ_output.mat')
    subprocess.check_call(['Rscript', 'Rcode.R'])
    AJ_mats[data_type] = sio.loadmat('AJ_output.mat')
    os.remove('AJ_output.mat')
os.remove('Rcode.R')
os.remove('data.xlsx')
