import os
import sys
import subprocess
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib
from matplotlib.gridspec import GridSpec
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')


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
    
    #outcome_groups = ['neuro', 'cardio', 'psych', 'mortality']
    #outcome_groups_txt = ['Neurological outcomes', 'Cardiovascular outcomes', 'Psychiatric outcomes', 'Mortality']
    
    outcomes = ['IschemicStroke', 'Myocardial_Infarction', 'Death']
    outcomes_txt = ['Ischemic stroke', 'Myocardial\ninfarction', 'Death']
        
    folder = '/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction (1)/code-haoqi/SHHS_validation/survival_results_NREM_AHIFALSE2'
    save_name = 'CI_curve_SHHS'
    
    sexs = [0,1]
    sexs_txt = ['Female', 'Male']
    curve_names = ['<Q1,sex=0', '[Q1--Q3],sex=0', '>Q3,sex=0', '<Q1,sex=1', '[Q1--Q3],sex=1', '>Q3,sex=1']
    
    colors = ['k','b','r']
    panel_xoffset = -0.07
    panel_yoffset = 1.03
    plt.close()
    fig = plt.figure(figsize=(12,4.5))
    gs = GridSpec(2,len(outcomes)+1) # +1 for legend
    import pdb;pdb.set_trace()
    for axi, outcome in enumerate(outcomes):
        print(outcome)
        outcome_txt = outcomes_txt[axi]
        
        if outcome in ['Death']:
            model_type = 'CoxPH'
        else:
            model_type = 'CoxPH_CompetingRisk'

        mat_path = os.path.join(folder, f'results_{outcome}_{model_type}.mat')
        if not os.path.exists(mat_path):
            continue
        mat = sio.loadmat(mat_path)
        
        for sexi, sex in enumerate(sexs):
            sex_txt = sexs_txt[sexi]
            
            # AJ
            aj_outcome_idx = [np.array(mat['AJ_curve_tes_states'].flatten()[i]).item() for i in range(len(mat['AJ_curve_tes_states']))].index('event1')
            aj_time = mat['AJ_curve_tes_time'].flatten()
            if mat['AJ_curve_tes_val'].ndim==2:
                mat['AJ_curve_tes_val'] = mat['AJ_curve_tes_val'][..., np.newaxis]
            aj_val_1 = mat['AJ_curve_tes_val'][:,curve_names.index(f'[Q1--Q3],sex={sex}'),aj_outcome_idx]
            aj_val_2 = mat['AJ_curve_tes_val'][:,curve_names.index(f'<Q1,sex={sex}'),aj_outcome_idx]
            aj_val_3 = mat['AJ_curve_tes_val'][:,curve_names.index(f'>Q3,sex={sex}'),aj_outcome_idx]
            
            # Cox
            
            cox_time = mat['cox_curve_tes_bt_time'].flatten()
            cox_outcome_idx = [np.array(mat['cox_curve_tes_bt_states'].flatten()[i]).item() for i in range(len(mat['cox_curve_tes_bt_states']))].index('event1')
            if mat['cox_curve_tes_bt_val'].ndim==3:
                mat['cox_curve_tes_bt_val'] = mat['cox_curve_tes_bt_val'][..., np.newaxis]
            cox_val_1 = mat['cox_curve_tes_bt_val'][:,0,curve_names.index(f'[Q1--Q3],sex={sex}'),cox_outcome_idx]
            cox_val_2 = mat['cox_curve_tes_bt_val'][:,0,curve_names.index(f'<Q1,sex={sex}'),cox_outcome_idx]
            cox_val_3 = mat['cox_curve_tes_bt_val'][:,0,curve_names.index(f'>Q3,sex={sex}'),cox_outcome_idx]
            
            cox_CI_1 = np.percentile(mat['cox_curve_tes_bt_val'][:,1:,curve_names.index(f'[Q1--Q3],sex={sex}'),cox_outcome_idx], (2.5, 97.5), axis=1) 
            cox_CI_2 = np.percentile(mat['cox_curve_tes_bt_val'][:,1:,curve_names.index(f'<Q1,sex={sex}'),cox_outcome_idx], (2.5, 97.5), axis=1) 
            cox_CI_3 = np.percentile(mat['cox_curve_tes_bt_val'][:,1:,curve_names.index(f'>Q3,sex={sex}'),cox_outcome_idx], (2.5, 97.5), axis=1)
            
            ax = fig.add_subplot(gs[sexi, axi])
            #aj_mat = sio.loadmat(f'../{result_folder}/AJ_output_{outcome}_{model_type}.mat')
            #aj_outcome_idx = [aj_mat['states'][i,0][0] for i in range(len(aj_mat['states']))].index('event1')
            #ax.fill_between(aj_mat['time'], aj_mat['lower'][:,aj_outcome_idx], aj_mat['upper'][:,aj_outcome_idx], step='pre', color='k', alpha=0.2)
            #ax.step(aj_mat['time'], aj_mat['val'][:,aj_outcome_idx], ls='--', c='k', label='Aalen-Johansen estimator (ground-truth)')
            for ii in [0,1,2]:
                cox_CI = eval(f'cox_CI_{ii+1}')
                #aj_mat = AJ_mats['mean(z)']
                #aj_outcome_idx = [aj_mat['states'][i,0][0] for i in range(len(aj_mat['states']))].index('event1')
                #ax.fill_between(aj_mat['time'], aj_mat['lower'][:,aj_outcome_idx]*100, aj_mat['upper'][:,aj_outcome_idx]*100, step='pre', color='k', alpha=0.2)
                #ax.step(aj_mat['time'], aj_mat['val'][:,aj_outcome_idx]*100, ls='--', c='k', label='Aalen-Johansen estimator: mean(z)')
                ax.fill_between(cox_time, cox_CI[0]*100, cox_CI[1]*100, step='pre', color=colors[ii], alpha=0.2)
            
            for ii in [0,1,2]:
                aj_val  = eval(f'aj_val_{ii+1}')
                ax.step(aj_time, aj_val*100, c=colors[ii], ls='--', lw=1)
            
            for ii in [0,1,2]:
                cox_val    = eval(f'cox_val_{ii+1}')
                ax.step(cox_time, cox_val*100, c=colors[ii], lw=2)
            
            ax.text(0.04, 0.95, f'{outcome_txt}\n{sex_txt}', ha='left', va='top', transform=ax.transAxes)
            
            ax.set_xlim([0, 10])
            ax.set_xticks([0,2,4,6,8,10])
            if outcome in ['IschemicStroke', 'Myocardial_Infarction']:
                ax.set_ylim([0, 12])
                ax.set_yticks([0,2,4,6,8,10])
            elif outcome in ['Hypertension', 'Depression', 'Death']:
                ax.set_ylim([0, 40])
                ax.set_yticks([0,10,20,30,40])
            #ax.set_ylim([-0.01, max(survprob_1.max(), survprob_2.max(), survprob_3.max())*100+0.1])
            #ax.legend(frameon=False, loc='upper left')
            if axi//2==1:
                ax.set_xlabel('Time since PSG (year)')
            else:
                if axi!=1:
                    ax.set_xticklabels([])
            if axi%2==0:
                ax.set_ylabel('P(outcome) (%)')
            ax.yaxis.grid(True)
            ax.text(panel_xoffset-0.07*(axi%2==0), panel_yoffset, chr(ord('A')+axi), ha='right', va='top', transform=ax.transAxes, fontweight='bold')
            sns.despine()

    # legend panel
    ax = fig.add_subplot(gs[1,len(outcomes)])
    ax.plot([0,1],[0,0], c='r', lw=2, label='Higher than Q3')#f'median(z)+{dz}')
    ax.plot([0,1],[0,0], c='k', lw=2, label='Within Q1 and Q3')
    ax.plot([0,1],[0,0], c='b', lw=2, label='Lower than Q1')
    ax.plot([0,1],[0,0], c='k', lw=1, ls='--', label='Aalen-Johansen')
    ax.fill_between([0,1],[0,0],[1,1], color='k', alpha=0.2, label='95% CI')
    ax.set_xlim(100,101)
    ax.legend(frameon=False, loc='center', bbox_to_anchor=(0.47,0.38))
    ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.14)
    if display_type=='pdf':
        plt.savefig(save_name+'.pdf', dpi=600, bbox_inches='tight', pad_inches=0.02)
    elif display_type=='png':
        plt.savefig(save_name+'.png', bbox_inches='tight', pad_inches=0.02)
    else:
        plt.show()
        
