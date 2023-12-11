import os
import sys
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib
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
    
    outcomes = [
        'IntracranialHemorrhage', 'IschemicStroke', 'Dementia', 'MCI+Dementia',
        'Atrial_Fibrillation', 'Myocardial_Infarction', 'DiabetesII', 'Hypertension',
        'Bipolar_Disorder', 'Depression',
        'Death']
    outcomes_txt = [
        'Intracranial\nhemorrhage', 'Ischemic stroke', 'Dementia', 'MCI or\nDementia',
        'Atrial fibrillation', 'Myocardial\ninfarction', 'Type 2\ndiabetes', 'Hypertension',
        'Bipolar disorder', 'Depression',
        'Death']
        
    folder = '/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction'
    curve_names = ['25%', '50%', '75%']
    
    color_nothing = (178/255, 216/255, 178/255)
    color_death   = (208/255, 153/255, 153/255)
    color_outcome = (243/255, 227/255, 187/255)
    panel_xoffset = -0.09
    panel_yoffset = 1.02
    figsize = (12,6.3)
    
    
    plt.close()
    fig = plt.figure(figsize=figsize)
    for axi, outcome in enumerate(outcomes):
        #print(outcome)
        outcome_txt = outcomes_txt[axi]
        if outcome in ['Death']:
            model_type = 'CoxPH'
        else:
            model_type = 'CoxPH_CompetingRisk'
            
        mat = sio.loadmat(os.path.join(folder, f'code-haoqi/survival_results_NREM_AHIFALSE/surv_curves_{outcome}_{model_type}.mat'))
        
        survtime = mat['survtime'].flatten()
        survprob = mat['survprob']*100
        if outcome in ['Death']:
            survprob = np.array([survprob, 100-survprob]).transpose(1,2,0)
            event_names = ['(s0)', 'event2']
        else:
            event_names = [np.array(mat['survstates'].flatten()[i]).item() for i in range(len(mat['survstates']))]
        
        idx = curve_names.index('50%')
        survprob_nothing = survprob[:,idx,event_names.index('(s0)')]
        if 'event1' in event_names:
            survprob_outcome = survprob[:,idx,event_names.index('event1')]
        else:
            survprob_outcome = np.zeros_like(survtime)
        survprob_death = survprob[:,idx,event_names.index('event2')]
        print(f'{outcome}:\tP(outcome)={survprob_outcome[-1]:.0f},\tP(death)={survprob_death[-1]:.0f},\tP(nothing)={survprob_nothing[-1]:.0f}')
        ax = fig.add_subplot(3,4,axi+1)
        ax.fill_between(survtime, 0, survprob_outcome, color=color_outcome)
        ax.fill_between(survtime, survprob_outcome, survprob_outcome+survprob_death, color=color_death)
        ax.fill_between(survtime, survprob_outcome+survprob_death, 100, color=color_nothing)
        
        ax.text(0.04, 0.95, outcome_txt, ha='left', va='top', transform=ax.transAxes)
        
        ax.set_xlim([0, 12.5])
        ax.set_xticks([0,2,4,6,8,10,12])
        if outcome in ['Hypertension', 'Depression']:
            ax.set_ylim([0, 21])
            ax.set_yticks([0,5,10,15,20])
            ax.set_yticklabels(['0','5','10','15',''])
        elif outcome in ['Dementia','MCI+Dementia', 'Atrial_Fibrillation', 'Myocardial_Infarction', 'DiabetesII', 'Bipolar_Disorder', 'Death']:
            ax.set_ylim([0, 10])
            ax.set_yticks([0,2,4,6,8,10])
            ax.set_yticklabels(['0','2','4','6','8',''])
        else:
            ax.set_ylim([0, 5])
        if axi//4==2:
            ax.set_xlabel('Time since PSG (year)')
        else:
            if axi!=7:
                ax.set_xticklabels([])
        if axi==4:
            ax.text(-0.18, 0.5, 'Cumulative state probability (%)', ha='right', va='center', transform=ax.transAxes, rotation=90)
        ax.text(panel_xoffset, panel_yoffset, chr(ord('A')+axi), ha='right', va='top', transform=ax.transAxes, fontweight='bold')#-0.12*(axi%4==0)
        sns.despine()
    
    # legend panel
    ax = fig.add_subplot(3,4,12)
    ax.fill_between([0,1], 0, 1, color=color_outcome, label='Outcome')
    ax.fill_between([0,1], 0, 1, color=color_death, label='Death')
    ax.fill_between([0,1], 0, 1, color=color_nothing, label='No event')
    ax.set_xlim(100,101)
    ax.legend(frameon=False, loc='center', bbox_to_anchor=(0.5,0.4))
    ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.21)
    if display_type=='pdf':
        plt.savefig(f'stacked_plots.pdf', dpi=600, bbox_inches='tight', pad_inches=0.02)
    elif display_type=='png':
        plt.savefig(f'stacked_plots.png', bbox_inches='tight', pad_inches=0.02)
    else:
        plt.show()

