import sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
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
    
    outcomes = [
        'IntracranialHemorrhage', 'IschemicStroke', 'Dementia', 'MCI+Dementia',
        'Atrial_Fibrillation', 'Myocardial_Infarction', 'DiabetesII', 'Hypertension',
        'Bipolar_Disorder', 'Depression',
        'Death']
    outcomes_txt = [
        'Intracranial\nhemorrhage', 'Ischemic stroke', 'Dementia', 'MCI or Dementia',
        'Atrial fibrillation', 'Myocardial infarction', 'Type 2 diabetes', 'Hypertension',
        'Bipolar disorder', 'Depression',
        'Death']
    
    panel_xoffset = -0.14
    panel_yoffset = 1.02
    save_name = 'z2risk_conversion'
    plt.close()
    fig = plt.figure(figsize=(13,7.5))
    for axi, outcome in enumerate(outcomes):
        print(outcome)
        outcome_txt = outcomes_txt[axi]        
        df = pd.read_csv(f'../code-haoqi/step6_output_z2risk_{outcome}.csv')
        
        #if axi==0:
        ax = fig.add_subplot(3,4,axi+1)
        #    ax0 = ax
        #else:
        #    ax = fig.add_subplot(3,4,axi+1,sharex=ax0)
        offset = 2
        df = df.iloc[offset:-offset].reset_index(drop=True)
        
        #ax.plot(df.z, df.Risk1Year*100, c='b')
        #ax.plot(df.z, df.Risk5Year*100, c='k')
        #ax.plot(df.z, df.Risk10Year*100, c='r')
        
        ax.plot(df.z, savgol_filter(df.Risk1Year*100,31,2), c='b', lw=2)
        ax.plot(df.z, savgol_filter(df.Risk5Year*100,31,2), c='k', lw=2)
        ax.plot(df.z, savgol_filter(df.Risk10Year*100,31,2), c='r', lw=2)

        ax.text(0.03, 0.95, outcome_txt, ha='left', va='top', transform=ax.transAxes, clip_on=False)
        if axi//4==2 or axi==7:
            ax.set_xlabel('z')
        if axi%4==0:
            ax.set_ylabel('Risk (%)')
        ax.set_xlim(df.z.min(), df.z.max())
        ymax = df[['Risk1Year','Risk5Year','Risk10Year']].values.max()*100
        #ymax = np.ceil(ymax/5)*5
        ax.set_ylim(0,ymax)
        ax.grid(which='both')
        ax.text(panel_xoffset-0.05*(axi%4==0), panel_yoffset, chr(ord('A')+axi), ha='right', va='top', transform=ax.transAxes, fontweight='bold')
        plt.minorticks_on()
        sns.despine()

    # legend panel
    ax = fig.add_subplot(3,4,12)
    ax.plot([0,1],[0,0], c='r', lw=2, label='10 year')
    ax.plot([0,1],[0,0], c='k', lw=2, label='5 year')
    ax.plot([0,1],[0,0], c='b', lw=2, label='1 year')
    ax.set_xlim(100,101)
    ax.legend(frameon=False, loc='center', bbox_to_anchor=(0.47,0.3))
    ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.18, wspace=0.24)
    if display_type=='pdf':
        plt.savefig(save_name+'.pdf', dpi=600, bbox_inches='tight', pad_inches=0.02)
    elif display_type=='png':
        plt.savefig(save_name+'.png', bbox_inches='tight', pad_inches=0.02)
    elif display_type=='svg':
        plt.savefig(save_name+'.svg', bbox_inches='tight', pad_inches=0.02)
    else:
        plt.show()
