import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts


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
        
    outcome = sys.argv[2]
    outcometxt = outcome

    if outcome == 'death':
        df = pd.read_excel('../SHHS_time2event_IschemicStroke.xlsx')
        df = df[(~pd.isna(df.cens_death)) & (~pd.isna(df.time_death)) & (df.time_death>=0)].reset_index(drop=True)
        y = {'event':~df.cens_death.values.astype(bool),
             'time':df.time_death.values,}
    else:
        df = pd.read_excel(f'../SHHS_time2event_{outcome}.xlsx')
        df = df[(~pd.isna(df.cens_outcome)) & (~pd.isna(df.time_outcome)) & (df.time_outcome>=0)].reset_index(drop=True)
        y = {'event':~df.cens_outcome.values.astype(bool),
             'time':df.time_outcome.values,}

    km = KaplanMeierFitter()
    km.fit(y['time'], y['event'])
    tt = np.arange(13)
    
    plt.close()
    fig = plt.figure(figsize=(10,5.6))

    ax = fig.add_subplot(111)
    ax.plot(list(km.survival_function_.index), km.survival_function_.values.flatten(), c='k')
    ax.fill_between(
        list(km.survival_function_.index),
        km.confidence_interval_survival_function_['KM_estimate_lower_0.95'],
        km.confidence_interval_survival_function_['KM_estimate_upper_0.95'],
        alpha=0.3, color='k')
    sns.despine()
    ax.set_xticks(tt)
    ax.set_xlim(tt.min(), tt.max())
    if outcome == 'death':
        ax.set_ylim(0.88,1)
    else:
        ax.set_ylim(0.9,1)
    ax.grid(True)
    ax.set_ylabel(f'Probability of not having {outcome}')
    ax.set_xlabel('Time since sleep recording (year)')
    add_at_risk_counts(km, ax=ax, labels=[''], ypos=-0.45)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3, left=0.17, right=0.97)
    if display_type=='pdf':
        plt.savefig(f'km_curve_SHHS_{outcome}.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
    elif display_type=='png':
        plt.savefig(f'km_curve_SHHS_{outcome}.png', bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()
        
