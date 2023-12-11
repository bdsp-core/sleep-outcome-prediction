from collections import defaultdict
import os
import numpy as np
import scipy.io as sio
import pandas as pd
import rpy2.robjects as robjects


outcomes = [
    'IntracranialHemorrhage', 'IschemicStroke', 'Dementia', 'MCI+Dementia',
    'Atrial_Fibrillation', 'Myocardial_Infarction', 'DiabetesII', 'Hypertension',
    'Bipolar_Disorder', 'Depression',
    'Death']

eeg_folder = '/data/cdac Dropbox/a_People_BIDMC/Haoqi/SleepBasedOutcomePrediction/code-haoqi/survival_results_NREM_bt1000'
ahi_folder = '/data/cdac Dropbox/a_People_BIDMC/Haoqi/SleepBasedOutcomePrediction/code-haoqi/survival_results_AHI_bt1000'
hb_folder = '/data/cdac Dropbox/a_People_BIDMC/Haoqi/SleepBasedOutcomePrediction/code-haoqi/survival_results_HB_bt1000'
red_folder = '/data/cdac Dropbox/a_People_BIDMC/Haoqi/SleepBasedOutcomePrediction/code-haoqi/survival_results_RED_bt1000'
bai_folder = '/data/cdac Dropbox/a_People_BIDMC/Haoqi/SleepBasedOutcomePrediction/code-haoqi/survival_results_BAI_bt1000'
se_folder = '/data/cdac Dropbox/a_People_BIDMC/Haoqi/SleepBasedOutcomePrediction/code-haoqi/survival_results_SE_bt1000'
remperc_folder = '/data/cdac Dropbox/a_People_BIDMC/Haoqi/SleepBasedOutcomePrediction/code-haoqi/survival_results_REMPerc_bt1000'
waso_folder = '/data/cdac Dropbox/a_People_BIDMC/Haoqi/SleepBasedOutcomePrediction/code-haoqi/survival_results_WASO_bt1000'

df = defaultdict(list)
for o in outcomes:
    print(o)
    if o=='Death':
        model = 'CoxPH'
    else:
        model = 'CoxPH_CompetingRisk'
    df['Outcome'].append(o)
    for datatype in ['eeg', 'ahi', 'hb', 'red', 'bai', 'se', 'remperc', 'waso']:
        #if datatype=='diff':
        #    df[].append()
        #else:
        if datatype=='eeg':
            folder = eeg_folder
            prefix = 'SleepEEG'
        elif datatype=='ahi':
            folder = ahi_folder
            prefix = 'AHI'
        elif datatype=='hb':
            folder = hb_folder
            prefix = 'HB'
        elif datatype=='red':
            folder = red_folder
            prefix = 'RED'
        elif datatype=='bai':
            folder = bai_folder
            prefix = 'BAI'
        elif datatype=='se':
            folder = se_folder
            prefix = 'SE'
        elif datatype=='remperc':
            folder = remperc_folder
            prefix = 'REMPerc'
        elif datatype=='waso':
            folder = waso_folder
            prefix = 'WASO'
        rmodel = robjects.r['readRDS'](os.path.join(folder, f'model_{o}_{model}.rda'))
        aic = np.mean([robjects.r['extractAIC'](rmodel[x])[1] for x in range(len(rmodel))])

        mat = sio.loadmat(os.path.join(folder, f'results_{o}_{model}.mat'))
        val, lb, ub = np.r_[mat['cindex_te'][0], np.percentile(mat['cindex_te'][1:], (2.5,97.5))]
        
        df[f'{prefix}_CIndex'].append(f'{val:.3f} ({lb:.3f}-{ub:.3f})')
        #df[f'{prefix}_AIC'].append(f'{aic:.1f}')
        
        #for yi, year in enumerate(range(2,11)):
        #    val, lb, ub = np.r_[mat['auc_te'][yi,0], np.percentile(mat['auc_te'][yi,1:], (2.5,97.5))]
        #    df[f'{prefix}_CDAUC_{year}y'].append(f'{val:.2f} ({lb:.2f}-{ub:.2f})')
df = pd.DataFrame(data=df)
print(df)
import pdb;pdb.set_trace()
df.to_excel('perf_table.xlsx', index=False)
