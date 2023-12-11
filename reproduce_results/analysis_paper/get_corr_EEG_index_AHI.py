from collections import defaultdict
import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import spearmanr


folder = r'survival_results_NREM_bt1000'

outcomes = [
    'IntracranialHemorrhage', 'IschemicStroke', 'Dementia', 'MCI+Dementia',
    'Atrial_Fibrillation', 'Myocardial_Infarction', 'DiabetesII', 'Hypertension',
    'Bipolar_Disorder', 'Depression',
    'Death' ]

df_res = defaultdict(list)
for outcome in outcomes:
    if outcome=='Death':
        method = 'CoxPH'
    else:
        method = 'CoxPH_CompetingRisk'

    result_path = os.path.join(folder, f'results_{outcome}_{method}.mat')
    mat = sio.loadmat(result_path)
    Xnames = np.array([x[0] for x in mat['xnames'].flatten()])

    df_path = os.path.join(folder, f'df_{outcome}_{method}.csv')
    df = pd.read_csv(df_path)

    coef_path = os.path.join(folder, f'coef_{outcome}_{method}.csv')
    coef = pd.read_csv(coef_path)
    coef = coef.rename(columns={'Unnamed: 0':'Name'})
    if method=='CoxPH_CompetingRisk':
        coef = coef[coef.Name.str.endswith('_1:2')].reset_index(drop=True)
        coef['Name'] = coef.Name.str.replace('_1:2','')
    coef = coef.set_index('Name')

    assert all(Xnames==coef.index.values)
    z = mat['zptr'].flatten()
    assert np.allclose(np.dot(df[Xnames], coef.coef), z)
    
    Xnames_exclude = ['Age', 'Sex', 'BMI', 'MedBenzo', 'MedAntiDep',
        'MedSedative', 'MedAntiEplipetic', 'MedStimulant']
    Xnames_eeg = [x for x in Xnames if x not in Xnames_exclude]
    z_eeg = np.dot(df[Xnames_eeg], coef.coef[Xnames_eeg])

    ahi = df.AHI.values
    ids = (~np.isnan(ahi))&(~np.isnan(z_eeg))
    ahi = ahi[ids]
    z_eeg = z_eeg[ids]
    corr, pval = spearmanr(z_eeg, ahi)

    df_res['Outcome'].append(outcome)
    df_res['Corr'].append(corr)
    df_res['PVal'].append(pval)
    df_res['N'].append(len(ahi))

df_res = pd.DataFrame(df_res)
print(df_res)
df_res.to_csv('corr_EEG_index_AHI.csv', index=False)

