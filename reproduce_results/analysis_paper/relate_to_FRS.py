import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt


if __name__=='__main__':
    outcomes = [
    'IntracranialHemorrhage', 'IschemicStroke', 'Dementia', 'MCI+Dementia',
    'Atrial_Fibrillation', 'Myocardial_Infarction', 'DiabetesII', 'Hypertension',
    'Bipolar_Disorder', 'Depression',
    'Death'
    ]

    # load FRS data
    df_frs1 = pd.read_excel('cohort_final_use_HIV_FRS.xlsx', sheet_name='Sheet1')
    df_frs2 = pd.read_excel('cohort_final_use_HIV_FRS.xlsx', sheet_name='FRS')
    assert len(df_frs1)==len(df_frs2) and np.all(df_frs1.StudyID==df_frs2.StudyID)
    df_frs = pd.concat([
        df_frs1[['MRN', 'CA', 'BAI_adj', 'Sex', 'Race', 'ESS', 'BMI']],
        df_frs2[['closest_HDL_val', 'closest_totchol_val', 'closest_sys_val', 'FRS', 'Imputed_FRS']]
        ], axis=1)
    df_frs = df_frs.rename(columns={'closest_HDL_val':'HDL', 'closest_totchol_val':'chol', 'closest_sys_val':'SBP'})

    for outcome in outcomes:
        print(outcome)
        model_type = 'CoxPH_CompetingRisk' if outcome!='Death' else 'CoxPH'

        # load outcome data
        if outcome!='Death':
            df_outcome = pd.read_excel(f'time2event_{outcome}.xlsx')
            df_outcome = df_outcome[(~pd.isna(df_outcome.cens_death))&(df_outcome.time_death>0)&(~pd.isna(df_outcome.cens_outcome))&(df_outcome.time_outcome>0)].reset_index(drop=True)
        else:
            df_outcome = pd.read_excel('time2event_IschemicStroke.xlsx')
            df_outcome = df_outcome[df_outcome.time_death>0].reset_index(drop=True)

        outcome_res = sio.loadmat(f'survival_results_NREM_bt1000/results_{outcome}_{model_type}.mat')
        # make covariates consstant
        Xnames = [outcome_res['xnames'][x,0].item().strip() for x in range(len(outcome_res['xnames']))]
        coef = outcome_res['coef'][:len(Xnames),0]
        Xtr = outcome_res['Xtr']
        Xmean = outcome_res['Xmean']
        Xstd  = outcome_res['Xstd']
        Xmean_names = [outcome_res['Xmean_names'][x,0].item().strip() for x in range(len(outcome_res['Xmean_names']))]
        assert len(Xtr)==len(df_outcome)
        df_outcome['zp_with_cov'] = outcome_res['zptr'].flatten()
        Xtr = KNNImputer(n_neighbors=10).fit_transform(Xtr)

        idx = Xmean_names.index('Age'); Xtr[:,idx] = (50-Xmean[idx])/Xstd[idx]
        idx = Xmean_names.index('Sex'); Xtr[:,idx] = (0.5-Xmean[idx])/Xstd[idx]
        idx = Xmean_names.index('BMI'); Xtr[:,idx] = (30-Xmean[idx])/Xstd[idx]
        idx = Xmean_names.index('MedBenzo'); Xtr[:,idx] = (0-Xmean[idx])/Xstd[idx]
        idx = Xmean_names.index('MedAntiDep'); Xtr[:,idx] = (0-Xmean[idx])/Xstd[idx]
        idx = Xmean_names.index('MedSedative'); Xtr[:,idx] = (0-Xmean[idx])/Xstd[idx]
        idx = Xmean_names.index('MedAntiEplipetic'); Xtr[:,idx] = (0-Xmean[idx])/Xstd[idx]
        idx = Xmean_names.index('MedStimulant'); Xtr[:,idx] = (0-Xmean[idx])/Xstd[idx]
        ids = [Xmean_names.index(x) for x in Xnames]
        df_outcome['zp'] = np.dot(Xtr[:,ids], coef)

        # merge and preprocess
        df = df_outcome.merge(df_frs, on='MRN', how='inner')
        df.loc[np.in1d(df.HDL, ['REFUSED','CANCELLED']), 'HDL'] = np.nan
        df.HDL = df.HDL.astype(float)
        df.loc[np.in1d(df.chol, ['REFUSED','CANCELLED']), 'chol'] = np.nan
        df.chol = df.chol.astype(float)

        #ids = (~pd.isna(df.zp))&(~pd.isna(df.HDL));print(np.sum(ids), spearmanr(df.zp[ids], df.HDL[ids]))
        #ids = (~pd.isna(df.zp))&(~pd.isna(df.chol));print(np.sum(ids),spearmanr(df.zp[ids], df.chol[ids]))
        #ids = (~pd.isna(df.zp))&(~pd.isna(df.SBP));print(np.sum(ids), spearmanr(df.zp[ids], df.SBP[ids]))
        ids = (~pd.isna(df.zp))&(~pd.isna(df.FRS));print(np.sum(ids), spearmanr(df.zp[ids], df.FRS[ids]))

        """
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ids = (~pd.isna(df.zp))&(~pd.isna(df.FRS))
        ax.scatter(df.FRS[ids], df.zp[ids], s=5, c='k')
        ax.set_xlabel('FRS')
        ax.set_ylabel('Z')

        plt.tight_layout()
        plt.show()
        """
