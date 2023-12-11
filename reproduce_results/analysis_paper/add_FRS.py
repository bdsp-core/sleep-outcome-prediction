import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt


if __name__=='__main__':
    # load FRS data
    df_frs1 = pd.read_excel('cohort_final_use_HIV_FRS.xlsx', sheet_name='Sheet1')
    df_frs2 = pd.read_excel('cohort_final_use_HIV_FRS.xlsx', sheet_name='FRS')
    assert len(df_frs1)==len(df_frs2) and np.all(df_frs1.StudyID==df_frs2.StudyID)
    df_frs = pd.concat([
        df_frs1[['MRN', 'study_date']].rename(columns={'study_date':'DateOfVisit'}),
        df_frs2[['Imputed_FRS']],
        ], axis=1)
    df_frs['DateOfVisit'] = pd.to_datetime(df_frs.DateOfVisit).dt.strftime('%Y-%m-%d')

    # load study data
    df = pd.read_csv('to_be_used_features_NREM.csv')
    df2 = df.merge(df_frs,on=['MRN','DateOfVisit'],how='left',validate='1:1')
    df2.to_csv('to_be_used_features_NREM.csv', index=False)

