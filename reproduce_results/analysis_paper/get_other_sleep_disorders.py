import datetime
import numpy as np
import pandas as pd

df = pd.read_csv('/data/Dropbox (BIDMC Dropbox Team)/Haoqi/SleepBasedOutcomePrediction/shared_data/MGH/to_be_used_features_NREM.csv')
df['DateOfVisit'] = pd.to_datetime(df.DateOfVisit)

df2 = pd.read_csv('/data/Dropbox (Partners HealthCare)/dementia_detection_ElissaYe/medical_data/RPDR_Dia_All.csv')
df2 = df2[df2.MRN_Type=='MGH'].reset_index(drop=True)
df2['MRN'] = df2.MRN.astype(int)
df2 = df2[np.in1d(df2.Code_Type, ['ICD9', 'ICD10', 'Phenotype'])].reset_index(drop=True)
df2['Diagnosis_Name'] = df2.Diagnosis_Name.astype(str)
df2['Code'] = df2.Code.astype(str)

diseases_keywords = {
    'insomnia': ['Diagnosis_Name', 'insom'],
    'hypersomnia': ['Diagnosis_Name', 'hypersom'],
    'circadian disorders': ['Diagnosis_Name', 'circadian'],
    'narcolepsy and cataplexy': ['Diagnosis_Name', '(?:narcol|catapl)'],
    'parasomnia': ['Diagnosis_Name', 'parasom'],
    'movement disorders': ['Code', '(?:G47.6[12]|327.5[12]|G25.81|333.94)'],
    }
diseases = list(diseases_keywords.keys())

df_res = df[['MRN', 'DateOfVisit']]
for disease, v in diseases_keywords.items():
    print(disease)
    ids = df2[v[0]].str.contains(v[1], case=False)
    df3 = df2[ids].reset_index(drop=True)
    df3['Date'] = pd.to_datetime(df3.Date)
    df4 = df.merge(df3, on='MRN', how='inner')
    ids = (df4.Date>=df4.DateOfVisit-datetime.timedelta(days=365*5))&(df4.Date<=df4.DateOfVisit+datetime.timedelta(days=365*1))
    mrns = df4[ids].MRN.unique()
    df5 = pd.DataFrame(data={'MRN':mrns, disease:[1]*len(mrns)})
    df_res = df_res.merge(df5, on='MRN',how='left')

df_res = df_res.fillna(0)
for d in diseases:
    df_res[d] = df_res[d].astype(int)

print(df_res[diseases].sum())
df_res.to_csv('other_sleep_disorders.csv', index=False)
"""
insomnia                    2212
hypersomnia                 1069
circadian disorders          148
narcolepsy and cataplexy     224
parasomnia                    45
movement disorders           861
"""

