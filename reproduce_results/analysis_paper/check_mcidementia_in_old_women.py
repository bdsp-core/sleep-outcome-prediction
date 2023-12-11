import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

outcome = 'MCI+Dementia'
dfy = pd.read_excel(f'/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction (1)/code-haoqi/time2event_{outcome}.xlsx')
dfX = pd.read_csv('/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction (1)/shared_data/MGH/to_be_used_features_NREM.csv')
assert np.all(dfX.MRN==dfy.MRN)
dfX = dfX.drop(columns=['MRN', 'DateOfVisit', 'PatientID'])
df = pd.concat([dfy, dfX], axis=1)

df = df[(df.Age>=70)&(df.Sex==0)&(~pd.isna(df.cens_outcome))].reset_index(drop=True)
df['MCIorDementiaIn5Year'] = ((df.cens_outcome==0)&(df.time_outcome<=5)).astype(int)

mod = smf.logit(formula='MCIorDementiaIn5Year ~ Age + alpha_bandpower_kurtosis_O_NREM', data=df).fit()
print(mod.summary())
