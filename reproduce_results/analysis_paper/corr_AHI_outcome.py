import numpy as np
import pandas as pd
from scipy.stats import ttest_ind,mannwhitneyu
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest
from sklearn.metrics import confusion_matrix


#outcome = 'IschemicStroke'
outcome = 'IntracranialHemorrhage'

aa=pd.read_excel(f'/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction/code-haoqi/time2event_{outcome}.xlsx')
bb=pd.read_csv('/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction/shared_data/MGH/to_be_used_features_NREM.csv')
assert np.all(aa.MRN==bb.MRN)
ids=pd.isna(aa.cens_outcome)|((aa.cens_outcome==0)&(aa.time_outcome<=5))
#ttest_ind(bb.AHI[ids].values,bb.AHI[~ids].values)
mannwhitneyu(bb.AHI[ids].values,bb.AHI[~ids].values)

kk=aa.cens_outcome[bb.AHI>30].values
kk2=aa.cens_outcome[bb.AHI<5].values
proportions_ztest([np.sum(pd.isna(kk)|(kk==0)),np.sum(pd.isna(kk2)|(kk2==0))], [len(kk),len(kk2)])
proportions_ztest([np.sum(pd.isna(kk)),np.sum(pd.isna(kk2))], [len(kk),len(kk2)])
proportions_ztest([np.sum(kk==0),np.sum(kk2==0)], [len(kk),len(kk2)])


ids=((bb.AHI>30)|(bb.AHI<5))
cf=confusion_matrix((bb.AHI[ids]>30).astype(int),(pd.isna(aa.cens_outcome[ids])|(aa.cens_outcome[ids]==0)).astype(int))
chi2_contingency(cf)
ids=((bb.AHI>20)|(bb.AHI<5))
cf=confusion_matrix((bb.AHI[ids]>20).astype(int),(pd.isna(aa.cens_outcome[ids])|(aa.cens_outcome[ids]==0)).astype(int))
chi2_contingency(cf)
cf=confusion_matrix((bb.AHI>30).astype(int),(pd.isna(aa.cens_outcome)|(aa.cens_outcome==0)).astype(int))

chi2_contingency(cf)
cf=confusion_matrix((bb.AHI>20).astype(int),(pd.isna(aa.cens_outcome)|(aa.cens_outcome==0)).astype(int))
chi2_contingency(cf)

