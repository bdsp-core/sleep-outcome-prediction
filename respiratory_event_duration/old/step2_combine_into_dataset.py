import numpy as np
import pandas as pd


cols = ['RED']

# MGH
df1 = pd.read_excel('dataset_RED.xlsx')
df1['DOVshifted'] = df1.DOVshifted.dt.strftime('%Y-%m-%d')

# de-id version
df2 = pd.read_csv('../features_MGH_deid.csv')
import pdb;pdb.set_trace()
df2 = df2.drop(columns=cols, errors='ignore')

df = df2.merge(df1[['HashID', 'DOVshifted']+cols], on=['HashID','DOVshifted'], how='left', validate='1:1')
df.to_csv('../features_MGH_deid.csv', index=False)

# id version
#df2 = pd.read_csv('/data/cdac Dropbox/a_People_BIDMC/Haoqi/SleepBasedOutcomePrediction/shared_data/MGH/to_be_used_features_NREM.csv')
#df3 = pd.concat([df2, df[cols]], axis=1)
#df3.to_csv('/data/cdac Dropbox/a_People_BIDMC/Haoqi/SleepBasedOutcomePrediction/shared_data/MGH/to_be_used_features_NREM.csv', index=False)


