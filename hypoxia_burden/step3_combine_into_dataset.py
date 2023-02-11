import numpy as np
import pandas as pd


cols = ['hypoxia_burden', 'hypoxia_T90', 'hypoxia_T90desat', 'hypoxia_T90nonspecific']

# MGH
df1 = pd.read_csv('outcomes_hypoxia.csv')
df1 = df1[df1.hypoxia_note=='all good'].reset_index(drop=True)

# de-id version
df2 = pd.read_csv('../features_MGH_deid.csv')
df2 = df2.drop(columns=cols+['AHI'], errors='ignore')

df_ahi = pd.read_excel('dataset_AHI_AHI_annot.xlsx')
df_ahi = df_ahi[(~pd.isna(df_ahi.AHI_annot))&(~pd.isna(df_ahi.AHI))].reset_index(drop=True)
df_ahi = df_ahi[np.abs(df_ahi.AHI_annot-df_ahi.AHI)<=5].reset_index(drop=True)
df_ahi['DOVshifted'] = df_ahi.DOVshifted.dt.strftime('%Y-%m-%d')

df = df2.merge(df1[['HashID', 'DOVshifted']+cols], on=['HashID','DOVshifted'], how='left', validate='1:1')
df = df.merge(df_ahi[['HashID', 'DOVshifted', 'AHI']], on=['HashID','DOVshifted'], how='left', validate='1:1')
df.loc[pd.isna(df.AHI), cols] = np.nan
df.to_csv('../features_MGH_deid.csv', index=False)

# id version
#df2 = pd.read_csv('/data/cdac Dropbox/a_People_BIDMC/Haoqi/SleepBasedOutcomePrediction/shared_data/MGH/to_be_used_features_NREM.csv')
#df3 = pd.concat([df2, df[cols+['AHI']]], axis=1)
#df3.to_csv('/data/cdac Dropbox/a_People_BIDMC/Haoqi/SleepBasedOutcomePrediction/shared_data/MGH/to_be_used_features_NREM.csv', index=False)

