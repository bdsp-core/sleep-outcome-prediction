import pandas as pd


cols = ['hypoxia_burden', 'hypoxia_T90', 'hypoxia_T90desat', 'hypoxia_T90nonspecific']

# MGH
df1 = pd.read_csv('/data/cdac Dropbox/a_People_BIDMC/Haoqi/SleepBasedOutcomePrediction/shared_data/MGH/outcomes_hypoxia.csv')
df1 = df1[df1.hypoxia_note=='all good'].reset_index(drop=True)

# de-id version
df2 = pd.read_csv('/data/cdac Dropbox/a_People_BIDMC/Haoqi/SleepBasedOutcomePrediction/github-repo/features_MGH_deid.csv')
for col in cols:
    if col in df2.columns:
        df2 = df2.drop(columns=col)
df = df2.merge(df1[['HashID', 'DOVshifted']+cols], on=['HashID','DOVshifted'], how='left', validate='1:1')
df.to_csv('/data/cdac Dropbox/a_People_BIDMC/Haoqi/SleepBasedOutcomePrediction/github-repo/features_MGH_deid.csv', index=False)

# id version
df2 = pd.read_csv('/data/cdac Dropbox/a_People_BIDMC/Haoqi/SleepBasedOutcomePrediction/shared_data/MGH/to_be_used_features_NREM.csv')
df3 = pd.concat([df2, df[cols]], axis=1)
df3.to_csv('/data/cdac Dropbox/a_People_BIDMC/Haoqi/SleepBasedOutcomePrediction/shared_data/MGH/to_be_used_features_NREM.csv', index=False)
