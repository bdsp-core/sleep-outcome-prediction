import os
import numpy as np
import pandas as pd

        
folder = '/data/cdac Dropbox/a_People_BIDMC/Haoqi/SleepBasedOutcomePrediction'
df_res = {'Variable':[], 'MGH':[], 'SHHS':[]}

df_mgh = pd.read_csv(os.path.join(folder, 'shared_data/MGH/to_be_used_features.csv'))
df_shhs = pd.read_csv(os.path.join(folder, 'shared_data/SHHS/to_be_used_features.csv'))

df_shhs2 = pd.read_csv('/media/sunhaoqi/Seagate Backup Plus Drive/SHHS/datasets-0.14.0/shhs2-dataset-0.14.0.csv', encoding='latin1')
df_shhs = df_shhs.merge(df_shhs2[['nsrrid','sh318a','sh318b']],on='nsrrid', how='left',validate='1:1')

assert len(df_mgh)==len(df_mgh.MRN.unique())
assert len(df_shhs)==len(df_shhs.nsrrid.unique())
df_res['Variable'].append('Number of participants')
df_res['MGH'].append(len(df_mgh))
df_res['SHHS'].append(len(df_shhs))

df_res['Variable'].append('Age (year) (mean, min-max)')
df_res['MGH'].append(f'{df_mgh.Age.mean():.0f}, {df_mgh.Age.min():.0f}-{df_mgh.Age.max():.0f}')
df_res['SHHS'].append(f'{df_shhs.Age.mean():.0f}, {df_shhs.Age.min():.0f}-{df_shhs.Age.max():.0f}')

df_res['Variable'].append('Male sex, n (%)')
df_res['MGH'].append(f'{df_mgh.Sex.sum()} ({df_mgh.Sex.mean()*100:.0f}%)')
df_res['SHHS'].append(f'{df_shhs.Sex.sum()} ({df_shhs.Sex.mean()*100:.0f}%)')

df_res['Variable'].append('BMI (kg/m2) (mean, std)')
df_res['MGH'].append(f'{df_mgh.BMI.mean():.0f} ({df_mgh.BMI.std():.0f})')
df_res['SHHS'].append(f'{df_shhs.BMI.mean():.0f} ({df_shhs.BMI.std():.0f})')

#df_res['Variable'].append('ESS (mean, min-max)')
#df_res['MGH'].append(f'{df_mgh.ESS.mean():.0f}, {df_mgh.ESS.min():.0f}-{df_mgh.ESS.max():.0f}')
#df_res['SHHS'].append(f'{df_shhs.ESS.mean():.0f}, {df_shhs.ESS.min():.0f}-{df_shhs.ESS.max():.0f}')

df_res['Variable'].append('No apnea: <5/hour, n (%)')
df_res['MGH'].append(f'{np.sum(df_mgh.AHI<5)} ({np.mean(df_mgh.AHI<5)*100:.0f}%)')
df_res['SHHS'].append(f'{np.sum(df_shhs.AHI<5)} ({np.mean(df_shhs.AHI<5)*100:.0f}%)')

df_res['Variable'].append('Mild apnea: 5-15/hour, n (%)')
df_res['MGH'].append(f'{np.sum((df_mgh.AHI>=5)&(df_mgh.AHI<15))} ({np.mean((df_mgh.AHI>=5)&(df_mgh.AHI<15))*100:.0f}%)')
df_res['SHHS'].append(f'{np.sum((df_shhs.AHI>=5)&(df_shhs.AHI<15))} ({np.mean((df_shhs.AHI>=5)&(df_shhs.AHI<15))*100:.0f}%)')

df_res['Variable'].append('Moderate apnea: 15-30/hour, n (%)')
df_res['MGH'].append(f'{np.sum((df_mgh.AHI>=15)&(df_mgh.AHI<30))} ({np.mean((df_mgh.AHI>=15)&(df_mgh.AHI<30))*100:.0f}%)')
df_res['SHHS'].append(f'{np.sum((df_shhs.AHI>=15)&(df_shhs.AHI<30))} ({np.mean((df_shhs.AHI>=15)&(df_shhs.AHI<30))*100:.0f}%)')

df_res['Variable'].append('Severe apnea: >30/hour, n (%)')
df_res['MGH'].append(f'{np.sum(df_mgh.AHI>30)} ({np.mean(df_mgh.AHI>30)*100:.0f}%)')
df_res['SHHS'].append(f'{np.sum(df_shhs.AHI>30)} ({np.mean(df_shhs.AHI>30)*100:.0f}%)')

df_res['Variable'].append('Benzodiazepine, n (%)')
df_res['MGH'].append(f'{np.sum(df_mgh.MedBenzo)} ({np.mean(df_mgh.MedBenzo)*100:.0f}%)')
df_res['SHHS'].append(f'{np.sum(df_shhs.MedBenzo)} ({np.mean(df_shhs.MedBenzo)*100:.0f}%)')

df_res['Variable'].append('Antidepressant, n (%)')
df_res['MGH'].append(f'{np.sum(df_mgh.MedAntiDep)} ({np.mean(df_mgh.MedAntiDep)*100:.0f}%)')
df_res['SHHS'].append(f'{np.sum(df_shhs.MedAntiDep)} ({np.mean(df_shhs.MedAntiDep)*100:.0f}%)')

df_res['Variable'].append('Sedative, n (%)')
df_res['MGH'].append(f'{np.sum(df_mgh.MedSedative)} ({np.mean(df_mgh.MedSedative)*100:.0f}%)')
df_res['SHHS'].append('N.A.')

df_res['Variable'].append('Anti-seizure, n (%)')
df_res['MGH'].append(f'{np.sum(df_mgh.MedAntiEplipetic)} ({np.mean(df_mgh.MedAntiEplipetic)*100:.0f}%)')
df_res['SHHS'].append('N.A.')

df_res['Variable'].append('Stimulant, n (%)')
df_res['MGH'].append(f'{np.sum(df_mgh.MedStimulant)} ({np.mean(df_mgh.MedStimulant)*100:.0f}%)')
df_res['SHHS'].append(f'{np.sum(df_shhs.MedStimulant)} ({np.mean(df_shhs.MedStimulant)*100:.0f}%)')

df_res['Variable'].append('Insomnia, n (%)')
df_res['MGH'].append('2212 (26%)')
df_res['SHHS'].append(f'{df_shhs.sh318a.sum()} ({df_shhs.sh318a.sum()/len(df_shhs)*100:.0f}%)')

df_res['Variable'].append('Hypersomnia, n (%)')
df_res['MGH'].append('1069 (12%)')
df_res['SHHS'].append('N.A.')

df_res['Variable'].append('Restless leg syndrome, n (%)')
df_res['MGH'].append('861 (10%)')
df_res['SHHS'].append(f'{df_shhs.sh318b.sum()} ({df_shhs.sh318b.sum()/len(df_shhs)*100:.0f}%)')

df_res = pd.DataFrame(data=df_res)
print(df_res)
df_res.to_excel('table_S1.xlsx', index=False)


