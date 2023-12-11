from collections import defaultdict
import os
import numpy as np
import pandas as pd


if __name__=='__main__':
    outcomes = ['Overall', 'IschemicStroke', 'Myocardial_Infarction', 'Death']
    outcomes_txt = ['Overall', 'Ischemic stroke', 'Myocardial infarction', 'Death']
        
    folder = '/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction (1)'
            
    df_table = defaultdict(list)
    for axi, outcome in enumerate(outcomes):
        outcome_txt = outcomes_txt[axi]
        if outcome=='Overall':
            df_all = pd.read_csv(os.path.join(folder, 'shared_data/SHHS/to_be_used_features.csv'))
            df_all['TotalSleepTime'] = df_all.TotalSleepTime/60. # convert to hour
            df = df_all
        else:
            if outcome=='Death':
                df = pd.read_excel(os.path.join(folder, 'code-haoqi/SHHS_validation/SHHS_time2event_IschemicStroke.xlsx'))
                df = df.drop(columns=['cens_outcome', 'time_outcome'])
                df = df.rename(columns={'cens_death':'cens_outcome', 'time_death':'time_outcome'})
            else:
                df = pd.read_excel(os.path.join(folder, f'code-haoqi/SHHS_validation/SHHS_time2event_{outcome}.xlsx'))
                
            assert np.all(df.nsrrid==df_all.nsrrid)
            assert len(df.nsrrid.unique())==len(df)
            df = pd.concat([df_all[['Age', 'Sex', 'BMI', 'AHI', 'TotalSleepTime', 'SleepEfficiency', 'N1PercTST', 'N2PercTST', 'N3PercTST', 'REMPercTST']], df], axis=1)
            df = df[df.cens_outcome==0].reset_index(drop=True)
        
        df_table['Outcome'].append(outcome_txt)
        df_table['# Subject'].append(len(set(df.nsrrid)))
        #df_table['# PSG'].append(len(df))
        if outcome=='Overall':
            df_table['Average time to event (year)'].append(np.nan)
        else:
            df_table['Average time to event (year)'].append(df.time_outcome[df.cens_outcome==0].mean())
        df_table['Age (year)'].append(f'{df.Age.mean():.1f} ({df.Age.std():.1f})')
        df_table['Sex (%Male)'].append(f'{(df.Sex==1).mean()*100:.1f}%')
        df_table['BMI (kg/m2)'].append(f'{df.BMI.mean():.1f} ({df.BMI.std():.1f})')
        df_table['AHI (/hour)'].append(f'{df.AHI.mean():.1f} ({df.AHI.std():.1f})')
        df_table['TST (hour)'].append(f'{df.TotalSleepTime.mean():.1f} ({df.TotalSleepTime.std():.1f})')
        df_table['N1%'].append(f'{df.N1PercTST.mean():.0f} ({df.N1PercTST.std():.0f})')
        df_table['N2%'].append(f'{df.N2PercTST.mean():.0f} ({df.N2PercTST.std():.0f})')
        df_table['N3%'].append(f'{df.N3PercTST.mean():.0f} ({df.N3PercTST.std():.0f})')
        df_table['REM%'].append(f'{df.REMPercTST.mean():.0f} ({df.REMPercTST.std():.0f})')
        if outcome in ['Overall', 'Death']:
            df_table['# First outcome then deceased'].append(np.nan)
        else:
            df_table['# First outcome then deceased'].append(len(set(df.nsrrid[df.cens_death==0])))
    
    df_table = pd.DataFrame(data=df_table)
    df_table.to_excel('table1_SHHS.xlsx', index=False)
    
