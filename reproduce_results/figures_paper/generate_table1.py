from collections import defaultdict
import os
import numpy as np
import pandas as pd


if __name__=='__main__':
    outcomes = [
        'Overall',
        'IntracranialHemorrhage', 'IschemicStroke', 'Dementia', 'MCI+Dementia',
        'Atrial_Fibrillation', 'Myocardial_Infarction', 'DiabetesII', 'Hypertension',
        'Bipolar_Disorder', 'Depression',
        'Death'
    ]
    outcomes_txt = [
        'Overall',
        'Intracranial hemorrhage', 'Ischemic stroke', 'Dementia', 'MCI or Dementia',
        'Atrial fibrillation', 'Myocardial infarction', 'Type 2 diabetes', 'Hypertension',
        'Bipolar disorder', 'Depression',
        'Death'
    ]
        
    folder = '/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction (1)'
            
    df_table = defaultdict(list)
    for axi, outcome in enumerate(outcomes):
        outcome_txt = outcomes_txt[axi]
        if outcome=='Overall':
            df_all = pd.read_csv(os.path.join(folder, 'shared_data/MGH/to_be_used_features.csv'))
            df_all['TotalSleepTime'] = df_all.TotalSleepTime/60. # convert to hour
            df_all['DateOfVisit'] = pd.to_datetime(df_all.DateOfVisit)
            df = df_all
            assert not any(df[['MRN','DateOfVisit']].duplicated())
            
            # get race and ethnicity
            df2 = pd.concat([pd.read_csv('/data/interesting_side_projects/sleepdata_generate_mastersheet_with_LightsOffOn_ESS_BMI_race_eth/grass_studies_list_with_age_lightsoffon_ESS_BMI_race_eth_twindataid.csv'), pd.read_csv('/data/interesting_side_projects/sleepdata_generate_mastersheet_with_LightsOffOn_ESS_BMI_race_eth/natus_studies_list_with_age_lightsoffon_ESS_BMI_race_eth_twindataid.csv')], axis=0, ignore_index=True)
            ids = (~pd.isna(df2.MRN))&(~df2.MRN.astype(str).str.contains('(?:X|/)'))
            df2 = df2[ids]
            df2['MRN'] = df2.MRN.astype(int)
            df2['DateOfVisit'] = pd.to_datetime(df2.DateOfVisit)
            df2 = df2.drop_duplicates(['MRN','DateOfVisit'],ignore_index=True)
            
            df = df.merge(df2[['MRN','DateOfVisit', 'Race','Ethnicity']], on=['MRN','DateOfVisit'], how='left')
            import pdb;pdb.set_trace()
            """
Counter({'White': 6652, 'Black': 527, 'Other': 415, 'Hispanic': 414, 'Asian': 332, nan: 315, 'American-Native': 17, 'Middle-Eastern': 1})
White: 6652/8673, 77%
Black: 527/8673, 6%
Hispanic: 414/8673, 5%
Asian: 332/8673, 4%
American-Native: 17/8673, 0.2%
Middle-Eastern: 1/8673, 0.01%
Unknown: 730/8673, 8%

Counter({'Non-Hispanic': 7487, 'Hispanic': 456, 'Other': 415, nan: 315})
Non-Hispanic: 7487/8673, 86%
Hispanic: 456/8673, 5%
Unknown: 730/8673, 8%
            """
        else:
            if outcome=='Death':
                df = pd.read_excel(os.path.join(folder, 'code-haoqi/time2event_IntracranialHemorrhage.xlsx'))
                df = df.drop(columns=['cens_outcome', 'time_outcome'])
                df = df.rename(columns={'cens_death':'cens_outcome', 'time_death':'time_outcome'})
            else:
                df = pd.read_excel(os.path.join(folder, f'code-haoqi/time2event_{outcome}.xlsx'))
                
            assert np.all(df.MRN==df_all.MRN)
            assert len(df.MRN.unique())==len(df)
            df = pd.concat([df_all[['Age', 'Sex', 'BMI', 'AHI', 'TotalSleepTime', 'SleepEfficiency', 'N1PercTST', 'N2PercTST', 'N3PercTST', 'REMPercTST']], df], axis=1)
            df = df[df.cens_outcome==0].reset_index(drop=True)
        
        df_table['Outcome'].append(outcome_txt)
        df_table['# Subject'].append(len(set(df.MRN)))
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
            df_table['# First outcome then deceased'].append(len(set(df.MRN[df.cens_death==0])))
    
    df_table = pd.DataFrame(data=df_table)
    df_table.to_excel('table1.xlsx', index=False)
    
