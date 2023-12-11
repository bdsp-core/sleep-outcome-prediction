from dateutil.parser import parse
import datetime
from itertools import product
import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from statsmodels.stats.proportion import proportions_ztest
from tqdm import tqdm
   
    
if __name__=='__main__':
    outcome = str(sys.argv[1])
    
    if outcome in ['IntracranialHemorrhage', 'IschemicStroke']:
        outcome_col = 'stk_date'
        prev_outcome_col = 'prev_stk'
    elif outcome=='Myocardial_Infarction':
        outcome_col = 'mi_date'
        prev_outcome_col = 'prev_mi'
    elif outcome=='CongestiveHeartFailure':
        outcome_col = 'chf_date'
        prev_outcome_col = 'prev_chf'
    elif outcome=='Atrial_Fibrillation':
        outcome_col = None
        prev_outcome_col = 'afibprevalent'
    elif outcome=='DiabetesII':
        outcome_col = None
        prev_outcome_col = 'ParRptDiab'
    else:
        raise NotImplementedError(outcome)
        
    # read all pts
    df_pt = pd.read_csv('SHHS_features_NREM.csv')
    
    """
    df_pt_MGH = pd.read_csv('../../shared_data/MGH/to_be_used_features.csv')
    
    df_pt_sex_mask = {0:df_pt.Sex==0, 1:df_pt.Sex==1} # save time
    Nattempt = 10
    assert len(df_pt_MGH)>len(df_pt)
    for N in tqdm(np.arange(500,len(df_pt),100)[::-1]):
        #
        sample_ids = []
        for k in range(Nattempt):
            np.random.seed(2022+k)
            ids = np.random.choice(len(df_pt_MGH), N, replace=False)
            p1 = ks_2samp(df_pt_MGH.Age.values, df_pt_MGH.Age.iloc[ids].values).pvalue
            p2 = ks_2samp(df_pt_MGH.BMI.values, df_pt_MGH.BMI.iloc[ids].values).pvalue
            p3 = proportions_ztest([np.sum(df_pt_MGH.Sex==0), np.sum(df_pt_MGH.Sex.iloc[ids]==0)], [len(df_pt_MGH), len(ids)])[1]
            if min(p1,p2,p3)>0.05:
                sample_ids.append(ids)
        if len(sample_ids)==0:
            print(f'sample {N} out of {len(df_pt)} did not find same distribution of Age, BMI, and Sex')
            continue
        
        for ids in sample_ids:
            notfoundmatch = False
            match_mask = np.zeros(len(df_pt)).astype(bool)
            for ii in ids:
                age = df_pt_MGH.Age.iloc[ii]
                sex = df_pt_MGH.Sex.iloc[ii]
                bmi = df_pt_MGH.BMI.iloc[ii]
                match_ids = (np.abs(df_pt.Age-age)<=5)&(np.abs(df_pt.BMI-bmi)<=5)&df_pt_sex_mask[sex]
                match_ids = np.where(match_ids)[0]
                if len(match_ids)==0:
                    notfoundmatch = True
                    print(1)
                    break
                #
                if np.all(match_mask[match_ids]):
                    notfoundmatch = True
                    print(2)
                    break
                else:
                    false_id = np.where(np.logical_not(match_mask[match_ids]))[0][0]
                    match_mask[match_ids[false_id]] = True
            if not notfoundmatch:
                break
        if notfoundmatch:
            print(f'{N}: not found match')
        else:
            print(f'{N}: found match!')
            break
    assert not notfoundmatch
    """
    
    # read variables
    shhs_dir = '/media/sunhaoqi/Seagate Backup Plus Drive/SHHS/datasets-0.14.0'
    df_outcome = pd.read_csv(os.path.join(shhs_dir, 'shhs-cvd-summary-dataset-0.14.0.csv'))
    if outcome in ['IschemicStroke', 'IntracranialHemorrhage']:
        df_outcome2 = pd.read_csv(os.path.join(shhs_dir, 'shhs-cvd-events-dataset-0.14.0.csv'))
        if outcome=='IschemicStroke':
            # stk_type==11 is ischemic
            df_outcome2 = df_outcome2[df_outcome2.stk_type==11].drop_duplicates('nsrrid', ignore_index=True)
        else:
            # stk_type==6 Intra-cerebral hemorrhage
            # stk_type==12 Sub-arachnoid hemorrhage
            df_outcome2 = df_outcome2[np.in1d(df_outcome2.stk_type, [6,12])].drop_duplicates('nsrrid', ignore_index=True)
        df_outcome = df_outcome.merge(df_outcome2[['stk_type','nsrrid']], on='nsrrid', how='left')
        
    elif outcome=='DiabetesII':
        df_outcome2 = pd.read_csv(os.path.join(shhs_dir, 'shhs1-dataset-0.14.0.csv'))
        df_outcome = df_outcome.merge(df_outcome2[['ParRptDiab','nsrrid']], on='nsrrid', how='left')
    
    """
    Outcome
    happened before sleep study: cens_outcomes = 0, time2outcome = -
    happened after sleep study:  cens_outcomes = 0, time2outcome = +
    did not happen:              cens_outcomes = 1, time2outcome = time2death/censor
    does not know happened:      cens_outcomes = nan, time2outcome = nan
    
    Death
    happened after sleep study:  cens_death = 0, time2death = time2death/censor
    did not happen:              cens_death = 1, time2death = time2death/censor
    does not know happened:      cens_death = nan, time2death = nan
    """
    
    # generate time to event
    cens_outcomes = []
    time2outcomes = []
    cens_deaths = []
    time2deaths = []
    for i in range(len(df_pt)):
        sid = df_pt.nsrrid.iloc[i]
        
        ids = np.where(df_outcome.nsrrid==sid)[0]
        assert len(ids)==1
            
        cens_outcome = np.nan
        time2outcome = np.nan
        cens_death = np.nan
        time2death = np.nan
        
        outcomes = df_outcome.iloc[ids[0]]
        # check existing condition
        if outcomes[prev_outcome_col]>0:
            cens_outcome = 0
            time2outcome = -1
        # check future condition
        else:
            if outcome_col is not None:  # know future condition
                if outcome=='IschemicStroke':
                    cens_outcome = int(np.isnan(outcomes[outcome_col])|(outcomes.stk_type!=11))
                elif outcome=='IntracranialHemorrhage':
                    #cens_outcome = int(np.isnan(outcomes[outcome_col])|(~np.in1d(outcomes.stk_type, [6,12])))
                    cens_outcome = int(np.isnan(outcomes[outcome_col])|((outcomes.stk_type!=6)&(outcomes.stk_type!=12)))
                else:
                    cens_outcome = int(np.isnan(outcomes[outcome_col]))
                if cens_outcome==1:
                    time2outcome = outcomes.censdate/365
                else:
                    time2outcome = outcomes[outcome_col]/365.
            elif outcomes[prev_outcome_col]==0:   # know did not happen before sleep study, does not know happen after sleep study, so actually don't know, but for classifying existing conditions, we need to indicate if it happened before sleep study
                cens_outcome = 1
                time2outcome = outcomes.censdate/365
            #else:   # does not know happened before sleep study, does not know happen after sleep study, so don't know
            #    all nan
        
        if not pd.isna(outcomes.vital):
            cens_death = outcomes.vital  # 0: Dead  1: Alive
            time2death = outcomes.censdate/365
            
        if time2outcome>time2death:
            print(f'{sid}: cens_outcome={cens_outcome}, time2outcome={time2outcome}, cens_death={cens_death}, time2death={time2death}')
            time2death = time2outcome
            
        cens_outcomes.append(cens_outcome)
        time2outcomes.append(time2outcome)
        cens_deaths.append(cens_death)
        time2deaths.append(time2death)
    
    df_pt['cens_outcome'] = cens_outcomes
    df_pt['time_outcome'] = time2outcomes
    df_pt['cens_death'] = cens_deaths
    df_pt['time_death'] = time2deaths
    df_time2event = df_pt[['nsrrid', 'cens_outcome', 'time_outcome', 'cens_death', 'time_death']]
            
    # check time_outcome<0 & cens_outcome=1, do not exist
    assert np.sum((df_time2event.time_outcome<0)&(df_time2event.cens_outcome==1))==0
    # make time_death<=0 to nan
    ids = df_time2event.time_death<=0
    df_time2event.loc[ids, 'cens_death'] = np.nan
    df_time2event.loc[ids, 'time_death'] = np.nan
    
    #df_time2event = df_time2event.dropna().reset_index(drop=True)
    df_time2event.to_excel('SHHS_time2event_%s.xlsx'%outcome, index=False)
    
