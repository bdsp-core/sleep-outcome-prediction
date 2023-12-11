import os
import numpy as np
import pandas as pd
import scipy.io as sio
import subprocess
from tqdm import tqdm
#import matplotlib.pyplot as plt


def get_CIC(ids, folder, outcome, model_type, rcode_filename='tmp.R', output_filename='tmp.mat', event1_only=True):
    ids_R = ','.join(map(str,ids+1))
    save_path = os.path.join(folder, output_filename)
    rcode = f"""library(survival)
library(readxl)
library(R.matlab)
library(reticulate)  # for sklearn.impute.KNNImputer

if ('{outcome}'=='Death') {{
  outcome2 <- 'IschemicStroke'
}} else {{
  outcome2 <- '{outcome}'
}}
dfy <- read_excel(file.path('{folder}', sprintf('code-haoqi/time2event_%s.xlsx',outcome2)))
dfX <- read.csv(file.path('{folder}', 'shared_data/MGH/to_be_used_features_NREM.csv'))
dfX <- subset(dfX, select=-c(DateOfVisit, TwinDataID, Path) )
dfy <- subset(dfy, select=-c(PatientID, MRN, DateOfVisit) )
df <- cbind(dfX, dfy)
if ('{outcome}'=='Death') {{
  ids <- df$time_death>0
}} else {{
  ids <- (!is.na(df$cens_death))&(df$time_death>0)&(!is.na(df$cens_outcome))&(df$time_outcome>0)
}}
df <- df[ids,]

mat<- readMat(file.path('{folder}', 'code-haoqi/survival_results_NREM_bt1000/results_{outcome}_{model_type}.mat'))
Xnames <- unlist(mat$Xmean.names)
X <- as.matrix(df[,Xnames])
X <- ( X-t(replicate(nrow(X), mat$Xmean)) ) / t(replicate(nrow(X), mat$Xstd))
sklearn.impute <- import("sklearn.impute")
knn.model <- sklearn.impute$KNNImputer(n_neighbors=10L)
knn.model$fit(X)
df[,Xnames] <- knn.model$transform(X)

model <- readRDS(file.path('{folder}', 'code-haoqi/survival_results_NREM_bt1000/model_{outcome}_{model_type}.rda'))
Nmodel <- length(model)
res <- list()
for (i in 1:Nmodel) {{
  res_ <- survfit(model[[i]], df[c({ids_R}),])
  res_ <- summary(res_)
  res[[length(res)+1]] <- res_
}}
survtime <- res[[1]]$time
if ('{model_type}'=='CoxPH') {{
  survprob <- 0
  for (i in 1:Nmodel) {{
    survprob <- survprob + 1-res[[i]]$surv
  }}
  survprob <- survprob/Nmodel
  survstate <- c('event1')
}} else {{
  survprob <- 0
  for (i in 1:Nmodel) {{
    survprob <- survprob + res[[i]]$pstate
  }}
  survprob <- survprob/Nmodel
  survstate <- res[[1]]$states
}}
writeMat('{save_path}', survtime=survtime, survprob=survprob, survstate=survstate)
"""
    rcode_path = os.path.join(folder, rcode_filename)
    with open(rcode_path, 'w') as ff:
        ff.write(rcode)
    subprocess.check_call(['Rscript', rcode_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res = sio.loadmat(save_path)
    
    os.remove(rcode_path)
    os.remove(save_path)
    
    survstate = [x.item() for x in res['survstate'].flatten()]
    if event1_only:
        idx = survstate.index('event1')
        if res['survprob'].ndim==2:
            res['survprob'] = res['survprob'][...,np.newaxis]
        return res['survtime'], res['survprob'][:,:,idx]
    else:
        return res['survtime'], res['survprob'], survstate
        
        
if __name__=='__main__':
    outcomes = [
        'IntracranialHemorrhage', 'IschemicStroke', 'Dementia', 'MCI+Dementia',
        'Atrial_Fibrillation', 'Myocardial_Infarction', 'DiabetesII', 'Hypertension',
        'Bipolar_Disorder', 'Depression',
        'Death'
    ]
    outcomes_txt = [
        'Intracranial hemorrhage', 'Ischemic stroke', 'Dementia', 'MCI or Dementia',
        'Atrial fibrillation', 'Myocardial infarction', 'Type 2 diabetes', 'Hypertension',
        'Bipolar disorder', 'Depression',
        'Death'
    ]
    
    cov_names = [
        'Age', 'Sex', 'MedBenzo', 'MedAntiDep',#, 'BMI'
        'MedSedative', 'MedAntiEplipetic', 'MedStimulant']
    cov_names_txt = [
        'Age (year)', 'Sex (female=0, male=1)', 'Taking benzodiazepine medication (yes=1, no=0)', 'Taking antidepressant medication (yes=1, no=0)',
        'Taking sedative medication (yes=1, no=0)', 'Taking anticonvulsant medication (yes=1, no=0)', 'Taking stimulant medication (yes=1, no=0)']
    # ignore BMI since it has small coefficient, and we do not expect monotonic relationship
    
    folder = '/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction (1)'
    dfX = pd.read_csv(os.path.join(folder, 'shared_data/MGH/to_be_used_features_NREM.csv'))
    df_scores = []
    for oi, outcome in enumerate(outcomes):
        print(outcome)
        
        # load df
        if outcome=='Death':
            dfy = pd.read_excel(os.path.join(folder, 'code-haoqi/time2event_IschemicStroke.xlsx'))
        else:
            dfy = pd.read_excel(os.path.join(folder, f'code-haoqi/time2event_{outcome}.xlsx'))
        assert np.all(dfX.MRN==dfy.MRN)
        assert np.all(dfX.DateOfVisit==dfy.DateOfVisit)
        df = pd.concat([dfX, dfy.drop(columns=['PatientID', 'MRN', 'DateOfVisit'])], axis=1)
        
        # load fitting results
        if outcome=='Death':
            model_type = 'CoxPH'
        else:
            model_type = 'CoxPH_CompetingRisk'
        mat = sio.loadmat(os.path.join(folder, f'code-haoqi/survival_results_NREM/results_{outcome}_{model_type}.mat'))        
        Xmean_names = np.array([x.item() for x in mat['Xmean_names'].flatten()])
        Xmean = mat['Xmean']
        Xstd  = mat['Xstd']
        Xcoef_names = np.array([x.item() for x in mat['xnames'].flatten()])
        if outcome=='Death':
            coef = mat['coef'][:, 0]
        else:
            coef = mat['coef'][:len(mat['coef'])//2, 0]
        
        # validate correctness
        df2 = df.copy()
        df2.loc[:,Xmean_names] = (df2.loc[:,Xmean_names].values - Xmean)/Xstd
        z_all = np.dot(df2[Xcoef_names], coef)
        
        if outcome=='Death':
            is_future_ids = df.time_death>0
        else:
            is_future_ids = (~pd.isna(df.cens_death))&(df.time_death>0)&(~pd.isna(df.cens_outcome))&(df.time_outcome>0)
        z = z_all[is_future_ids]
        ids = ~np.isnan(z)
        assert np.allclose(z[ids], mat['zptr'][ids])
        
        # get z(sleep) and coef of unscaled covariates
        sleep_coef_ids = ~np.in1d(Xcoef_names, cov_names)
        zsleep = np.dot(df2[Xcoef_names[sleep_coef_ids]], coef[sleep_coef_ids])
        coef_unscaled_cov = [coef[Xcoef_names==x][0]/Xstd[Xmean_names==x][0] for x in cov_names]
        #constant = 
        
        # get FRS-style score table
        if oi==0:
            cov_range_txt = [f'{np.nanmin(df[x]):.1g} to {np.nanmax(df[x]):.1g}' for x in cov_names]
            data={
            'Variable name':['Sleep EEG composite']+cov_names_txt,
            'Valid range':[f'{np.nanmin(z_all):.1g} to {np.nanmax(z_all):.1g}']+cov_range_txt,
            f'{outcomes_txt[oi]} weight':[f'{x:.1g}' for x in [1]+coef_unscaled_cov]}
        else:
            data={f'{outcomes_txt[oi]} weight':[f'{x:.1g}' for x in [1]+coef_unscaled_cov]}
        df_score = pd.DataFrame(data=data)
        df_scores.append(df_score)
        
        # get percentiles of z and its X-year risk
        #plt.hist(mat['zptr'],bins=50);plt.show()
        Nlevel = 100
        smooth_halfwindow = 5
        z2_all = zsleep + np.dot(df[cov_names].values, coef_unscaled_cov)
        z2 = z2_all[is_future_ids]
        notnan_ids = np.where(~np.isnan(z2))[0]
        level_center_ids = np.round(np.linspace(0,len(z2[notnan_ids])-1,Nlevel)).astype(int)
        z_levels = z2[notnan_ids][np.argsort(z2[notnan_ids])[level_center_ids]]
        risks_1y = []
        risks_5y = []
        risks_10y = []
        for ii in tqdm(level_center_ids):
            ids = ii+np.arange(-smooth_halfwindow,smooth_halfwindow+1)
            ids = ids[(ids>=0)&(ids<len(z2[notnan_ids]))]
            survtime, survprob = get_CIC(notnan_ids[np.argsort(z2[notnan_ids])[ids]], folder, outcome, model_type, rcode_filename='tmp2.R', output_filename='tmp2.mat')
            risks_1y.append( survprob[np.argmin(np.abs(survtime-1))].mean() )
            risks_5y.append( survprob[np.argmin(np.abs(survtime-5))].mean() )
            risks_10y.append( survprob[np.argmin(np.abs(survtime-10))].mean() )
        df_z2risk = pd.DataFrame(data={
            'z':z_levels,
            'Risk1Year':risks_1y,
            'Risk5Year':risks_5y,
            'Risk10Year':risks_10y,})
        df_z2risk.to_csv(f'step6_output_z2risk_{outcome}.csv', index=False)
        
    df_scores = pd.concat(df_scores, axis=1)
    #df_scores = df_scores[df_scores['Variable name']!='BMI'].reset_index(drop=True)
    df_scores.to_excel('step6_output_score_table.xlsx', index=False)
    
