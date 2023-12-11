import os
import subprocess
import sys
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')


if __name__=='__main__':
    if len(sys.argv)>=2:
        if 'pdf' in sys.argv[1].lower():
            display_type = 'pdf'
        elif 'png' in sys.argv[1].lower():
            display_type = 'png'
        else:
            display_type = 'show'
    else:
        raise SystemExit('python %s show/png/pdf'%__file__)
        
    result_folder = '../results'
    model_type = 'CoxPH_CompetingRisk'
    outcome = str(sys.argv[2])
    
    result_mat = sio.loadmat(f'{result_folder}/SHHS_prediction_{outcome}_{model_type}.mat')
    predicted_curve_names = ['mean(z)', 'mean(z)-stdev(z)', 'mean(z)+stdev(z)', '0%', '0.5%','1%', '2.5%', '10%', '25%', '50%', '75%', '90%', '97.5%', '99%', '99.5%', '100%']
    
    cox_outcome_idx = [result_mat['survstates'][i,0][0] for i in range(len(result_mat['survstates']))].index('event1')
    survtime = result_mat['survtime']
    survprob_middle = result_mat['survprob'][:,predicted_curve_names.index('50%'),cox_outcome_idx]
    survprob_lower = result_mat['survprob'][:,predicted_curve_names.index('25%'),cox_outcome_idx]
    survprob_upper = result_mat['survprob'][:,predicted_curve_names.index('75%'),cox_outcome_idx]
    zp = result_mat['zp']
    
    ##
    #goodids = np.ones(len(zp), dtype=bool)
    #goodids[[8240, 8727, 8971, 9041, 9048]] = False
    #zp = zp[goodids]
    
    ## get AJ estimate of CIF
    
    df = pd.read_excel(f'../SHHS_time2event_{outcome}.xlsx')
    df = df[(~pd.isna(df.cens_death)) & (df.time_death>=0) & (~pd.isna(df.cens_outcome)) & (df.time_outcome>=0)].reset_index(drop=True)
    
    base_path = '/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction/code-haoqi/SHHS_validation/figures'
    rcode = f"""library(survival)
library(R.matlab)
library(readxl)

# get data
df <- read_excel(file.path('{base_path}', 'data.xlsx'))
    
# inverse cens to occur
names(df)[names(df)=="time_death"] <- "event2_time"
names(df)[names(df)=="time_outcome"] <- "event1_time"
names(df)[names(df)=="cens_death"] <- "event2_occur"
names(df)[names(df)=="cens_outcome"] <- "event1_occur"
df$event2_occur = 1-df$event2_occur
df$event1_occur = 1-df$event1_occur
df$id = 1:nrow(df)
rownames(df) <- NULL

# AJ estimate of cumulative incidence for competing risk
etime <- with(df, ifelse(event1_occur==0, event2_time, event1_time))
event <- with(df, ifelse(event1_occur==0, 2*event2_occur, 1))
event <- factor(event, 0:2, labels=c("censor", "event1", "event2"))
AJ_fit <- survfit(Surv(etime, event) ~ 1, data=df, id=id)
writeMat(file.path('{base_path}', 'AJ_output.mat'), time=AJ_fit$time, val=AJ_fit$pstate, lower=AJ_fit$lower, upper=AJ_fit$upper, states=AJ_fit$states)
    """
    with open('Rcode.R', 'w') as ff:
        ff.write(rcode)
    data_types = ['50%', '25%', '75%']
    z_range = np.std(zp)/10
    AJ_mats = {}
    for data_type in data_types:
        if data_type=='50%':
            target_z = np.median(zp)
        elif data_type=='25%':
            target_z = np.percentile(zp, 25)
        elif data_type=='75%':
            target_z = np.percentile(zp, 75)
        perc = np.mean(np.sort(zp)<=target_z)*100
        upper = np.percentile(zp, min(perc+10, 100)) # +/-10% percentile around this value
        lower = np.percentile(zp, max(perc-10, 0))
        df2 = df[(zp>=lower)&(zp<=upper)].reset_index(drop=True)
        df2.to_excel('data.xlsx', index=False)
        #if os.path.exists('AJ_output.mat'):
        #    os.remove('AJ_output.mat')
        subprocess.check_call(['Rscript', 'Rcode.R'])
        AJ_mats[data_type] = sio.loadmat('AJ_output.mat')
        os.remove('AJ_output.mat')
    os.remove('Rcode.R')
    os.remove('data.xlsx')
    
    years = [1,2,3,4,5,6,7,8]
    rr_m_stds = []
    rr_p_stds = []
    for year in years:
        idx = np.argmin(np.abs(survtime-year))
        rr_m_stds.append( survprob_lower[idx] / survprob_middle[idx] )
        rr_p_stds.append( survprob_upper[idx] / survprob_middle[idx] )
    years.append('Average')
    rr_m_stds.append(np.mean(rr_m_stds))
    rr_p_stds.append(np.mean(rr_p_stds))
    df = pd.DataFrame(data={
            'year':years,
            'RR(Q3 : median)':rr_p_stds,
            'RR(Q1 : median)':rr_m_stds,})
    df.to_csv(f'risk_ratio_table_{model_type}_{outcome}.csv', index=False)
    
    
    plt.close()
    fig = plt.figure(figsize=(10,7))
    
    ax = fig.add_subplot(111)
    
    aj_mat = AJ_mats['50%']
    aj_outcome_idx = [aj_mat['states'][i,0][0] for i in range(len(aj_mat['states']))].index('event1')
    #ax.fill_between(aj_mat['time'], aj_mat['lower'][:,aj_outcome_idx]*100, aj_mat['upper'][:,aj_outcome_idx]*100, step='pre', color='k', alpha=0.2)
    ax.step(aj_mat['time'], aj_mat['val'][:,aj_outcome_idx]*100, ls='--', c='k', label='Aalen-Johansen estimator: median')
    ax.step(survtime, survprob_middle*100, c='k', label='Cox PH with competing risk: median')
    
    aj_mat = AJ_mats['25%']
    aj_outcome_idx = [aj_mat['states'][i,0][0] for i in range(len(aj_mat['states']))].index('event1')
    #ax.fill_between(aj_mat['time'], aj_mat['lower'][:,aj_outcome_idx]*100, aj_mat['upper'][:,aj_outcome_idx]*100, step='pre', color='b', alpha=0.2)
    ax.step(aj_mat['time'], aj_mat['val'][:,aj_outcome_idx]*100, ls='--', c='b', label='Aalen-Johansen estimator: Q1')
    ax.step(survtime, survprob_lower*100, c='b', label='Cox PH with competing risk: Q1')
    
    aj_mat = AJ_mats['75%']
    aj_outcome_idx = [aj_mat['states'][i,0][0] for i in range(len(aj_mat['states']))].index('event1')
    #ax.fill_between(aj_mat['time'], aj_mat['lower'][:,aj_outcome_idx]*100, aj_mat['upper'][:,aj_outcome_idx]*100, step='pre', color='r', alpha=0.2)
    ax.step(aj_mat['time'], aj_mat['val'][:,aj_outcome_idx]*100, ls='--', c='r', label='Aalen-Johansen estimator: Q3')
    ax.step(survtime, survprob_upper*100, c='r', label='Cox PH with competing risk: Q3')
    
    ax.set_xlim([-0.1, 12.5])
    #ax.set_ylim([-0.01, max(survprob_middle.max(), survprob_lower.max(), survprob_upper.max())*100+0.1])
    ax.legend(frameon=False, loc='upper left')
    ax.set_xlabel('Time since sleep recording (year)')
    ax.set_ylabel(f'Probability of {outcome} (%)')
    seaborn.despine()

    plt.tight_layout()
    #plt.subplots_adjust(wspace=0.18)
    if display_type=='pdf':
        plt.savefig(f'SHHS_cumulative_hazard_{outcome}_{model_type}.pdf', dpi=600, bbox_inches='tight', pad_inches=0.01)
    elif display_type=='png':
        plt.savefig(f'SHHS_cumulative_hazard_{outcome}_{model_type}.png', bbox_inches='tight', pad_inches=0.01)
    else:
        plt.show()
