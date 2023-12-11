import os
import numpy as np
import scipy.io as sio
import pandas as pd


folder = '/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction (1)/code-haoqi/SHHS_validation/survival_results_NREM2'
sexs = [0,1]
sexs_txt = ['female', 'male']
curve_names = ['<Q1,sex=0', '[Q1--Q3],sex=0', '>Q3,sex=0', '<Q1,sex=1', '[Q1--Q3],sex=1', '>Q3,sex=1']

outcomes = [
    'IschemicStroke', 'Myocardial_Infarction', 'Death'
]
outcomes_txt = [
    'Ischemic stroke', 'Myocardial infarction', 'Death'
]

for si, sex in enumerate(sexs):
    print(sexs_txt[si])
    q1q3_idx = curve_names.index(f'[Q1--Q3],sex={sex}')
    q1_idx = curve_names.index(f'<Q1,sex={sex}')
    q3_idx = curve_names.index(f'>Q3,sex={sex}')
    
    suffix = '_'+sexs_txt[si]
    rr1_cox = []
    rr2_cox = []
    rr1_aj = []
    rr2_aj = []
    val1_cox = []
    val2_cox = []
    val3_cox = []
    val1_aj = []
    val2_aj = []
    val3_aj = []
    val1_diff = []
    val2_diff = []
    val3_diff = []
    for ii, outcome in enumerate(outcomes):
        print(outcome)
        
        if outcome in ['Death']:
            model_type = 'CoxPH'
        else:
            model_type = 'CoxPH_CompetingRisk'
        AJ_mat = sio.loadmat(os.path.join(folder, f'results_{outcome}_{model_type}.mat'))
        cox_mat = sio.loadmat(os.path.join(folder, f'results_{outcome}_{model_type}.mat'))
        
        ## AJ
        
        vals_bt = np.array([
            AJ_mat['AJ_curve_tes_val'],
            AJ_mat['AJ_curve_tes_lower'],
            AJ_mat['AJ_curve_tes_upper'],
            ])*100
        if vals_bt.ndim==4:
            vals_bt = vals_bt.transpose(1,0,2,3)
        else:
            vals_bt = vals_bt.transpose(1,0,2)
            vals_bt = vals_bt[..., np.newaxis]
        outcome_idx = [np.array(AJ_mat['AJ_curve_tes_states'].flatten()[i]).item() for i in range(len(AJ_mat['AJ_curve_tes_states']))].index('event1')
        vals_bt = vals_bt[..., outcome_idx]  # shape = (#T, 3)
        
        survtime = AJ_mat['AJ_curve_tes_time'].flatten()
        time_idx = np.argmin(np.abs(survtime-10))
        
        val1_aj_mean = vals_bt[time_idx,0,q1_idx]
        val1_aj_lb = vals_bt[time_idx,1,q1_idx]
        val1_aj_ub = vals_bt[time_idx,2,q1_idx]
        val1_aj.append(f'{val1_aj_mean:.1f} ({val1_aj_lb:.1f} - {val1_aj_ub:.1f})')
        
        val2_aj_mean = vals_bt[time_idx,0,q1q3_idx]
        val2_aj_lb = vals_bt[time_idx,1,q1q3_idx]
        val2_aj_ub = vals_bt[time_idx,2,q1q3_idx]
        val2_aj.append(f'{val2_aj_mean:.1f} ({val2_aj_lb:.1f} - {val2_aj_ub:.1f})')
        
        val3_aj_mean = vals_bt[time_idx,0,q3_idx]
        val3_aj_lb = vals_bt[time_idx,1,q3_idx]
        val3_aj_ub = vals_bt[time_idx,2,q3_idx]
        val3_aj.append(f'{val3_aj_mean:.1f} ({val3_aj_lb:.1f} - {val3_aj_ub:.1f})')
        
        # for EEG, use poor(q1) as ref, rr1 is good(q3) vs poor(q1), rr2 is avg(q2) vs poor(q1)
        rr1_mean = vals_bt[time_idx,0,q3_idx] / vals_bt[time_idx,0,q1q3_idx]
        rr1_lb = vals_bt[time_idx,1,q3_idx] / vals_bt[time_idx,2,q1q3_idx]
        rr1_ub = vals_bt[time_idx,2,q3_idx] / vals_bt[time_idx,1,q1q3_idx]
        rr1_aj.append(f'{rr1_mean:.1f} ({rr1_lb:.1f} - {rr1_ub:.1f})')
        
        rr2_mean = vals_bt[time_idx,0,q1q3_idx] / vals_bt[time_idx,0,q1_idx]
        rr2_lb = vals_bt[time_idx,1,q1q3_idx] / vals_bt[time_idx,2,q1_idx]
        rr2_ub = vals_bt[time_idx,2,q1q3_idx] / vals_bt[time_idx,1,q1_idx]
        rr2_aj.append(f'{rr2_mean:.1f} ({rr2_lb:.1f} - {rr2_ub:.1f})')
        
        ## Cox
        
        vals_bt = cox_mat['cox_curve_tes_bt_val']*100
        if vals_bt.ndim==3:
            vals_bt = vals_bt[..., np.newaxis]
        outcome_idx = [np.array(cox_mat['cox_curve_tes_bt_states'].flatten()[i]).item() for i in range(len(cox_mat['cox_curve_tes_bt_states']))].index('event1')
        vals_bt = vals_bt[..., outcome_idx]  # shape = (#T, #bt, 3)
        
        survtime = cox_mat['cox_curve_tes_bt_time'].flatten()
        time_idx = np.argmin(np.abs(survtime-10))
        
        rr1_mean = vals_bt[time_idx,0,q3_idx] / vals_bt[time_idx,0,q1q3_idx]
        rr1_ci = vals_bt[time_idx,1:,q3_idx] / vals_bt[time_idx,1:,q1q3_idx]
        rr1_lb, rr1_ub = np.nanpercentile(rr1_ci, (2.5, 97.5))
        rr1_cox.append(f'{rr1_mean:.1f} ({rr1_lb:.1f} - {rr1_ub:.1f})')
        
        rr2_mean = vals_bt[time_idx,0,q1q3_idx] / vals_bt[time_idx,0,q1_idx]
        rr2_ci = vals_bt[time_idx,1:,q1q3_idx] / vals_bt[time_idx,1:,q1_idx]
        rr2_lb, rr2_ub = np.nanpercentile(rr2_ci, (2.5, 97.5))
        rr2_cox.append(f'{rr2_mean:.1f} ({rr2_lb:.1f} - {rr2_ub:.1f})')
        
        val1_cox_mean = vals_bt[time_idx,0,q1_idx]
        val1_cox_lb, val1_cox_ub = np.nanpercentile(vals_bt[time_idx,1:,q1_idx], (2.5, 97.5))
        if val1_cox_ub<val1_aj_lb or val1_cox_lb>val1_aj_ub:
            sig = ' *'
        else:
            sig = ''
        val1_cox.append(f'{val1_cox_mean:.1f} ({val1_cox_lb:.1f} - {val1_cox_ub:.1f}){sig}')
        val1_diff.append(np.abs(val1_cox_mean-val1_aj_mean))
        
        val2_cox_mean = vals_bt[time_idx,0,q1q3_idx]
        val2_cox_lb, val2_cox_ub = np.nanpercentile(vals_bt[time_idx,1:,q1q3_idx], (2.5, 97.5))
        if val2_cox_ub<val2_aj_lb or val2_cox_lb>val2_aj_ub:
            sig = ' *'
        else:
            sig = ''
        val2_cox.append(f'{val2_cox_mean:.1f} ({val2_cox_lb:.1f} - {val2_cox_ub:.1f}){sig}')
        val2_diff.append(np.abs(val2_cox_mean-val2_aj_mean))
        
        val3_cox_mean = vals_bt[time_idx,0,q3_idx]
        val3_cox_lb, val3_cox_ub = np.nanpercentile(vals_bt[time_idx,1:,q3_idx], (2.5, 97.5))
        if val3_cox_ub<val3_aj_lb or val3_cox_lb>val3_aj_ub:
            sig = ' *'
        else:
            sig = ''
        val3_cox.append(f'{val3_cox_mean:.1f} ({val3_cox_lb:.1f} - {val3_cox_ub:.1f}){sig}')
        val3_diff.append(np.abs(val3_cox_mean-val3_aj_mean))
        
    pd.set_option('display.max_columns', None)

    df = pd.DataFrame(data={'Outcome':outcomes_txt, 'RR1_AJ':rr1_aj, 'RR1_cox':rr1_cox, 'RR2_AJ':rr2_aj, 'RR2_cox':rr2_cox})
    print(df)

    df2 = pd.DataFrame(data={'Outcome':outcomes_txt,
            'val1_aj':val1_aj, 'val2_aj':val2_aj, 'val3_aj':val3_aj,
            'val1_cox':val1_cox, 'val2_cox':val2_cox, 'val3_cox':val3_cox,
            'val1_diff':val1_diff, 'val2_diff':val2_diff, 'val3_diff':val3_diff,
            })
    print(df2)

    df.to_excel(f'table_effect_size{suffix}_SHHS.xlsx', index=False)
    df2.to_excel(f'table_val{suffix}_SHHS.xlsx', index=False)

