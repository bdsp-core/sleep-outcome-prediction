from collections import defaultdict
import os
import numpy as np
import scipy.io as sio
import pandas as pd


data_type = 'NREM'
folder = rf'C:\Users\hocke\Downloads\survival_results_{data_type}_bt1000'
sexs = [0,1]
sexs_txt = ['female', 'male']
curve_names = ['<Q1,sex=0', '[Q1--Q3],sex=0', '>Q3,sex=0', '<Q1,sex=1', '[Q1--Q3],sex=1', '>Q3,sex=1']

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

df_cindex = defaultdict(list)
for ii, outcome in enumerate(outcomes):
    print(outcome)
    if outcome in ['Death']:
        model_type = 'CoxPH'
    else:
        model_type = 'CoxPH_CompetingRisk'
    mat_path = os.path.join(folder, f'results_{outcome}_{model_type}.mat')
    mat = sio.loadmat(mat_path)
    cindex = mat['cindex_te']
    lb, ub = np.percentile(cindex[1:], (2.5,97.5))
    df_cindex['Outcome'].append(outcome)
    df_cindex[data_type].append(f'{cindex[0]:.3f} ({lb:.3f}-{ub:.3f})')
df_cindex = pd.DataFrame(df_cindex)
print(df_cindex)
df_cindex.to_excel(f'table_Cindex_{data_type}.xlsx', index=False)


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
    actual_outcomes = []
    for ii, outcome in enumerate(outcomes):
        print(outcome)
        
        if outcome in ['Death']:
            model_type = 'CoxPH'
        else:
            model_type = 'CoxPH_CompetingRisk'
        mat_path = os.path.join(folder, f'results_{outcome}_{model_type}.mat')
        if not os.path.exists(mat_path):
            continue
        mat = sio.loadmat(mat_path)
        actual_outcomes.append(outcomes_txt[ii])
        
        ## AJ
        
        vals_bt = np.array([
            mat['AJ_curve_tes_val'],
            mat['AJ_curve_tes_lower'],
            mat['AJ_curve_tes_upper'],
            ])*100
        if vals_bt.ndim==4:
            vals_bt = vals_bt.transpose(1,0,2,3)
        else:
            vals_bt = vals_bt.transpose(1,0,2)
            vals_bt = vals_bt[..., np.newaxis]
        outcome_idx = [np.array(mat['AJ_curve_tes_states'].flatten()[i]).item() for i in range(len(mat['AJ_curve_tes_states']))].index('event1')
        vals_bt = vals_bt[..., outcome_idx]  # shape = (#T, 3)
        
        survtime = mat['AJ_curve_tes_time'].flatten()
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
        
        rr1_mean = vals_bt[time_idx,0,q3_idx] / vals_bt[time_idx,0,q1q3_idx]
        rr1_lb = vals_bt[time_idx,1,q3_idx] / vals_bt[time_idx,2,q1q3_idx]
        rr1_ub = vals_bt[time_idx,2,q3_idx] / vals_bt[time_idx,1,q1q3_idx]
        rr1_aj.append(f'{rr1_mean:.1f} ({rr1_lb:.1f} - {rr1_ub:.1f})')
        rr2_mean = vals_bt[time_idx,0,q1q3_idx] / vals_bt[time_idx,0,q1_idx]
        rr2_lb = vals_bt[time_idx,1,q1q3_idx] / vals_bt[time_idx,2,q1_idx]
        rr2_ub = vals_bt[time_idx,2,q1q3_idx] / vals_bt[time_idx,1,q1_idx]
        rr2_aj.append(f'{rr2_mean:.1f} ({rr2_lb:.1f} - {rr2_ub:.1f})')
        
        ## Cox
        
        vals_bt = mat['cox_curve_tes_bt_val']*100
        if vals_bt.ndim==3:
            vals_bt = vals_bt[..., np.newaxis]
        outcome_idx = [np.array(mat['cox_curve_tes_bt_states'].flatten()[i]).item() for i in range(len(mat['cox_curve_tes_bt_states']))].index('event1')
        vals_bt = vals_bt[..., outcome_idx]  # shape = (#T, #bt, 3)
        vals_bt[vals_bt<0.001] = np.nan
        
        survtime = mat['cox_curve_tes_bt_time'].flatten()
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

    rr1_aj.append('%.1f, %.1f%%'%(
        np.mean([float(x.split(' ')[0]) for x in rr1_aj]),
        np.mean([float(x.split(' ')[1][1:])>1 or float(x.split(' ')[3][:-1])<1 for x in rr1_aj])*100,
        ))
    rr2_aj.append('%.1f, %.1f%%'%(
        np.mean([float(x.split(' ')[0]) for x in rr2_aj]),
        np.mean([float(x.split(' ')[1][1:])>1 or float(x.split(' ')[3][:-1])<1 for x in rr2_aj])*100,
        ))
    rr1_cox.append('%.1f, %.1f%%'%(
        np.mean([float(x.split(' ')[0]) for x in rr1_cox]),
        np.mean([float(x.split(' ')[1][1:])>1 or float(x.split(' ')[3][:-1])<1 for x in rr1_cox])*100,
        ))
    rr2_cox.append('%.1f, %.1f%%'%(
        np.mean([float(x.split(' ')[0]) for x in rr2_cox]),
        np.mean([float(x.split(' ')[1][1:])>1 or float(x.split(' ')[3][:-1])<1 for x in rr2_cox])*100,
        ))
    df = pd.DataFrame(data={'Outcome':actual_outcomes+['Summary'], 'RR1_AJ':rr1_aj, 'RR1_cox':rr1_cox, 'RR2_AJ':rr2_aj, 'RR2_cox':rr2_cox})
    print(df)

    df2 = pd.DataFrame(data={'Outcome':actual_outcomes,
            'val1_aj':val1_aj, 'val2_aj':val2_aj, 'val3_aj':val3_aj,
            'val1_cox':val1_cox, 'val2_cox':val2_cox, 'val3_cox':val3_cox,
            'val1_diff':val1_diff, 'val2_diff':val2_diff, 'val3_diff':val3_diff,
            })
    print(df2)

    df.to_excel(f'table_effect_size_{data_type}{suffix}.xlsx', index=False)
    df2.to_excel(f'table_val_{data_type}{suffix}.xlsx', index=False)

