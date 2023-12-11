import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter


if __name__=='__main__':
    # load MI data
    df = pd.read_excel('/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction (1)/code-haoqi/time2event_DiabetesII.xlsx')
    dfX = pd.read_csv('/data/Dropbox (Partners HealthCare)/SleepBasedOutcomePrediction (1)/shared_data/MGH/to_be_used_features_NREM.csv')
    assert np.all(df.MRN==dfX.MRN)
    df = pd.concat([df[['cens_outcome', 'time_outcome', 'cens_death', 'time_death']], dfX], axis=1)
    df['event_outcome']=1-df['cens_outcome']

    df = df[(~pd.isna(df.cens_death))&(df.time_death>0)&(~pd.isna(df.cens_outcome))&(df.time_outcome>0)].reset_index(drop=True)
    ids1 = df.AHI>30
    ids2 = df.AHI<5
    results = logrank_test(
        df.time_outcome[ids1].values,
        df.time_outcome[ids2].values,
        event_observed_A=1-df.cens_outcome[ids1].values,
        event_observed_B=1-df.cens_outcome[ids2].values,
        )

    kmf1 = KaplanMeierFitter()
    kmf1.fit(df.time_outcome[ids1].values, 1-df.cens_outcome[ids1].values)
    kmf2 = KaplanMeierFitter()
    kmf2.fit(df.time_outcome[ids2].values, 1-df.cens_outcome[ids2].values)

    results.print_summary()
    #print(results.p_value)
    #print(results.test_statistic)

    plt.plot(kmf1.survival_function_.index.values,kmf1.survival_function_.KM_estimate,c='r')
    plt.plot(kmf2.survival_function_.index.values,kmf2.survival_function_.KM_estimate,c='b')
    plt.show()

    """
    df['groupA']=0
    df.loc[ids1,'groupA']=1
    df.loc[ids2,'groupA']=0
    cox=CoxPHFitter().fit(X, 'time_outcome','event_outcome')
    X = df[ids1|ids2].reset_index(drop=True)[['time_outcome','event_outcome','groupA']]
    cox.print_summary()
    """
