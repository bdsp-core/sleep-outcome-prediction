import os
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '../sleep_general')
from mgh_sleeplab import read_psg_from_bdsp


def main():
    # read mastersheet
    df = pd.read_excel('mastersheet_outcome_deid.xlsx')
    print(df)

    # specify which subject you need

    i = 134
    sid = df.HashID.iloc[i]
    dov = df.DOVshifted.iloc[i]
    signals, sleep_stages, params = read_psg_from_bdsp(sid, dov, channels=['F3-?[AM]', 'F4-?[AM]', 'C3-?[AM]', 'C4-?[AM]', 'O1-?[AM]', 'O2-?[AM]'])

    print(signals.shape)
    print(sleep_stages.shape)
    print(params)


if __name__=='__main__':
    main()

