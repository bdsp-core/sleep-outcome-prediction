import os
import pandas as pd
import sys
sys.path.insert(0, '../sleep_general')
from mgh_sleeplab import load_mgh_signal, annotations_preprocess, vectorize_sleep_stages


def read_psg_from_bdsp(sid, dov):
    """
    Parameters:
    sid: str, HashID
    dov: str, shifted date in YYYMMDD format

    Returns:
    signals: np.ndarray, shape = (#channel, #sample points)
    sleep_stages: np.ndarray, shape = (#sample points,)
    params: dict, contains Fs, channel_names, start_time
    """
    base_folder = '/home/ec2-user/studies/BDSP-CORE/PSG/data/S0001'
    data_folders = os.listdir(base_folder)

    data_folder = [x for x in data_folders if f'{sid}_{dov}' in x]; assert len(data_folder)==1
    data_files = os.listdir(os.path.join(base_folder, data_folder[0]))
    signal_path = [x for x in data_files if 'signal_' in x.lower() and x.lower().endswith('.mat')]; assert len(signal_path)==1
    sleep_stage_path = [x for x in data_files if x.lower().endswith('_annotations.csv')]; assert len(sleep_stage_path)==1

    signal_path = os.path.join(base_folder, data_folder[0], signal_path[0])
    sleep_stage_path = os.path.join(base_folder, data_folder[0], sleep_stage_path[0])

    # load signal
    import pdb;pdb.set_trace()
    signals, params = load_mgh_signal(signal_path, channels=['C3.?[AM]', 'C4.?[AM]'], return_signal_dtype='numpy')
    Fs = params['Fs']

    # load sleep stages
    annot = annotations_preprocess(pd.read_csv(sleep_stage_path), Fs)
    sleep_stages = vectorize_sleep_stages(annot, signals.shape[1])

    return signals, sleep_stages, params


def main():
    # read mastersheet
    df = pd.read_excel('mastersheet_outcome_deid.xlsx')
    print(df)

    # specify which subject you need

    i = 0
    sid = df.HashID.iloc[i]
    dov = df.DOVshifted.iloc[i].strftime('%Y%m%d')
    signals, sleep_stages, params = read_psg_from_bdsp(sid, dov)

    print(signals.shape)
    print(sleep_stages.shape)
    print(params)


if __name__=='__main__':
    main()
