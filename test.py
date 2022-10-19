import os
import pandas as pd


def read_psg_from_bdsp(sid, dov):
    """
    sid is a string, HashID
    dov is a string, shifted date in YYYMMDD format
    """
    base_folder = '/home/ec2-user/studies/BDSP-CORE/PSG/data/S0001'
    data_folders = os.listdir(base_folder)

    data_folder = [x for x in data_folders if f'{sid}_{dov}' in x]
    import pdb;pdb.set_trace()
    assert len(data_folder)==1
    data_folder = data_folder[0]
    data_files = os.listdir(os.path.join(base_folder, data_folder))
    signal_path = o


def main():
    # read mastersheet
    df = pd.read_excel('mastersheet_outcome_deid.xlsx')
    print(df)

    # specify which subject you need

    i = 0
    sid = df.HashID.iloc[i]
    dov = df.DOVshifted.iloc[i].strftime('%Y%m%d')


if __name__=='__main__':
    main()

