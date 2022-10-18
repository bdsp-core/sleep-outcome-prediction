import os
import h5py

with h5py.File('../BAI-EMU/0a0c9ac72a7577ece4de9448bb8178b04c08f5d45941d27375097a69d23419bd_20210524_141512/0a0c9ac72a7577ece4de9448bb8178b04c08f5d45941d27375097a69d23419bd_0_20210524_141512.mat', 'r') as ff:
    print(ff.keys())
    print(ff['data'].shape)
