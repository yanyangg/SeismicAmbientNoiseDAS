# This script does preprocessing for ambient noise cross-correlation, including:
# differentiation, detrend, bandpass filter, decimation, demean (of channels), temporal normalization
# Yan Yang 2022-07-10

import numpy as np
import h5py
from time import time
import os
from tqdm import tqdm
from func_PyCC_test import *
from glob import glob
from joblib import Parallel, delayed

#%% parameters for Ridgecrest_ODH3
output_preprocessed = '/net/han/ssd-tmp-nobak3/yyang7/AWS/Ridgecrest_DAS/SEG-Y/preprocessed'
filelist = glob('/net/han/ssd-tmp-nobak3/yyang7/AWS/Ridgecrest_DAS/SEG-Y/hourly/*segy')
filelist.sort()

fs = 250 # sampling frequency
f1, f2 = 0.1, 20 # bandpass filter in preprocessing
Decimation = 5 # if not 1, decimation factor after filtering
Diff = True # whether differentiate strain to strain rate
ram_win = 1 # if 0, one-bit; otherwise temporal normalization windowm, usually  1/f1/5 ~ 1/f1/2 #
min_length = 60 # length of the segment in preprocessing, in sec, if shorter than this length, skip the file
min_npts = int(min_length*fs)
njobs = 5 # number f jobs if parallel

def preprocess(x, fs, f1, f2, Decimation, Diff=Diff, ram_win = ram_win):
    '''
    :param x: input data shape (nch, npts)
    :param fs, f1, f2, Decimation, Diff, ram_win: see above
    :return:
    '''
    if Diff:
        x = np.gradient(x, axis=-1) * fs
    x = detrend(x, axis=-1)
    x = filter(x, fs, f1, f2)
    x = x[:, ::Decimation]
    fs_deci = fs / Decimation
    x = x - np.median(x, 0) # common mode noise
    x = temporal_normalization(x, fs_deci, ram_win)
    x = x.astype('float32')
    return x

#%% Preprocessing: read raw data, decimation, differentiation, bandpass filter, demean
if not os.path.exists(output_preprocessed):
    os.mkdir(output_preprocessed)

for ifile in tqdm(filelist):
    outputname = os.path.join(output_preprocessed, os.path.basename(ifile).replace('.segy', '.h5'))
    # try not overlap
    if os.path.exists(outputname):
        print(outputname, 'exists')
        continue

    data, das_time = read_PASSCAL_segy(ifile)
    nch = data.shape[0]
    npts = data.shape[1]

    nchunk = int(np.ceil(npts / min_npts))
    out = Parallel(n_jobs=njobs)(
        delayed(preprocess)(data[:, int(min_length * fs * i): int(min_length * fs * (i + 1))],
                            fs, f1, f2, Decimation) for i in range(nchunk))
    data_out = np.concatenate(out, axis=-1)

    fs_deci = fs / Decimation
    output_h5 = h5py.File(outputname, 'w')
    output_data = output_h5.create_dataset('Data', data=data_out)
    output_data.attrs['fs'] = fs_deci
    output_data.attrs['nt'] = data_out.shape[1]
    output_data.attrs['nCh'] = data_out.shape[0]
    output_h5.close()

