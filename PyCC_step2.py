# This script does temporal normalization, spatial whitening, and cc in frequency domain
from glob import glob
import numpy as np
import h5py
from time import time
import pandas as pd
import os
from func_PyCC import *
from tqdm import tqdm
#%% parameters
path_preprocessed = '/kuafu/scratch/yyang7/DASCCtest/preprocessed_data/'
output_CC = '/kuafu/scratch/yyang7/DASCCtest/cc/'
if not os.path.exists(output_CC):
    os.mkdir(output_CC)

# starting and ending dates to compute stacked CC
dates = [x.strftime('%Y%m%d') for x in pd.date_range(start='01/01/1970', end='01/01/1970', freq='1D')]

# Now give the No. of channel pairs to calculate CC.
# below is an example of all possible pairs
nch = 500
# below is an example of common-shot gather
ch_src = 400
pair_channel1 = ch_src * np.ones(nch)
pair_channel2 = np.arange(nch)
# # below is an example of all possible pairs
# pair_channel1 = np.repeat(np.arange(nch), nch)
# pair_channel2 = np.tile(np.arange(nch), nch)
npair = len(pair_channel1)
#############

f1, f2 = 0.1, 10 # frequency band in spectral whitening
fs = 50 # sampling frequency
is_spectral_whitening = True
window_freq = 0 # 0 means aggresive spectral whitening, otherwise running mean
max_lag = 30 # in sec, the time lag of the output CC
npts_lag = int(max_lag*fs)
xcorr_seg = 60 # in sec, the length of the segment to compute CC, slightly larger than max_lag is good
npts_seg = int(xcorr_seg*fs)

device = "cuda"
gpu_ids = [0] # GPU device id
npair_chunk = 500 # depends on # of channels, sampling frequency, and xcorr_seg, needs to be adaptive
nchunk = int(np.ceil(npair/ npair_chunk))


#%% spectral whitening, CC
t1 = time()
for idate in tqdm(dates):
    ccall = np.zeros((npair, int(max_lag * fs * 2 + 1)))
    output_file_tmp = f'{output_CC}/{idate}.npy'
    if os.path.exists(output_file_tmp):
        print(output_file_tmp, 'exists, skip')
        continue

    filelist = glob(os.path.join(path_preprocessed,idate+'*h5'))
    filelist.sort()

    if len(filelist) == 0:
        print(f'{idate}: no file')
        continue
    flag_mean = 0

    for ifile in filelist:
        fid = h5py.File(ifile, 'r')
        data = fid['Data'][:]
        fid.close()

        nch = data.shape[0]
        npts = data.shape[1]

        npts = npts//npts_seg * npts_seg
        if npts< npts_seg:
            continue

        data = data[:, :npts]

        nseg = int(npts / npts_seg)
        flag_mean += nseg

        # use n gpus
        torch.cuda.empty_cache()
        model_conf = {
            "is_spectral_whitening": is_spectral_whitening,
            "whitening_params": [fs, window_freq, f1, f2],
        }
        model = Torch_cross_correlation(**model_conf)
        model = nn.DataParallel(model, device_ids=gpu_ids)
        model.to(device)
        model.eval()

        with torch.no_grad():
            data = torch.from_numpy(data).to(torch.float32)  # .to(device)

        for ichunk in range(nchunk):

            print(idate, ifile, ichunk, '/', nchunk, time()-t1)

            ich1 = pair_channel1[npair_chunk * ichunk: npair_chunk * (ichunk + 1)]
            ich2 = pair_channel2[npair_chunk * ichunk: npair_chunk * (ichunk + 1)]
            data1 = data[ich1, :].reshape(-1, npts_seg)
            data2 = data[ich2, :].reshape(-1, npts_seg)

            cc = model(data1, data2)
            cc = cc.cpu().numpy()
            cc = np.sum(cc.reshape(len(ich1), nseg, -1), 1)
            ccall[npair_chunk * ichunk: npair_chunk * (ichunk + 1), :] += cc[:,
                                                                          npts_seg - npts_lag - 1:npts_lag - npts_seg + 1]

        torch.cuda.empty_cache()

    if flag_mean >0:
        ccall /= flag_mean
        np.save(output_file_tmp, ccall)



#%% plot
import matplotlib.pyplot as plt
ccall = np.load(output_file_tmp)
ccf = filter(ccall,50, 2, 10)
vmax = np.percentile(np.abs(ccf), 90)
plt.imshow(ccf,aspect='auto', vmax=vmax,vmin=-vmax, extent=(-max_lag, max_lag, ccall.shape[0],0),cmap='RdBu');
plt.xlabel('Time lag (s)')
plt.ylabel('Channel')
plt.show()

