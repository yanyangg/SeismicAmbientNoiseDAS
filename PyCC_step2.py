# This script does spatial whitening, and cc in frequency domain
# Yan Yang 2022-07-10
from glob import glob
import numpy as np
import h5py
from time import time
import pandas as pd
import os
from func_PyCC import *
from tqdm import tqdm
import obspy
#%% parameters
dasinfo = pd.read_csv('/export/ssd-tmp-nobak3/yyang7/AWS/Ridgecrest_DAS/das_info.csv')
path_preprocessed = '/net/han/ssd-tmp-nobak3/yyang7/AWS/Ridgecrest_DAS/SEG-Y/preprocessed/'
output_CC = '/net/han/ssd-tmp-nobak3/yyang7/AWS/Ridgecrest_DAS/SEG-Y/CC/'
if not os.path.exists(output_CC):
    os.mkdir(output_CC)

# starting and ending dates to compute stacked CC
dates = [x.strftime('%Y%m%d') for x in pd.date_range(start='06/20/2020', end='07/29/2020', freq='1D')]

# Now give the No. of channel pairs to calculate CC.
# below is an example of all possible pairs
nch = 1250
# common-shot gather
# ch_src = 600
# pair_channel1 = ch_src * np.ones(nch)
# pair_channel2 = np.arange(nch)
# common offset
chdist = obspy.geodetics.base.degrees2kilometers(np.array([obspy.geodetics.base.locations2degrees(
    dasinfo['latitude'][x], dasinfo['longitude'][x],dasinfo['latitude'], dasinfo['longitude']) * np.sign(
    np.arange(len(dasinfo)) - x) for x in range(len(dasinfo))] ))

seginkm = np.array([0.6, 0.3]) # common-offset in km

pair_channel1 = []
pair_channel2 = []
for iseginkm in range(len(seginkm)):
    offset = seginkm[iseginkm]
    idx_end = np.argmin(np.abs(offset - chdist[:,-1]))
    pair_channel1.extend(dasinfo['index'][:idx_end+1])
    for i in range(int(idx_end+1)):
        pair_channel2.append(dasinfo['index'][np.argmin(np.abs(offset - chdist[i,:]))])

npair = len(pair_channel1)
#############

f1, f2 = 0.1, 20 # frequency band in spectral whitening
fs = 50 # sampling frequency
is_spectral_whitening = True
window_freq = 0 # 0 means aggresive spectral whitening, otherwise running mean
max_lag = 30 # in sec, the time lag of the output CC
npts_lag = int(max_lag*fs)
xcorr_seg = 60 # in sec, the length of the segment to compute CC, slightly larger than max_lag is good
npts_seg = int(xcorr_seg*fs)

device = "cuda"
gpu_ids = [0,1] # GPU device id
npair_chunk = 1250 # depends on # of channels, sampling frequency, and xcorr_seg, needs to be adaptive
nchunk = int(np.ceil(npair/ npair_chunk))


#%%
for idate in tqdm(dates):
    ccall = np.zeros((npair, int(max_lag * fs * 2 + 1)))
    output_file_tmp = f'{output_CC}/{idate}.npy'
    if os.path.exists(output_file_tmp):
        print(output_file_tmp)
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
        if npts< npts_seg:# or nch!=nch_end:
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
            ich1 = pair_channel1[npair_chunk * ichunk: npair_chunk * (ichunk + 1)]
            ich2 = pair_channel2[npair_chunk * ichunk: npair_chunk * (ichunk + 1)]
            data1 = data[ich1, :].reshape(-1, npts_seg)
            data2 = data[ich2, :].reshape(-1, npts_seg)

            cc = model(data1, data2)
            cc = cc.cpu().numpy()
            cc = np.sum(cc.reshape(len(ich1), nseg, -1), 1)
            ccall[npair_chunk * ichunk: npair_chunk * (ichunk + 1), :] += cc[:,npts_seg - npts_lag - 1:npts_lag - npts_seg + 1]

        torch.cuda.empty_cache()



    if flag_mean >0:
        ccall /= flag_mean
        np.save(output_file_tmp, ccall)



