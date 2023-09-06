import numpy as np
from scipy.signal import tukey, butter, filtfilt, resample
import torch
import torch.fft
from scipy import sparse
from scipy.sparse.linalg import lsmr

def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

def filter(data, fs, f1, f2, alpha=0.05):
    window = tukey(data.shape[1], alpha=alpha)
    passband = [f1*2/fs, f2*2/fs]
    b, a = butter(4, passband, 'bandpass')
    dataf = filtfilt(b,a,data * window)
    return dataf

def mean_filter(data, nch):
    # data shape: nch*... any dimension
    # median filter over i-nch//2, i+nch//2
    datarme = np.zeros_like(data)
    for i in range(data.shape[0]):
        datarme[i] = np.nanmean(data[max(0, i - nch//2):min(i + nch//2 + 1, data.shape[0]), ...], 0)
    return datarme

def torch_xcorr(signal_1, signal_2):
    if len(signal_1.shape)<2 | len(signal_2.shape)<2:
        print('input dimension must be ntrace*npts !')
        return 0
    else:
        signal_length=signal_1.shape[-1]
        x_cor_sig_length = signal_length*2 - 1
        fast_length = nextpow2(x_cor_sig_length)

        # the last signal_ndim axes will be transformed
        fft_1 = torch.fft.rfft(signal_1, fast_length, dim=-1)
        fft_2 = torch.fft.rfft(signal_2, fast_length, dim=-1)

        # take the complex conjugate of one of the spectrums. Which one you choose depends on domain specific conventions
        fft_multiplied = torch.conj(fft_1) * fft_2

        # back to time domain.
        prelim_correlation = torch.fft.irfft(fft_multiplied, dim=-1)

        # shift the signal to make it look like a proper crosscorrelation,
        # and transform the output to be purely real
        final_result = torch.roll(prelim_correlation, fast_length//2, dims=-1)[:,  fast_length//2-x_cor_sig_length//2:fast_length//2-x_cor_sig_length//2+x_cor_sig_length]
        return final_result

def MCCC(data, dt, cc_thres, damp, return_all = False ):

    # remember to first put data in cuda: data = torch.from_numpy(data).to("cuda:0")
    nch = data.shape[0]
    npts = data.shape[1]

    # normalize so the maxCC=1
    data = (data - torch.mean(data,1).reshape(nch,1).repeat(1,npts)) / torch.std(
        data,1).reshape(nch,1).repeat(1,npts)

    # CC with torch on gpu
    ccmax = torch.zeros(nch, nch)
    dtmax = torch.zeros(nch, nch)
    # ccmax_interp = torch.zeros(nch, nch)
    # dtmax_interp = torch.zeros(nch, nch)
    for i in range(nch):
        signal1 = data
        signal2 = data[i,:].reshape(1,npts).repeat(nch,1)
        cc = torch_xcorr(signal1, signal2)/npts
        ccmax[i,:], idx = torch.max(cc,1)
        dtmax[i,:] = (idx - (npts-1)) * dt



#
# # # ########################
# interpolate to get higher resolution
#         y1 = cc.index_select(1, torch.maximum(idx-1,torch.zeros_like(idx))).diag()
#         y2 = cc.index_select(1, idx).diag()
#         y3 = cc.index_select(1, torch.minimum(idx+1,(cc.shape[-1]-1)*torch.ones_like(idx))).diag()
#         dt_interp_max = dt * ((y3 - y1) / (2 * (2 * y2 - y1 - y3)))
#         dtmax_interp[i,:] = dtmax[i,:] + dt_interp_max
#         ccmax_interp[i,:] = y2 + ((y3 - y1) ** 2) / (8 * (2 * y2 - y1 - y3))
#     dtmax = dtmax_interp
#     ccmax = ccmax_interp
# # # ##########################


    # LSQR inversion
    x0, y0 = torch.where(ccmax>cc_thres) # only use CC>cc_thres
    x, y = x0[x0<y0], y0[x0<y0] # avoid repeated pairs
    nrow = len(x) +1
    ncol = nch

    row = np.tile(np.arange(nrow - 1), 2)
    col = np.concatenate((x, y))
    val = np.concatenate((np.ones(nrow - 1), -np.ones(nrow - 1)))
    row = np.concatenate((row, (nrow - 1) * np.ones(ncol)))
    col = np.concatenate((col, np.arange(ncol)))
    val = np.concatenate((val, np.ones(ncol)))

    G_cc = sparse.coo_matrix((val, (row, col)), shape=(nrow, ncol))
    dt_obs = dtmax[x, y]

    # regularization (lines below are from Ettore)
    D = (np.diag(np.ones(ncol)) - np.diag(np.ones(ncol - 1), k=-1))[1:, :]
    D = sparse.csr_matrix(D) * damp
    d = np.concatenate((dt_obs, np.zeros(D.shape[0]+1)))
    G = sparse.vstack((G_cc, D))

    m = lsmr(G, d)[0] # better than lsqr

    # now compute uncertainty (VANDECAR AND CROSSON, 1990)
    # ccmax[ccmax < cc_thres] = np.nan
    # dtmax[ccmax < cc_thres] = np.nan
    if (return_all):
        return m, np.array(ccmax), np.array(dtmax)
    else:
        return m

#%%
def run_MCCC_single(data, pickt,cc_thres, tshift, dt,fs, f1, f2,day_smooth,device,flag):
    # if flag % 100 ==0:
    #     print(flag)
    t1, t2 = np.maximum(pickt - tshift, 0), pickt + tshift
    upsample_factor = int(np.ceil(5/(t2-t1)))
    idxt1, idxt2 = int(np.floor(t1 / dt * upsample_factor)), int(np.ceil(t2 / dt * upsample_factor))

    midpt = data.shape[-1] // 2
    data = (np.fliplr(data[:, :midpt + 1]) + data[:, midpt:]) / 2
    # data = np.fliplr(data[:, :midpt + 1])
    # data = data[:,midpt:]
    dataf = filter(data, fs, f1, f2)

    # upsample
    dataf = resample(dataf, dataf.shape[-1] * upsample_factor, axis=-1)
    # filter

    # mean filter smoothed over weeks
    dataf_smo = np.zeros_like(dataf)
    for i in range(dataf.shape[0]):
        dataf_smo[i, :] = np.nanmean(
            dataf[max(0, i - day_smooth // 2):min(i + day_smooth // 2 + 1, dataf.shape[0]), :], 0)

    datatmp = torch.from_numpy(dataf_smo[:, idxt1:idxt2] * tukey(idxt2-idxt1, alpha=0.5)).to(device)
    m, ccmax, dtmax = MCCC(datatmp, dt / upsample_factor, cc_thres, 0.0, True)
    dt_t = m / pickt
    # cc_num_used = np.sum(ccmax > cc_thres, 0)
    cc_median = np.median(ccmax, 0)
    ccmax[ccmax < cc_thres] = np.nan
    dtmax[ccmax < cc_thres] = np.nan

    ccmax_mean = np.nanmean(ccmax, 0)
    ti_tj = m.reshape(len(m), 1).repeat(len(m), 1) - m.reshape(1, len(m)).repeat(len(m), 0)
    dt_t_err = np.sqrt(np.nanmean((dtmax - ti_tj) ** 2, 0)) / pickt

    return dt_t, ccmax_mean, dt_t_err, cc_median

