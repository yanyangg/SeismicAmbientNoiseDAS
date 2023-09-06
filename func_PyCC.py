# This script contains function for filter, temporal normalization, spatial whitening,
# and cross-correlation in frequency domain. Pytorch required.
# Yan Yang 2022-07-10

import numpy as np
from scipy.signal import tukey, butter, filtfilt, detrend, convolve
import torch
import torch.fft
from torch import nn
import gzip
import struct
#%%
# Function to load the DAS data
def read_PASSCAL_segy(infile, nTraces=1250, nSample=900000, TraceOff=0):
    """Function to read PASSCAL segy raw data
    For Ridgecrest data, there are 1250 channels in total,
    Sampling rate is 250 Hz so for one hour data: 250 * 3600 samples
    """
    fs = nSample/ 3600 # sampling rate
    data = np.zeros((nTraces, nSample), dtype=np.float32)
    gzFile = False
    if infile.split(".")[-1] == "segy":
        fid = open(infile, 'rb')
    elif infile.split(".")[-1] == "gz":
        gzFile = True
        fid = gzip.open(infile, 'rb')
    fid.seek(3600)
    # Skipping traces if necessary
    fid.seek(TraceOff*(240+nSample*4),1)
    # Looping over traces
    for ii in range(nTraces):
        fid.seek(240, 1)
        if gzFile:
            # np.fromfile does not work on gzip file
            BinDataBuffer = fid.read(nSample*4) # read binary bytes from file
            data[ii, :] = struct.unpack_from(">"+('f')*nSample, BinDataBuffer)
        else:
            data[ii, :] = np.fromfile(fid, dtype=np.float32, count=nSample)
    fid.close()

    # Convert the phase-shift to strain (in nanostrain)
    Ridgecrest_conversion_factor = 1550.12 / (0.78 * 4 * np.pi * 1.46 * 8)
    data = data * Ridgecrest_conversion_factor

    das_time = np.arange(0, data.shape[1]) * 1 / fs

    return data, das_time

def filter(data, fs, f1, f2, alpha=0.05):
    '''
    :param data: shape: nch * npts
    :param fs: sampling frequency
    :param f1: low frequency
    :param f2: high frequency
    :param alpha: taper length
    :return: filtered data
    '''
    window = tukey(data.shape[1], alpha=alpha)
    passband = [f1*2/fs, f2*2/fs]
    b, a = butter(4, passband, 'bandpass')

    dataf = filtfilt(b,a,data * window)
    return dataf

def running_absolute_mean(trace, nwin):
    '''
    reference: refer to noisepy package, but need to be improved faster on GPU
    :param trace: 1d array shape: npts
    :param nwin: # of points in moving window
    :return: smoothed data
    '''
    npts = len(trace)
    tmp = np.zeros(npts + 2 * nwin)
    tmp[nwin:-nwin] = np.abs(trace)
    tmp[:nwin] = tmp[nwin]
    tmp[-nwin:] = tmp[-nwin - 1]
    return np.nan_to_num(trace/convolve(tmp, np.ones(nwin) / nwin, mode='same')[nwin: -nwin])


def temporal_normalization(data, fs, window_time):
    '''
    running absolute mean normalization or one-bit, depending on window_time
    :param data: shape: nch * npts
    :param fs: sampling frequency
    :param window_time: running window length, in seconds. recommended: half the longest period
    :return: normalized data
    '''
    if window_time == 0: # 1-bit
        return np.sign(data)
    else:
        nwin = int(fs * window_time)
        nch = data.shape[0]
        for i in range(nch):
            data[i,:] = running_absolute_mean(data[i,:], nwin)
        return data


def spectral_whitening(rfftdata, df, window_freq, f1, f2):
    '''
    phase-only or running absolute mean spectral whitening, depending on window_freq
    :param rfftdata: shape: nch * npts, !!torch.tensor!!
    :param df: frequency interval
    :param window_freq: running window length, in Hz.
    :return: whitened spectra
    '''
    idxf1 = int(np.floor(f1 / df))
    idxf2 = int(np.ceil(f2 / df))
    rfftdata_angle = torch.angle(rfftdata)

    if window_freq == 0: # phase-only
        rfftdata = torch.exp(1j * rfftdata_angle)

    else: # running absolute mean
        nwin = int(window_freq / df)
        nch = rfftdata.shape[0]
        for i in range(nch):
            rfftdata[i,:] = torch.from_numpy(running_absolute_mean(rfftdata[i,:].cpu().numpy(), nwin))

    rfftdata[:, :idxf1] = torch.cos(torch.linspace(np.pi / 2, np.pi, idxf1, device=rfftdata.device)) ** 2 * rfftdata[:, :idxf1]
    rfftdata[:, idxf2:] = torch.cos(torch.linspace(np.pi, np.pi / 2, rfftdata.shape[-1] - idxf2, device=rfftdata.device)) ** 2 * rfftdata[:, idxf2:]

    return rfftdata

def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

def cross_correlation(signal_1, signal_2, is_spectral_whitening = False, whitening_params = 0):
    '''
    :param signal_1: data1: shape: nch * npts
    :param signal_2: data2: shape: nch * npts
    :param spectral_whitening: whether to apply spectral whitening
    :param whitening_params: fs, window_freq, f1, f2
    :return: CC: nch * (2*npts-1)
    '''
    if len(signal_1.shape)<2 | len(signal_2.shape)<2:
        print('input dimension must be ntrace*npts !')
        return 0
    else:
        signal_length=signal_1.shape[-1]
        x_cor_sig_length = signal_length*2 - 1
        fast_length = nextpow2(x_cor_sig_length)

        fft_1 = torch.fft.rfft(signal_1, fast_length, dim=-1)
        fft_2 = torch.fft.rfft(signal_2, fast_length, dim=-1)

        if(is_spectral_whitening):
            fs, window_freq, f1, f2 = whitening_params
            df = fs/fast_length
            fft_1 = spectral_whitening(fft_1, df, window_freq, f1, f2)
            fft_2 = spectral_whitening(fft_2, df, window_freq, f1, f2)

        # take the complex conjugate of one of the spectrums. Which one you choose depends on domain specific conventions
        fft_multiplied = torch.conj(fft_1) * fft_2

        # back to time domain.
        prelim_correlation = torch.fft.irfft(fft_multiplied, dim=-1)

        # shift the signal to make it look like a proper crosscorrelation,
        # and transform the output to be purely real
        final_result = torch.roll(prelim_correlation, fast_length//2, dims=-1)[:,  fast_length//2-x_cor_sig_length//2 : fast_length//2-x_cor_sig_length//2+x_cor_sig_length]
        return final_result


class Torch_cross_correlation(nn.Module):
    # https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    def __init__(self, **kwargs):
        super().__init__()
        self.is_spectral_whitening = kwargs["is_spectral_whitening"]
        self.whitening_params = kwargs["whitening_params"]

    def forward(self, data1, data2):
        cc = cross_correlation(data1, data2, is_spectral_whitening = self.is_spectral_whitening, whitening_params=self.whitening_params)
        return cc


def cross_correlation_full(data, ich1, ich2, is_spectral_whitening = False, whitening_params = 0):
    '''
    :param signal_1: data1: shape: nch * npts
    :param signal_2: data2: shape: nch * npts
    :param spectral_whitening: whether to apply spectral whitening
    :param whitening_params: fs, window_freq, f1, f2
    :return: CC: nch * (2*npts-1)
    '''
    if len(signal_1.shape)<2 | len(signal_2.shape)<2:
        print('input dimension must be ntrace*npts !')
        return 0
    else:
        signal_length=signal_1.shape[-1]
        x_cor_sig_length = signal_length*2 - 1
        fast_length = nextpow2(x_cor_sig_length)

        fft_1 = torch.fft.rfft(signal_1, fast_length, dim=-1)
        fft_2 = torch.fft.rfft(signal_2, fast_length, dim=-1)

        if(is_spectral_whitening):
            fs, window_freq, f1, f2 = whitening_params
            df = fs/fast_length
            fft_1 = spectral_whitening(fft_1, df, window_freq, f1, f2)
            fft_2 = spectral_whitening(fft_2, df, window_freq, f1, f2)

        # take the complex conjugate of one of the spectrums. Which one you choose depends on domain specific conventions
        fft_multiplied = torch.conj(fft_1) * fft_2

        # back to time domain.
        prelim_correlation = torch.fft.irfft(fft_multiplied, dim=-1)

        # shift the signal to make it look like a proper crosscorrelation,
        # and transform the output to be purely real
        final_result = torch.roll(prelim_correlation, fast_length//2, dims=-1)[:,  fast_length//2-x_cor_sig_length//2 : fast_length//2-x_cor_sig_length//2+x_cor_sig_length]
        return final_result