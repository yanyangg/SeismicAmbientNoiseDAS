'''
Functions for surface wave phase velocity extraction.
Beamforming is used to generate f-v image.
Automatic local maximum is used to track dispersion curves. based on Huajian Yao's MATLAB code
Yan Yang 2019-01-20
'''
#%%
import numpy as np
from scipy.signal import iirpeak, tukey, butter, filtfilt, hilbert

#%%
def filter(data, fs, f1, f2, alpha=0.05):
    window = tukey(data.shape[1], alpha=alpha)
    passband = [f1*2/fs, f2*2/fs]
    b, a = butter(2, passband, 'bandpass')
    dataf = filtfilt(b,a,data * window)
    return dataf
def envelop(data, alpha=0.05):
    datae = np.abs(hilbert(data)) * tukey(data.shape[1], alpha=alpha)
    return datae
def narrow_filter(data, fs, f, w, alpha=0.05):
    window = tukey(data.shape[1], alpha=alpha)
    b, a = iirpeak(f, w, fs=fs)
    dataf = filtfilt(b, a, data* window)
    return dataf


#%%
def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

def shift_signal(datin, nshift):
    '''
    # shift signal in frequency domain
    # very slow for fft of large array. Not use for now.
    # if n > 0, shift to right; n < 0, shift to left.
    # input:
    # datin - signal(npts * nx)
    # nshift - number of points to shift(row vector, 1 * nx), unit: samples
    '''
    npts = datin.shape[1]
    nx = datin.shape[0]
    N = nextpow2(npts)
    S0 = np.fft.fft(datin, N)
    SS = S0 * np.exp(nshift.reshape(-1,1).dot((-2j * np.pi/ N * np.arange(N)).reshape(1,-1)))
    ss = np.real(np.fft.ifft(SS, N))
    datout = ss[:,:npts]
    return datout

def CalcDisp_shift_stack(f, v, fs,recdist, beamdist, data, nbeamwidth=None, nepicdist=None, wavelen=None):
    '''
    # calculate dispersion using shift and stack (shift in frequency domain)
    # Input:
    # f - frequency
    # v - phase velocity
    # fs - sample rate
    # wavelen - wavelength corresponding to f
    # recdist - epicentral distance of the beam center
    # beamdist - epicentral distance of the stations in the beam (receiver-source)
    # data - waveform, length(gcarc) * npts
    # nbeamwidth, nepicdist, wavelen - parameters for setting new beams
    # Output:
    # disp - dispersion matrix, length(v) * length(f)
    '''


    # data = data / np.abs(data).max(axis=1)[:, np.newaxis]

    nf = len(f)
    nv = len(v)
    disp = np.empty((nv, nf))
    disp[:] = np.nan



    for i in range(nf):
        newbeam = np.arange(len(beamdist))#np.where((np.abs(beamdist) > nepicdist * wavelen[i]) & (
        #         np.abs(beamdist-recdist) <  nbeamwidth * wavelen[i]))[0]

        bb, aa = iirpeak(f[i], 3, fs=fs) # need to check the smoothness we want, 3 is pretty large actually
        dataf = filtfilt(bb, aa, data[newbeam,:])
        for j in range(nv):
            nshift = (beamdist[newbeam]-recdist)/v[j]*fs
            dataf_shift = np.zeros_like(dataf)
            for k in range(len(newbeam)):
                dataf_shift[k, :] = np.roll(dataf[k, :], -nshift[k].astype(int))
            # dataf_env = np.abs(hilbert(dataf_shift))
            disp[j, i] = np.linalg.norm(np.sum(dataf_shift, 0)) + 1e-4
            # disp[j, i] = np.max(np.sum(dataf_env, 0))
        disp[:, i] = disp[:, i] / np.max(disp[:, i])
    # vout = extr_disp([f, v, disp])
    # vmax = v[np.argmax(disp, 0)]

    return disp #[vout, vmax]

#%%
def extr_disp(f, v, disp, f_ref_set = [1], vmax_set= [1]):
    '''
    # Extract dispersion curve from a frequency-velocity image
    # Input:
    # f - frequency
    # v - phase velocity
    # disp - amplitude at each (f,v) point (nv*nf)
    # f_ref_set - reference frequency for the starting point of dispersion curve picking
    # vmax_set - corresponding to f_ref_set, max allowed velocity at that frequency
    '''

    xpt =[]
    ypt = []
    for k in range(len(f_ref_set)):
        f_ref = f_ref_set[k]
        vmax = vmax_set[k]
        index = np.where(f >= f_ref)[0][0]
        f_ref = f[index]
        disp_ref = disp[:, index]

        v_ref = v[np.argmax(disp_ref[v<vmax])]
        ypt.append(np.abs(v - v_ref).argmin())
        xpt.append(np.abs(f - f_ref).argmin())

    ArrPt = AutoSearch(ypt[0], xpt[0], disp)
    # ArrPt = AutoSearchMultiplePoints(np.array(ypt), np.array(xpt), disp)
    voutput = v[ArrPt]
    return voutput
#%%
def AutoSearch(InitialY, InitialX, ImageData, step=5):
    '''
    # Credit: Huajian Yao's MATLAB code
    # picking dispersion curve (local maxima) from
    # a starting point in the image.
    # step: Center_T search up
    '''
    YSize = ImageData.shape[0]
    XSize = ImageData.shape[1]

    ArrPt = np.zeros(XSize,dtype=int)


    for i in range(InitialX,XSize):
        index1 = 0
        index2 = 0
        point_left = InitialY
        point_right = InitialY
        while index1 == 0:
            point_left_new = max(0, point_left - step)
            if ImageData[point_left, i] < ImageData[point_left_new, i]:
                point_left = point_left_new
            else:
                index1 = 1
                point_left = point_left_new

        while index2 == 0:
            point_right_new = min(point_right + step, YSize-1)
            if ImageData[point_right, i] < ImageData[point_right_new, i]:
                point_right = point_right_new
            else:
                index2 = 1
                point_right = point_right_new


        index_max = np.argmax(ImageData[point_left:point_right+1, i])
        ArrPt[i] = index_max + point_left
        InitialY = ArrPt[i]

    # Center_T search down
    InitialY = ArrPt[InitialX]
    for i in range(InitialX - 1,-1,-1):
        index1 = 0
        index2 = 0
        point_left = InitialY
        point_right = InitialY

        while index1 == 0:
            point_left_new = max(0, point_left - step)
            if ImageData[point_left, i] < ImageData[point_left_new, i]:
                point_left = point_left_new
            else:
                index1 = 1
                point_left = point_left_new

        while index2 == 0:
            point_right_new = min(point_right + step, YSize-1)
            if ImageData[point_right, i] < ImageData[point_right_new, i]:
                point_right = point_right_new
            else:
                index2 = 1
                point_right = point_right_new


        index_max = np.argmax(ImageData[point_left:point_right+1, i])
        ArrPt[i] = index_max + point_left
        InitialY = ArrPt[i]

    return ArrPt
#%%
def AutoSearchMultiplePoints(ptY, ptX, ImageData, step = 5):
    '''
    # Credit: Huajian Yao's MATLAB code
    # picking dispersion curve (local maxima) from
    # multiple starting points in the image.
    '''
    nPt = len(ptX)
    II = np.argsort(ptX)
    ptX = ptX[II]
    ptY = ptY[II]

    YSize = ImageData.shape[0]
    XSize = ImageData.shape[1]

    ArrPt = np.zeros(XSize, dtype=int)


    point_left = 0
    point_right = 0

    InitialX = ptX[nPt-1]
    InitialY = ptY[nPt-1]
    for i in range(InitialX,XSize):
        index1 = 0
        index2 = 0
        point_left = InitialY
        point_right = InitialY
        while index1 == 0:
            point_left_new = max(0, point_left - step)
            if ImageData[point_left, i] < ImageData[point_left_new, i]:
                point_left = point_left_new
            else:
                index1 = 1
                point_left = point_left_new
        while index2 == 0:
            point_right_new = min(point_right + step, YSize-1)
            if ImageData[point_right, i] < ImageData[point_right_new, i]:
                point_right = point_right_new
            else:
                index2 = 1
                point_right = point_right_new

        index_max = np.argmax(ImageData[point_left:point_right+1, i])
        ArrPt[i] = index_max + point_left
        InitialY = ArrPt[i]

    InitialX = ptX[nPt-1]
    InitialY = ArrPt[ptX[nPt-1]]
    midX = ptX[nPt - 2]
    midY = ptY[nPt - 2]
    kk = 0

    for i in range(ptX[nPt-1],-1,-1):
        index1 = 0
        index2 = 0

        if i == midX:
            InitialY = midY
            kk = kk + 1
            if (nPt - kk) > 1:
                midX = ptX[nPt - kk - 2]
                midY = ptY[nPt - kk - 2]
        point_left = InitialY
        point_right = InitialY
        while index1 == 0:
            point_left_new = max(0, point_left - step)
            if ImageData[point_left, i] < ImageData[point_left_new, i]:
                point_left = point_left_new
            else:
                index1 = 1
                point_left = point_left_new
        while index2 == 0:
            point_right_new = min(point_right + step, YSize-1)
            if ImageData[point_right, i] < ImageData[point_right_new, i]:
                point_right = point_right_new
            else:
                index2 = 1
                point_right = point_right_new

        index_max = np.argmax(ImageData[point_left:point_right + 1, i])
        ArrPt[i] = index_max + point_left
        InitialY = ArrPt[i]

    return ArrPt
