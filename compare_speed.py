#%%
import numpy as np
import torch
import scipy
import scipy.fft
from time import time
from tqdm import tqdm
def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

def cross_correlation_torch(signal_1, signal_2):
    '''
    :param signal_1: data1: shape: nch * npts
    :param signal_2: data2: shape: nch * npts
    :param spectral_whitening: whether to apply spectral whitening
    :param whitening_params: fs, window_freq, f1, f2
    :return: CC: nch * (2*npts-1)
    '''

    signal_length=signal_1.shape[-1]
    x_cor_sig_length = signal_length*2 - 1
    fast_length = nextpow2(x_cor_sig_length)

    fft_1 = torch.fft.rfft(signal_1, fast_length, dim=-1)
    fft_2 = torch.fft.rfft(signal_2, fast_length, dim=-1)

    # take the complex conjugate of one of the spectrums. Which one you choose depends on domain specific conventions
    fft_multiplied = torch.conj(fft_1) * fft_2

    # back to time domain.
    prelim_correlation = torch.fft.irfft(fft_multiplied, dim=-1)

    # shift the signal to make it look like a proper crosscorrelation,
    # and transform the output to be purely real
    final_result = torch.roll(prelim_correlation, fast_length//2, dims=-1)[:,  fast_length//2-x_cor_sig_length//2 : fast_length//2-x_cor_sig_length//2+x_cor_sig_length]
    return final_result

def cross_correlation_numpy(signal_1, signal_2):
    '''
    :param signal_1: data1: shape: nch * npts
    :param signal_2: data2: shape: nch * npts
    :param spectral_whitening: whether to apply spectral whitening
    :param whitening_params: fs, window_freq, f1, f2
    :return: CC: nch * (2*npts-1)
    '''

    signal_length=signal_1.shape[-1]
    x_cor_sig_length = signal_length*2 - 1
    fast_length = nextpow2(x_cor_sig_length)

    fft_1 = np.fft.rfft(signal_1, fast_length, -1)
    fft_2 = np.fft.rfft(signal_2, fast_length, -1)

    # take the complex conjugate of one of the spectrums. Which one you choose depends on domain specific conventions
    fft_multiplied = np.conj(fft_1) * fft_2

    # back to time domain.
    prelim_correlation = np.fft.irfft(fft_multiplied, axis=-1)

    # shift the signal to make it look like a proper crosscorrelation,
    # and transform the output to be purely real
    final_result = np.roll(prelim_correlation, fast_length//2, -1)[:,  fast_length//2-x_cor_sig_length//2 : fast_length//2-x_cor_sig_length//2+x_cor_sig_length]
    return final_result

def cross_correlation_scipy(signal_1, signal_2):
    '''
    :param signal_1: data1: shape: nch * npts
    :param signal_2: data2: shape: nch * npts
    :param spectral_whitening: whether to apply spectral whitening
    :param whitening_params: fs, window_freq, f1, f2
    :return: CC: nch * (2*npts-1)
    '''

    signal_length=signal_1.shape[-1]
    x_cor_sig_length = signal_length*2 - 1
    fast_length = nextpow2(x_cor_sig_length)

    fft_1 = scipy.fft.rfft(signal_1, fast_length, -1)
    fft_2 = scipy.fft.rfft(signal_2, fast_length, -1)

    # take the complex conjugate of one of the spectrums. Which one you choose depends on domain specific conventions
    fft_multiplied = np.conj(fft_1) * fft_2

    # back to time domain.
    prelim_correlation = scipy.fft.irfft(fft_multiplied, axis=-1)

    # shift the signal to make it look like a proper crosscorrelation,
    # and transform the output to be purely real
    final_result = np.roll(prelim_correlation, fast_length//2, -1)[:,  fast_length//2-x_cor_sig_length//2 : fast_length//2-x_cor_sig_length//2+x_cor_sig_length]
    return final_result
#%%
t_torch_cpu = []
t_torch_gpu = []
t_torch_cpu_move = []
t_torch_gpu_move = []
t_numpy = []
t_scipy = []
torch.cuda.set_device('cuda:3')
for i in tqdm(range(100)):


    signal_1 = np.random.rand(1250, 32768)
    signal_2 = np.random.rand(1250, 32768)


    t2 = time()
    signal_1_cpu = torch.from_numpy(signal_1)
    signal_2_cpu = torch.from_numpy(signal_2)
    t1 = time()
    cc_torch_cpu = cross_correlation_torch(signal_1_cpu, signal_2_cpu)
    t_torch_cpu.append(time()-t1)
    cc_torch_cpu = cc_torch_cpu.numpy()
    t_torch_cpu_move.append(time() - t2)

    t2 = time()
    signal_1_cuda = torch.from_numpy(signal_1).to("cuda:3")
    signal_2_cuda = torch.from_numpy(signal_2).to("cuda:3")
    t1 = time()
    cc_torch_gpu = cross_correlation_torch(signal_1_cuda, signal_2_cuda)
    t_torch_gpu.append(time()-t1)
    # print(torch.cuda.memory_allocated())  # Returns the current GPU memory usage by tensors in bytes
    # print(torch.cuda.memory_cached())
    cc_torch_gpu = cc_torch_gpu.cpu().numpy()
    t_torch_gpu_move.append(time() - t2)

    t1 = time()
    cc_numpy = cross_correlation_numpy(signal_1, signal_2)
    t_numpy.append(time()-t1)
    t1 = time()
    cc_scipy = cross_correlation_scipy(signal_1, signal_2)
    t_scipy.append(time() - t1)

#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(5,4))
plt.plot([np.mean(t_numpy), np.mean(t_scipy), np.mean(t_torch_cpu) , np.mean(t_torch_gpu)], 'o-', label='Computation time only')
plt.xticks([0,1,2,3], ['NumPy', 'SciPy', 'PyTorch_CPU', 'PyTorch_GPU'])
plt.ylabel('Computation time (s)')
plt.grid()
plt.gca().set_yscale('log')
plt.tight_layout()
plt.show()