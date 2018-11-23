#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 13:32:49 2018

@author: antonio
"""

from obspy.core import read
import numpy as np
from scipy import signal as scp
import matplotlib.pyplot as plt

root = '/data01/antonio/Dropbox/crsmex_cuda/benchmark/'
A = read(root + 'CAIG.HHZ..4-16Hz_data/CAIG.IG.HHZ..D.2011.010.084532.922.sac')
B = read(root + 'CAIG.HHZ..4-16Hz_data/CAIG.IG.HHZ..D.2011.118.124547.954.sac')


# Power A
npts       = len(A[0].data)
fft_A      = np.fft.fft(A[0].data)
fft_A_conj = np.conj(fft_A)
npts       = 1.0
print('len(fft_A) = ', len(fft_A))

# Scipy function
corr_A      = scp.correlate(A[0].data, A[0].data)/ npts
corr_A_max  = corr_A.max()
# CUDA-type implementation
corr_A_CUDA     = np.fft.ifft(fft_A*fft_A_conj) /npts
corr_A_CUDA_max = corr_A_CUDA.max().real
ind_PowerA      = np.argmax(corr_A_CUDA.real)

Power1         = corr_A_CUDA_max

# Power B
npts       = len(B[0].data)
fft_B      = np.fft.fft(B[0].data)
fft_B_conj = np.conj(fft_B)
npts       = 1.0
print('npts = ', npts)

# Scipy function
corr_B      = scp.correlate(B[0].data, B[0].data)/ npts
corr_B_max  = corr_B.max()
# CUDA-type implementation
corr_B_CUDA = np.fft.ifft(fft_B*fft_B_conj) /npts
corr_B_CUDA_max = corr_B_CUDA.max().real
ind_PowerB      = np.argmax(corr_B_CUDA.real)

Power2           = corr_B_CUDA_max

# Correlation A and B
corr_AB      = scp.correlate(A[0].data, B[0].data)/ (npts*np.sqrt(Power1*Power2))
corr_AB_max  = corr_AB.max()
# CUDA-type implementation
corr_AB_CUDA = np.fft.ifft(fft_A*fft_B_conj)/ (npts*np.sqrt(Power1*Power2))
corr_AB_CUDA_max = corr_AB_CUDA.max().real

print('Power A      = ', Power1)
print('ind(Power_A) = ',ind_PowerA)
print('Power B = ', Power2)
print('ind(Power_B) = ',ind_PowerB)

print('CC = ', corr_AB_CUDA_max)

#plt.plot(corr_A_CUDA.real)
#plt.show()
np.savetxt('powerA_py.dat', corr_A_CUDA.real)
np.savetxt('powerB_py.dat', corr_B_CUDA.real)
#np.savetxt('fft_A.dat',abs(fft_A))
