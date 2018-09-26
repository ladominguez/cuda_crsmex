import numpy as np
import matplotlib.mlab
import scipy.signal
import matplotlib.pyplot as plt

S1  = np.load('S1.npy')
S2  = np.load('S2.npy')

plt.figure()
plt.plot(S1)
plt.xlabel('Samples')
plt.title('Signal 1')
plt.savefig('Signal1.png')
plt.figure()
plt.plot(S2)
plt.xlabel('Samples')
plt.title('Signal 2')
plt.xlabel('Samples')
plt.savefig('Signal2.png')
plt.figure()
fs     = 20
Cxy1, f1  = matplotlib.mlab.cohere(S1,S2,Fs=fs,NFFT=256)
f2, Cxy2 = scipy.signal.coherence(S1, S2, fs=20.)

plt.figure()
plt.semilogy(f1,abs(Cxy1))
plt.xlabel('Frequency')
plt.title('Coherence - matplotlib')
plt.savefig('coherence_matplotlib.png')
np.savetxt('S1.txt', S1, fmt = '%5.2f')
np.savetxt('S2.txt', S2, fmt = '%5.2f')
