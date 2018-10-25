from obspy.core import read
import numpy as np
from scipy import signal as scp

traces = read('*CAIG*HHZ.sac')
traces.filter('bandpass',freqmin=1.0, freqmax = 8.0, corners=2, zerophase=True)
N      = len(traces)
npts   = 1024    # 2^11

print('Number of traces = ', N)

for k in range(N-1):
	master   = traces[k]
	tp       = master.stats.sac.a  # Get the p wave time
	time1    = np.linspace(master.stats.sac.b, master.stats.sac.e, master.stats.npts )
	ind      = np.where (time1>= master.stats.sac.a)
	signal1  = master.data[ind]
	signal1  = signal1[0:npts]
	Power_S1 = max(scp.correlate(signal1,signal1))/npts
	for j in range(k+1,N):
		test    = traces[j]
		tp2     = test.stats.sac.a  # Get the p wave time
		time2   = np.linspace(test.stats.sac.b, test.stats.sac.e, test.stats.npts )
		ind     = np.where (time2>= test.stats.sac.a) 
		signal2 = test.data[ind]
		signal2 = signal2[0:npts]
		Power_S2 = max(scp.correlate(signal2,signal2))/npts 
		A        = scp.correlate(signal1,signal2,'full')/(npts*np.sqrt(Power_S1*Power_S2))
		CC       = A.max()
		print('Waveform1 = ', Power_S1 , ' Waveform2 = ', Power_S2, ' CC = ', CC)
		
		

exit()


