from obspy.core import read
import numpy as np
from scipy import signal as scp

# reads data, sac files. 
traces = read('./CAIG.HHZ..4-16Hz_data/*.sac')
N      = len(traces)

print('Number of traces = ', N)

for k in range(N-1):
	master   = traces[k]
	npts     = master.stats.sac.npts
	signal1  = master.data
	Power_S1 = max(scp.correlate(signal1,signal1))/npts
	for j in range(k+1,N):
		test     = traces[j]
		signal2  = test.data
		Power_S2 = max(scp.correlate(signal2,signal2))/npts 
		A        = scp.correlate(signal1,signal2,'full')/(npts*np.sqrt(Power_S1*Power_S2))
		CC       = A.max()
		if CC > 0.6:
			print('Waveform1 = ', int(master.stats.sac.kevnm) , ' Waveform2 = ', int(test.stats.sac.kevnm), ' CC = ', CC)
		
		

exit()


