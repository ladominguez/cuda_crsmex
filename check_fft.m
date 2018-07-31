clear all
close all


sac = rsac('20110106064638.IG.CAIG.HHZ.sac');
A   = load('output0.dat');
N    = 2*size(A,1);
nfft = (N/2) + 1
Ac  = A(:,1) + i*A(:,2);
t = sac.t(1:N);
y = sac.d(1:N);

f = fft(y);

subplot(4, 1, 1)
plot(t, y);

subplot(4,1,2)
title('Matlab')
loglog(abs(f(1:nfft)));

subplot(4,1,3)
title('CUDA')
loglog(abs(Ac))

subplot(4,1,4)
loglog(abs(f(1:nfft)));
hold on
loglog(abs(Ac), 'r')


