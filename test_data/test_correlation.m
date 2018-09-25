clear;
close all;

A = rsac('20110110084332.IG.CAIG.HHZ.sac');
B = rsac('20110113002147.IG.CAIG.HHZ.sac');

subplot(3,2,1)
plot(A.t, A.d);
title('Signal 1')
axis tight 

subplot(3,2,3)
plot(B.t, B.d);
title('Signal 2')
axis tight

subplot(3,2,5)
plot(A.t, A.d);
hold all
plot(B.t, B.d);
axis tight

Af = filter_sac(A, 1, Inf, 2);
Bf = filter_sac(B, 1, Inf, 2);

subplot(3,2,2)
plot(Af.t, Af.d);
title('Signal 1')
axis tight 

subplot(3,2,4)
plot(Bf.t, Bf.d);
title('Signal 2')
axis tight

subplot(3,2,6)
plot(Af.t, Af.d);
hold all
plot(Bf.t, Bf.d);
axis tight



