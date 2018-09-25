clear
close all

A = rsac('20011105104504.IG.PLIG.BHZ.sac');
B = rsac('20040403234914.IG.PLIG.BHZ.sac');

Af = filter_sac(A, 1, Inf, 2);
Bf = filter_sac(B, 1, Inf, 2);

[dsamp,maxc,xcor] = xcorrshift(Af.d, Bf.d)

%[CorrelationCoefficient tshift S1 S2] = get_correlation_coefficient(Af,Bf,25.5, 'combined',1);

s1 = 