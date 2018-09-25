function sac=filter_sac(sac,low,high,npoles)
% function sac=filter_sac(sac,low,high,npoles)
if nargin < 4
	npoles=2;
end

Nyquist=0.5*(1/sac.dt);

if low == 0
	[b,a]=butter(npoles,high./Nyquist,'low');
elseif high ==Inf
	[b,a]=butter(npoles,low./Nyquist,'high');
else
	[b,a]=butter(npoles,[low high]./Nyquist);
end
 
sac.d=filter(b,a,sac.d);
 

