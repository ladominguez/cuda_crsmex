clear all
close all

input = 'filenames.dat';

sac_input = load(input, '-ascii');
Nf        = numel(sac_input)
pos       = [114          73         696        2036]; 

for k = 1:Nf
	figure(k);
	file_sac = [num2str(sac_input(k)) '.IG.CAIG.HHZ.sac'];
        file_fft = ['output' num2str(k-1) '.dat'];
	sac      = rsac(file_sac);
	A        = load(file_fft);
	N        = 2*size(A,1);
	nfft     = (N/2) + 1
	Ac       = A(:,1) + i*A(:,2);

	if N > sac.npts
		N = sac.npts;
	end

	t = sac.t(1:N);
	y = sac.d(1:N);

	f = fft(y);

	subplot(4, 1, 1)
	plot(t, y);

	subplot(4,1,2)
	loglog(abs(f(1:nfft)));
	title('Matlab')

	subplot(4,1,3)
    whos A
	loglog(abs(Ac))
	title('CUDA')

	subplot(4,1,4)
	loglog(abs(f(1:nfft)));
	hold on
	loglog(abs(Ac), 'r')

    pos (1) = pos(1) + 696;
    if pos(1) > 3500
        pos(1) = 114;
    end
	setwin(pos)
    clear A sac f
end
