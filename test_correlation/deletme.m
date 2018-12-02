clear all
close all

 A = load('tmp.dat');
% 
% plot(A(513:end,2))

corr_real = load('corr_AB_AUX_real.dat');
corr_imag = load('corr_AB_AUX_imag.dat');

% corr_cuda = sortrows(load('tmp.dat'));
% corr_cuda_real = corr_cuda(:,2);
% corr_cuda_imag = corr_cuda(:,3);

subplot(2,1,1)
plot(corr_real)
hold on
plot(A(1025:end,1)/1024,'ro')
subplot(2,1,2)
plot(corr_imag)
% hold on
% plot(A(513:end,3),'r')
