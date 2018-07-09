clear variable

A = load('check_dara.dat');
N = size(A,1);
[files Nf] = ValidateComponent('Z');
for k =1:N
    sac =rsac(files(k+1).name);
    subplot(3,1,1)
    plot(A(k,:))
    subplot(3,1,2)
    plot(sac.d)
    subplot(3,1,3)

    pause(1)
    
end