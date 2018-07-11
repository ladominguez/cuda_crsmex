clear variable
close all
A = load('data.dat');
N = size(A,1);
[files Nf] = ValidateComponent('Z');


for k =1:N
    
    
    sac =rsac(files(k+1).name);
    %sac = rsac('20110210143930.IG.CAIG.HHZ.sac');
    %subplot(3,1,1)
    %plot(A(k,:))
    %axis tight
    %subplot(3,1,2)
    plot(sac.d)
    hold on 
    plot(A(k,:))
    title(files(k+1).name);
    axis tight
    %subplot(3,1,3)
   
    pause
    clf
    
end
