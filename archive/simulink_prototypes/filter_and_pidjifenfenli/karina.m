close all;
clear all;

ts=0.001;
Q=tf(1,[0.04,1]);
dQ=c2d(Q,ts,'tustin');
[numQ,denQ]=tfdata(dQ,'v');

sys=tf(5.235e005,[1,87.35,1.047e004,0]);
dsys=c2d(sys,ts,'z');
[num,den]=tfdata(dsys,'v');
ki=0.01;
kp=0.20;
