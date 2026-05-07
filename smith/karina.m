close all;
clear all;
sys=tf(1,[60,1]);
ts=0.1;
dsys=c2d(sys,ts,'z');
[num,den]=tfdata(dsys,'v');