clear all;
%close all;

ts=0.001;
x=[0 0 0]';
y_1=0;y_2=0;
u_1=0;u_2=0;
% wkp_1=0.10;
% wki_1=0.10;
% wkd_1=0.10;
wkp_1=rand;
wki_1=rand;
wkd_1=rand;
xitep=0.40;
xitei=0.35;
xited=0.40;
error1=0;
error2=0;

for k=1:1:1000
    time(k)=k*ts;

yd(k)=0.5*sign(sin(4*pi*k*ts));
y(k)=0.368*y_1+0.26*y_2+0.1*u_1+0.632*u_2;
error(k)=yd(k)-y(k);

M= 1;

if M==1
    wkp(k)=wkp_1+xitep*(2*error(k)-error1)*u_1;
    wki(k)=wki_1+xitei*(2*error(k)-error1)*u_1;
    wkd(k)=wkd_1+xited*(2*error(k)-error1)*u_1;
    K=0.06;
elseif M==2
    wkp(k)=wkp_1+xitep*error(k)*(2*error(k)-error1);%这里用的是delta公式，效果还不如用u1
    wki(k)=wki_1+xitei*error(k)*(2*error(k)-error1);
    wkd(k)=wkd_1+xited*error(k)*(2*error(k)-error1);
    K=0.12;
elseif M==3
    wkp(k)=wkp_1+xitep*x(1)*u_1*error(k);
    wki(k)=wki_1+xitei*x(2)*u_1*error(k);
    wkd(k)=wkd_1+xited*x(3)*u_1*error(k);
    K=0.12;
elseif M==4
    wkp(k)=wkp_1+xitep*(2*error(k)-error1)*u_1*error(k);
    wki(k)=wki_1+xitei*(2*error(k)-error1)*u_1*error(k);
    wkd(k)=wkd_1+xited*(2*error(k)-error1)*u_1*error(k);
    K=0.12;
end

x(1)=error(k)-error1;
x(2)=error(k);
x(3)=error(k)-2*error1+error2;

wadd(k)=abs(wkp(k))+abs(wki(k))+abs(wkd(k));
w11(k)=wkp(k)/wadd(k);
w22(k)=wki(k)/wadd(k);
w33(k)=wkd(k)/wadd(k);
w=[w11(k) w22(k) w33(k)];
u(k)=u_1+K*w*x;

error2=error1;
error1=error(k);
u_2=u_1;u_1=u(k);
y_2=y_1;y_1=y(k);

wkp_1=wkp(k);
wki_1=wki(k);
wkd_1=wkd(k);
end

figure(1);
plot(time,yd,'r',time,y,'k','linewidth',2);
xlabel('time(s)');
ylabel('yd,y');
legend('ideal position','position tracking');
figure(2);
plot(time,u,'r','linewidth',2);
xlabel('time(s)');
ylabel('control input');















