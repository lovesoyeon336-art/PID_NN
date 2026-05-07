close all;
t=out.t;
y=out.y;
plot(t,y(:,1),'r',t,y(:,2),'k');
xlabel("time(s)");
ylabel("yd/y");
legend("ideal","tracking");
grid on;
