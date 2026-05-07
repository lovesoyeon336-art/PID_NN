function [sys,x0,str,ts] = exp_pidf(t,x,u,flag)
switch flag,
    case 0
        [sys,x0,str,ts] = mdlInitializeSizes;
    case 2
        sys = mdlUpdates(x,u);
    case 3
        sys = mdlOutputs(t,x,u);
    case {1, 4, 9}
        sys = [];
    otherwise
        error(['unhandled flag = ',num2str(flag)]);
end

function [sys,x0,str,ts] = mdlInitializeSizes
sizes = simsizes;
sizes.NumContStates  = 0;
sizes.NumDiscStates  = 3;
sizes.NumOutputs    = 4;
sizes.NumInputs      = 7;
sizes.DirFeedthrough = 1;
sizes.NumSampleTimes = 1;
sys = simsizes(sizes);
x0  = zeros(3,1);
str = [];
ts  = [0.001 0];

function sys = mdlUpdates(x,u)
T=0.001;
x=[u(5);x(2)+u(5)*T;(u(5)-u(4))/T];
sys=[x(1);x(2);x(3)];

function sys = mdlOutputs(t,x,u)

 persistent wi wo wi_1 wi_2 wo_1 wo_2
  IN=3;H=5;OUT=3;T=0.001;
    if t == 0
        wi = rand(H, IN);
        wo = rand(OUT, H);
        wi_1 = wi; wi_2 = wi;
        wo_1 = wo; wo_2 = wo;
        disp('Initial wi:'); disp(wi);
    disp('Initial wo:'); disp(wo);
    end
  
    %xite=0.0001;
    xite_base = 1e-4;
xite = xite_base * (1 + abs(u(5)));  % 误差大时学习更快
    alfa=0.05;
    
    Oh=zeros(5,1);
    I=Oh;
    %xi=[u(1) u(3) u(5) ];
    % xi=[x(1) x(2) x(3) ];
    
    persistent integral_e
if t==0, integral_e=0; end
integral_e = integral_e + u(5)*T;
xi = [u(5), integral_e, (u(5)-u(4))/T];
    xi = xi ./ (max(abs(xi)) + eps); 
    
    epid=[x(1);x(2);x(3)];
    epid = epid ./ (max(abs(xi)) + eps); 
    I=xi*wi';
    for j=1:1:H
       Oh(j) = tanh(I(j));        
    end
    K1=wo*Oh;
    for i=1:1:OUT
        %K(i)=exp(K1(i))/(exp(K1(i))+exp(-K1(i)));
        K(i)  = 1 / (1 + exp(-2*K1(i)));
    end
    u_k=K*epid;

    %dyu=sign(((u(3)-u(2)))/(u(7)-u(6)+0.0001));
    persistent dyu_filt
if t==0
    dyu_filt = 0;
end
    dy_raw = (u(3)-u(2)) / (u(7)-u(6) + 1e-6);
dy_raw = max(min(dy_raw, 1), -1);
dyu = 0.9*dyu_filt + 0.1*dy_raw;
dyu_filt = dyu;

    for j=1:1:OUT
        %dK(j)=2/(exp(K1(j))+exp(-K1(j)))^2;
        dK(j) = 2*K(j)*(1-K(j));    % 对应导数，无需重新计算 exp
    end
    for i=1:1:OUT
        delta3(i)=u(5)*dyu_filt*epid(i)*dK(i);
    end
    d_wo = xite * delta3' * Oh' + alfa*(wo_1-wo_2);
    wo = wo_1+d_wo;
   
    for i=1:1:H
        %dO(i)=4/(exp(I(i))+exp(-I(i)))^2;
        dO(i) = 1 - tanh(I(i))^2;   % sech²(x) 的等价形式
    end
    segma=delta3*wo;
    delta2=dO.*segma;
    d_wi=xite*delta2'*xi+alfa*(wi_1-wi_2);
    wi=wi_1+d_wi;
    wo_2=wo_1;
    wo_1=wo;
    wi_2=wi_1;
    wi_1=wi;
    Kp=K(1)*10;Ki=K(2)*5;Kd=K(3)*2;
    sys=[u_k, Kp, Ki, Kd];