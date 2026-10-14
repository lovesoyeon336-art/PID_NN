% run_engine_test.m
% Engine Model 4 — BP-PI + K_ff前馈 + D_cancel解耦
% Kp1=7.0 Ki1=3.0 | 积分冻结 | 相位超前 | 增益调度

if ~exist('skip_clear','var') || ~skip_clear, clear; end; clc;

Ts=0.01; N_sim=1000;

A=[-0.301,-0.050;0.325,-0.375];
B_mat=[0.079,0.099;0.133,0.302];
C_mat=[1,0;0.038,0.67];
D_mat=[0,0;0.19,-1.31];

%% BP参数（11-8-4: Kp1,Ki1,Kp2,Ki2 + 频率特征）
IN=9; H_bp=8; Out_bp=4;
Kp1_base=7.00; Ki1_base=3.00;
Kp2_base=0.80; Ki2_base=6.50;
K_delta=[18.0,14.0,0.75,20.0];
K_lower=[0.50,0.05,0.05,0.05];
K_upper=[30.0,20.0,2.00,30.0];
Kp2_sign=-1; Ki2_sign=-1;
du_max=8.0; u_sat=30;
beta1=1.00; beta2=0.20;
e_freeze=0.10;  % 积分冻结阈值

%% 前馈 + 解耦
G_ss = D_mat - C_mat * (A \ B_mat);
K_ff = inv(G_ss);
alpha_ff = 0.20;
D_cancel = -D_mat(2,1)/D_mat(2,2);
tau_lead = 0.8;  dr_max = 0.25;  % 相位超前

%% 参考预滤波 + 增益调度
tau_ref = 0.12;  alpha_ref = Ts/(Ts+tau_ref);
K_gain_boost = 1.5;  e1_thresh_gs = 0.3;

fprintf('K_ff=[%.1f %.1f; %.1f %.1f]  D_cancel=%.3f  tau_ref=%.2fs  tau_lead=%.1fs\n',...
    K_ff(1,1),K_ff(1,2),K_ff(2,1),K_ff(2,2),D_cancel,tau_ref,tau_lead);
fprintf('Kp1=%.1f Ki1=%.1f Kp2=%.1f Ki2=%.1f  aw=%.2f\n',...
    Kp1_base,Ki1_base,Kp2_base,Ki2_base,e_freeze);

%% BP权重: 优先加载预训练
pretrain_file = fullfile(fileparts(mfilename('fullpath')), 'bp_mimo_pretrained.mat');
if isfile(pretrain_file)
    ld = load(pretrain_file, 'w1', 'w2');
    w1_global = ld.w1; w2_global = ld.w2;
    fprintf('BP: 加载预训练权重\n');
else
    rng(42);
    w1_global = 0.2 * randn(H_bp, IN);
    w2_global = 0.15 * randn(Out_bp, H_bp);
    fprintf('BP: 随机初始化 (无预训练)\n');
end

%% 场景
t_vec=(0:N_sim-1)*Ts; scenarios=[];
sc.name='Step r1 Only'; sc.r1=ones(1,N_sim); sc.r2=zeros(1,N_sim); sc.N_sim=N_sim; scenarios=[scenarios,sc];
sc.name='Step r2 Only'; sc.r1=zeros(1,N_sim); sc.r2=ones(1,N_sim); sc.N_sim=N_sim; scenarios=[scenarios,sc];
sc.name='r1 Ramp'; sc.r1=min(1,t_vec/4); sc.r2=zeros(1,N_sim); sc.N_sim=N_sim; scenarios=[scenarios,sc];
sc.name='r2 Ramp'; sc.r1=zeros(1,N_sim); sc.r2=min(1,t_vec/4); sc.N_sim=N_sim; scenarios=[scenarios,sc];
sc.name='r1 Sine'; sc.r1=0.5*sin(2*pi*0.06*t_vec); sc.r2=zeros(1,N_sim); sc.N_sim=N_sim; scenarios=[scenarios,sc];
sc.name='r2 Sine'; sc.r1=zeros(1,N_sim); sc.r2=0.5*sin(2*pi*0.08*t_vec); sc.N_sim=N_sim; scenarios=[scenarios,sc];

% 正弦场景延长至30s, 确保2个完整周期
N_sim_long = 3000;
t_vec_long = (0:N_sim_long-1)*Ts;
sc.name='r1 Sine (30s)'; sc.r1=0.5*sin(2*pi*0.06*t_vec_long); sc.r2=zeros(1,N_sim_long); sc.N_sim=N_sim_long; scenarios=[scenarios,sc];
sc.name='r2 Sine (30s)'; sc.r1=zeros(1,N_sim_long); sc.r2=0.5*sin(2*pi*0.08*t_vec_long); sc.N_sim=N_sim_long; scenarios=[scenarios,sc];

% 删除原10s正弦场景 (已替换为30s)
scenarios([5,6]) = [];
sc.name='Dual Step'; sc.r1=0.5*ones(1,N_sim); sc.r2=0.5*ones(1,N_sim); sc.N_sim=N_sim; scenarios=[scenarios,sc];

fprintf('========== 7-scenario test ==========\n');
all_results={};

for sidx=1:length(scenarios)
    sc=scenarios(sidx);
    clear plant_engine_mimo;
    rbf_identifier_mimo(0,0,0,0,'freeze');
    fprintf('  Scenario: %s\n',sc.name);

    % 每场景从预训练权重起步，在线微调
    w1=w1_global; w2=w2_global;

    best_mae=inf; N_ep=1;
    for ep=1:N_ep

    y1_1=0;y2_1=0;y1_2=0;y2_2=0;
    u1_1=0;u1_2=0;u1_3=0;u2_1=0;u2_2=0;u2_3=0;
    u1_pid=0;u2_pid=0;u1_ff=0;r1_1=0;r2_1=0;
    r1_filt=0;
    e1_sp_1=0;e2_sp_1=0;

    nk = sc.N_sim; t_arr=zeros(1,nk);y1_arr=zeros(1,nk);y2_arr=zeros(1,nk);
    u1_arr=zeros(1,nk);u2_arr=zeros(1,nk);
    kp1_a=zeros(1,nk);ki1_a=zeros(1,nk);
    kp2_a=zeros(1,nk);ki2_a=zeros(1,nk);
    mae_sum1=0;mae_sum2=0;

    for k=1:nk
        t_arr(k)=k*Ts; r1_raw=sc.r1(k); r2_raw=sc.r2(k);
        r1_filt = alpha_ref*r1_raw + (1-alpha_ref)*r1_filt;
        r1_k=r1_filt; r2_k=r2_raw;
        e1_filt=r1_k-y1_1; e2_filt=r2_k-y2_1;

        %% BP前向 (9-8-4, 权重冻结)
        I1=[r1_k,y1_1,e1_filt,r1_k-r1_1,r2_k,y2_1,e2_filt,r2_k-r2_1,1];
        I2=I1*w1'; O2=tanh(I2); I3=w2*O2'; I3_t=I3';
        O3=tanh(I3_t);
        gain_raw=[Kp1_base,Ki1_base,Kp2_base,Ki2_base]+K_delta.*O3;
        gain_abs=max(K_lower,min(K_upper,gain_raw));

        Kp1=gain_abs(1); Ki1=gain_abs(2);
        Kp2=Kp2_sign*gain_abs(3); Ki2=Ki2_sign*gain_abs(4);
        Kp1_eff = Kp1 * (1 + K_gain_boost * exp(-abs(e1_filt)/e1_thresh_gs));
        kp1_a(k)=Kp1_eff;ki1_a(k)=Ki1;kp2_a(k)=Kp2;ki2_a(k)=Ki2;

        %% 增量式PI
        e1_sp=beta1*r1_k-y1_1; e2_sp=beta2*r2_k-y2_1;
        de1_sp=e1_sp-e1_sp_1; de2_sp=e2_sp-e2_sp_1;
        du1=Kp1_eff*de1_sp;
        if abs(e1_filt)<=e_freeze, du1=du1+Ki1*Ts*e1_filt; end
        du2=Kp2*de2_sp+Ki2*Ts*e2_filt;
        du1=max(-du_max,min(du_max,du1));
        du2=max(-du_max,min(du_max,du2));
        u1_pid=u1_pid+du1; u2_pid=u2_pid+du2;

        %% Ch1前馈 (相位超前)
        dr1 = (r1_k - r1_1)/Ts;
        dr1_clip = max(-dr_max, min(dr_max, dr1));
        r1_lead = r1_k + tau_lead * dr1_clip;
        u1_ff_raw = K_ff(1,1)*r1_lead + K_ff(1,2)*r2_k;
        u1_ff = alpha_ff*u1_ff_raw + (1-alpha_ff)*u1_ff;

        %% 总控制量
        u1_k=max(-u_sat,min(u_sat,u1_ff + u1_pid));
        u2_k=max(-u_sat,min(u_sat,u2_pid + D_cancel*u1_k));
        u1_arr(k)=u1_k;u2_arr(k)=u2_k;

        %% 被控对象
        [y1_k,y2_k]=plant_engine_mimo(y1_1,y2_1,u1_1,u1_2,u1_3,u2_1,u2_2,u2_3);
        y1_arr(k)=y1_k; y2_arr(k)=y2_k;
        mae_sum1=mae_sum1+abs(r1_k-y1_k); mae_sum2=mae_sum2+abs(r2_k-y2_k);

        y1_2=y1_1; y1_1=y1_k; y2_2=y2_1; y2_1=y2_k;
        u1_3=u1_2;u1_2=u1_1;u1_1=u1_k;u2_3=u2_2;u2_2=u2_1;u2_1=u2_k;
        r1_1=r1_k; r2_1=r2_k;
        e1_sp_1=e1_sp; e2_sp_1=e2_sp;
    end

    mae_avg=(mae_sum1+mae_sum2)/(2*nk);
    if mae_avg<best_mae
        best_mae=mae_avg; best_y1=y1_arr; best_y2=y2_arr;
        best_u1=u1_arr; best_u2=u2_arr; best_t=t_arr;
        best_kp1=kp1_a;best_ki1=ki1_a;best_kp2=kp2_a;best_ki2=ki2_a;
    end
    end
    fprintf('  Best Ep MAE1=%.4f MAE2=%.4f\n',...
        mean(abs(sc.r1-best_y1)),mean(abs(sc.r2-best_y2)));

    results=calc_metrics(best_t,sc.r1,best_y1,sc.r2,best_y2,best_u1,best_u2);
    results.scenario_name=sc.name;results.t=best_t;
    results.r1=sc.r1;results.y1=best_y1;results.e1=sc.r1-best_y1;results.u1=best_u1;
    results.r2=sc.r2;results.y2=best_y2;results.e2=sc.r2-best_y2;results.u2=best_u2;
    results.kp11=best_kp1;results.ki11=best_ki1;
    results.kp22=best_kp2;results.ki22=best_ki2;
    all_results{end+1}=results;
end

fprintf('\n========== Summary ==========\n');
fprintf('%-18s %8s %8s %8s %8s %8s %8s\n','Scenario','MAE1','MAE2','peak1','peak2','ess1','ess2');
fprintf('%s\n',repmat('-',1,78));
for i=1:length(all_results)
    r=all_results{i};
    fprintf('%-18s %8.4f %8.4f %8.3f %8.3f %8.4f %8.4f\n',...
        r.scenario_name,r.MAE1,r.MAE2,r.peak_pct1,r.peak_pct2,r.ess1,r.ess2);
end
data_dir = fileparts(mfilename('fullpath'));
save(fullfile(data_dir,'engine_test_data.mat'),'all_results');
fprintf('Data: %s\n', fullfile(data_dir,'engine_test_data.mat'));

%% 出图改用 plot_figures_all.py (手动运行)
% py_path = fullfile(fileparts(mfilename('fullpath')), 'plot_figures.py');
% system(['D:/Python312/python.exe "' py_path '"']);

function m=calc_metrics(t,r1,y1,r2,y2,u1,u2)
    N=length(t); e1=r1-y1; e2=r2-y2;
    m.MAE1=mean(abs(e1)); m.MAE2=mean(abs(e2));
    ss=round(0.7*N):N;
    m.ess1=mean(abs(e1(ss))); m.ess2=mean(abs(e2(ss)));
    m.peak_pct1=calc_peak(y1,r1); m.peak_pct2=calc_peak(y2,r2);
    m.delta_u1_rms=sqrt(mean(diff(u1).^2));
    m.delta_u2_rms=sqrt(mean(diff(u2).^2));
    m.peak_u1=max(abs(u1)); m.peak_u2=max(abs(u2));
end
function pk=calc_peak(y,r)
    ss=round(0.7*length(r)):length(r); r_tail=r(ss);
    r_ss=mean(r_tail);
    if max(r_tail)-min(r_tail)<0.02
        if abs(r_ss)<0.01, pk=max(abs(y));
        else, pk=max(0,(max(y)-r_ss)/abs(r_ss)*100); end
    else
        amp=max(abs(r)); if amp<0.01, pk=max(abs(y)); return; end
        pk=max(abs(y-r))/amp*100;
    end
end
