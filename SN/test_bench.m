clear; close all;

%% ==================== 加载参数 ====================
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));

tuned1 = load(fullfile(script_dir, 'pid_tuned_params.mat'));
tuned2 = load(fullfile(script_dir, 'pid_tuned_params_plant2.mat'));

sn1 = load(fullfile(script_dir, 'sn_tuned_params_plant1.mat'));
sn2 = load(fullfile(script_dir, 'sn_tuned_params_plant2.mat'));

fprintf('对象1: PID Kp=%.2f Ki=%.2f Kd=%.2f  |  SN K=%.2f eta=%.2f\n', ...
    tuned1.Kp_opt, tuned1.Ki_opt, tuned1.Kd_opt, sn1.K_opt, sn1.eta_opt);
fprintf('对象2: PID Kp=%.2f Ki=%.2f Kd=%.2f  |  SN K=%.2f eta=%.2f\n', ...
    tuned2.Kp_opt, tuned2.Ki_opt, tuned2.Kd_opt, sn2.K_opt, sn2.eta_opt);

%% ==================== 16 场景测试矩阵 ====================
N = 2000;

sc_defs = {
    'plant1', 'step',        '1.基本阶跃';
    'plant1', 'sine_low',    '2.正弦低频';
    'plant1', 'sine_high',   '3.正弦高频';
    'plant1', 'ramp',        '4.斜坡跟踪';
    'plant1', 'composite',   '5.复合信号';
    'plant1', 'perturb',     '6.参数摄动';
    'plant1', 'disturb',     '7.输出扰动';
    'plant1', 'noise',       '8.量测噪声';
    'plant2', 'step',        '9.对象2阶跃';
    'plant2', 'sine_low',    '10.对象2正弦';
    'plant2', 'square',      '11.对象2方波';
};
nS = size(sc_defs, 1);

SN_MAE=zeros(1,nS); Fix_MAE=zeros(1,nS);
SN_ISE=zeros(1,nS); Fix_ISE=zeros(1,nS);
SN_ITAE=zeros(1,nS); Fix_ITAE=zeros(1,nS);
SN_Stl=zeros(1,nS); Fix_Stl=zeros(1,nS);
SN_dU=zeros(1,nS);  Fix_dU=zeros(1,nS);

for i = 1:nS
    pid = sc_defs{i,1};
    sc  = sc_defs{i,2};
    r_seq = get_r_array(N, sc);

    switch pid
        case 'plant1'
            KpF=tuned1.Kp_opt; KiF=tuned1.Ki_opt; KdF=tuned1.Kd_opt;
            Ksn=sn1.K_opt; eta_sn=sn1.eta_opt;
        case 'plant2'
            KpF=tuned2.Kp_opt; KiF=tuned2.Ki_opt; KdF=tuned2.Kd_opt;
            Ksn=sn2.K_opt; eta_sn=sn2.eta_opt;
    end

    [y_sn, e_sn, u_sn] = sim_sn_pid(N, sc, pid, Ksn, eta_sn);
    [y_fix, e_fix, u_fix] = sim_fix_pid(N, sc, pid, KpF, KiF, KdF);

    [SN_MAE(i),Fix_MAE(i), SN_ISE(i),Fix_ISE(i), SN_ITAE(i),Fix_ITAE(i), ...
     SN_Stl(i),Fix_Stl(i), SN_dU(i),Fix_dU(i)] = ...
        metrics2(y_sn, e_sn, r_seq, y_fix, e_fix, u_sn, u_fix);
end

%% ==================== 输出对比表 ====================
fprintf('\n========== 单神经元 PID 测试结果 ==========\n');
fprintf('%-16s | %-8s %-8s | %-8s %-8s\n', '场景', 'SN_MAE', 'Fix_MAE', 'SN_ISE', 'Fix_ISE');
fprintf('%s\n', repmat('-', 1, 70));
for i = 1:nS
    fprintf('%-16s | %8.4f %8.4f | %8.4f %8.4f\n', ...
        sc_defs{i,3}, SN_MAE(i), Fix_MAE(i), SN_ISE(i), Fix_ISE(i));
end

fprintf('\n综合 MAE: SN=%.4f  Fix=%.4f  倍率=%.2f\n', sum(SN_MAE), sum(Fix_MAE), sum(SN_MAE)/max(sum(Fix_MAE),1e-9));
fprintf('综合 ISE: SN=%.4f  Fix=%.4f\n', sum(SN_ISE), sum(Fix_ISE));
fprintf('综合 ITAE: SN=%.2f  Fix=%.2f\n', sum(SN_ITAE), sum(Fix_ITAE));

%% ==================== 保存 ====================
save(fullfile(script_dir, 'sn_test_results.mat'), ...
    'sc_defs','N','SN_MAE','Fix_MAE','SN_ISE','Fix_ISE','SN_ITAE','Fix_ITAE', ...
    'SN_Stl','Fix_Stl','SN_dU','Fix_dU');
fprintf('\n结果已保存至 sn_test_results.mat\n');

%% ==================== 单神经元 PID 仿真 ====================

function [y, error, u] = sim_sn_pid(N, scenario, plant_id, K_neuron, eta)
    w = [0.3, 0.3, 0.3];
    y_1 = 0;  y_2 = 0;  u_1 = 0;  error_1 = 0;  error_2 = 0;
    y = zeros(1, N);  error = zeros(1, N);  u = zeros(1, N);
    y_true = 0;
    rng(42);

    for k = 1:N
        r_k = get_target(k, scenario);
        y_fb = get_feedback(y_true, k, scenario);
        e_cur = r_k - y_fb;
        error(k) = e_cur;

        x1 = e_cur;
        x2 = e_cur - error_1;
        x3 = e_cur - 2*error_1 + error_2;

        w_sum = abs(w(1)) + abs(w(2)) + abs(w(3)) + 1e-6;
        delta_u = K_neuron * (w(1)/w_sum*x1 + w(2)/w_sum*x2 + w(3)/w_sum*x3);
        delta_u = max(-0.5, min(0.5, delta_u));
        u_k = u_1 + delta_u;
        u(k) = u_k;

        a_override = get_ak(k, scenario);
        y_true = plant_dynamics(plant_id, y_1, y_2, u_k, u_1, k, a_override);
        y(k) = y_true;

        w(1) = w(1) + eta * e_cur * x1;
        w(2) = w(2) + eta * e_cur * x2;
        w(3) = w(3) + eta * e_cur * x3;

        u_1 = u_k;  y_2 = y_1;  y_1 = y_true;
        error_2 = error_1;  error_1 = e_cur;
    end
end

%% ==================== 固定 PID 仿真 ====================

function [y, error, u] = sim_fix_pid(N, scenario, plant_id, Kp, Ki, Kd)
    y_1 = 0;  y_2 = 0;  u_1 = 0;  ei = 0;  last_e = 0;
    y = zeros(1, N);  error = zeros(1, N);  u = zeros(1, N);
    y_true = 0;
    rng(42);

    for k = 1:N
        r_k = get_target(k, scenario);
        y_fb = get_feedback(y_true, k, scenario);
        e_cur = r_k - y_fb;
        error(k) = e_cur;

        ei = ei + e_cur;
        ei_clamped = max(-3, min(3, ei));
        ed = e_cur - last_e;
        delta_u = Kp*e_cur + Ki*ei_clamped + Kd*ed - u_1;
        delta_u = max(-0.5, min(0.5, delta_u));
        u_k = u_1 + delta_u;
        u(k) = u_k;

        a_override = get_ak(k, scenario);
        y_true = plant_dynamics(plant_id, y_1, y_2, u_k, u_1, k, a_override);
        y(k) = y_true;

        last_e = e_cur;
        y_2 = y_1;  y_1 = y_true;
        u_1 = u_k;
    end
end

%% ==================== 指标函数 ====================

function [sn_mae,fix_mae, sn_ise,fix_ise, sn_itae,fix_itae, sn_stl,fix_stl, sn_du,fix_du] = ...
        metrics2(y_sn, e_sn, r_seq, y_fix, e_fix, u_sn, u_fix)
    N = length(e_sn);
    k_vec = 1:N;

    sn_mae=mean(abs(e_sn)); fix_mae=mean(abs(e_fix));
    sn_ise=sum(e_sn.^2)/N; fix_ise=sum(e_fix.^2)/N;
    sn_itae=sum(k_vec.*abs(e_sn)); fix_itae=sum(k_vec.*abs(e_fix));

    band=0.05;
    sn_last=find(abs(y_sn-r_seq)>band.*abs(r_seq+1e-6),1,'last');
    fix_last=find(abs(y_fix-r_seq)>band.*abs(r_seq+1e-6),1,'last');
    sn_stl=sn_last; if isempty(sn_stl),sn_stl=0;end
    fix_stl=fix_last; if isempty(fix_stl),fix_stl=0;end

    sn_du=sqrt(mean(diff(u_sn).^2));
    fix_du=sqrt(mean(diff(u_fix).^2));
end

%% ==================== 场景参数函数 ====================

function r_seq = get_r_array(N, scenario)
    r_seq = zeros(1, N);
    for k = 1:N
        r_seq(k) = get_target(k, scenario);
    end
end

function r = get_target(k, scenario)
    persistent rand_seq rand_vals rand_amp
    switch scenario
        case 'varying_r'
            if k<=500, r=1; elseif k<=1000, r=2; elseif k<=1500, r=0.5; else, r=1.5; end
        case 'square'
            if mod(floor((k-1)/100),2)==0, r=1; else, r=2; end
        case 'sine_low'
            r = 1 + 0.5*sin(2*pi*0.005*k);
        case 'sine_high'
            r = 1 + 0.5*sin(2*pi*0.02*k);
        case 'ramp'
            r = min(1, k/500);
        case 'composite'
            r = 1 + 0.6*sin(2*pi*0.005*k) + 0.3*sin(2*pi*0.03*k);
        case 'random_step'
            if isempty(rand_seq) || mod(k,200)==1
                rng(k); rand_amp = 0.5 + 1.5*rand();
            end
            r = rand_amp;
        otherwise
            r = 1;
    end
end

function a = get_ak(k, scenario)
    a0 = 1.2 * (1 - 0.8*exp(-0.1*k));
    switch scenario
        case 'perturb'
            if k>500 && k<=1000, a=a0*1.3; elseif k>1500, a=a0*0.7; else, a=a0; end
        case 'drift'
            if k>500, a=a0*0.5; else, a=a0; end
        otherwise
            a = a0;
    end
end

function y_fb = get_feedback(y_true, k, scenario)
    switch scenario
        case 'disturb'
            if k==500, y_fb=y_true+0.5; else, y_fb=y_true; end
        case 'noise'
            y_fb = y_true + (rand-0.5)*2*0.02;
        case 'combo'
            y_fb = y_true + (rand-0.5)*2*0.02;
            if k==300, y_fb=y_fb+0.5; end
            if k==700, y_fb=y_fb-0.3; end
        otherwise
            y_fb = y_true;
    end
end
