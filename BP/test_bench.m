clear; close all;

%% ==================== 加载整定参数和预训练权重 ====================
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));

tuned = load(fullfile(script_dir, 'pid_tuned_params.mat'));
Kp_fix = tuned.Kp_opt;
Ki_fix = tuned.Ki_opt;
Kd_fix = tuned.Kd_opt;

pretrain1 = load(fullfile(script_dir, 'bp_pretrained_weights.mat'));
pretrain2 = load(fullfile(script_dir, 'bp_pretrained_weights_plant2.mat'));
fprintf('对象1 PID: Kp=%.4f  Ki=%.4f  Kd=%.4f\n', Kp_fix, Ki_fix, Kd_fix);

%% ==================== 加载各对象固定PID ====================
tuned2 = load(fullfile(script_dir, 'pid_tuned_params_plant2.mat'));
fprintf('对象2 PID: Kp=%.4f  Ki=%.4f  Kd=%.4f\n', tuned2.Kp_opt, tuned2.Ki_opt, tuned2.Kd_opt);

%% ==================== 16 场景测试矩阵 ====================

N = 2000;

% {plant_id, scenario, display_name}
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

BP_MAE=zeros(1,nS); Fix_MAE=zeros(1,nS);
BP_ISE=zeros(1,nS); Fix_ISE=zeros(1,nS);
BP_ITAE=zeros(1,nS); Fix_ITAE=zeros(1,nS);
BP_Stl=zeros(1,nS); Fix_Stl=zeros(1,nS);
BP_dU=zeros(1,nS);  Fix_dU=zeros(1,nS);
BP_PeakU=zeros(1,nS); Fix_PeakU=zeros(1,nS);
BP_Ov=zeros(1,nS); Fix_Ov=zeros(1,nS);
BP_Pk=zeros(1,nS); Fix_Pk=zeros(1,nS);
BP_Ss=zeros(1,nS); Fix_Ss=zeros(1,nS);

Y_bp=cell(1,nS); Y_fix=cell(1,nS); R_seq=cell(1,nS);

for i = 1:nS
    pid = sc_defs{i,1};
    sc  = sc_defs{i,2};
    r_seq = get_r_array(N, sc);

    switch pid
        case 'plant1'
            KpF = Kp_fix; KiF = Ki_fix; KdF = Kd_fix;
            w1p = pretrain1.w1; w2p = pretrain1.w2;
        case 'plant2'
            KpF = tuned2.Kp_opt; KiF = tuned2.Ki_opt; KdF = tuned2.Kd_opt;
            w1p = pretrain2.w1; w2p = pretrain2.w2;
    end

    [y_bp, e_bp, u_bp] = sim_bp_pid(N, sc, pid, w1p, w2p);
    [y_fix, e_fix, u_fix] = sim_fix_pid(N, sc, pid, KpF, KiF, KdF);

    [BP_MAE(i),Fix_MAE(i), BP_ISE(i),Fix_ISE(i), ...
     BP_ITAE(i),Fix_ITAE(i), ...
     BP_Stl(i),Fix_Stl(i), ...
     BP_dU(i),Fix_dU(i), ...
     BP_PeakU(i),Fix_PeakU(i), ...
     BP_Ov(i),Fix_Ov(i), ...
     BP_Pk(i),Fix_Pk(i), ...
     BP_Ss(i),Fix_Ss(i)] = ...
        metrics2(y_bp,e_bp,r_seq, y_fix,e_fix, u_bp,u_fix);

    Y_bp{i}=y_bp; Y_fix{i}=y_fix; R_seq{i}=r_seq;
end

%% ==================== 输出对比表 ====================
fprintf('\n========== 测试结果汇总 ==========\n');
fprintf('%-16s | %-8s %-8s | %-8s %-8s\n', ...
    '场景', 'BP_MAE', 'Fix_MAE', 'BP_ISE', 'Fix_ISE');
fprintf('%s\n', repmat('-', 1, 70));
for i = 1:nS
    fprintf('%-16s | %8.4f %8.4f | %8.4f %8.4f\n', ...
        sc_defs{i,3}, BP_MAE(i), Fix_MAE(i), BP_ISE(i), Fix_ISE(i));
end

fprintf('\n----- 动态性能指标 -----\n');
fprintf('%-16s | %-8s %-8s | %-8s %-8s | %-8s %-8s\n', ...
    '场景', 'BP_超调', 'Fix_超调', 'BP_峰值e', 'Fix_峰值e', 'BP_稳态e', 'Fix_稳态e');
fprintf('%s\n', repmat('-', 1, 90));
for i = 1:nS
    fprintf('%-16s | %8.4f %8.4f | %8.4f %8.4f | %8.4f %8.4f\n', ...
        sc_defs{i,3}, BP_Ov(i), Fix_Ov(i), BP_Pk(i), Fix_Pk(i), BP_Ss(i), Fix_Ss(i));
end

fprintf('\n----- 综合 MAE -----\n');
fprintf('BP-PID=%.4f  固定PID=%.4f\n', sum(BP_MAE), sum(Fix_MAE));
fprintf('BP/Fix=%.2f\n', sum(BP_MAE)/max(sum(Fix_MAE),1e-9));
fprintf('\n----- 综合 ISE -----\n');
fprintf('BP-PID=%.4f  固定PID=%.4f\n', sum(BP_ISE), sum(Fix_ISE));
fprintf('\n----- 综合 ITAE -----\n');
fprintf('BP-PID=%.2f  固定PID=%.2f\n', sum(BP_ITAE), sum(Fix_ITAE));

%% ==================== 绘图 ====================
figure('Name', 'MAE/ISE/ITAE/Δu/PeakU 指标对比', 'NumberTitle', 'off');
metrics_names = {'MAE', 'ISE', 'ITAE', 'RMS(Δu)', 'Peak |u|'};
for p = 1:5
    subplot(2,3,p);
    switch p
        case 1, bp_d=BP_MAE; fix_d=Fix_MAE;
        case 2, bp_d=BP_ISE; fix_d=Fix_ISE;
        case 3, bp_d=BP_ITAE; fix_d=Fix_ITAE;
        case 4, bp_d=BP_dU;  fix_d=Fix_dU;
        case 5, bp_d=BP_PeakU; fix_d=Fix_PeakU;
    end
    bar([bp_d(:),fix_d(:)]);
    set(gca,'XTickLabel',sc_defs(:,3));
    xtickangle(45);
    if p==1, legend('BP-PID','固定PID','Location','best'); end
    title(metrics_names{p}); grid on;
end

figure('Name', '超调/峰值误差/稳态误差 对比', 'NumberTitle', 'off');
dyn_names = {'超调量', '峰值动态误差', '稳态误差'};
for p = 1:3
    subplot(1,3,p);
    switch p
        case 1, bp_d=BP_Ov; fix_d=Fix_Ov;
        case 2, bp_d=BP_Pk; fix_d=Fix_Pk;
        case 3, bp_d=BP_Ss; fix_d=Fix_Ss;
    end
    bar([bp_d(:),fix_d(:)]);
    set(gca,'XTickLabel',sc_defs(:,3));
    xtickangle(45);
    if p==1, legend('BP-PID','固定PID','Location','best'); end
    title(dyn_names{p}); grid on;
end

fx=[1,min(500,N)];
nFigs = ceil(nS/6);
for fig=1:nFigs
    figure('Name',sprintf('时域响应 %d',fig),'NumberTitle','off');
    s0=(fig-1)*6;
    for s=1:min(6,nS-s0)
        subplot(2,3,s); idx=s0+s;
        r_plot=R_seq{idx}; yb=Y_bp{idx}; yf=Y_fix{idx};
        plot(1:N,r_plot,'r',1:N,yf,'g',1:N,yb,'b--','LineWidth',0.8);
        xlim(fx); xlabel('步'); ylabel('y');
        if s==1, legend('目标','固定PID','BP-PID','Location','best'); end
        title(sc_defs{idx,3}); grid on;
    end
end

%% ==================== 保存 ====================
save(fullfile(script_dir, 'test_results.mat'), ...
    'sc_defs','N','BP_MAE','Fix_MAE','BP_ISE','Fix_ISE', ...
    'BP_ITAE','Fix_ITAE','BP_Stl','Fix_Stl', ...
    'BP_dU','Fix_dU','BP_PeakU','Fix_PeakU', ...
    'BP_Ov','Fix_Ov','BP_Pk','Fix_Pk','BP_Ss','Fix_Ss');
fprintf('\n结果已保存至 test_results.mat\n');

%% ==================== 指标函数（两控制器） ====================

function [bp_mae,fix_mae, bp_ise,fix_ise, ...
          bp_itae,fix_itae, bp_stl,fix_stl, ...
          bp_du,fix_du, bp_peak,fix_peak, ...
          bp_ov,fix_ov, bp_pk,fix_pk, bp_ss,fix_ss] = ...
        metrics2(y_bp, e_bp, r_seq, y_fix, e_fix, u_bp, u_fix)
    N = length(e_bp);

    % MAE
    bp_mae=mean(abs(e_bp)); fix_mae=mean(abs(e_fix));

    % ISE (积分平方误差)
    bp_ise=sum(e_bp.^2)/N; fix_ise=sum(e_fix.^2)/N;

    % ITAE (时间加权)
    k_vec=1:N;
    bp_itae=sum(k_vec.*abs(e_bp)); fix_itae=sum(k_vec.*abs(e_fix));

    % 调节时间
    band=0.05;
    bp_last=find(abs(y_bp-r_seq)>band.*abs(r_seq+1e-6),1,'last');
    fix_last=find(abs(y_fix-r_seq)>band.*abs(r_seq+1e-6),1,'last');
    bp_stl=bp_last; if isempty(bp_stl),bp_stl=0;end
    fix_stl=fix_last; if isempty(fix_stl),fix_stl=0;end

    % RMS(Δu)
    bp_du=sqrt(mean(diff(u_bp).^2));
    fix_du=sqrt(mean(diff(u_fix).^2));

    % 峰值控制量
    bp_peak=max(abs(u_bp)); fix_peak=max(abs(u_fix));

    % 超调量（输出超出参考的最大正值）
    bp_ov=max(0, max(y_bp - r_seq)); fix_ov=max(0, max(y_fix - r_seq));

    % 峰值动态误差
    bp_pk=max(abs(e_bp)); fix_pk=max(abs(e_fix));

    % 稳态误差（最后 200 步平均绝对误差）
    ss_N = min(200, N);
    bp_ss=mean(abs(e_bp(end-ss_N+1:end)));
    fix_ss=mean(abs(e_fix(end-ss_N+1:end)));
end

%% ==================== BP-PID 仿真（含场景参数） ====================

function [y, error, u] = sim_bp_pid(N, scenario, plant_id, w1, w2)
    IN = 5;  H = 5;  Out = 3;
    switch plant_id
        case 'plant1'
            rate = 0.005;  Kp_max = 2.0;  Ki_max = 2.0;  Kd_max = 1.0;
            jac_type = 'fd';
        case 'plant2'
            rate = 0.002;  Kp_max = 10.0; Ki_max = 1.0;  Kd_max = 25.0;
            jac_type = 'model';  jac_val = 2.5;
    end
    rate2 = 0.01;
    scale_vec = [Kp_max, Ki_max, Kd_max];

    w1_1 = w1;  w1_2 = w1;
    w2_1 = w2;  w2_2 = w2;

    y_1 = 0;  y_2 = 0;  u_1 = 0;  error_1 = 0;  error_2 = 0;
    y = zeros(1, N);  error = zeros(1, N);  u = zeros(1, N);
    y_true = 0;

    rng(42);

    for k = 1:N
        r_k = get_target(k, scenario);
        y_fb = get_feedback(y_true, k, scenario);

        error(k) = r_k - y_fb;

        % 前向传播
        I1 = [r_k, y_fb, error(k), error_1, 1];
        I2 = I1 * w1';
        O2 = tanh(I2);
        I3 = w2 * O2';
        I3_t = I3';
        O3 = zeros(1, Out);
        for l = 1:Out
            if I3_t(l) > 0
                O3(l) = I3_t(l);
            else
                O3(l) = 0.2 * I3_t(l);
            end
        end

        kp = Kp_max * O3(1);  ki = Ki_max * O3(2);  kd = Kd_max * O3(3);
        Kpid = [kp, ki, kd];

        e_pid = [error(k) - error_1;
                 error(k);
                 error(k) - 2*error_1 + error_2];
        delta_u = Kpid * e_pid;
        delta_u = max(-0.5, min(0.5, delta_u));
        u_k = u_1 + delta_u;
        u(k) = u_k;

        % 被控对象
        a_override = get_ak(k, scenario);
        y_true = plant_dynamics(plant_id, y_1, y_2, u_k, u_1, k, a_override);
        y(k) = y_true;
        error(k) = r_k - y_true;

        % 反向传播（误差死区）
        dead_zone = 0.01;
        if abs(error(k)) >= dead_zone
        dO3 = zeros(1, Out);
        for j = 1:Out
            if O3(j) > 0
                dO3(j) = 1;
            else
                dO3(j) = 0.2;
            end
        end
        if strcmp(jac_type, 'fd')
            dydu = (y_true - y_1) / (u_k - u_1 + 0.0001);
            du_sys = max(-1, min(1, dydu));
        else
            du_sys = get_model_jacobian(plant_id, u_k);
        end

        delta3 = zeros(1, Out);
        for l = 1:Out
            delta3(l) = error(k) * du_sys * scale_vec(l) * e_pid(l) * dO3(l);
        end

        d_w2 = zeros(Out, H);
        for l = 1:Out
            for i = 1:H
                d_w2(l, i) = rate * delta3(l) * O2(i);
            end
        end
        w2 = w2_1 + d_w2 + rate2 * 2 * (w2_1 - w2_2);

        dO2 = 1 - tanh(I2).^2;
        a_back = delta3 * w2;
        delta2 = dO2 .* a_back;
        d_w1 = rate * delta2' * I1;
        w1 = w1_1 + d_w1 + rate2 * (w1_1 - w1_2);
        end  % 误差死区

        % 状态缓存
        u_1 = u_k;  y_2 = y_1;  y_1 = y_true;
        w2_2 = w2_1;  w2_1 = w2;
        w1_2 = w1_1;  w1_1 = w1;
        error_2 = error_1;
        error_1 = error(k);
    end
end

%% ==================== 固定 PID 仿真 ====================

function [y, error, u] = sim_fix_pid(N, scenario, plant_id, Kp, Ki, Kd)
    % 增量式 PID（与 BP-PID 结构一致）
    y_1 = 0;  y_2 = 0;  u_1 = 0;
    e_1 = 0;  e_2 = 0;
    y = zeros(1, N);  error = zeros(1, N);  u = zeros(1, N);
    y_true = 0;

    rng(42);

    for k = 1:N
        r_k = get_target(k, scenario);
        y_fb = get_feedback(y_true, k, scenario);

        e_cur = r_k - y_fb;

        delta_u = Kp*(e_cur - e_1) + Ki*e_cur + Kd*(e_cur - 2*e_1 + e_2);
        delta_u = max(-0.5, min(0.5, delta_u));
        u_k = u_1 + delta_u;
        u(k) = u_k;

        a_override = get_ak(k, scenario);
        y_true = plant_dynamics(plant_id, y_1, y_2, u_k, u_1, k, a_override);
        y(k) = y_true;
        error(k) = r_k - y_true;   % 指标用真实误差

        e_2 = e_1;  e_1 = e_cur;   % PID 状态仍用控制器看到的误差
        y_2 = y_1;  y_1 = y_true;
        u_1 = u_k;
    end
end

%% ==================== 参考序列 ====================

function r_seq = get_r_array(N, scenario)
    r_seq = zeros(1, N);
    for k = 1:N
        r_seq(k) = get_target(k, scenario);
    end
end

%% ==================== 场景参数函数 ====================

function r = get_target(k, scenario)
    persistent rand_seq rand_vals rand_amp
    switch scenario
        case 'varying_r'
            if k <= 500,       r = 1;
            elseif k <= 1000,  r = 2;
            elseif k <= 1500,  r = 0.5;
            else,              r = 1.5;
            end
        case 'square'
            if mod(floor((k-1)/100), 2) == 0, r = 1; else, r = 2; end
        case 'sine_low'
            r = 1 + 0.5 * sin(2*pi*0.005*k);     % 低频正弦, 周期200步
        case 'sine_high'
            r = 1 + 0.5 * sin(2*pi*0.02*k);       % 高频正弦, 周期50步
        case 'ramp'
            r = min(1, k / 500);                   % 0→1 斜坡
        case 'composite'
            r = 1 + 0.6*sin(2*pi*0.005*k) + 0.3*sin(2*pi*0.03*k);  % 多频率复合（DC=1）
        case 'random_step'
            if isempty(rand_seq) || mod(k, 200) == 1
                rng(k); rand_amp = 0.5 + 1.5*rand();
            end
            r = rand_amp;
        otherwise
            r = 1;
    end
end

function a = get_ak(k, scenario)
    a0 = 1.2 * (1 - 0.8 * exp(-0.1 * k));
    switch scenario
        case 'perturb'
            if k > 500 && k <= 1000
                a = a0 * 1.3;
            elseif k > 1500
                a = a0 * 0.7;
            else
                a = a0;
            end
        case 'drift'
            if k > 500, a = a0 * 0.5; else, a = a0; end
        otherwise
            a = a0;
    end
end

function y_fb = get_feedback(y_true, k, scenario)
    switch scenario
        case 'disturb'
            if k == 500, y_fb = y_true + 0.5; else, y_fb = y_true; end
        case 'noise'
            y_fb = y_true + (rand - 0.5) * 2 * 0.02;
        case 'combo'
            y_fb = y_true + (rand - 0.5) * 2 * 0.02;
            if k == 300, y_fb = y_fb + 0.5; end
            if k == 700, y_fb = y_fb - 0.3; end
        otherwise
            y_fb = y_true;
    end
end

%% ==================== 模型解析 Jacobian ====================
function du_sys = get_model_jacobian(plant_id, u_k)
    switch plant_id
        case 'plant2'
            du_sys = 1.5;                           % 二阶对象稳态增益
        otherwise
            du_sys = 1.0;
    end
end
