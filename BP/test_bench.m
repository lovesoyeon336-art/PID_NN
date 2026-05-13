clear; close all;

%% ==================== 加载整定参数和预训练权重 ====================
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));

tuned = load(fullfile(script_dir, 'pid_tuned_params.mat'));
Kp_fix = tuned.Kp_opt;
Ki_fix = tuned.Ki_opt;
Kd_fix = tuned.Kd_opt;

pretrain = load(fullfile(script_dir, 'bp_pretrained_weights.mat'));
w1_init = pretrain.w1;
w2_init = pretrain.w2;

fprintf('固定 PID: Kp=%.4f  Ki=%.4f  Kd=%.4f\n', Kp_fix, Ki_fix, Kd_fix);

%% ==================== 5 场景测试 ====================
scenario_names = {'1.基本阶跃', '2.时变参考', '3.参数摄动', '4.输出扰动', '5.量测噪声', ...
                  '6.对象永久变异', '7.周期方波', '8.复合扰动+噪声'};
nS = 8;
BP_MAE  = zeros(1,nS);  Fix_MAE = zeros(1,nS);
BP_MaxE = zeros(1,nS);  Fix_MaxE = zeros(1,nS);
BP_Ovr  = zeros(1,nS);  Fix_Ovr  = zeros(1,nS);
BP_Stl  = zeros(1,nS);  Fix_Stl  = zeros(1,nS);
BP_RMS  = zeros(1,nS);  Fix_RMS  = zeros(1,nS);
BP_dU   = zeros(1,nS);  Fix_dU   = zeros(1,nS);

Y_bp  = cell(1,nS);   Y_fix = cell(1,nS);   R_seq = cell(1,nS);

N = 2000;
idx = 0;

%% ---- 场景 1: 基本阶跃 r=1 ----
idx = idx + 1;
sc = 'step';  r_seq = get_r_array(N, sc);
[y_bp, e_bp, u_bp] = sim_bp_pid(N, sc, w1_init, w2_init);
[y_fix, e_fix, u_fix] = sim_fix_pid(N, sc, Kp_fix, Ki_fix, Kd_fix);
[BP_MAE(idx), Fix_MAE(idx), BP_MaxE(idx), Fix_MaxE(idx), ...
 BP_Ovr(idx), Fix_Ovr(idx), BP_Stl(idx), Fix_Stl(idx), ...
 BP_RMS(idx), Fix_RMS(idx), BP_dU(idx), Fix_dU(idx)] = ...
    metrics(y_bp, e_bp, r_seq, y_fix, e_fix, u_bp, u_fix);
Y_bp{idx} = y_bp;  Y_fix{idx} = y_fix;  R_seq{idx} = r_seq;

%% ---- 场景 2: 时变参考 ----
idx = idx + 1;
sc = 'varying_r';  r_seq = get_r_array(N, sc);
[y_bp, e_bp, u_bp] = sim_bp_pid(N, sc, w1_init, w2_init);
[y_fix, e_fix, u_fix] = sim_fix_pid(N, sc, Kp_fix, Ki_fix, Kd_fix);
[BP_MAE(idx), Fix_MAE(idx), BP_MaxE(idx), Fix_MaxE(idx), ...
 BP_Ovr(idx), Fix_Ovr(idx), BP_Stl(idx), Fix_Stl(idx), ...
 BP_RMS(idx), Fix_RMS(idx), BP_dU(idx), Fix_dU(idx)] = ...
    metrics(y_bp, e_bp, r_seq, y_fix, e_fix, u_bp, u_fix);
Y_bp{idx} = y_bp;  Y_fix{idx} = y_fix;  R_seq{idx} = r_seq;

%% ---- 场景 3: 对象参数摄动 ----
idx = idx + 1;
sc = 'perturb';  r_seq = get_r_array(N, sc);
[y_bp, e_bp, u_bp] = sim_bp_pid(N, sc, w1_init, w2_init);
[y_fix, e_fix, u_fix] = sim_fix_pid(N, sc, Kp_fix, Ki_fix, Kd_fix);
[BP_MAE(idx), Fix_MAE(idx), BP_MaxE(idx), Fix_MaxE(idx), ...
 BP_Ovr(idx), Fix_Ovr(idx), BP_Stl(idx), Fix_Stl(idx), ...
 BP_RMS(idx), Fix_RMS(idx), BP_dU(idx), Fix_dU(idx)] = ...
    metrics(y_bp, e_bp, r_seq, y_fix, e_fix, u_bp, u_fix);
Y_bp{idx} = y_bp;  Y_fix{idx} = y_fix;  R_seq{idx} = r_seq;

%% ---- 场景 4: 输出扰动 ----
idx = idx + 1;
sc = 'disturb';  r_seq = get_r_array(N, sc);
[y_bp, e_bp, u_bp] = sim_bp_pid(N, sc, w1_init, w2_init);
[y_fix, e_fix, u_fix] = sim_fix_pid(N, sc, Kp_fix, Ki_fix, Kd_fix);
[BP_MAE(idx), Fix_MAE(idx), BP_MaxE(idx), Fix_MaxE(idx), ...
 BP_Ovr(idx), Fix_Ovr(idx), BP_Stl(idx), Fix_Stl(idx), ...
 BP_RMS(idx), Fix_RMS(idx), BP_dU(idx), Fix_dU(idx)] = ...
    metrics(y_bp, e_bp, r_seq, y_fix, e_fix, u_bp, u_fix);
Y_bp{idx} = y_bp;  Y_fix{idx} = y_fix;  R_seq{idx} = r_seq;

%% ---- 场景 5: 量测噪声 ----
idx = idx + 1;
sc = 'noise';  r_seq = get_r_array(N, sc);
[y_bp, e_bp, u_bp] = sim_bp_pid(N, sc, w1_init, w2_init);
[y_fix, e_fix, u_fix] = sim_fix_pid(N, sc, Kp_fix, Ki_fix, Kd_fix);
[BP_MAE(idx), Fix_MAE(idx), BP_MaxE(idx), Fix_MaxE(idx), ...
 BP_Ovr(idx), Fix_Ovr(idx), BP_Stl(idx), Fix_Stl(idx), ...
 BP_RMS(idx), Fix_RMS(idx), BP_dU(idx), Fix_dU(idx)] = ...
    metrics(y_bp, e_bp, r_seq, y_fix, e_fix, u_bp, u_fix);
Y_bp{idx} = y_bp;  Y_fix{idx} = y_fix;  R_seq{idx} = r_seq;

%% ---- 场景 6: 对象永久变异 ----
idx = idx + 1;
sc = 'drift';  r_seq = get_r_array(N, sc);
[y_bp, e_bp, u_bp] = sim_bp_pid(N, sc, w1_init, w2_init);
[y_fix, e_fix, u_fix] = sim_fix_pid(N, sc, Kp_fix, Ki_fix, Kd_fix);
[BP_MAE(idx), Fix_MAE(idx), BP_MaxE(idx), Fix_MaxE(idx), ...
 BP_Ovr(idx), Fix_Ovr(idx), BP_Stl(idx), Fix_Stl(idx), ...
 BP_RMS(idx), Fix_RMS(idx), BP_dU(idx), Fix_dU(idx)] = ...
    metrics(y_bp, e_bp, r_seq, y_fix, e_fix, u_bp, u_fix);
Y_bp{idx} = y_bp;  Y_fix{idx} = y_fix;  R_seq{idx} = r_seq;

%% ---- 场景 7: 周期方波 ----
idx = idx + 1;
sc = 'square';  r_seq = get_r_array(N, sc);
[y_bp, e_bp, u_bp] = sim_bp_pid(N, sc, w1_init, w2_init);
[y_fix, e_fix, u_fix] = sim_fix_pid(N, sc, Kp_fix, Ki_fix, Kd_fix);
[BP_MAE(idx), Fix_MAE(idx), BP_MaxE(idx), Fix_MaxE(idx), ...
 BP_Ovr(idx), Fix_Ovr(idx), BP_Stl(idx), Fix_Stl(idx), ...
 BP_RMS(idx), Fix_RMS(idx), BP_dU(idx), Fix_dU(idx)] = ...
    metrics(y_bp, e_bp, r_seq, y_fix, e_fix, u_bp, u_fix);
Y_bp{idx} = y_bp;  Y_fix{idx} = y_fix;  R_seq{idx} = r_seq;

%% ---- 场景 8: 复合扰动+噪声 ----
idx = idx + 1;
sc = 'combo';  r_seq = get_r_array(N, sc);
[y_bp, e_bp, u_bp] = sim_bp_pid(N, sc, w1_init, w2_init);
[y_fix, e_fix, u_fix] = sim_fix_pid(N, sc, Kp_fix, Ki_fix, Kd_fix);
[BP_MAE(idx), Fix_MAE(idx), BP_MaxE(idx), Fix_MaxE(idx), ...
 BP_Ovr(idx), Fix_Ovr(idx), BP_Stl(idx), Fix_Stl(idx), ...
 BP_RMS(idx), Fix_RMS(idx), BP_dU(idx), Fix_dU(idx)] = ...
    metrics(y_bp, e_bp, r_seq, y_fix, e_fix, u_bp, u_fix);
Y_bp{idx} = y_bp;  Y_fix{idx} = y_fix;  R_seq{idx} = r_seq;

%% ==================== 输出对比表 ====================
fprintf('\n========== 测试结果汇总 ==========\n');
fprintf('%-12s | %-8s %-8s | %-8s %-8s | %-8s %-8s | %-8s %-8s\n', ...
    '场景', 'BP_MAE', 'Fix_MAE', 'BP_RMSu', 'Fix_RMSu', 'BP_dU', 'Fix_dU', 'BP_调节', 'Fix_调节');
fprintf('%s\n', repmat('-', 1, 100));
for i = 1:nS
    fprintf('%-12s | %8.4f %8.4f | %8.4f %8.4f | %8.4f %8.4f | %8d %8d\n', ...
        scenario_names{i}, ...
        BP_MAE(i), Fix_MAE(i), ...
        BP_RMS(i), Fix_RMS(i), ...
        BP_dU(i), Fix_dU(i), ...
        BP_Stl(i), Fix_Stl(i));
end

fprintf('\n综合 MAE: BP-PID=%.4f  固定PID=%.4f  倍率=%.2f\n', sum(BP_MAE), sum(Fix_MAE), sum(BP_MAE)/max(sum(Fix_MAE),1e-9));
fprintf('综合 RMS(u): BP-PID=%.4f  固定PID=%.4f\n', sum(BP_RMS), sum(Fix_RMS));
fprintf('综合 RMS(Δu): BP-PID=%.4f  固定PID=%.4f\n', sum(BP_dU), sum(Fix_dU));

%% ==================== 绘图 ====================
figure('Name', '5场景全指标对比', 'NumberTitle', 'off');

titles = {'MAE', 'RMS(u)', 'RMS(Δu)', '调节时间 (步)'};
bp_data = {BP_MAE, BP_RMS, BP_dU, BP_Stl};
fix_data = {Fix_MAE, Fix_RMS, Fix_dU, Fix_Stl};

for p = 1:4
    subplot(2, 2, p);
    bar_data = [bp_data{p}(:), fix_data{p}(:)];
    bar(bar_data);
    set(gca, 'XTickLabel', scenario_names);
    if p == 1, legend('BP-PID', '固定PID', 'Location', 'best'); end
    ylabel(titles{p});
    title(titles{p});
    grid on;
end

%% ==================== 曲线图: 窗口2 (场景 1-4) ====================
figure('Name', '时域响应: 场景1-4', 'NumberTitle', 'off');
fx = [1, min(300, N)];
for s = 1:4
    subplot(2, 2, s);
    r_plot = R_seq{s};  yb = Y_bp{s};  yf = Y_fix{s};
    plot(1:N, r_plot, 'r', 1:N, yf, 'k:', 1:N, yb, 'b--', 'LineWidth', 1);
    xlim(fx);
    xlabel('时间步'); ylabel('y');
    if s==1, legend('目标 r', '固定PID', 'BP-PID', 'Location', 'best'); end
    title(scenario_names{s}); grid on;
end

%% ==================== 曲线图: 窗口3 (场景 5-8) ====================
figure('Name', '时域响应: 场景5-8', 'NumberTitle', 'off');
for s = 5:8
    subplot(2, 2, s-4);
    r_plot = R_seq{s};  yb = Y_bp{s};  yf = Y_fix{s};
    plot(1:N, r_plot, 'r', 1:N, yf, 'k:', 1:N, yb, 'b--', 'LineWidth', 1);
    xlim(fx);
    xlabel('时间步'); ylabel('y');
    if s==5, legend('目标 r', '固定PID', 'BP-PID', 'Location', 'best'); end
    title(scenario_names{s}); grid on;
end

%% ==================== 保存结果 ====================
save(fullfile(script_dir, 'test_results.mat'), ...
    'scenario_names', 'N', 'Kp_fix', 'Ki_fix', 'Kd_fix', ...
    'BP_MAE', 'Fix_MAE', 'BP_MaxE', 'Fix_MaxE', ...
    'BP_Ovr', 'Fix_Ovr', 'BP_Stl', 'Fix_Stl', ...
    'BP_RMS', 'Fix_RMS', 'BP_dU', 'Fix_dU', ...
    'Y_bp', 'Y_fix', 'R_seq');
fprintf('结果已保存至 test_results.mat\n');

%% ==================== 辅助函数 ====================

function [bp_mae, fix_mae, bp_maxe, fix_maxe, bp_ovr, fix_ovr, bp_stl, fix_stl, ...
          bp_rmse, fix_rmse, bp_du, fix_du] = ...
        metrics(y_bp, e_bp, r_seq, y_fix, e_fix, u_bp, u_fix)
    bp_mae  = mean(abs(e_bp));
    fix_mae = mean(abs(e_fix));
    bp_maxe = max(abs(e_bp));
    fix_maxe = max(abs(e_fix));

    % 超调
    bp_dev = y_bp - r_seq;
    fix_dev = y_fix - r_seq;
    bp_ovr = max(bp_dev);
    fix_ovr = max(fix_dev);

    % 调节时间
    band = 0.05;
    bp_out = abs(y_bp - r_seq) > band .* abs(r_seq + 1e-6);
    fix_out = abs(y_fix - r_seq) > band .* abs(r_seq + 1e-6);
    bp_last = find(bp_out, 1, 'last');
    fix_last = find(fix_out, 1, 'last');
    bp_stl = bp_last;  if isempty(bp_stl), bp_stl = 0; end
    fix_stl = fix_last; if isempty(fix_stl), fix_stl = 0; end

    % 控制能耗
    bp_rmse = sqrt(mean(u_bp.^2));
    fix_rmse = sqrt(mean(u_fix.^2));

    % 控制变化率
    bp_du = sqrt(mean(diff(u_bp).^2));
    fix_du = sqrt(mean(diff(u_fix).^2));
end

%% ==================== BP-PID 仿真（含场景参数） ====================

function [y, error, u] = sim_bp_pid(N, scenario, w1, w2)
    IN = 4;  H = 5;  Out = 3;
    rate  = 0.001;  rate2 = 0.01;
    Kp_max = 2.0;  Ki_max = 0.5;  Kd_max = 1.0;
    scale_vec = [Kp_max, Ki_max, Kd_max];

    w1_1 = w1;  w1_2 = w1;
    w2_1 = w2;  w2_2 = w2;

    y_1 = 0;  u_1 = 0;  error_1 = 0;  error_2 = 0;
    y = zeros(1, N);  error = zeros(1, N);  u = zeros(1, N);
    y_true = 0;

    rng(42);

    for k = 1:N
        r_k = get_target(k, scenario);
        y_fb = get_feedback(y_true, k, scenario);

        error(k) = r_k - y_fb;

        % 前向传播
        I1 = [r_k, y_fb, error(k), 1];
        I2 = I1 * w1';
        O2 = tanh(I2);
        I3 = w2 * O2';
        O3 = sigmoid(I3');

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
        a_k = get_ak(k, scenario);
        y_true = a_k / (1 + y_1^2) * y_1 + u_k;
        y(k) = y_true;
        error(k) = r_k - y_true;

        % 反向传播
        dO3 = sigmoidGradient(O3);
        dydu = (y_true - y_1) / (u_k - u_1 + 0.0001);
        du_sys = max(-1, min(1, dydu));

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

        % 状态缓存
        u_1 = u_k;  y_1 = y_true;
        w2_2 = w2_1;  w2_1 = w2;
        w1_2 = w1_1;  w1_1 = w1;
        error_2 = error_1;
        error_1 = error(k);
    end
end

%% ==================== 固定 PID 仿真 ====================

function [y, error, u] = sim_fix_pid(N, scenario, Kp, Ki, Kd)
    y_1 = 0;  u_1 = 0;  ei = 0;  last_e = 0;
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
        delta_u = Kp * e_cur + Ki * ei_clamped + Kd * ed - u_1;
        delta_u = max(-0.5, min(0.5, delta_u));
        u_k = u_1 + delta_u;
        u(k) = u_k;

        a_k = get_ak(k, scenario);
        y_true = a_k / (1 + y_1^2) * y_1 + u_k;
        y(k) = y_true;

        last_e = e_cur;
        y_1 = y_true;
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
    switch scenario
        case 'varying_r'
            if k <= 500,       r = 1;
            elseif k <= 1000,  r = 2;
            elseif k <= 1500,  r = 0.5;
            else,              r = 1.5;
            end
        case 'square'
            if mod(floor((k-1)/100), 2) == 0, r = 1; else, r = 2; end
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
