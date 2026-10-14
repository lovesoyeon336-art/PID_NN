clear; close all;

%% ==================== Plant2 监督预训练：模仿 FixPID ====================
% 核心思路：收集 FixPID 在各场景的运行轨迹，BP 直接学习输出 FixPID 的 Kp/Ki/Kd。
% BP 从 FixPID 行为基线出发，在线学习只需微调而非从头探索。

IN = 5;   H = 5;  Out = 3;
N = 2000;
epochs = 200;   rate_sup = 0.005;   rate2 = 0.01;
batch_size = 500;  % mini-batch SGD

% FixPID 统一参数 (来自 pid_tuned_params_plant2.mat)
Kp_fix = 6.2979;  Ki_fix = 0.7554;  Kd_fix = 6.3626;

% Plant2 残差架构
Kp_base = 1.0115;  Kp_delta = 11.0;
Ki_base = 0.1089;  Ki_delta = 3.0;
Kd_base = 0;       Kd_delta = 5.0;

% 归一化目标 O3（使 Kp = base + delta * O3）
O3_target = [(Kp_fix - Kp_base) / Kp_delta;
             (Ki_fix - Ki_base) / Ki_delta;
             (Kd_fix - Kd_base) / Kd_delta];
fprintf('FixPID 目标 O3: [%.4f, %.4f, %.4f]\n', O3_target);

%% ==================== FixPID 仿真参数 ====================
ff_gain = 0.667;  beta_sp = 0.85;  du_max = 0.5;  u_sat = 5.0;

scenarios = {'step','sine_low','sine_high','ramp','disturb','noise','square'};

%% ==================== 收集训练数据 ====================
fprintf('\n===== 收集 FixPID 运行数据 =====\n');
X_train = [];  Y_train = [];

for s = 1:length(scenarios)
    sc = scenarios{s};
    [I1_data, ~] = collect_fixpid_trajectory(N, sc, Kp_fix, Ki_fix, Kd_fix);

    % 目标 O3 对所有步相同（FixPID 参数恒定）
    T_data = repmat(O3_target, 1, N);

    X_train = [X_train; I1_data];  % N×5
    Y_train = [Y_train; T_data'];  % N×3
    fprintf('  %s: %d 样本\n', sc, N);
end

N_sample = size(X_train, 1);
fprintf('总样本数: %d\n', N_sample);

%% ==================== Xavier 半规模初始化 ====================
rng(1);
w1 = 0.5 * sqrt(2/(IN+H)) * randn(H, IN);
w2 = 0.5 * sqrt(2/(H+Out)) * randn(Out, H);

w1_1 = w1;  w1_2 = w1;
w2_1 = w2;  w2_2 = w2;

%% ==================== 监督训练 ====================
fprintf('\n===== 监督训练 =====\n');
best_loss = inf;  plateau_cnt = 0;
for ep = 1:epochs
    idx = randperm(N_sample);
    total_loss = 0;

    for b_start = 1:batch_size:N_sample
        b_end = min(b_start + batch_size - 1, N_sample);
        batch_idx = idx(b_start:b_end);
        n_batch = length(batch_idx);

        % 累积梯度
        dW1_sum = zeros(H, IN);
        dW2_sum = zeros(Out, H);
        batch_loss = 0;

        for bi = 1:n_batch
            i = batch_idx(bi);
            I1 = X_train(i, :);   % 1×5
            T  = Y_train(i, :)';  % 3×1

            % 前向传播
            I2 = I1 * w1';        % 1×H
            O2 = tanh(I2);        % 1×H
            I3_t = w2 * O2';      % Out×1
            O3 = zeros(Out, 1);
            dO3 = zeros(Out, 1);
            for l = 1:Out
                if I3_t(l) > 0
                    O3(l) = I3_t(l);
                    dO3(l) = 1;
                else
                    O3(l) = 0.2 * I3_t(l);
                    dO3(l) = 0.2;
                end
            end

            % MSE 损失梯度
            err = O3 - T;          % Out×1
            batch_loss = batch_loss + sum(err.^2);

            delta3 = err .* dO3;   % Out×1

            % w2 梯度
            dW2 = rate_sup * delta3 * O2;  % Out×H

            % w1 梯度
            dO2 = 1 - tanh(I2).^2;         % 1×H
            a_back = delta3' * w2_1;       % 1×H
            delta2 = dO2 .* a_back;        % 1×H
            dW1 = rate_sup * delta2' * I1; % H×IN

            dW1_sum = dW1_sum + dW1;
            dW2_sum = dW2_sum + dW2;
        end

        % 平均梯度 + 动量更新
        w2 = w2_1 - dW2_sum / n_batch + rate2 * (w2_1 - w2_2);
        w1 = w1_1 - dW1_sum / n_batch + rate2 * (w1_1 - w1_2);

        w2_2 = w2_1;  w2_1 = w2;
        w1_2 = w1_1;  w1_1 = w1;

        total_loss = total_loss + batch_loss / n_batch;
    end

    avg_loss = total_loss / ceil(N_sample / batch_size);
    if mod(ep, 20) == 0 || ep == 1
        fprintf('  Epoch %d/%d  MSE=%.6f  rate=%.4f\n', ep, epochs, avg_loss, rate_sup);
    end

    % 平台检测 + 学习率衰减
    if avg_loss < best_loss * 0.999
        best_loss = avg_loss;
        plateau_cnt = 0;
    else
        plateau_cnt = plateau_cnt + 1;
    end
    if plateau_cnt >= 30
        rate_sup = rate_sup * 0.5;
        plateau_cnt = 0;
        fprintf('  Epoch %d: 学习率衰减至 %.4f\n', ep, rate_sup);
        if rate_sup < 1e-5, break; end
    end
end

%% ==================== 验证：输出是否接近 FixPID ====================
fprintf('\n===== 验证 =====\n');
for s = 1:length(scenarios)
    sc = scenarios{s};
    [I1_v, ~] = collect_fixpid_trajectory(min(200, N), sc, Kp_fix, Ki_fix, Kd_fix);

    mae_kp = 0;  mae_ki = 0;  mae_kd = 0;
    n_v = size(I1_v, 1);
    for i = 1:n_v
        I1 = I1_v(i, :);
        I2 = I1 * w1';  O2 = tanh(I2);
        I3 = w2 * O2';
        O3 = zeros(Out, 1);
        for l = 1:Out
            if I3(l) > 0, O3(l) = I3(l); else, O3(l) = 0.2 * I3(l); end
        end
        kp_bp = Kp_base + Kp_delta * O3(1);
        ki_bp = Ki_base + Ki_delta * O3(2);
        kd_bp = Kd_base + Kd_delta * O3(3);
        mae_kp = mae_kp + abs(kp_bp - Kp_fix);
        mae_ki = mae_ki + abs(ki_bp - Ki_fix);
        mae_kd = mae_kd + abs(kd_bp - Kd_fix);
    end
    fprintf('  %-10s: ΔKp=%.3f  ΔKi=%.4f  ΔKd=%.3f\n', ...
        sc, mae_kp/n_v, mae_ki/n_v, mae_kd/n_v);
end

%% ==================== 保存 ====================
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));
save(fullfile(script_dir, 'bp_pretrained_weights_plant2_supervised.mat'), 'w1', 'w2');
fprintf('\n权重已保存至 bp_pretrained_weights_plant2_supervised.mat\n');

%% ==================== 数据采集函数 ====================

function [X_out, Y_out] = collect_fixpid_trajectory(N_sim, scenario, Kp, Ki, Kd)
    ff_gain = 0.667;  beta_sp = 0.85;  du_max = 0.5;  u_sat = 5.0;

    is_sine = any(strcmp(scenario, {'sine_low','sine_high'}));
    if is_sine
        r_start = 1 + 0.5 * sin(2*pi*0.005);
        y_1 = r_start;  y_2 = r_start;  y_true = r_start;
    else
        y_1 = 0;  y_2 = 0;  y_true = 0;
    end
    u_1 = 0;  r_1 = 0;  e_sp_1 = 0;  e_sp_2 = 0;

    X_out = zeros(N_sim, 5);  % [r, y, e, dr, bias]
    Y_out = zeros(N_sim, 1);  % unused (target is constant)

    rng(42);
    for k = 1:N_sim
        r_k = get_r_sup(k, scenario);
        y_fb = get_yfb(y_true, k, scenario);
        e_cur = r_k - y_fb;
        e_sp_k = beta_sp * r_k - y_fb;
        Kd_eff = Kd; if strcmp(scenario, 'square'), Kd_eff = 0; end
        delta_u = Kp*(e_sp_k - e_sp_1) + Ki*e_cur + Kd_eff*(e_sp_k - 2*e_sp_1 + e_sp_2);
        dr_v = r_k - r_1;
        if abs(dr_v) <= 0.1, delta_u = delta_u + ff_gain * dr_v; end
        delta_u = max(-du_max, min(du_max, delta_u));
        u_k = max(-u_sat, min(u_sat, u_1 + delta_u));

        y_true = plant_dynamics('plant2', y_1, y_2, u_1, u_1, k);

        % 记录 BP 网络输入状态
        X_out(k, :) = [r_k, y_fb, e_cur, r_k - r_1, 1];

        e_sp_2 = e_sp_1;  e_sp_1 = e_sp_k;
        y_2 = y_1;  y_1 = y_true;  u_1 = u_k;  r_1 = r_k;
    end
end

function r_k = get_r_sup(k, scenario)
    switch scenario
        case 'step',       r_k = 1;
        case 'sine_low',   r_k = 1 + 0.5 * sin(2*pi*0.005*k);
        case 'sine_high',  r_k = 1 + 0.5 * sin(2*pi*0.02*k);
        case 'ramp',       r_k = min(1, k / 500);
        case 'square'
            if mod(floor((k-1)/100), 2) == 0, r_k = 1; else, r_k = 2; end
        otherwise,         r_k = 1;
    end
end

function y_fb = get_yfb(y_true, k, scenario)
    switch scenario
        case 'disturb'
            if k == 500, y_fb = y_true + 0.5; else, y_fb = y_true; end
        case 'noise'
            y_fb = y_true + (rand - 0.5) * 2 * 0.02;
        otherwise
            y_fb = y_true;
    end
end
