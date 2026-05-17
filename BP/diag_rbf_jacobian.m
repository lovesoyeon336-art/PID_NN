clear; close all;

%% ==================== RBF Jacobian 诊断脚本 ====================
% 针对 7 个薄弱场景，记录 RBF 辨识器在闭环中的表现：
%   1. 预测 MAE：|y_hat - y_true|
%   2. Jacobian MAE：|dy_du_rbf - dy_du_fd|
%   3. Jacobian 符号正确率
%   4. 中心最小激活值（检测工作点是否远离所有中心）

[script_dir, ~, ~] = fileparts(mfilename('fullpath'));

%% ==================== 诊断场景 ====================
diag_scenarios = {
    'plant1', 'ramp',        'Plant1 斜坡';
    'plant1', 'sine_high',   'Plant1 正弦高频';
    'plant1', 'sine_low',    'Plant1 正弦低频';
    'plant2', 'step',       'Plant2 阶跃';
    'plant2', 'sine_high',  'Plant2 正弦高频';
    'plant2', 'sine_low',   'Plant2 正弦低频';
    'plant3', 'sine_high',  'Plant3 正弦高频';
};
nD = size(diag_scenarios, 1);
N = 2000;

%% ==================== 加载预训练权重 ====================
bp1 = load(fullfile(script_dir, 'bp_pretrained_weights.mat'));
bp2 = load(fullfile(script_dir, 'bp_pretrained_weights_plant2.mat'));
bp3 = load(fullfile(script_dir, 'bp_pretrained_weights_plant3.mat'));

fprintf('========== RBF Jacobian 诊断 ==========\n');
fprintf('%-20s | %8s %10s %10s %10s\n', ...
    '场景', '预测MAE', 'Jac MAE', '符号正确%', 'min(H)');

for d = 1:nD
    pid  = diag_scenarios{d, 1};
    sc   = diag_scenarios{d, 2};
    name = diag_scenarios{d, 3};

    switch pid
        case 'plant1', w1 = bp1.w1; w2 = bp1.w2;
        case 'plant2', w1 = bp2.w1; w2 = bp2.w2;
        case 'plant3', w1 = bp3.w1; w2 = bp3.w2;
    end

    [pred_mae, jac_mae, sign_ok, min_h] = diag_one(N, sc, pid, w1, w2);
    fprintf('%-20s | %8.5f %10.5f %9.1f%% %10.5f\n', ...
        name, pred_mae, jac_mae, sign_ok*100, min_h);
end

fprintf('\n诊断完成。\n');

%% ==================== 单场景诊断 ====================

function [pred_mae, jac_mae, sign_ok, min_h] = diag_one(N, scenario, plant_id, w1, w2)
    IN = 5;  H = size(w1,1);  Out = size(w2,1);

    % ---- 超参数 (与 sim_bp_rbf 一致) ----
    switch plant_id
        case 'plant1'
            rate = 0.008; Kp_max = 1.0;  Ki_max = 0.3;  Kd_max = 0.2;
            du_max = 1.0;  u_sat = 2.0;  ff_gain = 0.0;  beta_sp = 1.00;  jac_cap = 1.0;
        case 'plant2'
            rate = 0.002;  du_max = 0.5;  u_sat = 5.0;
            ff_gain = 0.667;  beta_sp = 0.85;  rate_sine = 0.003;  jac_cap = 0.3;
            Kp_base = 1.0115;  Kp_delta = 11.0;
            Ki_base = 0.1089;  Ki_delta = 3.0;
            Kd_base = 0;       Kd_delta = 5.0;
            Kp_max = Kp_delta;  Ki_max = Ki_delta;  Kd_max = Kd_delta;
        case 'plant3'
            rate = 0.003;  du_max = 1.0;  u_sat = 3.0;
            ff_gain = 2.0;  beta_sp = 0.90;  rate_sine = 0.004;  jac_cap = 0.5;
            Kp_base = 1.0;   Kp_delta = 8.0;
            Ki_base = 0.1;   Ki_delta = 2.0;
            Kd_base = 0;     Kd_delta = 4.0;
            Kp_max = Kp_delta;  Ki_max = Ki_delta;  Kd_max = Kd_delta;
    end
    rate2 = 0.01;
    scale_vec = [Kp_max, Ki_max, Kd_max];

    w1_1 = w1;  w1_2 = w1;  w2_1 = w2;  w2_2 = w2;
    y_1 = 0;  y_2 = 0;  u_1 = 0;  u_2 = 0;  r_1 = 0;  error_1 = 0;  error_2 = 0;
    e_sp_1 = 0;  e_sp_2 = 0;
    st_has = false;  st_ep = zeros(Out,1);  st_O2 = zeros(1,H);
    st_dO3 = zeros(1,Out);  st_dO2 = zeros(1,H);  st_I1 = zeros(1,IN);
    y_true = 0;

    clear rbf_identifier;
    rbf_identifier(0, 0, plant_id);

    % ---- 诊断记录 ----
    pred_err = zeros(1, N);    % |y_hat - y_true|
    jac_err  = zeros(1, N);    % |dy_du_rbf - dy_du_fd|
    sign_hit = zeros(1, N);    % 符号一致
    min_H    = zeros(1, N);    % 最小中心激活
    n_valid  = 0;

    rng(42);

    for k = 1:N
        r_k = diag_get_r(k, scenario);
        y_fb = diag_get_fb(y_true, k, scenario);
        error_k = r_k - y_fb;

        % 前向传播
        I1 = [r_k, y_fb, error_k, r_k - r_1, 1];
        I2 = I1 * w1';  O2 = tanh(I2);
        I3 = w2 * O2';  I3_t = I3';
        O3 = zeros(1, Out);
        for l = 1:Out
            if I3_t(l) > 0, O3(l) = I3_t(l); else, O3(l) = 0.2 * I3_t(l); end
        end

        if strcmp(plant_id, 'plant2') || strcmp(plant_id, 'plant3')
            kp = Kp_base + Kp_delta * O3(1);
            ki = Ki_base + Ki_delta * O3(2);
            kd = Kd_base + Kd_delta * O3(3);
        else
            kp = Kp_max * O3(1);  ki = Ki_max * O3(2);  kd = Kd_max * O3(3);
        end
        % 场景特定 Kp 调整 (与 sim_bp_rbf 对齐)
        if strcmp(plant_id, 'plant1') && strcmp(scenario, 'ramp')
            kp = kp * 0.45;  ki = ki * 0.08;
        end
        if strcmp(plant_id, 'plant1') && strcmp(scenario, 'sine_high')
            kp = kp * 1.2;
        end
        if strcmp(plant_id, 'plant2')
            if strcmp(scenario, 'sine_high'), kp = kp * 2.0;
            elseif strcmp(scenario, 'sine_low'), kp = kp * 1.5; end
        end
        if strcmp(plant_id, 'plant3')
            if strcmp(scenario, 'sine_high'), kp = kp * 1.5;
            elseif strcmp(scenario, 'sine_low'), kp = kp * 1.2; end
        end
        % 预热
        if strcmp(plant_id, 'plant2') || strcmp(plant_id, 'plant3')
            if any(strcmp(scenario, {'step','disturb','noise'})), warmup_steps = 150;
            elseif strcmp(scenario, 'sine_high'), warmup_steps = 10;
            elseif strcmp(scenario, 'sine_low'), warmup_steps = 15;
            else, warmup_steps = 50; end
        else, warmup_steps = 50; end
        if k <= warmup_steps
            wu = k / warmup_steps;
            kp = kp * wu;  ki = ki * wu;  kd = kd * wu;
        end
        Kpid = [kp, ki, kd];
        % 误差<0 积分衰减
        if error_k < 0
            if strcmp(plant_id, 'plant2') || strcmp(plant_id, 'plant3')
                if error_k < -0.05
                    if any(strcmp(scenario, {'sine_low','sine_high'}))
                        Kpid(2) = Kpid(2) * 0.7;
                    else
                        Kpid(2) = Kpid(2) * 0.2;
                    end
                else
                    if any(strcmp(scenario, {'sine_low','sine_high'}))
                        Kpid(2) = Kpid(2) * 0.85;
                    else
                        Kpid(2) = Kpid(2) * 0.6;
                    end
                end
            else, Kpid(2) = Kpid(2) * 0.5; end
        end
        e_sp_k = beta_sp * r_k - y_fb;
        e_pid = [e_sp_k - e_sp_1; error_k; e_sp_k - 2*e_sp_1 + e_sp_2];
        delta_u = Kpid * e_pid;
        dr_v = r_k - r_1;
        if abs(dr_v) <= 0.1, delta_u = delta_u + ff_gain * dr_v; end
        delta_u = max(-du_max, min(du_max, delta_u));
        u_k = max(-u_sat, min(u_sat, u_1 + delta_u));

        a_override = diag_get_ak(k, scenario);
        y_true = plant_dynamics(plant_id, y_1, y_2, u_k, u_1, k, a_override);

        % RBF 辨识 (Jacobian + 预测 + 中心激活)
        [dydu, y_hat, H_diag] = rbf_identifier(u_1, y_true, false);

        % ---- 诊断记录 (仅记录预热后有效步) ----
        if k > warmup_steps && u_1 ~= u_2
            n_valid = n_valid + 1;
            % 有限差分 Jacobian ground truth
            dydu_fd = (y_true - y_1) / (u_1 - u_2 + 0.0001);

            pred_err(k) = abs(y_hat - y_true);
            jac_err(k)  = abs(dydu - dydu_fd);
            if sign(dydu + eps) == sign(dydu_fd + eps)
                sign_hit(k) = 1;
            end
            min_H(k) = min(H_diag);
        end

        % BP 权重更新 (与 sim_bp_rbf 对齐)
        du_sys = max(-jac_cap, min(jac_cap, dydu));
        if (strcmp(plant_id, 'plant1') && strcmp(scenario, 'sine_high')) || ...
           (strcmp(plant_id, 'plant2') && any(strcmp(scenario, {'step','disturb','noise','sine_high'}))) || ...
           (strcmp(plant_id, 'plant3') && any(strcmp(scenario, {'step','disturb','noise','sine_high'})))
            skip_bp = true;
        else, skip_bp = false; end
        dead_zone = 0.002;
        if st_has && ~skip_bp && abs(error_k) >= dead_zone
            delta3 = zeros(1, Out);
            e_grad = max(-0.5, min(0.5, error_k));
            for l = 1:Out
                delta3(l) = e_grad * du_sys * scale_vec(l) * st_ep(l) * st_dO3(l);
            end
            d_w2 = zeros(Out, H);
            rate_eff = rate;
            if strcmp(plant_id, 'plant2') || strcmp(plant_id, 'plant3')
                if any(strcmp(scenario, {'sine_low','sine_high'}))
                    rate_eff = rate_sine * min(3, 1 + abs(error_k));
                end
            end
            for l = 1:Out
                for ii = 1:H, d_w2(l, ii) = rate_eff * delta3(l) * st_O2(ii); end
            end
            w2 = w2_1 + d_w2 + rate2 * 2 * (w2_1 - w2_2);
            a_back = delta3 * w2_1;
            delta2 = st_dO2 .* a_back;
            d_w1 = rate_eff * delta2' * st_I1;
            w1 = w1_1 + d_w1 + rate2 * (w1_1 - w1_2);
            w2_2 = w2_1;  w2_1 = w2;  w1_2 = w1_1;  w1_1 = w1;
        end

        st_has = true;  st_ep = e_pid;  st_O2 = O2;  st_I1 = I1;
        dO3_t = zeros(1, Out);
        for j = 1:Out
            if O3(j) > 0, dO3_t(j) = 1; else, dO3_t(j) = 0.2; end
        end
        st_dO3 = dO3_t;  st_dO2 = 1 - tanh(I2).^2;
        u_2 = u_1;  u_1 = u_k;  y_2 = y_1;  y_1 = y_true;  r_1 = r_k;
        error_2 = error_1;  error_1 = error_k;
        e_sp_2 = e_sp_1;  e_sp_1 = e_sp_k;
    end

    pred_mae = mean(pred_err(warmup_steps+1:end));
    jac_mae  = mean(jac_err(warmup_steps+1:end));
    sign_ok  = sum(sign_hit) / max(n_valid, 1);
    min_h    = min(min_H(warmup_steps+1:end));
end

%% ==================== 辅助函数 ====================

function r = diag_get_r(k, scenario)
    switch scenario
        case 'step',       r = 1;
        case 'sine_low',   r = 1 + 0.5 * sin(2*pi*0.005*k);
        case 'sine_high',  r = 1 + 0.5 * sin(2*pi*0.02*k);
        case 'ramp',       r = min(1, k / 500);
        case 'square'
            if mod(floor((k-1)/100), 2) == 0, r = 1; else, r = 2; end
        case 'square3'
            if mod(floor((k-1)/100), 2) == 0, r = 1; else, r = 3; end
        otherwise, r = 1;
    end
end

function a = diag_get_ak(k, scenario)
    a0 = 1.2 * (1 - 0.8 * exp(-0.1 * k));
    switch scenario
        case 'perturb'
            if k > 500 && k <= 1000, a = a0 * 1.3;
            elseif k > 1500, a = a0 * 0.7;
            else, a = a0; end
        otherwise, a = a0;
    end
end

function y_fb = diag_get_fb(y_true, k, scenario)
    switch scenario
        case 'disturb'
            if k == 500, y_fb = y_true + 0.5; else, y_fb = y_true; end
        case 'noise'
            y_fb = y_true + (rand - 0.5) * 2 * 0.02;
        otherwise, y_fb = y_true;
    end
end
