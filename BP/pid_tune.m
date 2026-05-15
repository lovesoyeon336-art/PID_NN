clear; close all;

%% ==================== 逐场景独立整定 PID —— 对象1 ====================

N_tune = 2000;
scenarios = {'step','sine_low','sine_high','ramp','perturb','disturb','noise'};
nS = length(scenarios);

init_2d = [
    log(0.5),  log(0.1);
    log(1.0),  log(0.5);
    log(2.0),  log(0.2);
    log(0.2),  log(0.05);
    log(1.5),  log(0.3);
    log(0.1),  log(0.8);
    log(3.0),  log(0.05);
];
init_3d = [
    log(0.5),  log(0.2), log(0.05);
    log(1.0),  log(0.3), log(0.1);
    log(2.0),  log(0.1), log(0.02);
    log(0.3),  log(0.5), log(0.03);
    log(1.5),  log(0.4), log(0.2);
    log(3.0),  log(0.8), log(0.01);
    log(0.8),  log(0.15),log(0.3);
];
opts = optimset('Display', 'off', 'MaxIter', 500, 'TolX', 1e-6);

Kp_opt = zeros(1, nS);
Ki_opt = zeros(1, nS);
Kd_opt = zeros(1, nS);

for s = 1:nS
    sc = scenarios{s};
    is_sine = any(strcmp(sc, {'sine_low','sine_high'}));

    if is_sine
        % 正弦场景：3D搜索(Kp,Ki,Kd) + ISE代价 + 前馈 + 热启动
        best_cost = inf;  best_x = [];
        for i = 1:size(init_3d, 1)
            cost_fn = @(x) pid_cost_sine(x, N_tune, sc);
            [x_opt, cost_val] = fminsearch(cost_fn, init_3d(i, :), opts);
            if cost_val < best_cost
                best_cost = cost_val;  best_x = x_opt;
            end
        end
        Kp_opt(s) = exp(best_x(1));
        Ki_opt(s) = exp(best_x(2));
        Kd_opt(s) = exp(best_x(3));
    else
        % 非正弦场景：2D搜索(Kp,Ki) + ITAE代价 + 超调惩罚
        best_cost = inf;  best_x = [];
        for i = 1:size(init_2d, 1)
            cost_fn = @(x) pid_cost_step(x, N_tune, sc);
            [x_opt, cost_val] = fminsearch(cost_fn, init_2d(i, :), opts);
            if cost_val < best_cost
                best_cost = cost_val;  best_x = x_opt;
            end
        end
        Kp_opt(s) = exp(best_x(1));
        Ki_opt(s) = exp(best_x(2));
        Kd_opt(s) = 0;
    end

    [y_hist, e_hist] = sim_pid(N_tune, sc, Kp_opt(s), Ki_opt(s), Kd_opt(s));
    fprintf('%-10s  Kp=%.4f  Ki=%.4f  Kd=%.4f  MAE=%.4f  max|e|=%.4f\n', ...
        sc, Kp_opt(s), Ki_opt(s), Kd_opt(s), mean(abs(e_hist)), max(abs(e_hist)));
end

%% ==================== 保存 ====================
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));
save(fullfile(script_dir, 'pid_tuned_params.mat'), ...
    'scenarios', 'Kp_opt', 'Ki_opt', 'Kd_opt');
fprintf('\n逐场景PID参数已保存至 pid_tuned_params.mat\n');

%% ==================== 代价函数 ====================

function cost = pid_cost_step(x, N_sim, scenario)
    Kp = exp(x(1));  Ki = exp(x(2));  Kd = 0;
    [y_hist, e_hist] = sim_pid(N_sim, scenario, Kp, Ki, Kd);
    itae = sum((1:N_sim) .* abs(e_hist));
    ov_penalty = 0;
    if any(strcmp(scenario, {'step','perturb','disturb','noise'}))
        ov = max(0, max(y_hist) - 1);
        ov_penalty = 5000 * ov^2;
    end
    cost = itae/1000 + ov_penalty;
end

function cost = pid_cost_sine(x, N_sim, scenario)
    Kp = exp(x(1));  Ki = exp(x(2));  Kd = exp(x(3));
    if Kp < 0.01, cost = 1e10; return; end  % 禁止Kp归零
    [~, e_hist] = sim_pid(N_sim, scenario, Kp, Ki, Kd);
    mae = mean(abs(e_hist));
    pk  = max(abs(e_hist));
    cost = mae + 0.1 * pk;
end

%% ==================== 仿真函数 ====================

function [y_hist, e_hist] = sim_pid(N_sim, scenario, Kp, Ki, Kd)
    is_sine = any(strcmp(scenario, {'sine_low','sine_high'}));
    % 正弦场景：启用前馈 + 热启动
    if is_sine
        ff_gain = 0.5;  beta_sp = 1.00;
        r_start = 1 + 0.5 * sin(2*pi*0.005);  % ≈1.016 (sine_low初始值, sine_high同理)
        y_1 = r_start;  % 热启动：消除冷启动瞬态
    else
        ff_gain = 0.0;  beta_sp = 1.00;
        y_1 = 0;
    end
    du_max = 1.0;
    u_1 = 0;  r_1 = 0;  e_1 = 0;  e_2 = 0;  e_sp_1 = 0;  e_sp_2 = 0;
    y_hist = zeros(1, N_sim);  e_hist = zeros(1, N_sim);
    y_true = y_1;
    rng(42);
    for k = 1:N_sim
        r_k = get_r(k, scenario);
        y_fb = get_yfb(y_true, k, scenario);
        a_override = get_ak_scenario(k, scenario);
        e_cur = r_k - y_fb;
        e_sp_k = beta_sp * r_k - y_fb;
        delta_u = Kp*(e_sp_k - e_sp_1) + Ki*e_cur + Kd*(e_sp_k - 2*e_sp_1 + e_sp_2);
        dr = r_k - r_1;
        if abs(dr) <= 0.1, delta_u = delta_u + ff_gain * dr; end
        delta_u = max(-du_max, min(du_max, delta_u));
        u_k = u_1 + delta_u;
        y_true = plant_dynamics('plant1', y_1, 0, u_1, u_1, k, a_override);
        y_hist(k) = y_true;  e_hist(k) = r_k - y_true;
        e_2 = e_1;  e_1 = e_cur;  e_sp_2 = e_sp_1;  e_sp_1 = e_sp_k;
        y_1 = y_true;  u_1 = u_k;  r_1 = r_k;
    end
end

function r_k = get_r(k, scenario)
    switch scenario
        case 'step',       r_k = 1;
        case 'sine_low',   r_k = 1 + 0.5 * sin(2*pi*0.005*k);
        case 'sine_high',  r_k = 1 + 0.5 * sin(2*pi*0.02*k);
        case 'ramp',       r_k = min(1, k / 500);
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

function a = get_ak_scenario(k, scenario)
    a0 = 1.2 * (1 - 0.8 * exp(-0.1 * k));
    switch scenario
        case 'perturb'
            if k > 500 && k <= 1000, a = a0 * 1.3;
            elseif k > 1500, a = a0 * 0.7;
            else, a = a0; end
        otherwise, a = a0;
    end
end
