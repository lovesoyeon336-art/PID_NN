clear; close all;

%% ==================== 统一整定 PID —— Plant3 (Hammerstein) ====================

N_tune = 2000;
scenarios = {'step','sine_low','sine_high','ramp','disturb','noise','square'};
nS = length(scenarios);

init_3d = [
    log(1.0),  log(0.1), log(0.05);
    log(3.0),  log(0.3), log(0.2);
    log(5.0),  log(0.5), log(0.5);
    log(0.5),  log(0.8), log(0.01);
    log(2.0),  log(0.4), log(0.3);
    log(7.0),  log(0.1), log(1.0);
    log(1.5),  log(1.0), log(0.1);
    log(0.2),  log(0.2), log(0.02);
    log(10.0), log(0.05),log(1.5);
    log(0.8),  log(0.15),log(0.05);
    log(0.5),  log(2.0), log(0.01);
    log(5.0),  log(1.5), log(0.3);
    log(2.0),  log(3.0), log(0.1);
    log(3.0),  log(0.05), log(2.0);
    log(4.0),  log(0.10), log(3.0);
    log(5.0),  log(0.15), log(4.0);
    log(6.0),  log(0.20), log(5.0);
    log(3.0),  log(0.20), log(1.5);
    log(7.0),  log(0.08), log(3.5);
    log(2.5),  log(0.12), log(2.5);
];
opts = optimset('Display', 'iter', 'MaxIter', 2000, 'TolX', 1e-6);

Kp_opt = 0;  Ki_opt = 0;  Kd_opt = 0;

%% ==================== 多起点统一搜索 ====================
best_cost = inf;  best_x = [];
for i = 1:size(init_3d, 1)
    cost_fn = @(x) pid_cost_unified(x, N_tune, scenarios);
    [x_opt, cost_val] = fminsearch(cost_fn, init_3d(i, :), opts);
    fprintf('  起点 %2d: cost=%.4f  Kp=%.4f  Ki=%.4f  Kd=%.4f\n', ...
        i, cost_val, exp(x_opt(1)), exp(x_opt(2)), exp(x_opt(3)));
    if cost_val < best_cost
        best_cost = cost_val;  best_x = x_opt;
    end
end

Kp_opt = exp(best_x(1));
Ki_opt = exp(best_x(2));
Kd_opt = exp(best_x(3));

fprintf('\n========== Plant3 统一PID最优参数 ==========\n');
fprintf('Kp=%.4f  Ki=%.4f  Kd=%.4f\n\n', Kp_opt, Ki_opt, Kd_opt);

%% ==================== 逐场景验证 ====================
fprintf('%-10s  %8s  %8s  %8s\n', '场景', '超调%', 'MAE', 'max|e|');
fprintf('%s\n', repmat('-', 1, 50));
for s = 1:nS
    sc = scenarios{s};
    [y_hist, e_hist] = sim_pid(N_tune, sc, Kp_opt, Ki_opt, Kd_opt);
    r_seq = zeros(1, N_tune);
    for k = 1:N_tune, r_seq(k) = get_r(k, sc); end
    if strcmp(sc, 'square'), ref_nom = 2; else, ref_nom = 1; end
    ov = max(0, max(y_hist - r_seq));
    ov_pct = ov / ref_nom * 100;
    fprintf('%-10s  %7.2f%%  %8.4f  %8.4f\n', ...
        sc, ov_pct, mean(abs(e_hist)), max(abs(e_hist)));
end

%% ==================== 保存 ====================
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));
save(fullfile(script_dir, 'pid_tuned_params_plant3.mat'), ...
    'scenarios', 'Kp_opt', 'Ki_opt', 'Kd_opt');
fprintf('\nPlant3 统一PID参数已保存至 pid_tuned_params_plant3.mat\n');

%% ==================== 统一代价函数 ====================

function cost = pid_cost_unified(x, N_sim, scenarios)
    Kp = exp(x(1));  Ki = exp(x(2));  Kd = exp(x(3));
    if Kp < 0.01, cost = 1e10; return; end

    nS = length(scenarios);
    tracking_costs = zeros(1, nS);
    ov_pcts = zeros(1, nS);

    for s = 1:nS
        sc = scenarios{s};
        [y_hist, e_hist] = sim_pid(N_sim, sc, Kp, Ki, Kd);

        r_seq = zeros(1, N_sim);
        for k = 1:N_sim, r_seq(k) = get_r(k, sc); end

        if strcmp(sc, 'square'), ref_nominal = 2; else, ref_nominal = 1; end
        ov = max(0, max(y_hist - r_seq));
        ov_pcts(s) = ov / ref_nominal * 100;

        if any(strcmp(sc, {'sine_low','sine_high','ramp'}))
            tracking_costs(s) = mean(abs(e_hist));
        else
            tracking_costs(s) = sum((1:N_sim) .* abs(e_hist)) / 1000;
        end
    end

    ov_penalties = zeros(1, nS);
    for s = 1:nS
        ov = ov_pcts(s);
        sc = scenarios{s};
        ov_lim = 12; if strcmp(sc, 'square'), ov_lim = 60; end
        if ov > ov_lim
            ov_penalties(s) = 20*(ov-ov_lim)^2 + 100*(ov-ov_lim);
        else
            ov_penalties(s) = 0.5*ov^2;
        end
    end
    cost = sum(tracking_costs) + sum(ov_penalties);
end

%% ==================== 仿真函数 ====================

function [y_hist, e_hist] = sim_pid(N_sim, scenario, Kp, Ki, Kd)
    is_sine = any(strcmp(scenario, {'sine_low','sine_high'}));
    ff_gain = 2.0;  beta_sp = 0.90;  du_max = 1.0;  u_sat = 3.0;
    if is_sine
        r_start = 1 + 0.5 * sin(2*pi*0.005);
        y_1 = r_start;  y_2 = r_start;
    else
        y_1 = 0;  y_2 = 0;
    end
    u_1 = 0;  r_1 = 0;  e_1 = 0;  e_2 = 0;  e_sp_1 = 0;  e_sp_2 = 0;
    y_hist = zeros(1, N_sim);  e_hist = zeros(1, N_sim);
    y_true = y_1;
    rng(42);
    for k = 1:N_sim
        r_k = get_r(k, scenario);
        y_fb = get_yfb(y_true, k, scenario);
        e_cur = r_k - y_fb;
        e_sp_k = beta_sp * r_k - y_fb;
        Kd_eff = Kd; if strcmp(scenario, 'square'), Kd_eff = 0; end
        delta_u = Kp*(e_sp_k - e_sp_1) + Ki*e_cur + Kd_eff*(e_sp_k - 2*e_sp_1 + e_sp_2);
        dr = r_k - r_1;
        if abs(dr) <= 0.1, delta_u = delta_u + ff_gain * dr; end
        delta_u = max(-du_max, min(du_max, delta_u));
        u_k = max(-u_sat, min(u_sat, u_1 + delta_u));
        y_true = plant_dynamics('plant3', y_1, y_2, u_1, u_1, k);
        y_hist(k) = y_true;  e_hist(k) = r_k - y_true;
        e_2 = e_1;  e_1 = e_cur;  e_sp_2 = e_sp_1;  e_sp_1 = e_sp_k;
        y_2 = y_1;  y_1 = y_true;  u_1 = u_k;  r_1 = r_k;
    end
end

function r_k = get_r(k, scenario)
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
