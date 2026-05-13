clear; close all;

%% ==================== ITAE 整定固定 PID ====================

N_tune = 2000;          % 整定仿真步数
r_target = 1;

%% ==================== 多起点搜索 ====================
init_guesses = [
    log(0.5),  log(0.1), log(0.01);
    log(1.0),  log(0.5), log(0.1);
    log(2.0),  log(0.2), log(0.5);
    log(0.2),  log(0.05),log(0.5);
    log(1.5),  log(0.3), log(0.3);
];

best_cost = inf;
best_x = [];

opts = optimset('Display', 'off', 'MaxIter', 500, 'TolX', 1e-6);

for i = 1:size(init_guesses, 1)
    cost_fn = @(x) pid_cost(x, N_tune, r_target);
    [x_opt, cost_val] = fminsearch(cost_fn, init_guesses(i, :), opts);
    if cost_val < best_cost
        best_cost = cost_val;
        best_x = x_opt;
    end
end

Kp_opt = exp(best_x(1));
Ki_opt = exp(best_x(2));
Kd_opt = exp(best_x(3));

%% ==================== 验证最优参数 ====================
y_1 = 0;  u_1 = 0;  ei = 0;  last_e = 0;
y_hist = zeros(1, N_tune);
u_hist = zeros(1, N_tune);
e_hist = zeros(1, N_tune);

for k = 1:N_tune
    e_cur = r_target - y_1;
    ei = ei + e_cur;
    ei_clamped = max(-3, min(3, ei));
    ed = e_cur - last_e;
    u_k = Kp_opt * e_cur + Ki_opt * ei_clamped + Kd_opt * ed;

    a_k = 1.2 * (1 - 0.8 * exp(-0.1 * k));
    y_k = a_k / (1 + y_1^2) * y_1 + u_k;

    y_hist(k) = y_k;
    u_hist(k) = u_k;
    e_hist(k) = e_cur;

    last_e = e_cur;
    y_1 = y_k;
    u_1 = u_k;
end

fprintf('===== ITAE 整定结果 =====\n');
fprintf('Kp = %.4f   Ki = %.4f   Kd = %.4f\n', Kp_opt, Ki_opt, Kd_opt);
fprintf('ITAE = %.2f\n', best_cost);
fprintf('MAE  = %.6f\n', sum(abs(e_hist)) / N_tune);
fprintf('超调 = %.2f%%\n', (max(y_hist) - r_target) * 100);

settle_band = 0.05;
above = find(abs(y_hist - r_target) > settle_band, 1, 'last');
if isempty(above)
    fprintf('调节时间 (±5%%) : 已收敛\n');
else
    fprintf('调节时间 (±5%%) : step %d\n', above);
end

%% ==================== 保存 ====================
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));
save(fullfile(script_dir, 'pid_tuned_params.mat'), 'Kp_opt', 'Ki_opt', 'Kd_opt');
fprintf('\n最优参数已保存至 pid_tuned_params.mat\n');

%% ==================== 绘图 ====================
fx = [1, min(500, N_tune)];
figure('Name', '整定后 PID 阶跃响应', 'NumberTitle', 'off');
plot(1:N_tune, r_target * ones(1, N_tune), 'r', 1:N_tune, y_hist, 'b--', 'LineWidth', 1.2);
xlim(fx);
xlabel('时间步'); ylabel('温度');
legend('目标', '实际', 'Location', 'best');
title(sprintf('ITAE 整定 PID (Kp=%.2f, Ki=%.2f, Kd=%.2f)', Kp_opt, Ki_opt, Kd_opt));
grid on;

%% ==================== 局部函数 ====================

function cost = pid_cost(x, N_sim, r_target)
    Kp = exp(x(1));
    Ki = exp(x(2));
    Kd = exp(x(3));

    y_1 = 0;  u_1 = 0;  ei = 0;  last_e = 0;
    ITAE = 0;

    for k = 1:N_sim
        e_cur = r_target - y_1;
        ei = ei + e_cur;
        ei_clamped = max(-3, min(3, ei));
        ed = e_cur - last_e;

        u_k = Kp * e_cur + Ki * ei_clamped + Kd * ed;

        a_k = 1.2 * (1 - 0.8 * exp(-0.1 * k));
        y_k = a_k / (1 + y_1^2) * y_1 + u_k;

        ITAE = ITAE + k * abs(e_cur);

        last_e = e_cur;
        y_1 = y_k;
        u_1 = u_k;
    end
    cost = ITAE;
end
