clear; close all;

%% ==================== 固定 PID 参数 ====================
Kp = 0.5;       % 比例增益
Ki = 0.1;       % 积分增益
Kd = 0.01;      % 微分增益
T  = 1;         % 控制周期
N  = 35000;     % 仿真步数

%% ==================== 状态初始化 ====================
r_target = 1;       % 阶跃目标
y     = zeros(N, 1);       % 被控对象输出序列
y_1   = 0;                 % 上一时刻输出
u_1   = 0;                 % 上一时刻控制量
ei    = 0;                 % 积分累加
last_e = 0;                % 上一时刻误差

%% ==================== 预分配 ====================
time = zeros(1, N);
r    = zeros(1, N);
u    = zeros(1, N);

%% ==================== 主循环 ====================
for k = 1:N
    time(k) = k;
    r(k) = r_target;

    % 误差
    e_cur = r(k) - y_1;

    % 积分累加
    ei = ei + e_cur * T;

    % 微分
    ed = (e_cur - last_e) / T;

    % 位置式 PID
    u(k) = Kp * e_cur + Ki * ei + Kd * ed;

    % 非线性时变被控对象
    a_k = 1.2 * (1 - 0.8 * exp(-0.1 * k));
    y(k) = a_k / (1 + y_1^2) * y_1 + u(k);

    % 状态更新
    last_e = e_cur;
    y_1 = y(k);
    u_1 = u(k);
end

%% ==================== 结果输出 ====================
fprintf('===== 固定 PID 仿真结果 =====\n');
fprintf('Kp = %.2f, Ki = %.2f, Kd = %.2f\n', Kp, Ki, Kd);
fprintf('平均绝对误差 MAE : %.6f\n', sum(abs(r - y')) / N);
fprintf('最终稳态误差     : %.6f\n', abs(r(end) - y(end)));
fprintf('超调量 (max)      : %.6f\n', max(y) - r_target);

%% ==================== 保存数据 ====================
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));
save(fullfile(script_dir, 'pid_fixed_result.mat'), 'time', 'r', 'y', 'u', 'Kp', 'Ki', 'Kd', 'N');

%% ==================== 绘图 ====================
fx = [1, min(2000, N)];

figure('Name', '固定PID 温度追踪', 'NumberTitle', 'off');
plot(time, r, 'r', time, y, 'b--', 'LineWidth', 1.2);
xlim(fx);
xlabel('时间步');  ylabel('温度');
legend('目标 r', '实际 y', 'Location', 'best');
title(sprintf('固定 PID 阶跃响应 (Kp=%.2f, Ki=%.2f, Kd=%.2f)', Kp, Ki, Kd));
grid on;
