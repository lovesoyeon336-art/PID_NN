clear; close all;

%% ==================== 超参数 ====================
IN = 4;   H = 5;   Out = 3;   % 输入/隐藏/输出 节点数
rate  = 0.001;                  % 学习率
rate2 = 0.01;                   % 动量系数
N = 35000;                      % 仿真步数

Kp_max = 2.0;                   % PID 参数缩放上限
Ki_max = 0.5;
Kd_max = 1.0;
scale_vec = [Kp_max, Ki_max, Kd_max];  % 链式法则用

%% ==================== 权重初始化 ====================
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));
pretrain_file = fullfile(script_dir, 'bp_pretrained_weights.mat');

if isfile(pretrain_file)
    load(pretrain_file, 'w1', 'w2');
    fprintf('已加载预训练权重: %s\n', pretrain_file);
else
    rng(1);
    w1 = sqrt(2/(IN+H))  * randn(H, IN);    % 5×4
    w2 = sqrt(2/(H+Out)) * randn(Out, H);   % 3×5
    fprintf('Xavier 初始化（未找到预训练权重）\n');
end

w1_1 = w1;  w1_2 = w1;                  % 动量缓存
w2_1 = w2;  w2_2 = w2;

%% ==================== 状态初始化 ====================
r_target = 1;           % 阶跃目标

y_1 = 0;                % 被控对象初始输出
u_1 = 0;                % 上一步控制量

error_1 = 0;            % e(k-1)
error_2 = 0;            % e(k-2)，增量式 PID 用

%% ==================== 预分配 ====================
time = zeros(1, N);
r    = zeros(1, N);
y    = zeros(1, N);
error = zeros(1, N);
u    = zeros(1, N);
kp   = zeros(1, N);
ki   = zeros(1, N);
kd   = zeros(1, N);

%% ==================== 主循环 ====================
for k = 1:N

    time(k) = k;

    % ---- ① 误差计算 ----
    r(k) = r_target;
    error(k) = r(k) - y_1;

    % ---- ② 前向传播 ----
    I1 = [r(k), y_1, error(k), 1];           % 输入向量 (1×4)
    I2 = I1 * w1';                             % 隐藏层输入 (1×5)

    O2 = zeros(1, H);
    for j = 1:H
        O2(j) = tanh(I2(j));                   % 隐藏层输出
    end

    I3 = w2 * O2';                             % 输出层输入 (3×1)
    O3 = zeros(1, Out);
    for l = 1:Out
        O3(l) = sigmoid(I3(l));                % 输出层输出
    end

    kp(k) = Kp_max * O3(1);
    ki(k) = Ki_max * O3(2);
    kd(k) = Kd_max * O3(3);
    Kpid = [kp(k), ki(k), kd(k)];              % 1×3

    % ---- ③ 增量式 PID ----
    e_pid = [error(k) - error_1;                % Δe  (P项)
             error(k);                          % e   (I项)
             error(k) - 2*error_1 + error_2];   % Δ²e (D项)
    delta_u = Kpid * e_pid;                     % 原始增量
    du_max = 0.5;                               % 单步限幅
    delta_u = max(-du_max, min(du_max, delta_u));
    u(k) = u_1 + delta_u;                       % u(k) = u(k-1) + Δu

    % ---- ④ 非线性时变被控对象 ----
    a_k = 1.2 * (1 - 0.8 * exp(-0.1 * k));
    y(k) = a_k / (1 + y_1^2) * y_1 + u(k);

    % 更新误差（用新 y 重新计算）
    error(k) = r(k) - y(k);

    % ---- ⑤ 反向传播 ----
    % 输出层灵敏度
    dO3 = zeros(1, Out);
    for j = 1:Out
        dO3(j) = sigmoidGradient(O3(j));        % sigmoid 导数
    end

    % Jacobian 幅值近似（clamp 防梯度爆炸）
    dydu = (y(k) - y_1) / (u(k) - u_1 + 0.0001);
    du_sys = max(-1, min(1, dydu));

    delta3 = zeros(1, Out);
    for l = 1:Out
        delta3(l) = error(k) * du_sys * scale_vec(l) * e_pid(l) * dO3(l);
    end

    % 输出层权重梯度 + 动量更新
    d_w2 = zeros(Out, H);
    for l = 1:Out
        for i = 1:H
            d_w2(l, i) = rate * delta3(l) * O2(i);
        end
    end
    w2 = w2_1 + d_w2 + rate2 * 2 * (w2_1 - w2_2);

    % 隐藏层灵敏度
    dO2 = zeros(1, H);
    for i = 1:H
        dO2(i) = 1 - tanh(I2(i))^2;             % tanh 导数
    end

    a_back = delta3 * w2;                        % 1×5
    delta2 = zeros(1, H);
    for i = 1:H
        delta2(i) = dO2(i) * a_back(i);
    end

    % 隐藏层权重梯度 + 动量更新
    d_w1 = rate * delta2' * I1;                 % 5×4
    w1 = w1_1 + d_w1 + rate2 * (w1_1 - w1_2);

    % ---- ⑥ 状态缓存 ----
    u_1 = u(k);
    y_1 = y(k);

    w2_2 = w2_1;  w2_1 = w2;
    w1_2 = w1_1;  w1_1 = w1;

    error_2 = error_1;
    error_1 = error(k);
end

%% ==================== 结果输出 ====================
fprintf('===== BP-PID 仿真结果 =====\n');
fprintf('平均绝对误差 MAE : %.6f\n', sum(abs(error)) / N);
fprintf('最终稳态误差     : %.6f\n', abs(error(end)));
fprintf('超调量 (max)      : %.6f\n', max(y) - r_target);

%% ==================== 保存数据 ====================
save(fullfile(script_dir, 'bp_pid_result.mat'), 'time', 'r', 'y', 'error', 'u', 'kp', 'ki', 'kd', 'N');

%% ==================== 绘图 ====================
fx = [1, min(2000, N)];    % 显示前 2000 步或全部

% --- 图1: 温度追踪 ---
figure('Name', 'BP-PID 温度追踪', 'NumberTitle', 'off');
plot(time, r, 'r', time, y, 'b--', 'LineWidth', 1.2);
xlim(fx);
xlabel('时间步');  ylabel('温度');
legend('目标 r', '实际 y', 'Location', 'best');
title('BP-PID 阶跃响应 (r=1)');
grid on;

% --- 图2: 控制量 ---
figure('Name', 'BP-PID 控制量', 'NumberTitle', 'off');
plot(time, u, 'r', 'LineWidth', 1);
xlim(fx);
xlabel('时间步');  ylabel('控制量 u');
title('BP-PID 控制量');
grid on;

% --- 图3: PID 参数在线变化 ---
figure('Name', 'BP-PID 参数变化', 'NumberTitle', 'off');
subplot(311);
plot(time, kp, 'r', 'LineWidth', 0.8);
xlim(fx);
xlabel('时间步');  ylabel('Kp');
title('PID 参数在线自适应');
grid on;
subplot(312);
plot(time, ki, 'g', 'LineWidth', 0.8);
xlim(fx);
xlabel('时间步');  ylabel('Ki');
grid on;
subplot(313);
plot(time, kd, 'b', 'LineWidth', 0.8);
xlim(fx);
xlabel('时间步');  ylabel('Kd');
grid on;
