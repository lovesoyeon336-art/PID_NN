clear; close all;

%% ==================== 超参数 ====================
IN = 5;   H = 5;   Out = 3;   % 输入/隐藏/输出 节点数
rate  = 0.01;                   % 学习率（提高以加速跟踪适应）
rate2 = 0.01;                   % 动量系数
N = 2000;                       % 仿真步数

Kp_max = 2.0;                   % PID 参数缩放上限
Ki_max = 0.2;
Kd_max = 0.0;
scale_vec = [Kp_max, Ki_max, Kd_max];  % 链式法则用
ff_gain = 0.0;  beta_sp = 1.00;

%% ==================== 权重初始化 ====================
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));
pretrain_file = fullfile(script_dir, 'bp_pretrained_weights.mat');

if isfile(pretrain_file)
    load(pretrain_file, 'w1', 'w2');
    fprintf('已加载预训练权重: %s\n', pretrain_file);
else
    rng(1);
    w1 = sqrt(2/(IN+H))  * randn(H, IN);    % 5×5
    w2 = sqrt(2/(H+Out)) * randn(Out, H);   % 3×5
    fprintf('Xavier 初始化（未找到预训练权重）\n');
end

w1_1 = w1;  w1_2 = w1;                  % 动量缓存
w2_1 = w2;  w2_2 = w2;

%% ==================== 状态初始化 ====================
r_target = 1;           % 阶跃目标

y_1 = 0;                % 被控对象初始输出
u_1 = 0;                % 上一步控制量
u_2 = 0;                % 上两步控制量 (FD Jacobian)
r_1 = r_target;         % r(k-1)，参考信号历史

error_1 = 0;            % e(k-1)
error_2 = 0;            % e(k-2)，增量式 PID 用
e_sp_1 = 0;  e_sp_2 = 0;

% 完整延时梯度存储：k 拍状态 → k+1 拍更新权重
st_has = false;  st_ep = zeros(3,1);  st_O2 = zeros(1,H);
st_dO3 = zeros(1,Out);  st_dO2 = zeros(1,H);  st_I1 = zeros(1,IN);

%% ==================== 预分配 ====================
time = zeros(1, N);
r    = zeros(1, N);
y    = zeros(1, N);
error = zeros(1, N);
u    = zeros(1, N);
kp   = zeros(1, N);
ki   = zeros(1, N);
kd   = zeros(1, N);

%% ==================== 主循环（完整延时梯度） ====================
% y(k) = g(y(k-1)) + u(k-1)：u(k) 的效果体现在 y(k+1)
% 因此权重更新延迟一拍：k+1 拍用 k 拍存储的状态 + e(k+1) 更新
for k = 1:N

    time(k) = k;

    % ---- ① 误差计算 ----
    r(k) = r_target;
    error(k) = r(k) - y_1;

    % ---- ② 前向传播 ----
    I1 = [r(k), y_1, error(k), r(k)-r_1, 1];   % Δr 替换 error_1 提供参考速度
    I2 = I1 * w1';                             % 隐藏层输入 (1×5)

    O2 = zeros(1, H);
    for j = 1:H
        O2(j) = tanh(I2(j));                   % 隐藏层输出
    end

    I3 = w2 * O2';                             % 输出层输入 (3×1)
    O3 = zeros(1, Out);
    for l = 1:Out
        if I3(l) > 0
            O3(l) = I3(l);                     % Leaky ReLU 正区
        else
            O3(l) = 0.2 * I3(l);              % Leaky ReLU 负区
        end
    end

    kp(k) = Kp_max * O3(1);
    ki(k) = Ki_max * O3(2);
    kd(k) = Kd_max * O3(3);
    Kpid = [kp(k), ki(k), kd(k)];              % 1×3

    % ---- ③ 增量式 PID ----
    e_sp_k = beta_sp * r(k) - y_1;
    e_pid = [e_sp_k - e_sp_1;                    % Δe  (P项，加权设定值)
             error(k);                            % e   (I项，完整误差)
             e_sp_k - 2*e_sp_1 + e_sp_2];         % Δ²e (D项，加权设定值)
    delta_u = Kpid * e_pid;
    dr = r(k) - r_1;
    if abs(dr) <= 0.1, delta_u = delta_u + ff_gain * dr; end
    du_max = 1.0;
    delta_u = max(-du_max, min(du_max, delta_u));
    u(k) = u_1 + delta_u;                       % u(k) = u(k-1) + Δu

    % ---- ④ 被控对象 (y(k) 由 u(k-1) 驱动) ----
    y(k) = plant_dynamics('plant1', y_1, 0, u_1, u_1, k);
    error(k) = r(k) - y(k);

    % ---- ⑤ 延时反向传播（用上一拍存储状态 + 当前误差） ----
    dead_zone = 0.002;  % 缩小死区，小误差也持续学习
    if st_has && abs(error(k)) >= dead_zone
        % Jacobian: ∂y(k)/∂u(k-1)
        dydu = (y(k) - y_1) / (u_1 - u_2 + 0.0001);
        du_sys = max(-1, min(1, dydu));

        delta3 = zeros(1, Out);
        for l = 1:Out
            delta3(l) = error(k) * du_sys * scale_vec(l) * st_ep(l) * st_dO3(l);
        end

        d_w2 = zeros(Out, H);
        for l = 1:Out
            for i = 1:H
                d_w2(l, i) = rate * delta3(l) * st_O2(i);
            end
        end
        w2 = w2_1 + d_w2 + rate2 * 2 * (w2_1 - w2_2);

        a_back = delta3 * w2_1;                  % 用上一拍权重
        delta2 = st_dO2 .* a_back;
        d_w1 = rate * delta2' * st_I1;
        w1 = w1_1 + d_w1 + rate2 * (w1_1 - w1_2);

        w2_2 = w2_1;  w2_1 = w2;
        w1_2 = w1_1;  w1_1 = w1;
    end

    % ---- ⑥ 存储当前步状态（供下一步延时梯度使用） ----
    st_has = true;
    st_ep  = e_pid;
    st_O2  = O2;
    st_I1  = I1;
    for j = 1:Out
        if O3(j) > 0, st_dO3(j) = 1; else, st_dO3(j) = 0.2; end
    end
    for i = 1:H
        st_dO2(i) = 1 - tanh(I2(i))^2;
    end

    % ---- ⑦ 状态缓存 ----
    u_2 = u_1;
    u_1 = u(k);
    y_1 = y(k);
    r_1 = r(k);
    error_2 = error_1;
    error_1 = error(k);
    e_sp_2 = e_sp_1;
    e_sp_1 = e_sp_k;
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
