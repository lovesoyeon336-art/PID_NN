function [u, Kp, Ki, Kd] = bp_pid_controller(r, y, ts)
% =========================================================================
%  BP 神经网络 PID 参数自整定控制器
%  用途：放入 Simulink 的 MATLAB Function 模块中使用
%
%  网络结构：
%    输入层  → 3 个神经元：[e(k), Δe(k), Δ²e(k)]
%    隐含层  → 6 个神经元，激活函数：tanh
%    输出层  → 3 个神经元：[Kp, Ki, Kd]，激活函数：sigmoid（保证正值）
%
%  控制律：增量式 PID
%    Δu(k) = Kp·Δe(k) + Ki·ts·e(k) + Kd/ts·Δ²e(k)
%    u(k)  = u(k-1) + Δu(k)
%
%  在线学习：每采样周期执行一次 BP 反向传播
%  Jacobian：采用差商近似 ∂y/∂u ≈ Δy/Δu（符号提取，增强鲁棒性）
%
%  输入：
%    r  - 参考指令（设定值）
%    y  - 被控对象测量输出
%    ts - 采样时间（由外部 Constant 模块接入）
%
%  输出：
%    u  - PID 控制量（输出给被控对象）
%    Kp - 比例系数（用于监控）
%    Ki - 积分系数（用于监控）
%    Kd - 微分系数（用于监控）
%
%  作者：自动生成  版本：v1.0
% =========================================================================

%% ============================= 超参数 ===================================
%  以下常量均可根据被控对象特性自行调整

lr       = 0.001;   % 学习率
u_max    = 10.0;    % 控制量上限（防积分饱和 / 执行器限幅）
u_min    = -10.0;   % 控制量下限
Kp_max   = 20.0;    % Kp 的最大允许值（sigmoid 输出乘以此系数）
Ki_max   = 5.0;     % Ki 的最大允许值
Kd_max   = 5.0;     % Kd 的最大允许值

% 神经网络结构参数（修改后需同步修改 persistent 变量维度）
n_in     = 3;       % 输入层神经元数
n_hid    = 6;       % 隐含层神经元数
n_out    = 3;       % 输出层神经元数（固定为 3，对应 Kp/Ki/Kd）

%% ========================== Persistent 变量 =============================
%  所有跨采样周期需要保留的状态均声明为 persistent

% 神经网络权值与偏置
persistent W1 B1    % 隐含层：W1(n_hid × n_in), B1(n_hid × 1)
persistent W2 B2    % 输出层：W2(n_out × n_hid), B2(n_out × 1)

% 历史误差与控制量（用于计算差分及 Jacobian）
persistent e_k1     % e(k-1)：上一时刻误差
persistent e_k2     % e(k-2)：上上时刻误差
persistent u_k1     % u(k-1)：上一时刻控制量
persistent y_k1     % y(k-1)：上一时刻被控对象输出（Jacobian 估算用）
persistent u_k2
% 初始化标志
persistent is_init

%% ========================= 首次运行初始化 ================================
 if isempty(is_init)

    % ---- 权值初始化：均匀分布小随机数，打破网络对称性 ----
    W1 = 0.2 * (rand(n_hid, n_in) - 0.5);
    B1 = 0.2 * (rand(n_hid, 1) - 0.5);
    W2 = 0.2 * (rand(n_out, n_hid) - 0.5);
    B2 = 0.2 * (rand(n_out, 1) - 0.5);

    % ---- 历史状态初始化 ----
    e_k1   = 0;
    e_k2   = 0;
    u_k1   = 0;
    y_k1   = 0;
    u_k2 = 0;
    is_init = true;
end



%% ======================= Step 1：误差计算 ================================

e   = r - y;                        % 当前误差 e(k)
de  = e - e_k1;                     % 误差一阶差分 Δe(k) = e(k) - e(k-1)
dde = e - 2.0*e_k1 + e_k2;         % 误差二阶差分 Δ²e(k)（用于 D 项）

%% ======================= Step 2：神经网络前向传播 ========================

% 输入向量（3×1）
x_in = [e; de; dde];

% -- 隐含层计算 --
z1 = W1 * x_in + B1;               % (n_hid × 1)
o1 = tanh(z1);                      % (n_hid × 1)

% -- 输出层计算 --
z2 = W2 * o1 + B2;                 % (n_out × 1)
o2 = 1.0 ./ (1.0 + exp(-z2));      % (n_out × 1)

% 将网络输出映射到实际 PID 参数范围
Kp = Kp_max * o2(1);
Ki = Ki_max * o2(2);
Kd = Kd_max * o2(3);

%% ======================= Step 3：增量式 PID 控制律 ======================

% 增量计算
delta_u = Kp * de  +  Ki * e * ts +  Kd * dde;

% 位置累加
u_raw = u_k1 + delta_u;

% -- 积分饱和处理（输出限幅）--
if u_raw > u_max
    u = u_max;
elseif u_raw < u_min
    u = u_min;
else
    u = u_raw;
end

%% ======================= Step 4：Jacobian 估算 ===========================

delta_y_obs  = y   - y_k1;         % Δy = y(k) - y(k-1)
% delta_u_obs  = u   - u_k1;         % Δu = u(k) - u(k-1)
delta_u_obs  = u_k1 - u_k2;
if abs(delta_u_obs) > 1e-6
    dydu_raw = delta_y_obs / delta_u_obs;
else
    dydu_raw = delta_y_obs * 1e3;
end

jac_sign = sign(dydu_raw);
if jac_sign == 0
    jac_sign = 1;
end

%% ======================= Step 5：BP 反向传播 =============================

dE_du = (-e) * jac_sign;           % ∂E/∂u（标量）

du_do2 = [Kp_max * de;
          Ki_max * ts * e;
          Kd_max * (dde)];    % (3×1)

% -- 输出层灵敏度 δ2（3×1）--
sig_grad = o2 .* (1.0 - o2);       % sigmoid 导数
delta2   = dE_du * du_do2 .* sig_grad;

% -- 输出层权值梯度 --
grad_W2 = delta2 * o1';            % (3×6)
grad_B2 = delta2;                  % (3×1)

% -- 隐含层灵敏度 δ1（6×1）--
tanh_grad = 1.0 - o1.^2;          % tanh 导数
delta1    = (W2' * delta2) .* tanh_grad; % (6×1)

% -- 隐含层权值梯度 --
grad_W1 = delta1 * x_in';          % (6×3)
grad_B1 = delta1;                  % (6×1)

%% ======================= Step 6：权值更新 ================================

W2 = W2 - lr * grad_W2;
B2 = B2 - lr * grad_B2;
W1 = W1 - lr * grad_W1;
B1 = B1 - lr * grad_B1;

%% ======================= Step 7：状态更新 ================================

e_k2 = e_k1;
e_k1 = e;
u_k2 = u_k1;
u_k1 = u;
y_k1 = y;
end
