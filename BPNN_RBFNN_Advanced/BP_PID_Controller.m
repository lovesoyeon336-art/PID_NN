function [u, Kp, Ki, Kd] = BP_PID_Controller(r, y, dydu)
% =========================================================================
% BP Neural Network Self-Tuning PID Controller
% =========================================================================
% 网络结构: 3输入 → M隐层(tanh) → 3输出(sigmoid→映射到PID参数范围)
%
% 输入:
%   r    - 设定值 (reference)
%   y    - 被控对象输出 (plant output, 含噪声的测量值)
%   dydu - RBF辨识器提供的雅可比 ∂y/∂u
%
% 输出:
%   u    - 控制量 (增量式PID累加, 含限幅)
%   Kp,Ki,Kd - 实时PID参数 (用于在线监控)
%
% 在线学习: 每步基于梯度链 ∂E/∂W 更新网络权值
%   E = 0.5*(r-y)^2 → BP反传 → W ← W - lr*∂E/∂W
% =========================================================================

%% ===== 超参数 =====
M        = 6;       % 隐层神经元数
lr       = 0.015;   % 学习率
momentum = 0.05;    % 动量系数
Ts       = 0.001;   % 采样周期 (需与Simulink固定步长一致)
grad_clip = 1.0;    % 梯度裁剪阈值

% PID参数映射范围 (sigmoid输出映射到此区间)
Kp_min = 0;    Kp_max = 50;
Ki_min = 0;    Ki_max = 20;
Kd_min = 0;    Kd_max = 10;

% 控制量限幅
u_min = -10;
u_max =  10;

%% ===== 持久变量 =====
persistent W1 W2 b1 b2               % 网络权值与偏置
persistent dW1_prev dW2_prev         % 上一步权值更新量(动量用)
persistent db1_prev db2_prev
persistent e1 e2 u_prev sum_e        % 历史状态
persistent is_init

nin  = 3;   % [e, de, sum_e]
nout = 3;   % [Kp, Ki, Kd]

%% ===== 初始化 =====
if isempty(is_init)
    W1 = (rand(M, nin)  - 0.5) * 2 * sqrt(2/nin);   % He初始化, 配合tanh
    W2 = (rand(nout, M) - 0.5) * 2 * sqrt(2/M);
    b1 = zeros(M, 1);
    b2 = zeros(nout, 1);

    dW1_prev = zeros(M, nin);
    dW2_prev = zeros(nout, M);
    db1_prev = zeros(M, 1);
    db2_prev = zeros(nout, 1);

    e1     = 0;
    e2     = 0;
    u_prev = 0;
    sum_e  = 0;
    is_init = true;
end

%% ===== 误差信号 =====
e    = r - y;
de   = e - e1;
d2e  = e - 2*e1 + e2;
sum_e = sum_e + e * Ts;

%% ===== 前向传播 =====
x = [e; de; sum_e];                     % 3×1 输入向量

net_h = W1 * x + b1;                    % M×1
h     = tanh(net_h);                    % 隐层激活

net_o = W2 * h + b2;                    % 3×1
o     = 1 ./ (1 + exp(-net_o));         % 输出层sigmoid, o∈(0,1)

% 映射到物理PID参数
Kp = Kp_min + (Kp_max - Kp_min) * o(1);
Ki = Ki_min + (Ki_max - Ki_min) * o(2);
Kd = Kd_min + (Kd_max - Kd_min) * o(3);

%% ===== 增量式PID =====
delta_u = Kp * de + Ki * e + Kd * d2e;
u = u_prev + delta_u;
u = max(u_min, min(u_max, u));

%% ===== 在线反向传播 =====
% 性能指标: E = 0.5*e^2

% dE/du = dE/dy * dy/du = -e * dydu
dE_du = -e * dydu;

% du/dKj: 增量PID对各参数的偏导数
%   ∂u/∂Kp ≈ de, ∂u/∂Ki ≈ e, ∂u/∂Kd ≈ d2e
du_dK = [de; e; d2e];

% dKj/doj: 线性映射的缩放因子
scale_K = [Kp_max - Kp_min; Ki_max - Ki_min; Kd_max - Kd_min];

% doj/dnet_oj: sigmoid导数
dsig = o .* (1 - o);    % sigmoid'(x) = sigmoid(x)*(1-sigmoid(x))

% 输出层误差信号 δo = dE/du * du/dK * dK/do * do/dnet_o
delta_o = dE_du * (du_dK .* scale_K .* dsig);

% 隐层误差信号 δh = W2'*δo .* tanh'(net_h)
delta_h = (W2' * delta_o) .* (1 - h.^2);   % tanh'(x) = 1 - tanh^2(x)

% 梯度裁剪
delta_o = max(-grad_clip, min(grad_clip, delta_o));
delta_h = max(-grad_clip, min(grad_clip, delta_h));

% 计算梯度
grad_W2 = delta_o * h';                    % 3×M
grad_b2 = delta_o;                         % 3×1
grad_W1 = delta_h * x';                    % M×3
grad_b1 = delta_h;                         % M×1

% 带动量的权值更新
dW2 = lr * grad_W2 + momentum * dW2_prev;
db2 = lr * grad_b2 + momentum * db2_prev;
dW1 = lr * grad_W1 + momentum * dW1_prev;
db1 = lr * grad_b1 + momentum * db1_prev;

W2 = W2 - dW2;
b2 = b2 - db2;
W1 = W1 - dW1;
b1 = b1 - db1;

% 保存动量项
dW2_prev = dW2;
db2_prev = db2;
dW1_prev = dW1;
db1_prev = db1;

%% ===== 更新历史状态 =====
e2     = e1;
e1     = e;
u_prev = u;

end
