function [dy_du, y_hat] = rbf_online_identification(u_prev, y, reset)
% RBF神经网络在线辨识 — 为 BP-PID 提供 ∂y/∂u
%
% 输入:
%   u_prev — u(k-1)，控制量经 Unit Delay 后接入
%   y      — y(k)，被控对象当前输出
%   reset  — 复位信号，接常数 0
%
% 输出:
%   dy_du  — 雅可比 ∂y/∂u
%   y_hat  — RBF 预测输出

persistent W C Sigma dW_prev dC_prev dSig_prev y1 y2 u2 gain init_flag

%% ===== 超参数 =====
M       = 10;     % RBF 隐层节点数（从 8 增至 10）
n       = 4;      % 输入维度 [y(k-1), y(k-2), u(k-1), u(k-2)]
eta_w   = 0.10;   % W 学习率（从 0.30 降至 0.10）
eta_c   = 0.08;   % C 学习率（从 0.05 提至 0.08）
eta_s   = 0.05;   % σ 学习率
mu_w    = 0.05;   % W 动量系数
mu_c    = 0.02;   % C 动量系数
mu_s    = 0.02;   % σ 动量系数
sig_min = 0.02;   % σ 下限

%% ===== 初始化 =====
if isempty(init_flag) || reset > 0.5
    % 输入归一化增益：将各维映射到约 [-1, 1]
    % y 范围 ~ [0, 2] → gain=1.0（以 0 为中心约 ±1）
    % u 范围 ~ [-10, 10] → gain=0.1（映射到 ±1）
    gain = [0.8; 0.8; 0.1; 0.1];

    W     = 0.1 * randn(1, M);
    % 中心在归一化空间初始化为 [-1, 1] 均匀分布
    rng(0);
    C     = 2*(rand(M, n)-0.5);   % [-1, 1]
    Sigma = 1.5 * ones(M, 1);
    dW_prev   = zeros(1, M);
    dC_prev   = zeros(M, n);
    dSig_prev = zeros(M, 1);
    y1 = 0; y2 = 0; u2 = 0;
    init_flag = true;
end

%% ===== 构造归一化输入 =====
x_raw = [y1; y2; double(u_prev); u2];  % 4×1
x = x_raw .* gain;                      % 归一化

%% ===== 前向传播 =====
D     = bsxfun(@minus, x', C);              % M×n
dist2 = sum(D .^ 2, 2);                     % M×1
H     = exp(-dist2 ./ (2 * Sigma .^ 2));    % M×1
y_hat = W * H;                              % 标量

%% ===== 辨识误差 =====
e = double(y) - y_hat;

%% ===== 梯度计算 =====
grad_W = -e * H';
scale  = (W .* H') ./ (Sigma .^ 2)';
grad_C = -e * bsxfun(@times, D, scale');
grad_S = -e * (W' .* H .* dist2 ./ (Sigma .^ 3));

%% ===== 带动量的参数更新 =====
dW = -eta_w * grad_W + mu_w * dW_prev;
dC = -eta_c * grad_C + mu_c * dC_prev;
dS = -eta_s * grad_S + mu_s * dSig_prev;

W     = W + dW;
C     = C + dC;
Sigma = max(Sigma + dS, sig_min);

dW_prev   = dW;
dC_prev   = dC;
dSig_prev = dS;

%% ===== 雅可比 ∂ŷ/∂u(k-1) =====
% 归一化空间的偏导 * gain(3) 还原到原始空间
dy_du_norm = sum(W' .* H .* (-D(:,3) ./ Sigma .^ 2));
dy_du = dy_du_norm * gain(3);

% 下限保护
if abs(dy_du) < 1e-4
    dy_du = sign(dy_du + eps) * 1e-4;
end

%% ===== 更新延迟寄存器 =====
y2 = y1;
y1 = double(y);
u2 = double(u_prev);

end
