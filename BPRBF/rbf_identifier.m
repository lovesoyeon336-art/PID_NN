function [dy_du, y_hat] = rbf_identifier(u_prev, y, reset)
% RBF在线辨识器 — 为 BP-PID 提供解析 Jacobian ∂ŷ/∂u
% 输入:
%   u_prev — u(k-1)，上一时刻控制量
%   y      — y(k)，被控对象当前输出
%   reset  — true 时重新初始化 RBF 参数
% 输出:
%   dy_du  — 雅可比 ∂y/∂u（解析梯度，不夹 dead-zone 以外的 clamp）
%   y_hat  — RBF 一步预测输出

persistent W C Sigma dW_prev dC_prev dSig_prev y1 y2 u2 gain init_flag

%% ===== 超参数 =====
M       = 10;
n       = 4;       % [y(k-1), y(k-2), u(k-1), u(k-2)]
eta_w   = 0.01;
eta_c   = 0.008;
eta_s   = 0.005;
mu_w    = 0.05;
mu_c    = 0.02;
mu_s    = 0.02;
sig_min = 0.1;

%% ===== 初始化 =====
if isempty(init_flag) || reset
    gain = [0.8; 0.8; 0.5; 0.5];  % u 归一化适配 ±2 范围
    rng(0);
    W     = 0.2 * ones(1, M) + 0.05 * randn(1, M);  % 正偏向初始 Jacobian > 0
    C     = 2*(rand(M, n)-0.5);
    Sigma = 3.0 * ones(M, 1);  % 初始宽度更大，覆盖更广
    dW_prev   = zeros(1, M);
    dC_prev   = zeros(M, n);
    dSig_prev = zeros(M, 1);
    y1 = 0; y2 = 0; u2 = 0;
    init_flag = true;
end

%% ===== 构造归一化输入 =====
x_raw = [y1; y2; u_prev; u2];
x = x_raw .* gain;

%% ===== 前向传播 =====
D     = bsxfun(@minus, x', C);              % M×n
dist2 = sum(D .^ 2, 2);                     % M×1
H     = exp(-dist2 ./ (2 * Sigma .^ 2));    % M×1 高斯激活
y_hat = W * H;                              % 标量预测

%% ===== 辨识误差 =====
e = y - y_hat;

%% ===== 梯度 =====
grad_W = -e * H';
scale  = (W .* H') ./ (Sigma .^ 2)';
grad_C = -e * bsxfun(@times, D, scale');
grad_S = -e * (W' .* H .* dist2 ./ (Sigma .^ 3));

%% ===== 带动量更新 =====
dW = -eta_w * grad_W + mu_w * dW_prev;
dC = -eta_c * grad_C + mu_c * dC_prev;
dS = -eta_s * grad_S + mu_s * dSig_prev;

W     = W + dW;
C     = C + dC;
Sigma = max(Sigma + dS, sig_min);

dW_prev   = dW;
dC_prev   = dC;
dSig_prev = dS;

%% ===== Jacobian ∂ŷ/∂u(k-1) =====
dy_du_norm = sum(W' .* H .* (-D(:,3) ./ Sigma .^ 2));
dy_du = dy_du_norm * gain(3);

% 下限保护
if abs(dy_du) < 1e-4
    dy_du = sign(dy_du + eps) * 1e-4;
end

%% ===== 延迟寄存器 =====
y2 = y1;
y1 = y;
u2 = u_prev;
end
