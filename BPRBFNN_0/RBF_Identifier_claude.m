function [dy_du, y_hat] = rbf_online_identification(u_prev, y, reset)
% RBF 神经网络在线辨识 —— Simulink MATLAB Function Block
%
% 输入:
%   u_prev —— u(k-1)，控制量经 Simulink Unit Delay 后接入，避免代数环
%   y      —— y(k)，被控对象当前输出
%   reset  —— 复位信号，接常数 0 即可
%
% 输出:
%   dy_du  —— 雅可比 ∂y/∂u，接 BP-PID 反传入口
%   y_hat  —— 预测输出，接 Scope 监控或 Terminator

persistent W C Sigma dW_prev dC_prev dSig_prev y1 y2 u2 init_flag

%% ===== 超参数 =====
M       = 8;      % RBF 隐层节点数
n       = 4;      % 输入维度
eta_w   = 0.30;   % W 学习率
eta_c   = 0.05;   % C 学习率
eta_s   = 0.05;   % σ 学习率
mu_w    = 0.05;   % W 动量系数
mu_c    = 0.02;   % C 动量系数
mu_s    = 0.02;   % σ 动量系数
sig_min = 0.01;   % σ 下限

%% ===== 初始化 =====
if isempty(init_flag) || reset > 0.5
    W         = 0.1 * randn(1, M);
    C         = repmat(linspace(-1, 1, M)', 1, n) + 0.1 * randn(M, n);
    Sigma     = 2.0 * ones(M, 1);
    dW_prev   = zeros(1, M);
    dC_prev   = zeros(M, n);
    dSig_prev = zeros(M, 1);
    y1 = 0; y2 = 0; u2 = 0;
    init_flag = true;
end

%% ===== 构造输入向量 =====
% x(k) = [y(k-1), y(k-2), u(k-1), u(k-2)]
% u(k-1) 直接来自输入端口，u(k-2) 用 persistent 变量 u2 存储
x = [y1; y2; double(u_prev); u2];   % 4×1

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

W         = W + dW;
C         = C + dC;
Sigma     = max(Sigma + dS, sig_min);

dW_prev   = dW;
dC_prev   = dC;
dSig_prev = dS;

%% ===== 计算雅可比 ∂ŷ/∂u(k-1) =====
% u(k-1) 在 x 中是第 3 个分量，对应 D 的第 3 列
dy_du = sum(W' .* H .* (-D(:,3) ./ Sigma .^ 2));

if abs(dy_du) < 1e-4
    dy_du = sign(dy_du + eps) * 1e-4;
end

%% ===== 更新延迟寄存器 =====
y2 = y1;           y1 = double(y);
u2 = double(u_prev);   % u(k-1) 下一拍变成 u(k-2)