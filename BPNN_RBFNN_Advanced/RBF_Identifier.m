function [dy_du, y_hat] = RBF_Identifier(u_prev, y)
% =========================================================================
% RBF Neural Network Online System Identifier
% =========================================================================
% 网络结构: 4输入 → M个RBF节点(高斯核) → 1输出
%
% 输入:
%   u_prev - u(k-1), 经Unit Delay的控制量 (避免代数环)
%   y      - y(k), 被控对象当前输出
%
% 输出:
%   dy_du  - 雅可比 ∂ŷ/∂u(k-1), 供BP-PID反传使用
%   y_hat  - RBF网络预测输出, 用于监控辨识精度
%
% 输入向量: x(k) = [y(k-1), y(k-2), u(k-1), u(k-2)]
%   u(k-1)来自外部端口, u(k-2)由persistent延迟寄存器提供
%   y(k-1), y(k-2)同理
%
% 在线学习: 每步基于辨识误差e_id = y - ŷ的梯度下降
% =========================================================================

%% ===== 超参数 =====
M       = 10;     % RBF隐层节点数
n       = 4;      % 输入维度 [y(k-1), y(k-2), u(k-1), u(k-2)]
eta_w   = 0.35;   % 输出权值 W 学习率
eta_c   = 0.06;   % 中心 C 学习率
eta_s   = 0.06;   % 宽度 σ 学习率
mu_w    = 0.05;   % W 动量系数
mu_c    = 0.03;   % C 动量系数
mu_s    = 0.03;   % σ 动量系数
sig_min = 0.01;   % σ 下界 (防止除零和数值不稳定)

%% ===== 持久变量 =====
persistent W C Sigma
persistent dW_prev dC_prev dSig_prev
persistent y1 y2 u2          % 延迟寄存器
persistent init_flag

%% ===== 初始化 =====
if isempty(init_flag)
    % W: 小随机值, 正负各半
    W = 0.15 * randn(1, M);
    % C: 在输入空间均匀分布 + 随机扰动
    C = repmat(linspace(-1.2, 1.2, M)', 1, n) + 0.1 * randn(M, n);
    % Sigma: 初始宽度覆盖输入空间
    Sigma = 1.5 * ones(M, 1);
    % 动量项初始化为零
    dW_prev   = zeros(1, M);
    dC_prev   = zeros(M, n);
    dSig_prev = zeros(M, 1);
    % 延迟寄存器
    y1 = 0;  y2 = 0;  u2 = 0;
    init_flag = true;
end

%% ===== 构造输入向量 =====
% x = [y(k-1); y(k-2); u(k-1); u(k-2)]
x = [y1; y2; double(u_prev); u2];   % 4×1

%% ===== 前向传播 =====
% D: M×n, 每行是输入向量到第i个中心的欧氏距离分量
D = bsxfun(@minus, x', C);              % M×n  (x' is 1×n, C is M×n)
dist2 = sum(D .^ 2, 2);                 % M×1  平方欧氏距离
H = exp(-dist2 ./ (2 * Sigma .^ 2));    % M×1  高斯核激活值
y_hat = W * H;                          % 标量  网络预测输出

%% ===== 辨识误差 =====
e_id = double(y) - y_hat;

%% ===== 梯度计算 =====
% ∂E/∂W = -e_id * H^T,  E=0.5*e_id^2
grad_W = -e_id * H';                    % 1×M

% ∂E/∂C 和 ∂E/∂σ 的公共缩放因子
scale = (W .* H') ./ (Sigma .^ 2)';     % 1×M

% ∂E/∂C: 利用bsxfun逐元素乘
grad_C = -e_id * bsxfun(@times, D, scale');    % M×n

% ∂E/∂σ = -e_id * W' .* H .* dist2 ./ σ^3
grad_S = -e_id * (W' .* H .* dist2 ./ (Sigma .^ 3));   % M×1

%% ===== 带动量的参数更新 =====
dW   = -eta_w * grad_W + mu_w * dW_prev;
dC   = -eta_c * grad_C + mu_c * dC_prev;
dS   = -eta_s * grad_S + mu_s * dSig_prev;

W     = W + dW;
C     = C + dC;
Sigma = max(Sigma + dS, sig_min);       % 保证σ>0

% 保存动量项
dW_prev   = dW;
dC_prev   = dC;
dSig_prev = dS;

%% ===== 计算雅可比 ∂ŷ/∂u(k-1) =====
% u(k-1) 在输入向量 x 中是第3个分量, 对应 D 的第3列
% ∂ŷ/∂u = Σ W_i * H_i * (-(u - c_{i,u}) / σ_i^2)
dy_du = sum(W' .* H .* (-D(:,3) ./ Sigma .^ 2));

% 雅可比下限: 避免梯度为零导致BP-PID停止学习
if abs(dy_du) < 1e-4
    dy_du = sign(dy_du + eps) * 1e-4;
end

%% ===== 更新延迟寄存器 =====
y2 = y1;
y1 = double(y);
u2 = double(u_prev);     % 当前u(k-1)在下一步变为u(k-2)

end
