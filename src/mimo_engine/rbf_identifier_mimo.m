function [J, y_hat] = rbf_identifier_mimo(u1_prev, u2_prev, y1, y2, mode)
% RBF 在线辨识器（MIMO 2×2）
% RBF 用于在线系统辨识（预测 y1_hat, y2_hat）并提供 2×2 Jacobian
% Jacobian 收敛于 D 矩阵: J≈[0, 0; 0.19, -1.31] (Model 4)

persistent W1 C1 Sigma1 W2 C2 Sigma2 ...
    dW1_prev dC1_prev dS1_prev dW2_prev dC2_prev dS2_prev ...
    y1_1 y1_2 y2_1 y2_2 u1_1 u1_2 u1_3 u2_1 u2_2 u2_3 ...
    gain1 gain2 init_flag freeze_flag

M = 10; n = 7;
eta_w = 0.01; eta_c = 0.008; eta_s = 0.005;
mu_w = 0.05; mu_c = 0.02; mu_s = 0.02;
sig_min = 0.5;

%% 初始化
is_load = ischar(mode) && (strcmp(mode, 'plant') || strcmp(mode, 'load') || strcmp(mode, 'freeze'));
if isempty(init_flag) || isequal(mode, true) || isequal(mode, 'reset') || is_load
    gain1 = [0.8; 0.5; 0.5; 0.5; 0.5; 0.5; 0.5];
    gain2 = [0.8; 0.5; 0.5; 0.5; 0.5; 0.5; 0.5];

    pretrain_file = fullfile(fileparts(mfilename('fullpath')), 'rbf_mimo_pretrained.mat');
    if isfile(pretrain_file)
        ld = load(pretrain_file, 'W1', 'C1', 'Sigma1', 'W2', 'C2', 'Sigma2');
        W1 = ld.W1; C1 = ld.C1; Sigma1 = ld.Sigma1;
        W2 = ld.W2; C2 = ld.C2; Sigma2 = ld.Sigma2;
        fprintf('  RBF MIMO loaded: %s\n', pretrain_file);
    else
        rng(0);
        W1 = 0.2 * ones(1, M) + 0.05 * randn(1, M);
        C1 = 2*(rand(M, n)-0.5);
        Sigma1 = 3.0 * ones(M, 1);
        W2 = 0.2 * ones(1, M) + 0.05 * randn(1, M);
        C2 = 2*(rand(M, n)-0.5);
        Sigma2 = 3.0 * ones(M, 1);
    end

    dW1_prev = zeros(1, M); dC1_prev = zeros(M, n); dS1_prev = zeros(M, 1);
    dW2_prev = zeros(1, M); dC2_prev = zeros(M, n); dS2_prev = zeros(M, 1);
    y1_1 = 0; y1_2 = 0; y2_1 = 0; y2_2 = 0;
    u1_1 = 0; u1_2 = 0; u1_3 = 0; u2_1 = 0; u2_2 = 0; u2_3 = 0;
    init_flag = true;

    if strcmp(mode, 'freeze'), freeze_flag = true; else, freeze_flag = false; end

    if ischar(mode)
        J = zeros(2); y_hat = [0; 0];
        return
    end
end

%% ===== RBF1 前向（预测 y1）=====
x1_raw = [y1_1; u1_prev; u1_1; u1_2; u2_prev; u2_1; u2_2];
x1 = x1_raw .* gain1;
D1 = bsxfun(@minus, x1', C1);
dist1_2 = sum(D1 .^ 2, 2);
H1 = exp(-dist1_2 ./ (2 * Sigma1 .^ 2));
y1_hat = W1 * H1;

%% ===== RBF2 前向（预测 y2）=====
x2_raw = [y2_1; u2_prev; u2_1; u2_2; u1_prev; u1_1; u1_2];
x2 = x2_raw .* gain2;
D2 = bsxfun(@minus, x2', C2);
dist2_2 = sum(D2 .^ 2, 2);
H2 = exp(-dist2_2 ./ (2 * Sigma2 .^ 2));
y2_hat = W2 * H2;

y_hat = [y1_hat; y2_hat];

%% ===== RBF Jacobian（2×2，权重更新前计算）=====
% RBF1: ∂y1/∂u1 (索引2), ∂y1/∂u2 (索引5)
d1_2 = -D1(:,2) ./ (Sigma1 .^ 2);
J11 = sum(W1' .* H1 .* d1_2) * gain1(2);
d1_5 = -D1(:,5) ./ (Sigma1 .^ 2);
J12 = sum(W1' .* H1 .* d1_5) * gain1(5);

% RBF2: ∂y2/∂u2 (索引2), ∂y2/∂u1 (索引5)
d2_2 = -D2(:,2) ./ (Sigma2 .^ 2);
J22 = sum(W2' .* H2 .* d2_2) * gain2(2);
d2_5 = -D2(:,5) ./ (Sigma2 .^ 2);
J21 = sum(W2' .* H2 .* d2_5) * gain2(5);

J = [J11, J12; J21, J22];

%% ===== RBF1 在线更新 =====
if ~freeze_flag
e1 = y1 - y1_hat;
grad_W1 = -e1 * H1';
scale1 = (W1 .* H1') ./ (Sigma1 .^ 2)';
grad_C1 = -e1 * bsxfun(@times, D1, scale1');
grad_S1 = -e1 * (W1' .* H1 .* dist1_2 ./ (Sigma1 .^ 3));

dW1 = -eta_w * grad_W1 + mu_w * dW1_prev;
dC1 = -eta_c * grad_C1 + mu_c * dC1_prev;
dS1 = -eta_s * grad_S1 + mu_s * dS1_prev;

W1 = W1 + dW1; C1 = C1 + dC1; Sigma1 = max(Sigma1 + dS1, sig_min);
dW1_prev = dW1; dC1_prev = dC1; dS1_prev = dS1;
end

%% ===== RBF2 在线更新 =====
if ~freeze_flag
e2 = y2 - y2_hat;
grad_W2 = -e2 * H2';
scale2 = (W2 .* H2') ./ (Sigma2 .^ 2)';
grad_C2 = -e2 * bsxfun(@times, D2, scale2');
grad_S2 = -e2 * (W2' .* H2 .* dist2_2 ./ (Sigma2 .^ 3));

dW2 = -eta_w * grad_W2 + mu_w * dW2_prev;
dC2 = -eta_c * grad_C2 + mu_c * dC2_prev;
dS2 = -eta_s * grad_S2 + mu_s * dS2_prev;

W2 = W2 + dW2; C2 = C2 + dC2; Sigma2 = max(Sigma2 + dS2, sig_min);
dW2_prev = dW2; dC2_prev = dC2; dS2_prev = dS2;
end

%% ===== 延迟寄存器更新 =====
y1_2 = y1_1; y1_1 = y1;
y2_2 = y2_1; y2_1 = y2;
u1_3 = u1_2; u1_2 = u1_1; u1_1 = u1_prev;
u2_3 = u2_2; u2_2 = u2_1; u2_1 = u2_prev;
end
