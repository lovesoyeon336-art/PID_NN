% rbf_engine_pretrain.m
% RBF 离线预训练 — Engine Model 4
% 改进版：宽激励范围 + 自动验证 Jacobian 质量
% 面向发动机状态空间模型的 I/O 辨识

clear; clc;
fprintf('===== RBF Engine Pretrain (改进版) =====\n');

%% 参数
N_excite = 10000; epochs = 50;
M = 10; n = 7;
gain1 = [0.8;0.5;0.5;0.5;0.5;0.5;0.5];
gain2 = [0.8;0.5;0.5;0.5;0.5;0.5;0.5];
eta_w = 0.005; eta_c = 0.004; eta_s = 0.003;
mu_w = 0.05; mu_c = 0.02; mu_s = 0.02;
sig_min = 0.5;

%% 激励信号（滤波随机噪声，无方波，±10 覆盖实际控制工作区）
rng(1);
u1_ex = zeros(1,N_excite); u2_ex = zeros(1,N_excite);
y1_ex = zeros(1,N_excite); y2_ex = zeros(1,N_excite);
y1_1=0;y2_1=0; u1_1=0;u1_2=0;u1_3=0; u2_1=0;u2_2=0;u2_3=0;

alpha_u = 0.12;  % 一阶低通平滑系数 (fc≈2Hz @ Ts=0.01s)
u1_ex(1) = (rand - 0.5) * 20;  % ±10
u2_ex(1) = (rand - 0.5) * 20;
for k = 2:N_excite
    u1_raw = (rand - 0.5) * 20;
    u2_raw = (rand - 0.5) * 20;
    u1_ex(k) = alpha_u * u1_raw + (1 - alpha_u) * u1_ex(k-1);
    u2_ex(k) = alpha_u * u2_raw + (1 - alpha_u) * u2_ex(k-1);
end

for k = 1:N_excite
    [y1_ex(k), y2_ex(k)] = plant_engine_mimo(y1_1,y2_1,u1_1,u1_2,u1_3,u2_1,u2_2,u2_3);
    u1_3=u1_2;u1_2=u1_1;u1_1=u1_ex(k);
    u2_3=u2_2;u2_2=u2_1;u2_1=u2_ex(k);
    y1_1=y1_ex(k); y2_1=y2_ex(k);
end

%% 样本
N_sample = N_excite - 3;
X1 = zeros(N_sample,n); Y1 = zeros(N_sample,1);
X2 = zeros(N_sample,n); Y2 = zeros(N_sample,1);
for k = 4:N_excite
    idx = k - 3;
    X1(idx,:) = [y1_ex(k-1),u1_ex(k-1),u1_ex(k-2),u1_ex(k-3),u2_ex(k-1),u2_ex(k-2),u2_ex(k-3)];
    Y1(idx) = y1_ex(k);
    X2(idx,:) = [y2_ex(k-1),u2_ex(k-1),u2_ex(k-2),u2_ex(k-3),u1_ex(k-1),u1_ex(k-2),u1_ex(k-3)];
    Y2(idx) = y2_ex(k);
end

%% RBF1 (预测 y1)
fprintf('--- RBF1 (y1) ---\n');
rng(0); W1 = 0.2*ones(1,M) + 0.05*randn(1,M);
C1 = 4*(rand(M,n)-0.5); Sigma1 = 5.0*ones(M,1);  % 更宽的核范围
dW_prev=zeros(1,M); dC_prev=zeros(M,n); dS_prev=zeros(M,1);
for ep = 1:epochs
    idx_p = randperm(N_sample); mae_ep = 0;
    for s = 1:N_sample
        i = idx_p(s); x = X1(i,:)' .* gain1; y_target = Y1(i);
        D = bsxfun(@minus,x',C1); dist2 = sum(D.^2,2);
        H = exp(-dist2./(2*Sigma1.^2)); y_hat = W1*H;
        e_pred = y_target - y_hat; mae_ep = mae_ep + abs(e_pred);
        grad_W = -e_pred * H';
        scale_p = (W1 .* H') ./ (Sigma1.^2)';
        grad_C = -e_pred * bsxfun(@times,D,scale_p');
        grad_S = -e_pred * (W1' .* H .* dist2 ./ (Sigma1.^3));
        dW = -eta_w*grad_W + mu_w*dW_prev;
        dC = -eta_c*grad_C + mu_c*dC_prev;
        dS = -eta_s*grad_S + mu_s*dS_prev;
        W1 = W1 + dW; C1 = C1 + dC; Sigma1 = max(Sigma1 + dS, sig_min);
        dW_prev=dW; dC_prev=dC; dS_prev=dS;
    end
    fprintf('  Ep %2d  MAE=%.5f\n', ep, mae_ep/N_sample);
end

%% RBF2 (预测 y2)
fprintf('--- RBF2 (y2) ---\n');
rng(0); W2 = 0.2*ones(1,M) + 0.05*randn(1,M);
C2 = 4*(rand(M,n)-0.5); Sigma2 = 5.0*ones(M,1);
dW_prev=zeros(1,M); dC_prev=zeros(M,n); dS_prev=zeros(M,1);
for ep = 1:epochs
    idx_p = randperm(N_sample); mae_ep = 0;
    for s = 1:N_sample
        i = idx_p(s); x = X2(i,:)' .* gain2; y_target = Y2(i);
        D = bsxfun(@minus,x',C2); dist2 = sum(D.^2,2);
        H = exp(-dist2./(2*Sigma2.^2)); y_hat = W2*H;
        e_pred = y_target - y_hat; mae_ep = mae_ep + abs(e_pred);
        grad_W = -e_pred * H';
        scale_p = (W2 .* H') ./ (Sigma2.^2)';
        grad_C = -e_pred * bsxfun(@times,D,scale_p');
        grad_S = -e_pred * (W2' .* H .* dist2 ./ (Sigma2.^3));
        dW = -eta_w*grad_W + mu_w*dW_prev;
        dC = -eta_c*grad_C + mu_c*dC_prev;
        dS = -eta_s*grad_S + mu_s*dS_prev;
        W2 = W2 + dW; C2 = C2 + dC; Sigma2 = max(Sigma2 + dS, sig_min);
        dW_prev=dW; dC_prev=dC; dS_prev=dS;
    end
    fprintf('  Ep %2d  MAE=%.5f\n', ep, mae_ep/N_sample);
end

%% 保存
data_dir = fileparts(mfilename('fullpath'));
save(fullfile(data_dir,'rbf_mimo_pretrained.mat'),'W1','C1','Sigma1','W2','C2','Sigma2');
fprintf('  -> %s\n', fullfile(data_dir,'rbf_mimo_pretrained.mat'));

%% ===== 自动验证 =====
fprintf('\n===== 预训练验证 =====\n');

% 生成独立测试信号（滤波随机噪声，无方波，u ∈ [−8, 8]）
N_test = 1000;
rng(99);
clear plant_engine_mimo;
u1_test = zeros(1,N_test); u2_test = zeros(1,N_test);
y1_test = zeros(1,N_test); y2_test = zeros(1,N_test);
y1_1=0;y2_1=0; u1_1=0;u1_2=0;u1_3=0; u2_1=0;u2_2=0;u2_3=0;

alpha_v = 0.10;  % 略小于训练集，产生更平滑的测试信号
u1_test(1) = (rand - 0.5) * 16;  % ±8
u2_test(1) = (rand - 0.5) * 16;
for k = 2:N_test
    u1_raw = (rand - 0.5) * 16;
    u2_raw = (rand - 0.5) * 16;
    u1_test(k) = alpha_v * u1_raw + (1 - alpha_v) * u1_test(k-1);
    u2_test(k) = alpha_v * u2_raw + (1 - alpha_v) * u2_test(k-1);
end

for k = 1:N_test
    [y1_test(k),y2_test(k)]=plant_engine_mimo(y1_1,y2_1,u1_1,u1_2,u1_3,u2_1,u2_2,u2_3);
    u1_3=u1_2;u1_2=u1_1;u1_1=u1_test(k);
    u2_3=u2_2;u2_2=u2_1;u2_1=u2_test(k);
    y1_1=y1_test(k); y2_1=y2_test(k);
end

% 加载预训练权重，做预测
rbf_identifier_mimo(0,0,0,0,'plant');
y1_hat=zeros(1,N_test); y2_hat=zeros(1,N_test);
J11_a=zeros(1,N_test); J12_a=zeros(1,N_test);
J21_a=zeros(1,N_test); J22_a=zeros(1,N_test);

for k = 1:N_test
    if k == 1, u1_p=0; u2_p=0; else, u1_p=u1_test(k-1); u2_p=u2_test(k-1); end
    [J_tmp, y_tmp] = rbf_identifier_mimo(u1_p, u2_p, y1_test(k), y2_test(k), false);
    y1_hat(k) = y_tmp(1); y2_hat(k) = y_tmp(2);
    J11_a(k) = J_tmp(1,1); J12_a(k) = J_tmp(1,2);
    J21_a(k) = J_tmp(2,1); J22_a(k) = J_tmp(2,2);
end

mae1 = mean(abs(y1_test - y1_hat));
mae2 = mean(abs(y2_test - y2_hat));
rmse1 = sqrt(mean((y1_test - y1_hat).^2));
rmse2 = sqrt(mean((y2_test - y2_hat).^2));

fprintf('预测精度 (N=%d, u∈[−8,8]):\n', N_test);
fprintf('  y1: MAE=%.4f  RMSE=%.4f\n', mae1, rmse1);
fprintf('  y2: MAE=%.4f  RMSE=%.4f\n', mae2, rmse2);

fprintf('Jacobian 均值 vs 理论 D 矩阵:\n');
fprintf('  J11 = %+.4f  (理论 D(1,1) = 0)\n', mean(J11_a));
fprintf('  J12 = %+.4f  (理论 D(1,2) = 0)\n', mean(J12_a));
fprintf('  J21 = %+.4f  (理论 D(2,1) = +0.190)\n', mean(J21_a));
fprintf('  J22 = %+.4f  (理论 D(2,2) = -1.310)\n', mean(J22_a));

fprintf('Jacobian 标准差:\n');
fprintf('  J11: %.4f  J12: %.4f  J21: %.4f  J22: %.4f\n', ...
    std(J11_a), std(J12_a), std(J21_a), std(J22_a));

J21_err = abs(mean(J21_a) - 0.190) / 0.190 * 100;
J22_err = abs(mean(J22_a) - (-1.310)) / 1.310 * 100;
fprintf('偏差: J21=%.1f%%  J22=%.1f%%  %s\n', J21_err, J22_err, ...
    cond_text(J21_err < 30 && J22_err < 30));

%% 保存验证数据供 Python 出图 (fig1_rbf_validation)
t_test = (0:N_test-1) * 0.01;
r2_1 = corr(y1_test', y1_hat')^2;
r2_2 = corr(y2_test', y2_hat')^2;
y1_t = y1_test; y1_h = y1_hat;
y2_t = y2_test; y2_h = y2_hat;
J21 = J21_a; J22 = J22_a;
save(fullfile(data_dir, 'figdata_rbf.mat'), ...
    't_test', 'y1_t', 'y1_h', 'y2_t', 'y2_h', ...
    'J21', 'J22', 'mae1', 'mae2', 'r2_1', 'r2_2', 'J21_err', 'J22_err');
fprintf('  -> %s\n', fullfile(data_dir, 'figdata_rbf.mat'));

function s = cond_text(ok)
    if ok, s = '达标'; else, s = '需改进'; end
end
