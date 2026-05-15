clear; close all;

%% ==================== RBF 离线辨识预训练（多目标：预测 + Jacobian 匹配） ====================

N_excite = 4000;
epochs = 20;

%% ==================== RBF 超参 ====================
M       = 10;
n       = 4;         % [y(k-1), y(k-2), u(k-1), u(k-2)]
gain    = [0.8; 0.8; 0.5; 0.5];
eta_w   = 0.01;   eta_c  = 0.008;  eta_s  = 0.005;
mu_w    = 0.05;   mu_c   = 0.02;   mu_s   = 0.02;
sig_min = 0.5;    % 增大下限，避免过宽的基函数吞掉 u 敏感度
lambda_jac = 0.50; % Jacobian 匹配权重（提高以改善闭环梯度精度）

for plant_idx = 1:2
    pid = sprintf('plant%d', plant_idx);
    fprintf('\n===== RBF 离线训练: 对象%d =====\n', plant_idx);

    %% ---- ① 激励信号 ----
    rng(plant_idx);
    u_ex = zeros(1, N_excite);
    y_ex = zeros(1, N_excite);
    y1 = 0; y2 = 0;

    hold_steps = 0;
    u_val = 0;
    for k = 1:N_excite
        if hold_steps <= 0
            u_val = (rand - 0.5) * 3.0;
            hold_steps = 20 + randi(40);
        end
        u_ex(k) = u_val;
        hold_steps = hold_steps - 1;
        y_ex(k) = plant_dynamics(pid, y1, y2, u_ex(k), max(k>1)*u_ex(max(k-1,1)), k);
        y2 = y1; y1 = y_ex(k);
    end

    %% ---- ② 训练样本 + 真实 Jacobian ----
    X = zeros(N_excite - 2, n);
    Y = zeros(N_excite - 2, 1);
    J = zeros(N_excite - 2, 1);  % 有限差分 Jacobian ground truth
    for k = 3:N_excite
        X(k-2, :) = [y_ex(k-1), y_ex(k-2), u_ex(k-1), u_ex(k-2)];
        Y(k-2) = y_ex(k);
        J(k-2) = (y_ex(k) - y_ex(k-1)) / (u_ex(k-1) - u_ex(k-2) + 0.001);
    end
    J = max(-1, min(1, J));  % clamp 同在线 du_sys
    N_sample = size(X, 1);

    %% ---- ③ 初始化 ----
    rng(0);
    W     = 0.2 * ones(1, M) + 0.05 * randn(1, M);
    C     = 2 * (rand(M, n) - 0.5);
    Sigma = 1.0 * ones(M, 1);   % 初始 σ=1（原 3.0 太宽）
    dW_prev = zeros(1, M);
    dC_prev = zeros(M, n);
    dSig_prev = zeros(M, 1);

    %% ---- ④ 多目标训练 ----
    for ep = 1:epochs
        idx = randperm(N_sample);
        mae_ep = 0;
        mae_jac_ep = 0;

        for s = 1:N_sample
            i = idx(s);
            x_raw = X(i, :)';
            x = x_raw .* gain;
            y_target = Y(i);
            j_target = J(i);

            % 前向
            D     = bsxfun(@minus, x', C);                       % M×n
            dist2 = sum(D .^ 2, 2);
            H     = exp(-dist2 ./ (2 * Sigma .^ 2));
            y_hat = W * H;
            e_pred = y_target - y_hat;
            mae_ep = mae_ep + abs(e_pred);

            % RBF Jacobian: ∂ŷ/∂u(k-1)
            d_j3  = -D(:,3) ./ (Sigma .^ 2);                      % M×1
            jac_rbf = sum(W' .* H .* d_j3) * gain(3);
            e_jac = jac_rbf - j_target;
            mae_jac_ep = mae_jac_ep + abs(e_jac);

            %% 预测损失梯度
            grad_W_pred = -e_pred * H';
            scale_pred  = (W .* H') ./ (Sigma .^ 2)';
            grad_C_pred = -e_pred * bsxfun(@times, D, scale_pred'); % M×n
            grad_S_pred = -e_pred * (W' .* H .* dist2 ./ (Sigma .^ 3));

            %% Jacobian 匹配损失梯度
            % dL_jac/dW: ∂/∂W_j (jac_rbf) = H_j * d_j3
            grad_W_jac = 2 * lambda_jac * e_jac * (H .* d_j3)' * gain(3);

            % dL_jac/dC
            grad_C_jac = zeros(M, n);
            for jj = 1:M
                d_jl_vec = -D(jj,:)' ./ (Sigma(jj)^2);            % n×1
                d_jl_vec(3) = d_j3(jj);
                dHdc = H(jj) .* d_jl_vec;                          % ∂H_j/∂c_j
                dd3dc = zeros(n, 1);
                dd3dc(3) = 1 / (Sigma(jj)^2);                      % ∂d_j3/∂c_j3
                djac_dc = W(jj) * (dHdc * d_j3(jj) + H(jj) * dd3dc);
                grad_C_jac(jj, :) = 2 * lambda_jac * e_jac * djac_dc' * gain(3);
            end

            % dL_jac/dSigma
            grad_S_jac = zeros(M, 1);
            for jj = 1:M
                djac_ds = W(jj) * H(jj) * d_j3(jj) * (dist2(jj)/(Sigma(jj)^3) - 2/Sigma(jj));
                grad_S_jac(jj) = 2 * lambda_jac * e_jac * djac_ds * gain(3);
            end

            %% 合并梯度 + 动量更新
            grad_W = grad_W_pred + grad_W_jac;
            grad_C = grad_C_pred + grad_C_jac;
            grad_S = grad_S_pred + grad_S_jac;

            dW = -eta_w * grad_W + mu_w * dW_prev;
            dC = -eta_c * grad_C + mu_c * dC_prev;
            dS = -eta_s * grad_S + mu_s * dSig_prev;

            W     = W + dW;
            C     = C + dC;
            Sigma = max(Sigma + dS, sig_min);

            dW_prev = dW; dC_prev = dC; dSig_prev = dS;
        end
        fprintf('  Epoch %d/%d  MAE_pred=%.5f  MAE_jac=%.4f\n', ...
            ep, epochs, mae_ep/N_sample, mae_jac_ep/N_sample);
    end

    %% ---- ⑤ 保存 ----
    [script_dir, ~, ~] = fileparts(mfilename('fullpath'));
    save(fullfile(script_dir, sprintf('rbf_pretrained_params_plant%d.mat', plant_idx)), ...
        'W', 'C', 'Sigma');
    fprintf('  参数已保存至 rbf_pretrained_params_plant%d.mat\n', plant_idx);
end

fprintf('\n===== RBF 离线预训练全部完成 =====\n');
