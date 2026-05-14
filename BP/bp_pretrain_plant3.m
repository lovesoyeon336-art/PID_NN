clear; close all;

%% ==================== 预训练 —— 对象3（Hammerstein 非线性） ====================
IN = 4;   H = 5;   Out = 3;
rate  = 0.015;
rate2 = 0.01;
N_pretrain = 1000;
epochs = 8;

Kp_max = 3.0;  Ki_max = 0.5;  Kd_max = 3.0;
scale_vec = [Kp_max, Ki_max, Kd_max];
r_target = 1;

%% ==================== Xavier 权重初始化 ====================
rng(1);
w1 = sqrt(2/(IN+H))  * randn(H, IN);
w2 = sqrt(2/(H+Out)) * randn(Out, H);

for ep = 1:epochs
    y_1 = 0;  y_2 = 0;  u_1 = 0;
    error_1 = 0;  error_2 = 0;

    w1_1 = w1;  w1_2 = w1;
    w2_1 = w2;  w2_2 = w2;

    time   = zeros(1, N_pretrain);
    r      = zeros(1, N_pretrain);
    y      = zeros(1, N_pretrain);
    error  = zeros(1, N_pretrain);
    u      = zeros(1, N_pretrain);

    for k = 1:N_pretrain
        time(k) = k;
        r(k) = r_target;
        error(k) = r(k) - y_1;

        % 前向传播
        I1 = [r(k), y_1, error(k), 1];
        I2 = I1 * w1';
        O2 = tanh(I2);
        I3 = w2 * O2';
        I3_t = I3';
        O3 = zeros(1, Out);
        for l = 1:Out
            if I3_t(l) > 0, O3(l) = I3_t(l); else, O3(l) = 0.2 * I3_t(l); end
        end

        kp_k = Kp_max * O3(1);  ki_k = Ki_max * O3(2);  kd_k = Kd_max * O3(3);
        Kpid = [kp_k, ki_k, kd_k];

        e_pid = [error(k) - error_1; error(k); error(k) - 2*error_1 + error_2];
        delta_u = Kpid * e_pid;
        delta_u = max(-0.5, min(0.5, delta_u));
        u(k) = u_1 + delta_u;

        y(k) = plant_dynamics('plant3', y_1, y_2, u(k), u_1, k);
        error(k) = r(k) - y(k);

        % 反向传播
        dO3 = zeros(1, Out);
        for j = 1:Out
            if O3(j) > 0, dO3(j) = 1; else, dO3(j) = 0.2; end
        end

        du_sys = 1.333 * (1 + 0.6*u(k) - 0.3*u(k)^2);  % 稳态增益 × ∂v/∂u

        delta3 = zeros(1, Out);
        for l = 1:Out
            delta3(l) = error(k) * du_sys * scale_vec(l) * e_pid(l) * dO3(l);
        end

        d_w2 = zeros(Out, H);
        for l = 1:Out
            for i = 1:H
                d_w2(l, i) = rate * delta3(l) * O2(i);
            end
        end
        w2 = w2_1 + d_w2 + rate2 * 2 * (w2_1 - w2_2);

        dO2 = 1 - tanh(I2).^2;
        a_back = delta3 * w2;
        delta2 = dO2 .* a_back;
        d_w1 = rate * delta2' * I1;
        w1 = w1_1 + d_w1 + rate2 * (w1_1 - w1_2);

        u_1 = u(k);  y_2 = y_1;  y_1 = y(k);
        w2_2 = w2_1;  w2_1 = w2;
        w1_2 = w1_1;  w1_1 = w1;
        error_2 = error_1;
        error_1 = error(k);
    end
    fprintf('Epoch %d/%d  MAE: %.6f\n', ep, epochs, sum(abs(error)) / N_pretrain);
end

[script_dir, ~, ~] = fileparts(mfilename('fullpath'));
save(fullfile(script_dir, 'bp_pretrained_weights_plant3.mat'), 'w1', 'w2');
fprintf('权重已保存至 bp_pretrained_weights_plant3.mat\n');
