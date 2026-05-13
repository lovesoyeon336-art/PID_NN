clear; close all;

%% ==================== 超参数 ====================
IN = 4;   H = 5;   Out = 3;
rate  = 0.001;
rate2 = 0.01;
N_pretrain = 1000;              % 预训练步数
epochs = 3;                     % 训练轮数

Kp_max = 2.0;  Ki_max = 0.5;  Kd_max = 1.0;
scale_vec = [Kp_max, Ki_max, Kd_max];
r_target = 1;

%% ==================== Xavier 权重初始化 ====================
rng(1);
w1 = sqrt(2/(IN+H))  * randn(H, IN);
w2 = sqrt(2/(H+Out)) * randn(Out, H);

%% ==================== 多轮预训练 ====================
for ep = 1:epochs

    % ---- 重置 plant 状态 ----
    y_1 = 0;  u_1 = 0;
    error_1 = 0;  error_2 = 0;

    % ---- 动量缓存 ----
    w1_1 = w1;  w1_2 = w1;
    w2_1 = w2;  w2_2 = w2;

    % ---- 预分配 ----
    time   = zeros(1, N_pretrain);
    r      = zeros(1, N_pretrain);
    y      = zeros(1, N_pretrain);
    error  = zeros(1, N_pretrain);
    u      = zeros(1, N_pretrain);
    kp_arr = zeros(1, N_pretrain);
    ki_arr = zeros(1, N_pretrain);
    kd_arr = zeros(1, N_pretrain);

    for k = 1:N_pretrain
        time(k) = k;
        r(k) = r_target;
        error(k) = r(k) - y_1;

        % 前向传播
        I1 = [r(k), y_1, error(k), 1];
        I2 = I1 * w1';
        O2 = tanh(I2);

        I3 = w2 * O2';
        O3 = sigmoid(I3');

        kp_arr(k) = Kp_max * O3(1);
        ki_arr(k) = Ki_max * O3(2);
        kd_arr(k) = Kd_max * O3(3);
        Kpid = [kp_arr(k), ki_arr(k), kd_arr(k)];

        % PID
        e_pid = [error(k) - error_1;
                 error(k);
                 error(k) - 2*error_1 + error_2];
        delta_u = Kpid * e_pid;
        du_max = 0.5;
        delta_u = max(-du_max, min(du_max, delta_u));
        u(k) = u_1 + delta_u;

        % 被控对象
        a_k = 1.2 * (1 - 0.8 * exp(-0.1 * k));
        y(k) = a_k / (1 + y_1^2) * y_1 + u(k);
        error(k) = r(k) - y(k);

        % 反向传播
        dO3 = sigmoidGradient(O3);

        dydu = (y(k) - y_1) / (u(k) - u_1 + 0.0001);
        du_sys = max(-1, min(1, dydu));

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

        % 状态缓存
        u_1 = u(k);
        y_1 = y(k);
        w2_2 = w2_1;  w2_1 = w2;
        w1_2 = w1_1;  w1_1 = w1;
        error_2 = error_1;
        error_1 = error(k);
    end

    fprintf('Epoch %d/%d  MAE: %.6f\n', ep, epochs, sum(abs(error)) / N_pretrain);
end

%% ==================== 保存预训练权重 ====================
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));
save(fullfile(script_dir, 'bp_pretrained_weights.mat'), 'w1', 'w2');
fprintf('权重已保存至 bp_pretrained_weights.mat\n');
