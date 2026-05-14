clear; close all;

%% ==================== 超参数 ====================
IN = 5;   H = 5;   Out = 3;
rate  = 0.005;
rate2 = 0.01;
N_pretrain = 1000;              % 预训练步数
epochs = 7;                     % 前4轮斜坡+第5-6轮阶跃+后1轮多样化

Kp_max = 2.0;  Ki_max = 2.0;  Kd_max = 1.0;
scale_vec = [Kp_max, Ki_max, Kd_max];

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
        r(k) = get_pretrain_r(k, N_pretrain, ep, epochs);
        error(k) = r(k) - y_1;

        % 前向传播
        I1 = [r(k), y_1, error(k), error_1, 1];
        I2 = I1 * w1';
        O2 = tanh(I2);

        I3 = w2 * O2';
        I3_t = I3';
        O3 = zeros(1, Out);
        for l = 1:Out
            if I3_t(l) > 0
                O3(l) = I3_t(l);
            else
                O3(l) = 0.2 * I3_t(l);
            end
        end

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
        y(k) = plant_dynamics('plant1', y_1, 0, u(k), u_1, k);
        error(k) = r(k) - y(k);

        % 反向传播（误差死区）
        dead_zone = 0.01;
        if abs(error(k)) >= dead_zone
        dO3 = zeros(1, Out);
        for j = 1:Out
            if O3(j) > 0
                dO3(j) = 1;
            else
                dO3(j) = 0.2;
            end
        end

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
        end  % 误差死区

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

%% ==================== 渐进式参考信号 ====================
function r_k = get_pretrain_r(k, N_total, ep, epochs)
    if ep <= 4                     % 前4轮：纯斜坡学积分
        r_k = 0.5 + 1.5 * min(1, k / N_total);
        return
    elseif ep <= 6                 % 第5-6轮：纯阶跃学比例
        r_k = 1;
        return
    end
    phase = k / N_total;           % 第7轮：多样化泛化
    if phase < 0.2
        r_k = 1;
    elseif phase < 0.4
        r_k = 1 + 0.5 * sin(2*pi*0.01*k);
    elseif phase < 0.6
        persistent rand_val rand_hold
        if isempty(rand_val) || rand_hold <= 0
            rand_val = 0.5 + 1.5 * rand();
            rand_hold = 80 + randi(70);
        end
        rand_hold = rand_hold - 1;
        r_k = rand_val;
    elseif phase < 0.8
        local_k = k - round(0.6 * N_total);
        local_N = round(0.2 * N_total);
        r_k = 0.5 + 1.5 * min(1, local_k / max(local_N, 1));
    else
        r_k = 1 + 0.3 * sin(2*pi*0.03*k);
    end
end
