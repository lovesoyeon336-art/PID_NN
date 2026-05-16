clear; close all;

%% ==================== 超参数 ====================
IN = 5;   H = 5;   Out = 3;
rate  = 0.003;  % 预训练保持保守
rate2 = 0.01;
N_pretrain = 2000;              % 预训练步数（翻倍增加样本）
epochs = 15;                    % 前5轮斜坡筑基 + 后10轮多样化泛化

Kp_max = 2.0;  Ki_max = 0.2;  Kd_max = 0.0;
scale_vec = [Kp_max, Ki_max, Kd_max];
ff_gain = 0.0;  beta_sp = 1.00;

%% ==================== Xavier 权重初始化 ====================
rng(1);
w1 = sqrt(2/(IN+H))  * randn(H, IN);
w2 = sqrt(2/(H+Out)) * randn(Out, H);

%% ==================== 多轮预训练 ====================
for ep = 1:epochs

    % ---- 重置 plant 状态 ----
    y_1 = 0;  u_1 = 0;  u_2 = 0;  r_1 = 0;
    error_1 = 0;  error_2 = 0;  e_sp_1 = 0;  e_sp_2 = 0;
    st_has = false;  st_ep = zeros(3,1);  st_O2 = zeros(1,H);
    st_dO3 = zeros(1,Out);  st_dO2 = zeros(1,H);  st_I1 = zeros(1,IN);

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
        I1 = [r(k), y_1, error(k), r(k)-r_1, 1];
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
        e_sp_k = beta_sp * r(k) - y_1;
        e_pid = [e_sp_k - e_sp_1;
                 error(k);
                 e_sp_k - 2*e_sp_1 + e_sp_2];
        delta_u = Kpid * e_pid;
        dr = r(k) - r_1;
        if abs(dr) <= 0.1, delta_u = delta_u + ff_gain * dr; end
        du_max = 1.0;
        delta_u = max(-du_max, min(du_max, delta_u));
        u(k) = u_1 + delta_u;

        % 被控对象 (y(k) 由 u(k-1) 驱动)
        y(k) = plant_dynamics('plant1', y_1, 0, u_1, u_1, k);
        error(k) = r(k) - y(k);

        % 延时反向传播（用上一拍存储状态 + 当前误差）
        dead_zone = 0.002;
        if st_has && abs(error(k)) >= dead_zone
        dydu_raw = (y(k) - y_1) / (u_1 - u_2 + 0.0001);
        du_sys = sign(dydu_raw) * 1.0;  % 符号 Jacobian

        delta3 = zeros(1, Out);
        for l = 1:Out
            e_grad = max(-0.5, min(0.5, error(k)));
            delta3(l) = e_grad * du_sys * scale_vec(l) * st_ep(l) * st_dO3(l);
        end

        d_w2 = zeros(Out, H);
        for l = 1:Out
            for i = 1:H
                d_w2(l, i) = rate * delta3(l) * st_O2(i);
            end
        end
        w2 = w2_1 + d_w2 + rate2 * 2 * (w2_1 - w2_2);

        a_back = delta3 * w2_1;
        delta2 = st_dO2 .* a_back;
        d_w1 = rate * delta2' * st_I1;
        w1 = w1_1 + d_w1 + rate2 * 2 * (w1_1 - w1_2);

        w2_2 = w2_1;  w2_1 = w2;
        w1_2 = w1_1;  w1_1 = w1;
        end  % 误差死区

        % 存储当前步状态
        st_has = true;
        st_ep  = e_pid;
        st_O2  = O2;
        st_I1  = I1;
        dO3_t = zeros(1, Out);
        for j = 1:Out
            if O3(j) > 0, dO3_t(j) = 1; else, dO3_t(j) = 0.2; end
        end
        st_dO3 = dO3_t;
        st_dO2 = 1 - tanh(I2).^2;

        % 状态缓存
        u_2 = u_1;
        u_1 = u(k);
        y_1 = y(k);
        r_1 = r(k);
        error_2 = error_1;
        error_1 = error(k);
        e_sp_2 = e_sp_1;
        e_sp_1 = e_sp_k;
    end

    fprintf('Epoch %d/%d  MAE: %.6f\n', ep, epochs, sum(abs(error)) / N_pretrain);
end

%% ==================== 保存预训练权重 ====================
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));
save(fullfile(script_dir, 'bp_pretrained_weights.mat'), 'w1', 'w2');
fprintf('权重已保存至 bp_pretrained_weights.mat\n');

%% ==================== 渐进式参考信号 ====================
function r_k = get_pretrain_r(k, N_total, ep, epochs)
    if ep <= 5                     % 前5轮：斜坡筑基
        r_k = 0.5 + 1.5 * min(1, k / N_total);
        return
    end
    phase = k / N_total;           % 第4-15轮：多样化泛化（12轮）
    if phase < 0.15
        r_k = 1;                                                  % 阶跃
    elseif phase < 0.30
        r_k = 1 + 0.5 * sin(2*pi*0.005*k);                       % sine_low 频率对齐
    elseif phase < 0.45
        r_k = 1 + 0.5 * sin(2*pi*0.02*k);                        % sine_high 频率对齐
    elseif phase < 0.55
        persistent rand_val rand_hold
        if isempty(rand_val) || rand_hold <= 0
            rand_val = 0.5 + 1.5 * rand();
            rand_hold = 80 + randi(70);
        end
        rand_hold = rand_hold - 1;
        r_k = rand_val;
    elseif phase < 0.70
        local_k = k - round(0.55 * N_total);
        local_N = round(0.15 * N_total);
        r_k = 0.5 + 1.5 * min(1, local_k / max(local_N, 1));     % 斜坡
    else
        r_k = 1 + 0.6*sin(2*pi*0.005*k) + 0.3*sin(2*pi*0.03*k); % composite 对齐
    end
end
