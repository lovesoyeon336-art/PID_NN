clear; close all;

%% ==================== 预训练 —— Plant3 (Hammerstein 非线性) ====================
IN = 5;   H = 5;  Out = 3;
rate  = 0.002;
rate2 = 0.01;
N_pretrain = 3000;
epochs = 12;

% 残差架构：Plant3 有效增益变化大（0.5~1.58），需更大 Kp 覆盖范围
Kp_base = 1.0;   Kp_delta = 8.0;
Ki_base = 0.1;   Ki_delta = 2.0;
Kd_base = 0;     Kd_delta = 4.0;
scale_vec = [Kp_delta, Ki_delta, Kd_delta];
ff_gain = 2.0;  beta_sp = 0.90;  u_sat = 3.0;

%% ==================== Xavier 初始化（半规模） ====================
rng(1);
w1 = 0.5 * sqrt(2/(IN+H)) * randn(H, IN);
w2 = 0.5 * sqrt(2/(H+Out)) * randn(Out, H);

for ep = 1:epochs
    y_1 = 0;  y_2 = 0;  u_1 = 0;  u_2 = 0;  r_1 = 0;
    error_1 = 0;  error_2 = 0;  e_sp_1 = 0;  e_sp_2 = 0;
    st_has = false;  st_ep = zeros(3,1);  st_O2 = zeros(1,H);
    st_dO3 = zeros(1,Out);  st_dO2 = zeros(1,H);  st_I1 = zeros(1,IN);

    w1_1 = w1;  w1_2 = w1;
    w2_1 = w2;  w2_2 = w2;

    time   = zeros(1, N_pretrain);
    r      = zeros(1, N_pretrain);
    y      = zeros(1, N_pretrain);
    error  = zeros(1, N_pretrain);
    u      = zeros(1, N_pretrain);

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
            if I3_t(l) > 0, O3(l) = I3_t(l); else, O3(l) = 0.2 * I3_t(l); end
        end

        kp_k = Kp_base + Kp_delta * O3(1);
        ki_k = Ki_base + Ki_delta * O3(2);
        kd_k = Kd_base + Kd_delta * O3(3);
        Kpid = [kp_k, ki_k, kd_k];

        e_sp_k = beta_sp * r(k) - y_1;
        e_pid = [e_sp_k - e_sp_1; error(k); e_sp_k - 2*e_sp_1 + e_sp_2];
        delta_u = Kpid * e_pid;
        dr = r(k) - r_1;
        if abs(dr) <= 0.1, delta_u = delta_u + ff_gain * dr; end
        delta_u = max(-1.0, min(1.0, delta_u));
        u(k) = max(-u_sat, min(u_sat, u_1 + delta_u));

        y(k) = plant_dynamics('plant3', y_1, y_2, u_1, u_1, k);
        error(k) = r(k) - y(k);

        % 延时反向传播
        dead_zone = 0.002;
        if st_has && abs(error(k)) >= dead_zone
            dydu_raw = (y(k) - y_1) / (u_1 - u_2 + 0.0001);
            du_sys = sign(dydu_raw) * 0.5;  % Plant3 Jacobian cap=0.5（线性区增益 0.5）

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
        end

        st_has = true;  st_ep = e_pid;  st_O2 = O2;  st_I1 = I1;
        dO3_t = zeros(1, Out);
        for j = 1:Out
            if O3(j) > 0, dO3_t(j) = 1; else, dO3_t(j) = 0.2; end
        end
        st_dO3 = dO3_t;
        st_dO2 = 1 - tanh(I2).^2;

        u_2 = u_1;  u_1 = u(k);
        y_2 = y_1;  y_1 = y(k);
        r_1 = r(k);
        error_2 = error_1;
        error_1 = error(k);
        e_sp_2 = e_sp_1;
        e_sp_1 = e_sp_k;
    end
    fprintf('Epoch %d/%d  MAE: %.6f\n', ep, epochs, sum(abs(error)) / N_pretrain);
end

[script_dir, ~, ~] = fileparts(mfilename('fullpath'));
save(fullfile(script_dir, 'bp_pretrained_weights_plant3.mat'), 'w1', 'w2');
fprintf('权重已保存至 bp_pretrained_weights_plant3.mat\n');

%% ==================== 参考信号（多幅值覆盖非线性区） ====================
function r_k = get_pretrain_r(k, N_total, ep, epochs)
    phase = mod(ep-1, 6);
    switch phase
        case 0  % 小信号阶跃 r=0.5（线性区）
            r_k = 0.5;
        case 1  % 中等阶跃 r=1（过渡区）
            r_k = 1;
        case 2  % 大信号阶跃 r=2（非线性区）
            r_k = 2;
        case 3  % 斜坡 0.5→2（跨区过渡）
            r_k = 0.5 + 1.5 * min(1, k / N_total);
        case 4  % 低频正弦 r=1+0.5sin
            r_k = 1 + 0.5 * sin(2*pi*0.005*k);
        case 5  % 斜坡+阶跃混合
            if k <= N_total/2, r_k = 0.5 + 1.5 * min(1, k/(N_total/2));
            else, r_k = 2; end
    end
end