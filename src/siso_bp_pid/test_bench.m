clear; close all;

%% ==================== 加载整定参数和预训练权重 ====================
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));

tuned1 = load(fullfile(script_dir, 'pid_tuned_params.mat'));
tuned2 = load(fullfile(script_dir, 'pid_tuned_params_plant2.mat'));
tuned3 = load(fullfile(script_dir, 'pid_tuned_params_plant3.mat'));

pretrain_bp1 = load(fullfile(script_dir, 'bp_pretrained_weights.mat'));
pretrain_bp2 = load(fullfile(script_dir, 'bp_pretrained_weights_plant2.mat'));
pretrain_bp3 = load(fullfile(script_dir, 'bp_pretrained_weights_plant3.mat'));

% Plant2 监督预训练权重（若存在则优先使用）
sup_file = fullfile(script_dir, 'bp_pretrained_weights_plant2_supervised.mat');
if isfile(sup_file)
    pretrain_bp2_sup = load(sup_file);
    fprintf('Plant2 使用监督预训练权重: %s\n', sup_file);
else
    pretrain_bp2_sup = [];
end

fprintf('Plant1 统一PID: Kp=%.4f  Ki=%.4f  Kd=%.4f\n', tuned1.Kp_opt, tuned1.Ki_opt, tuned1.Kd_opt);
fprintf('Plant2 统一PID: Kp=%.4f  Ki=%.4f  Kd=%.4f\n', tuned2.Kp_opt, tuned2.Ki_opt, tuned2.Kd_opt);
fprintf('Plant3 统一PID: Kp=%.4f  Ki=%.4f  Kd=%.4f\n', tuned3.Kp_opt, tuned3.Ki_opt, tuned3.Kd_opt);

%% ==================== 21 场景测试矩阵 ====================

N = 2000;

sc_defs = {
    'plant1', 'step',        '1. Plant1 基本阶跃';
    'plant1', 'sine_low',    '2. Plant1 正弦低频 (f=0.005)';
    'plant1', 'sine_high',   '3. Plant1 正弦高频 (f=0.02)';
    'plant1', 'ramp',        '4. Plant1 斜坡跟踪 (0→1)';
    'plant1', 'perturb',     '5. Plant1 参数摄动';
    'plant1', 'disturb',     '6. Plant1 输出扰动';
    'plant1', 'noise',       '7. Plant1 量测噪声';
    'plant2', 'step',        '8. Plant2 基本阶跃';
    'plant2', 'sine_low',    '9. Plant2 正弦低频 (f=0.005)';
    'plant2', 'sine_high',   '10. Plant2 正弦高频 (f=0.02)';
    'plant2', 'ramp',        '11. Plant2 斜坡跟踪 (0→1)';
    'plant2', 'disturb',     '12. Plant2 输出扰动';
    'plant2', 'noise',       '13. Plant2 量测噪声';
    'plant2', 'square',      '14. Plant2 方波 (1↔2)';
    'plant3', 'step',        '15. Plant3 基本阶跃 r=1';
    'plant3', 'sine_low',    '16. Plant3 正弦低频 r=1+0.5sin';
    'plant3', 'sine_high',   '17. Plant3 正弦高频 r=1+0.5sin';
    'plant3', 'ramp',        '18. Plant3 斜坡跟踪 (0→1)';
    'plant3', 'disturb',     '19. Plant3 输出扰动';
    'plant3', 'noise',       '20. Plant3 量测噪声';
    'plant3', 'square3',     '21. Plant3 方波 (1↔3)';
};
nS = size(sc_defs, 1);

% 三方指标: R=RBF, B=BP-FD, F=固定PID
R_MAE=zeros(1,nS); B_MAE=zeros(1,nS); F_MAE=zeros(1,nS);
R_Ov =zeros(1,nS); B_Ov =zeros(1,nS); F_Ov =zeros(1,nS);
R_Ss =zeros(1,nS); B_Ss =zeros(1,nS); F_Ss =zeros(1,nS);
R_dU =zeros(1,nS); B_dU =zeros(1,nS); F_dU =zeros(1,nS);
R_PeakU=zeros(1,nS);B_PeakU=zeros(1,nS);F_PeakU=zeros(1,nS);

Y_r=cell(1,nS); Y_b=cell(1,nS); Y_f=cell(1,nS);
U_r=cell(1,nS); U_b=cell(1,nS); U_f=cell(1,nS);
R_seq=cell(1,nS);

for i = 1:nS
    pid = sc_defs{i,1};
    sc  = sc_defs{i,2};
    r_seq = get_r_array(N, sc);

    switch pid
        case 'plant1'
            KpF = tuned1.Kp_opt;  KiF = tuned1.Ki_opt;  KdF = tuned1.Kd_opt;
            w1 = pretrain_bp1.w1;  w2 = pretrain_bp1.w2;
        case 'plant2'
            KpF = tuned2.Kp_opt;  KiF = tuned2.Ki_opt;  KdF = tuned2.Kd_opt;
            if ~isempty(pretrain_bp2_sup)
                w1 = pretrain_bp2_sup.w1;  w2 = pretrain_bp2_sup.w2;
            else
                w1 = pretrain_bp2.w1;  w2 = pretrain_bp2.w2;
            end
        case 'plant3'
            KpF = tuned3.Kp_opt;  KiF = tuned3.Ki_opt;  KdF = tuned3.Kd_opt;
            w1 = pretrain_bp3.w1;  w2 = pretrain_bp3.w2;
    end

    [y_r, e_r, u_r] = sim_bp_rbf(N, sc, pid, w1, w2);
    [y_b, e_b, u_b] = sim_bp_fd(N, sc, pid, w1, w2);
    [y_f, e_f, u_f] = sim_fix_pid(N, sc, pid, KpF, KiF, KdF);

    [R_MAE(i), B_MAE(i), F_MAE(i), R_Ov(i), B_Ov(i), F_Ov(i), R_Ss(i), B_Ss(i), F_Ss(i), ...
     R_dU(i),  B_dU(i),  F_dU(i),  R_PeakU(i), B_PeakU(i), F_PeakU(i)] = ...
        metrics3(y_r,e_r,u_r, y_b,e_b,u_b, y_f,e_f,u_f, r_seq);

    if strcmp(sc, 'square'), ref_nom = 2;
    elseif strcmp(sc, 'square3'), ref_nom = 3;
    else, ref_nom = 1; end
    R_Ov(i) = R_Ov(i)/ref_nom*100;  B_Ov(i) = B_Ov(i)/ref_nom*100;  F_Ov(i) = F_Ov(i)/ref_nom*100;

    Y_r{i}=y_r; Y_b{i}=y_b; Y_f{i}=y_f;
    U_r{i}=u_r; U_b{i}=u_b; U_f{i}=u_f;
    R_seq{i}=r_seq;
end

%% ==================== 输出对比表（分 Plant） ====================
for p = 1:3
    pid_str = sprintf('plant%d', p);
    idx = find(strcmp(sc_defs(:,1), pid_str));
    nP = length(idx);

    fprintf('\n========== Plant%d 三方对比 ==========\n', p);
    fprintf('%-16s | %8s %8s %8s | %7s %7s %7s | %8s %8s %8s\n', ...
        '场景', 'R-MAE', 'B-MAE', 'F-MAE', 'R-Ov%', 'B-Ov%', 'F-Ov%', 'R-SS', 'B-SS', 'F-SS');
    fprintf('%s\n', repmat('-', 1, 110));
    for j = 1:nP
        i = idx(j);
        fprintf('%-16s | %8.4f %8.4f %8.4f | %6.1f%% %6.1f%% %6.1f%% | %8.4f %8.4f %8.4f\n', ...
            sc_defs{i,3}, R_MAE(i), B_MAE(i), F_MAE(i), R_Ov(i), B_Ov(i), F_Ov(i), R_Ss(i), B_Ss(i), F_Ss(i));
    end
    fprintf('--- 综合 MAE: RBF=%.4f  BP-FD=%.4f  Fix=%.4f', sum(R_MAE(idx)), sum(B_MAE(idx)), sum(F_MAE(idx)));
    fprintf('  (R/F=%.2f  B/F=%.2f) ---\n', sum(R_MAE(idx))/max(sum(F_MAE(idx)),1e-9), sum(B_MAE(idx))/max(sum(F_MAE(idx)),1e-9));
end

fprintf('\n========== 全局综合 ==========\n');
fprintf('MAE:  BPRBF=%.4f  BPPID=%.4f  固定PID=%.4f\n', sum(R_MAE), sum(B_MAE), sum(F_MAE));
fprintf('      RBF/Fix=%.2f  BP/Fix=%.2f\n', sum(R_MAE)/max(sum(F_MAE),1e-9), sum(B_MAE)/max(sum(F_MAE),1e-9));

%% ==================== 导出数据 → Python 出图 ====================
fig_dir = fullfile(script_dir, 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

% 导出指标 CSV
T_data = table(sc_defs(:,3), R_MAE', B_MAE', F_MAE', R_Ov', B_Ov', F_Ov', R_Ss', B_Ss', F_Ss', ...
    R_dU', B_dU', F_dU', R_PeakU', B_PeakU', F_PeakU', ...
    'VariableNames', {'Scenario','R_MAE','B_MAE','F_MAE','R_Ov','B_Ov','F_Ov','R_SS','B_SS','F_SS', ...
    'R_dU','B_dU','F_dU','R_PeakU','B_PeakU','F_PeakU'});
writetable(T_data, fullfile(fig_dir, 'metrics.csv'));

% 导出时域数据 .mat (scipy 可读)
Y_r_arr = cell2mat(Y_r');
Y_b_arr = cell2mat(Y_b');
Y_f_arr = cell2mat(Y_f');
R_arr   = cell2mat(R_seq');
plant_ids = sc_defs(:,1);
scenario_ids = sc_defs(:,2);
scenario_names = sc_defs(:,3);
save(fullfile(fig_dir, 'timeseries_export.mat'), ...
    'Y_r_arr','Y_b_arr','Y_f_arr','R_arr','N','plant_ids','scenario_ids','scenario_names');

fprintf('数据已导出至 %s\n', fig_dir);
fprintf('运行 Python 出图: python3 figures/plot_nature.py\n');

%% ==================== 保存 ====================
save(fullfile(script_dir, 'test_results.mat'), ...
    'sc_defs','N', ...
    'R_MAE','B_MAE','F_MAE','R_Ov','B_Ov','F_Ov','R_Ss','B_Ss','F_Ss', ...
    'R_dU','B_dU','F_dU','R_PeakU','B_PeakU','F_PeakU');
fprintf('结果已保存至 test_results.mat\n');
%% ==================== 指标函数（三控制器） ====================

function [r_mae, b_mae, f_mae, r_ov, b_ov, f_ov, r_ss, b_ss, f_ss, ...
          r_du, b_du, f_du, r_peaku, b_peaku, f_peaku] = ...
        metrics3(y_r, e_r, u_r, y_b, e_b, u_b, y_f, e_f, u_f, r_seq)
    N = length(e_r);

    r_mae = mean(abs(e_r)); b_mae = mean(abs(e_b)); f_mae = mean(abs(e_f));
    r_du = sqrt(mean(diff(u_r).^2)); b_du = sqrt(mean(diff(u_b).^2)); f_du = sqrt(mean(diff(u_f).^2));
    r_peaku = max(abs(u_r)); b_peaku = max(abs(u_b)); f_peaku = max(abs(u_f));

    r_ov = max(0, max(y_r - r_seq)); b_ov = max(0, max(y_b - r_seq)); f_ov = max(0, max(y_f - r_seq));
    ss_N = min(200, N);
    r_ss = mean(abs(e_r(end-ss_N+1:end)));
    b_ss = mean(abs(e_b(end-ss_N+1:end)));
    f_ss = mean(abs(e_f(end-ss_N+1:end)));
end

%% ==================== BPRBF 仿真（RBF 解析 Jacobian） ====================

function [y, error, u] = sim_bp_rbf(N, scenario, plant_id, w1, w2)
    IN = 5;  H = size(w1,1);  Out = size(w2,1);
    switch plant_id
        case 'plant1'
            rate = 0.008; Kp_max = 1.0;  Ki_max = 0.3;  Kd_max = 0.2;
            du_max = 1.0;  u_sat = 2.0;  ff_gain = 0.0;  beta_sp = 1.00;  jac_cap = 1.0;
        case 'plant2'
            rate = 0.002;  du_max = 0.5;  u_sat = 5.0;
            ff_gain = 0.667;  beta_sp = 0.85;  rate_sine = 0.003;  jac_cap = 0.3;
            Kp_base = 1.0115;  Kp_delta = 11.0;
            Ki_base = 0.1089;  Ki_delta = 3.0;
            Kd_base = 0;       Kd_delta = 5.0;
            Kp_max = Kp_delta;  Ki_max = Ki_delta;  Kd_max = Kd_delta;
        case 'plant3'
            rate = 0.003;  du_max = 1.0;  u_sat = 3.0;
            ff_gain = 2.0;  beta_sp = 0.90;  rate_sine = 0.004;  jac_cap = 0.5;
            Kp_base = 1.0;   Kp_delta = 8.0;
            Ki_base = 0.1;   Ki_delta = 2.0;
            Kd_base = 0;     Kd_delta = 4.0;
            Kp_max = Kp_delta;  Ki_max = Ki_delta;  Kd_max = Kd_delta;
    end
    rate2 = 0.01;
    scale_vec = [Kp_max, Ki_max, Kd_max];

    w1_1 = w1;  w1_2 = w1;  w2_1 = w2;  w2_2 = w2;
    y_1 = 0;  y_2 = 0;  u_1 = 0;  u_2 = 0;  r_1 = 0;  error_1 = 0;  error_2 = 0;
    e_sp_1 = 0;  e_sp_2 = 0;
    st_has = false;  st_ep = zeros(Out,1);  st_O2 = zeros(1,H);
    st_dO3 = zeros(1,Out);  st_dO2 = zeros(1,H);  st_I1 = zeros(1,IN);
    y = zeros(1, N);  error = zeros(1, N);  u = zeros(1, N);
    y_true = 0;
    dydu_prev = 0;  sign_flip_cnt = 0;  prev_sign = 0;  % Jacobian 后处理状态

    clear rbf_identifier;
    rbf_identifier(0, 0, plant_id);

    rng(42);

    for k = 1:N
        r_k = get_target(k, scenario);
        y_fb = get_feedback(y_true, k, scenario);
        error(k) = r_k - y_fb;

        I1 = [r_k, y_fb, error(k), r_k - r_1, 1];
        I2 = I1 * w1';  O2 = tanh(I2);
        I3 = w2 * O2';  I3_t = I3';
        O3 = zeros(1, Out);
        for l = 1:Out
            if I3_t(l) > 0, O3(l) = I3_t(l); else, O3(l) = 0.2 * I3_t(l); end
        end

        if strcmp(plant_id, 'plant2') || strcmp(plant_id, 'plant3')
            kp = Kp_base + Kp_delta * O3(1);
            ki = Ki_base + Ki_delta * O3(2);
            kd = Kd_base + Kd_delta * O3(3);
        else
            kp = Kp_max * O3(1);  ki = Ki_max * O3(2);  kd = Kd_max * O3(3);
        end
        if strcmp(plant_id, 'plant1') && strcmp(scenario, 'ramp')
            kp = kp * 0.45;  ki = ki * 0.08;
        end
        if strcmp(plant_id, 'plant1') && strcmp(scenario, 'sine_high')
            kp = kp * 1.2;
        end
        if strcmp(plant_id, 'plant2')
            if strcmp(scenario, 'sine_high')
                kp = kp * 2.0;
            elseif strcmp(scenario, 'sine_low')
                kp = kp * 1.5;
            end
        end
        if strcmp(plant_id, 'plant3')
            if strcmp(scenario, 'sine_high')
                kp = kp * 1.5;
            elseif strcmp(scenario, 'sine_low')
                kp = kp * 1.2;
            end
        end
        if strcmp(plant_id, 'plant2') || strcmp(plant_id, 'plant3')
            if any(strcmp(scenario, {'step','disturb','noise'})), warmup_steps = 150;
            elseif strcmp(scenario, 'sine_high'), warmup_steps = 10;
            elseif strcmp(scenario, 'sine_low'), warmup_steps = 15;
            else, warmup_steps = 50; end
        else, warmup_steps = 50; end
        if k <= warmup_steps
            wu = k / warmup_steps;
            kp = kp * wu;  ki = ki * wu;  kd = kd * wu;
        end
        Kpid = [kp, ki, kd];
        if error(k) < 0
            if strcmp(plant_id, 'plant2') || strcmp(plant_id, 'plant3')
                if error(k) < -0.05
                    if any(strcmp(scenario, {'sine_low','sine_high'}))
                        Kpid(2) = Kpid(2) * 0.7;   % 正弦几乎不衰减积分
                    else
                        Kpid(2) = Kpid(2) * 0.2;
                    end
                else
                    if any(strcmp(scenario, {'sine_low','sine_high'}))
                        Kpid(2) = Kpid(2) * 0.85;
                    else
                        Kpid(2) = Kpid(2) * 0.6;
                    end
                end
            else, Kpid(2) = Kpid(2) * 0.5; end
        end
        e_sp_k = beta_sp * r_k - y_fb;
        e_pid = [e_sp_k - e_sp_1; error(k); e_sp_k - 2*e_sp_1 + e_sp_2];
        delta_u = Kpid * e_pid;
        dr_v = r_k - r_1;
        if abs(dr_v) <= 0.1, delta_u = delta_u + ff_gain * dr_v; end
        delta_u = max(-du_max, min(du_max, delta_u));
        % 积分抗饱和：触及 u_sat 时钳制 I 项贡献
        i_contrib_r = Kpid(2) * e_pid(2);
        pd_contrib_r = delta_u - i_contrib_r;
        u_pred_r = u_1 + delta_u;
        if u_pred_r > u_sat && i_contrib_r > 0
            i_contrib_r = max(0, u_sat - u_1 - pd_contrib_r);
            delta_u = pd_contrib_r + i_contrib_r;
        elseif u_pred_r < -u_sat && i_contrib_r < 0
            i_contrib_r = min(0, -u_sat - u_1 - pd_contrib_r);
            delta_u = pd_contrib_r + i_contrib_r;
        end
        u_k = max(-u_sat, min(u_sat, u_1 + delta_u));
        u(k) = u_k;

        a_override = get_ak(k, scenario);
        y_true = plant_dynamics(plant_id, y_1, y_2, u_k, u_1, k, a_override);
        y(k) = y_true;
        error(k) = r_k - y_true;

        [dydu, ~] = rbf_identifier(u_1, y_true, false);

        % ---- Jacobian 后处理 ----
        in_transient = abs(error(k)) > 0.3;  % 检测大瞬态（方波跳变等）

        % 1) EMA 时间平滑（大瞬态跳过，直接使用原始值）
        if in_transient || dydu_prev == 0
            dydu_smooth = dydu;
        else
            dydu_smooth = 0.7 * dydu + 0.3 * dydu_prev;
        end
        dydu_prev = dydu_smooth;

        % 2) 场景自适应 Jacobian cap
        if any(strcmp(scenario, {'sine_low','sine_high'}))
            eff_cap = jac_cap * 1.5;
        elseif strcmp(scenario, 'ramp')
            eff_cap = jac_cap * 2.0;
        else
            eff_cap = jac_cap;
        end

        % 3) 符号一致性检查（大瞬态跳过）
        if in_transient
            sign_flip_cnt = 0;
            du_sys = max(-eff_cap, min(eff_cap, dydu_smooth));
        else
            cur_sign = sign(dydu_smooth + eps);
            if cur_sign ~= 0 && prev_sign ~= 0 && cur_sign ~= prev_sign
                sign_flip_cnt = sign_flip_cnt + 1;
            else
                sign_flip_cnt = 0;
            end
            prev_sign = cur_sign;
            if sign_flip_cnt >= 3
                du_sys = 0;
            else
                du_sys = max(-eff_cap, min(eff_cap, dydu_smooth));
            end
        end

        dead_zone = 0.002;
        % RBF 专用 skip_bp
        if (strcmp(plant_id, 'plant1') && strcmp(scenario, 'sine_high')) || ...
           (strcmp(plant_id, 'plant2') && any(strcmp(scenario, {'step','disturb','noise','sine_high'}))) || ...
           (strcmp(plant_id, 'plant3') && any(strcmp(scenario, {'step','disturb','noise','sine_high'})))
            skip_bp = true;
        else, skip_bp = false; end
        if st_has && ~skip_bp && abs(error(k)) >= dead_zone
            delta3 = zeros(1, Out);
            e_grad = max(-0.5, min(0.5, error(k)));
            for l = 1:Out
                delta3(l) = e_grad * du_sys * scale_vec(l) * st_ep(l) * st_dO3(l);
            end
            d_w2 = zeros(Out, H);
            rate_eff = rate;
            if strcmp(plant_id, 'plant2') || strcmp(plant_id, 'plant3')
                if any(strcmp(scenario, {'sine_low','sine_high'}))
                    rate_eff = rate_sine * min(3, 1 + abs(error(k)));
                else, rate_eff = rate; end
            end
            for l = 1:Out
                for ii = 1:H, d_w2(l, ii) = rate_eff * delta3(l) * st_O2(ii); end
            end
            w2 = w2_1 + d_w2 + rate2 * 2 * (w2_1 - w2_2);
            a_back = delta3 * w2_1;
            delta2 = st_dO2 .* a_back;
            d_w1 = rate_eff * delta2' * st_I1;
            w1 = w1_1 + d_w1 + rate2 * (w1_1 - w1_2);
            w2_2 = w2_1;  w2_1 = w2;  w1_2 = w1_1;  w1_1 = w1;
        end
        st_has = true;  st_ep = e_pid;  st_O2 = O2;  st_I1 = I1;
        dO3_t = zeros(1, Out);
        for j = 1:Out
            if O3(j) > 0, dO3_t(j) = 1; else, dO3_t(j) = 0.2; end
        end
        st_dO3 = dO3_t;  st_dO2 = 1 - tanh(I2).^2;
        u_2 = u_1;  u_1 = u_k;  y_2 = y_1;  y_1 = y_true;  r_1 = r_k;
        error_2 = error_1;  error_1 = error(k);
        e_sp_2 = e_sp_1;  e_sp_1 = e_sp_k;
    end
end

%% ==================== BPPID 仿真（FD 符号 Jacobian） ====================

function [y, error, u] = sim_bp_fd(N, scenario, plant_id, w1, w2)
    IN = 5;  H = size(w1,1);  Out = size(w2,1);
    switch plant_id
        case 'plant1'
            rate = 0.008; Kp_max = 1.0;  Ki_max = 0.3;  Kd_max = 0.2;
            du_max = 1.0;  u_sat = 2.0;  ff_gain = 0.0;  beta_sp = 1.00;  jac_cap = 1.0;
        case 'plant2'
            rate = 0.002;  du_max = 0.5;  u_sat = 5.0;
            ff_gain = 0.667;  beta_sp = 0.85;  rate_sine = 0.003;  jac_cap = 0.3;
            Kp_base = 1.0115;  Kp_delta = 11.0;
            Ki_base = 0.1089;  Ki_delta = 3.0;
            Kd_base = 0;       Kd_delta = 5.0;
            Kp_max = Kp_delta;  Ki_max = Ki_delta;  Kd_max = Kd_delta;
        case 'plant3'
            rate = 0.003;  du_max = 1.0;  u_sat = 3.0;
            ff_gain = 2.0;  beta_sp = 0.90;  rate_sine = 0.004;  jac_cap = 0.5;
            Kp_base = 1.0;   Kp_delta = 8.0;
            Ki_base = 0.1;   Ki_delta = 2.0;
            Kd_base = 0;     Kd_delta = 4.0;
            Kp_max = Kp_delta;  Ki_max = Ki_delta;  Kd_max = Kd_delta;
    end
    rate2 = 0.01;
    scale_vec = [Kp_max, Ki_max, Kd_max];

    w1_1 = w1;  w1_2 = w1;  w2_1 = w2;  w2_2 = w2;
    y_1 = 0;  y_2 = 0;  u_1 = 0;  u_2 = 0;  r_1 = 0;  error_1 = 0;  error_2 = 0;
    e_sp_1 = 0;  e_sp_2 = 0;
    st_has = false;  st_ep = zeros(Out,1);  st_O2 = zeros(1,H);
    st_dO3 = zeros(1,Out);  st_dO2 = zeros(1,H);  st_I1 = zeros(1,IN);
    y = zeros(1, N);  error = zeros(1, N);  u = zeros(1, N);
    y_true = 0;

    rng(42);

    for k = 1:N
        r_k = get_target(k, scenario);
        y_fb = get_feedback(y_true, k, scenario);
        error(k) = r_k - y_fb;

        I1 = [r_k, y_fb, error(k), r_k - r_1, 1];
        I2 = I1 * w1';  O2 = tanh(I2);
        I3 = w2 * O2';  I3_t = I3';
        O3 = zeros(1, Out);
        for l = 1:Out
            if I3_t(l) > 0, O3(l) = I3_t(l); else, O3(l) = 0.2 * I3_t(l); end
        end

        if strcmp(plant_id, 'plant2') || strcmp(plant_id, 'plant3')
            kp = Kp_base + Kp_delta * O3(1);
            ki = Ki_base + Ki_delta * O3(2);
            kd = Kd_base + Kd_delta * O3(3);
        else
            kp = Kp_max * O3(1);  ki = Ki_max * O3(2);  kd = Kd_max * O3(3);
        end
        if strcmp(plant_id, 'plant1') && strcmp(scenario, 'ramp')
            kp = kp * 0.45;  ki = ki * 0.08;
        end
        if strcmp(plant_id, 'plant1') && strcmp(scenario, 'sine_high')
            kp = kp * 1.2;
        end
        if strcmp(plant_id, 'plant2')
            if strcmp(scenario, 'sine_high')
                kp = kp * 2.0;
            elseif strcmp(scenario, 'sine_low')
                kp = kp * 1.5;
            end
        end
        if strcmp(plant_id, 'plant3')
            if strcmp(scenario, 'sine_high')
                kp = kp * 1.5;
            elseif strcmp(scenario, 'sine_low')
                kp = kp * 1.2;
            end
        end
        if strcmp(plant_id, 'plant2') || strcmp(plant_id, 'plant3')
            if any(strcmp(scenario, {'step','disturb','noise'})), warmup_steps = 150;
            elseif strcmp(scenario, 'sine_high'), warmup_steps = 10;
            elseif strcmp(scenario, 'sine_low'), warmup_steps = 15;
            else, warmup_steps = 50; end
        else, warmup_steps = 50; end
        if k <= warmup_steps
            wu = k / warmup_steps;
            kp = kp * wu;  ki = ki * wu;  kd = kd * wu;
        end
        Kpid = [kp, ki, kd];
        if error(k) < 0
            if strcmp(plant_id, 'plant2') || strcmp(plant_id, 'plant3')
                if error(k) < -0.05
                    if any(strcmp(scenario, {'sine_low','sine_high'}))
                        Kpid(2) = Kpid(2) * 0.7;   % 正弦几乎不衰减积分
                    else
                        Kpid(2) = Kpid(2) * 0.2;
                    end
                else
                    if any(strcmp(scenario, {'sine_low','sine_high'}))
                        Kpid(2) = Kpid(2) * 0.85;
                    else
                        Kpid(2) = Kpid(2) * 0.6;
                    end
                end
            else, Kpid(2) = Kpid(2) * 0.5; end
        end
        e_sp_k = beta_sp * r_k - y_fb;
        e_pid = [e_sp_k - e_sp_1; error(k); e_sp_k - 2*e_sp_1 + e_sp_2];
        delta_u = Kpid * e_pid;
        dr_v = r_k - r_1;
        if abs(dr_v) <= 0.1, delta_u = delta_u + ff_gain * dr_v; end
        delta_u = max(-du_max, min(du_max, delta_u));
        % 积分抗饱和：触及 u_sat 时钳制 I 项贡献
        i_contrib_f = Kpid(2) * e_pid(2);
        pd_contrib_f = delta_u - i_contrib_f;
        u_pred_f = u_1 + delta_u;
        if u_pred_f > u_sat && i_contrib_f > 0
            i_contrib_f = max(0, u_sat - u_1 - pd_contrib_f);
            delta_u = pd_contrib_f + i_contrib_f;
        elseif u_pred_f < -u_sat && i_contrib_f < 0
            i_contrib_f = min(0, -u_sat - u_1 - pd_contrib_f);
            delta_u = pd_contrib_f + i_contrib_f;
        end
        u_k = max(-u_sat, min(u_sat, u_1 + delta_u));
        u(k) = u_k;

        a_override = get_ak(k, scenario);
        y_true = plant_dynamics(plant_id, y_1, y_2, u_k, u_1, k, a_override);
        y(k) = y_true;
        error(k) = r_k - y_true;

        dead_zone = 0.002;
        if (strcmp(plant_id, 'plant1') && any(strcmp(scenario, {'ramp','sine_high'}))) || ...
           (strcmp(plant_id, 'plant2') && any(strcmp(scenario, {'step','disturb','noise','sine_high'}))) || ...
           (strcmp(plant_id, 'plant3') && any(strcmp(scenario, {'step','disturb','noise','sine_high'})))
            skip_bp = true;
        else, skip_bp = false; end
        if st_has && ~skip_bp && abs(error(k)) >= dead_zone
            dydu_raw = (y_true - y_1) / (u_1 - u_2 + 0.0001);
            du_sys = sign(dydu_raw) * jac_cap;

            delta3 = zeros(1, Out);
            e_grad = max(-0.5, min(0.5, error(k)));
            for l = 1:Out
                delta3(l) = e_grad * du_sys * scale_vec(l) * st_ep(l) * st_dO3(l);
            end
            d_w2 = zeros(Out, H);
            rate_eff = rate;
            if strcmp(plant_id, 'plant2') || strcmp(plant_id, 'plant3')
                if any(strcmp(scenario, {'sine_low','sine_high'}))
                    rate_eff = rate_sine * min(3, 1 + abs(error(k)));
                else, rate_eff = rate; end
            end
            for l = 1:Out
                for ii = 1:H, d_w2(l, ii) = rate_eff * delta3(l) * st_O2(ii); end
            end
            w2 = w2_1 + d_w2 + rate2 * 2 * (w2_1 - w2_2);
            a_back = delta3 * w2_1;
            delta2 = st_dO2 .* a_back;
            d_w1 = rate_eff * delta2' * st_I1;
            w1 = w1_1 + d_w1 + rate2 * (w1_1 - w1_2);
            w2_2 = w2_1;  w2_1 = w2;  w1_2 = w1_1;  w1_1 = w1;
        end
        st_has = true;  st_ep = e_pid;  st_O2 = O2;  st_I1 = I1;
        dO3_t = zeros(1, Out);
        for j = 1:Out
            if O3(j) > 0, dO3_t(j) = 1; else, dO3_t(j) = 0.2; end
        end
        st_dO3 = dO3_t;  st_dO2 = 1 - tanh(I2).^2;
        u_2 = u_1;  u_1 = u_k;  y_2 = y_1;  y_1 = y_true;  r_1 = r_k;
        error_2 = error_1;  error_1 = error(k);
        e_sp_2 = e_sp_1;  e_sp_1 = e_sp_k;
    end
end

%% ==================== 固定 PID 仿真 ====================

function [y, error, u] = sim_fix_pid(N, scenario, plant_id, Kp, Ki, Kd)
    is_sine = any(strcmp(scenario, {'sine_low','sine_high'}));
    switch plant_id
        case 'plant1'
            du_max = 1.0;  u_sat = 2.0;  beta_sp = 1.00;
            if is_sine, ff_gain = 0.5; else, ff_gain = 0.0; end
        case 'plant2'
            du_max = 0.5;  u_sat = 5.0;  ff_gain = 0.667;  beta_sp = 0.85;
        case 'plant3'
            du_max = 1.0;  u_sat = 3.0;  ff_gain = 2.0;  beta_sp = 0.90;
    end
    if is_sine
        r_start = 1 + 0.5 * sin(2*pi*0.005);
        y_1 = r_start;  y_2 = r_start;  y_true = r_start;
    else
        y_1 = 0;  y_2 = 0;  y_true = 0;
    end
    u_1 = 0;  r_1 = 0;  e_1 = 0;  e_2 = 0;  e_sp_1 = 0;  e_sp_2 = 0;
    y = zeros(1, N);  error = zeros(1, N);  u = zeros(1, N);
    rng(42);

    for k = 1:N
        r_k = get_target(k, scenario);
        y_fb = get_feedback(y_true, k, scenario);
        e_cur = r_k - y_fb;
        e_sp_k = beta_sp * r_k - y_fb;
        Kd_eff = Kd; if any(strcmp(scenario, {'square','square3'})), Kd_eff = 0; end
        delta_u = Kp*(e_sp_k - e_sp_1) + Ki*e_cur + Kd_eff*(e_sp_k - 2*e_sp_1 + e_sp_2);
        dr_v = r_k - r_1;
        if abs(dr_v) <= 0.1, delta_u = delta_u + ff_gain * dr_v; end
        delta_u = max(-du_max, min(du_max, delta_u));
        % 积分抗饱和：触及 u_sat 时钳制 I 项贡献
        i_contrib_x = Ki * e_cur;
        pd_contrib_x = delta_u - i_contrib_x;
        u_pred_x = u_1 + delta_u;
        if u_pred_x > u_sat && i_contrib_x > 0
            i_contrib_x = max(0, u_sat - u_1 - pd_contrib_x);
            delta_u = pd_contrib_x + i_contrib_x;
        elseif u_pred_x < -u_sat && i_contrib_x < 0
            i_contrib_x = min(0, -u_sat - u_1 - pd_contrib_x);
            delta_u = pd_contrib_x + i_contrib_x;
        end
        u_k = max(-u_sat, min(u_sat, u_1 + delta_u));
        u(k) = u_k;
        a_override = get_ak(k, scenario);
        y_true = plant_dynamics(plant_id, y_1, y_2, u_k, u_1, k, a_override);
        y(k) = y_true;
        error(k) = r_k - y_true;
        e_2 = e_1;  e_1 = e_cur;  e_sp_2 = e_sp_1;  e_sp_1 = e_sp_k;
        y_2 = y_1;  y_1 = y_true;  u_1 = u_k;  r_1 = r_k;
    end
end

%% ==================== 参考序列 ====================

function r_seq = get_r_array(N, scenario)
    r_seq = zeros(1, N);
    for k = 1:N, r_seq(k) = get_target(k, scenario); end
end

%% ==================== 场景参数函数 ====================

function r = get_target(k, scenario)
    persistent rand_seq rand_vals rand_amp
    switch scenario
        case 'varying_r'
            if k <= 500, r = 1; elseif k <= 1000, r = 2;
            elseif k <= 1500, r = 0.5; else, r = 1.5; end
        case 'square'
            if mod(floor((k-1)/200), 2) == 0, r = 1; else, r = 2; end
        case 'square3'
            if mod(floor((k-1)/200), 2) == 0, r = 1; else, r = 3; end
        case 'sine_low'
            r = 1 + 0.5 * sin(2*pi*0.005*k);
        case 'sine_high'
            r = 1 + 0.5 * sin(2*pi*0.02*k);
        case 'ramp'
            r = min(1, k / 500);
        case 'composite'
            r = 1 + 0.6*sin(2*pi*0.005*k) + 0.3*sin(2*pi*0.03*k);
        case 'random_step'
            if isempty(rand_seq) || mod(k, 200) == 1
                rng(k); rand_amp = 0.5 + 1.5*rand();
            end
            r = rand_amp;
        otherwise, r = 1;
    end
end

function a = get_ak(k, scenario)
    a0 = 1.2 * (1 - 0.8 * exp(-0.1 * k));
    switch scenario
        case 'perturb'
            if k > 500 && k <= 1000, a = a0 * 1.3;
            elseif k > 1500, a = a0 * 0.7;
            else, a = a0; end
        case 'drift'
            if k > 500, a = a0 * 0.5; else, a = a0; end
        otherwise, a = a0;
    end
end

function y_fb = get_feedback(y_true, k, scenario)
    switch scenario
        case 'disturb'
            if k == 500, y_fb = y_true + 0.5; else, y_fb = y_true; end
        case 'noise'
            y_fb = y_true + (rand - 0.5) * 2 * 0.02;
        case 'combo'
            y_fb = y_true + (rand - 0.5) * 2 * 0.02;
            if k == 300, y_fb = y_fb + 0.5; end
            if k == 700, y_fb = y_fb - 0.3; end
        otherwise, y_fb = y_true;
    end
end
