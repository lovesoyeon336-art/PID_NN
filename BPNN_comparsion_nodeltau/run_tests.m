function run_tests()
% 综合性能测试：BP 神经网络 PID 控制器动态性能与鲁棒性

model = 'bpnn_claude';
fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% ===== 初始化 =====
open_system(model);
% 同步最新 bp_pid_controller.m 到 Simulink 内部
rt = sfroot;
chart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [model '/MATLAB Function']);
chart.Script = fileread(fullfile(fileparts(mfilename('fullpath')), 'bp_pid_controller.m'));
save_system(model);

% 切换到 BP PID（sw=1 选 MATLAB Function 输出）
set_param([model '/Manual Switch'], 'sw', '1');

% 存储全部结果
all_results = struct();

try
    %% ========== 测试 1: 阶跃响应 ==========
    fprintf('\n========== 测试 1: 阶跃响应 ==========\n');
    set_param([model '/Step1'], 'Time', '5', 'Before', '0', 'After', '1');
    out = sim(model, 'StopTime', '50');
    t = out.t; r = out.r; y = out.y; u = out.u;
    m1 = step_metrics(t, r, y, u);
    all_results.step = m1;
    fprintf('  超调量 σ = %.2f%%, 调节时间 ts = %.2f s, 稳态误差 ess = %.4f\n', ...
        m1.sigma*100, m1.ts, m1.ess);
    plot_step(t, r, y, u, out.p, out.i, out.d, m1, fig_dir);

    %% ========== 测试 2: 正弦跟踪 ==========
    fprintf('\n========== 测试 2: 正弦跟踪 ==========\n');
    freqs = [0.1, 1, 5];
    % 切换参考源：Step1 → Sine Wave1
    delete_line(model, 'Step1/1', 'Sum1/1');
    add_line(model, 'Sine Wave1/1', 'Sum1/1');
    for i = 1:length(freqs)
        f = freqs(i);
        set_param([model '/Sine Wave1'], 'Frequency', num2str(f), ...
            'Amplitude', '1', 'Bias', '0');
        out = sim(model, 'StopTime', '100');
        t = out.t; r = out.r; y = out.y;
        N = length(t); idx_ss = round(N*0.5):N;
        rms_err = rms(r(idx_ss) - y(idx_ss));
        [amp_r, ph_lag] = sine_metrics(t(idx_ss), r(idx_ss), y(idx_ss), f);
        all_results.sine(i) = struct('freq', f, 'rms_err', rms_err, ...
            'amp_ratio', amp_r, 'phase_lag_deg', ph_lag*180/pi);
        fprintf('  f=%.1f rad/s: RMS=%.4f, 幅值比=%.3f, 相位滞后=%.1f°\n', ...
            f, rms_err, amp_r, ph_lag*180/pi);
        plot_sine(t, r, y, out.u, f, fig_dir);
    end
    % 恢复 Step1
    delete_line(model, 'Sine Wave1/1', 'Sum1/1');
    add_line(model, 'Step1/1', 'Sum1/1');

    %% ========== 测试 3: 方波跟踪 ==========
    fprintf('\n========== 测试 3: 方波跟踪 ==========\n');
    add_block('simulink/Sources/Pulse Generator', [model '/SquarePulse']);
    set_param([model '/SquarePulse'], 'Amplitude', '1', 'Period', '20', ...
        'PulseWidth', '50', 'PhaseDelay', '5');
    delete_line(model, 'Step1/1', 'Sum1/1');
    add_line(model, 'SquarePulse/1', 'Sum1/1');
    out = sim(model, 'StopTime', '100');
    t = out.t; r = out.r; y = out.y; u = out.u;
    all_results.square = analyze_square(t, r, y, u);
    fprintf('  检测到 %d 次跳变, 上升超调=%.1f%%, 下降超调=%.1f%%\n', ...
        all_results.square.n_transitions, all_results.square.avg_sigma_up*100, ...
        all_results.square.avg_sigma_down*100);
    plot_square(t, r, y, u, fig_dir);
    % 恢复
    delete_line(model, 'SquarePulse/1', 'Sum1/1');
    delete_block([model '/SquarePulse']);
    add_line(model, 'Step1/1', 'Sum1/1');

    %% ========== 测试 4: 斜坡跟踪 ==========
    fprintf('\n========== 测试 4: 斜坡跟踪 ==========\n');
    add_block('simulink/Sources/Ramp', [model '/RampSrc']);
    set_param([model '/RampSrc'], 'slope', '0.05', 'start', '5', ...
        'InitialOutput', '0');
    delete_line(model, 'Step1/1', 'Sum1/1');
    add_line(model, 'RampSrc/1', 'Sum1/1');
    out = sim(model, 'StopTime', '100');
    t = out.t; r = out.r; y = out.y; u = out.u;
    N = length(t); idx_ss = round(N*0.5):N;
    ramp_err = mean(abs(r(idx_ss) - y(idx_ss)));
    all_results.ramp = struct('steady_err', ramp_err);
    fprintf('  斜坡稳态跟踪误差 = %.4f\n', ramp_err);
    plot_ramp(t, r, y, u, fig_dir);
    % 恢复
    delete_line(model, 'RampSrc/1', 'Sum1/1');
    delete_block([model '/RampSrc']);
    add_line(model, 'Step1/1', 'Sum1/1');

    %% ========== 测试 5: 参数摄动 ==========
    fprintf('\n========== 测试 5: 参数摄动 ==========\n');
    tf_cases = {
        '2', '[1 1.2 1]',   '增益加倍';
        '1', '[1 0.3 1]',   '欠阻尼';
        '1', '[1 3 1]',     '过阻尼'};
    baseline_num = '[1]'; baseline_den = '[1 1.2 1]';
    set_param([model '/Step1'], 'Time', '5', 'Before', '0', 'After', '1');

    figure('Name', '参数摄动鲁棒性', 'NumberTitle', 'off', 'Visible', 'off');
    colors = {'r', 'b', 'g'};
    for i = 1:size(tf_cases, 1)
        set_param([model '/Transfer Fcn'], 'Numerator', tf_cases{i,1}, ...
            'Denominator', tf_cases{i,2});
        out = sim(model, 'StopTime', '50');
        t = out.t; y = out.y;
        N = length(t); idx_ss = round(N*0.7):N;
        ess = mean(abs(out.r(idx_ss) - y(idx_ss)));
        sigma = max(0, (max(y) - 1) / 1);
        all_results.param(i) = struct('case', tf_cases{i,3}, 'ess', ess, 'sigma', sigma);
        fprintf('  %s: 超调=%.1f%%, 稳态误差=%.4f\n', tf_cases{i,3}, sigma*100, ess);
        plot(t, y, colors{i}, 'LineWidth', 1.2); hold on;
    end
    plot(t, out.r, 'k--', 'LineWidth', 1.2);
    xlabel('时间 t (s)'); ylabel('输出 y'); legend('增益加倍', '欠阻尼', '过阻尼', '设定值', 'Location', 'best');
    title('参数摄动鲁棒性'); grid on;
    saveas(gcf, fullfile(fig_dir, 'fig_rob_param.png')); close;
    % 恢复基准对象
    set_param([model '/Transfer Fcn'], 'Numerator', baseline_num, ...
        'Denominator', baseline_den);

    %% ========== 测试 6: 外加扰动 ==========
    fprintf('\n========== 测试 6: 外加扰动 ==========\n');
    % 在误差节点 Sum 注入输出端扰动：Sum = r - (y + d)
    % Sum 原本 Inputs=|+-，改为 |+-- 以容纳第三个输入（扰动，负号）
    % 脉冲扰动：t=30s 跳变到 2，持续 5s 后归零（Period 远大于仿真时长，等效单脉冲）
    add_block('simulink/Sources/Pulse Generator', [model '/DisturbPulse']);
    set_param([model '/DisturbPulse'], 'Amplitude', '2', 'Period', '1000', ...
        'PulseWidth', '0.5', 'PhaseDelay', '30');
    add_line(model, 'DisturbPulse/1', 'Sum1/2');
    out = sim(model, 'StopTime', '50');
    t = out.t; r = out.r; y = out.y; u = out.u;
    [max_dev, rec_time] = disturb_metrics(t, r, y, 30);
    all_results.disturb = struct('max_dev', max_dev, 'recovery_time', rec_time);
    fprintf('  最大偏差=%.4f, 恢复时间=%.2f s\n', max_dev, rec_time);
    plot_disturb(t, r, y, u, 30, fig_dir);
    % 断开并删除扰动脉冲块
    dplh = get_param([model '/DisturbPulse'], 'LineHandles');
    if dplh.Outport > 0, delete_line(dplh.Outport); end
    delete_block([model '/DisturbPulse']);

    %% ========== 测试 7: 测量噪声 ==========
    fprintf('\n========== 测试 7: 测量噪声 ==========\n');
    noise_vars = [0.01, 0.05];
    rng(0);
    for i = 1:length(noise_vars)
        nv = noise_vars(i);
        set_param([model '/Random Number'], 'Mean', '0', 'Variance', num2str(nv));
        add_line(model, 'Random Number/1', 'Sum1/2');
        rng(0);
        out = sim(model, 'StopTime', '50');
        t = out.t; r = out.r; y = out.y; u = out.u;
        N = length(t); idx_ss = round(N*0.3):N;
        rms_clean = rms(r(idx_ss) - y(idx_ss));
        u_std = std(u(idx_ss));
        all_results.noise(i) = struct('var', nv, 'rms_err', rms_clean, 'u_std', u_std);
        fprintf('  噪声方差 σ²=%.2f: RMS误差=%.4f, 控制量标准差=%.4f\n', nv, rms_clean, u_std);
        plot_noise(t, r, y, u, nv, fig_dir);
        rn_lh = get_param([model '/Random Number'], 'LineHandles');
        if rn_lh.Outport > 0, delete_line(rn_lh.Outport); end
    end

    %% ========== 测试 8: 纯滞后 ==========
    fprintf('\n========== 测试 8: 纯滞后 ==========\n');
    delays = [0.1, 0.3, 0.5];
    % 将 Transport Delay 串入前向通道（Manual Switch → Delay → Transfer Fcn）
    sw_lh = get_param([model '/Manual Switch'], 'LineHandles');
    tf_ph = get_param([model '/Transfer Fcn'], 'PortHandles');
    delete_line(sw_lh.Outport);  % 断开 Switch→TF
    add_line(model, 'Manual Switch/1', 'Transport Delay/1', 'autorouting', 'on');
    add_line(model, 'Transport Delay/1', 'Transfer Fcn/1', 'autorouting', 'on');

    figure('Name', '时滞鲁棒性', 'NumberTitle', 'off', 'Visible', 'off');
    for i = 1:length(delays)
        tau = delays(i);
        set_param([model '/Transport Delay'], 'DelayTime', num2str(tau));
        out = sim(model, 'StopTime', '50');
        t = out.t; y = out.y;
        N = length(t); idx_ss = round(N*0.7):N;
        ess = mean(abs(out.r(idx_ss) - y(idx_ss)));
        sigma = max(0, (max(y) - 1) / 1);
        all_results.delay(i) = struct('tau', tau, 'ess', ess, 'sigma', sigma);
        fprintf('  τ=%.1f s: 超调=%.1f%%, 稳态误差=%.4f\n', tau, sigma*100, ess);
        plot(t, y, 'LineWidth', 1.2); hold on;
    end
    plot(t, out.r, 'k--', 'LineWidth', 1.2);
    xlabel('时间 t (s)'); ylabel('输出 y');
    legend('τ=0.1', 'τ=0.3', 'τ=0.5', '设定值', 'Location', 'best');
    title('纯滞后鲁棒性'); grid on;
    saveas(gcf, fullfile(fig_dir, 'fig_rob_delay.png')); close;

    % 恢复前向通道
    delete_line(model, 'Transport Delay/1', 'Transfer Fcn/1');
    delete_line(model, 'Manual Switch/1', 'Transport Delay/1');
    add_line(model, 'Manual Switch/1', 'Transfer Fcn/1', 'autorouting', 'on');

    %% ===== 汇总 =====
    fprintf('\n========== 综合指标汇总 ==========\n');
    fprintf('1. 阶跃响应: σ=%.2f%%, ts=%.2fs, ess=%.4f\n', ...
        all_results.step.sigma*100, all_results.step.ts, all_results.step.ess);
    for i = 1:3
        fprintf('2. 正弦 f=%.1f: RMS=%.4f, 幅值比=%.3f, 相位=%.1f°\n', ...
            all_results.sine(i).freq, all_results.sine(i).rms_err, ...
            all_results.sine(i).amp_ratio, all_results.sine(i).phase_lag_deg);
    end
    fprintf('3. 方波: 平均上升超调=%.1f%%, 平均下降超调=%.1f%%\n', ...
        all_results.square.avg_sigma_up*100, all_results.square.avg_sigma_down*100);
    fprintf('4. 斜坡: 稳态跟踪误差=%.4f\n', all_results.ramp.steady_err);
    for i = 1:3
        fprintf('5. 参数摄动(%s): σ=%.1f%%, ess=%.4f\n', ...
            all_results.param(i).case, all_results.param(i).sigma*100, all_results.param(i).ess);
    end
    fprintf('6. 扰动抑制: max=%.4f, 恢复=%.2fs\n', ...
        all_results.disturb.max_dev, all_results.disturb.recovery_time);
    fprintf('8. 时滞: τ=0.1s(σ=%.1f%%) τ=0.3s(σ=%.1f%%) τ=0.5s(σ=%.1f%%)\n', ...
        all_results.delay(1).sigma*100, all_results.delay(2).sigma*100, all_results.delay(3).sigma*100);

    save(fullfile(fileparts(mfilename('fullpath')), 'test_results.mat'), 'all_results');

catch ME
    fprintf(2, '测试中断: %s\n', ME.message);
end
bdclose('all');
end

%% ==================== 指标计算函数 ====================

function m = step_metrics(t, r, y, u)
    N = length(t);
    % 找到阶跃时刻
    idx0 = find(r >= 0.5, 1);
    if isempty(idx0), idx0 = 1; end
    y_final = mean(y(round(N*0.8):N));
    % 超调量
    y_peak = max(y(idx0:end));
    m.sigma = max(0, (y_peak - y_final) / abs(y_final + eps));
    % 上升时间 10%→90%
    y10 = 0.1 * y_final; y90 = 0.9 * y_final;
    t10 = t(find(y(idx0:end) >= y10, 1) + idx0 - 1);
    t90 = t(find(y(idx0:end) >= y90, 1) + idx0 - 1);
    m.tr = t90 - t10;
    % 峰值时间
    [~, idx_peak] = max(y(idx0:end));
    m.tp = t(idx_peak + idx0 - 1) - t(idx0);
    % 调节时间 ±2%
    tol = 0.02 * abs(y_final);
    m.ts = NaN;
    for k = N:-1:idx0
        if abs(y(k) - y_final) > tol
            m.ts = t(k) - t(idx0);
            break;
        end
    end
    % 稳态误差
    m.ess = mean(abs(r(round(N*0.8):N) - y(round(N*0.8):N)));
end

function [amp_r, ph_lag] = sine_metrics(t, r, y, f)
    % 用互相关估计幅值比和相位滞后
    N = length(t);
    ref = sin(2*pi*f*t' / (2*pi));  % sin(f*t)
    % 简化：用正弦拟合
    A = [sin(f*t), cos(f*t), ones(N,1)];
    theta_r = A \ r;
    theta_y = A \ y;
    amp_r_in = sqrt(theta_r(1)^2 + theta_r(2)^2);
    amp_y_in = sqrt(theta_y(1)^2 + theta_y(2)^2);
    amp_r = amp_y_in / (amp_r_in + eps);
    phase_r = atan2(theta_r(2), theta_r(1));
    phase_y = atan2(theta_y(2), theta_y(1));
    ph_lag = phase_r - phase_y;
    if ph_lag > pi, ph_lag = ph_lag - 2*pi; end
    if ph_lag < -pi, ph_lag = ph_lag + 2*pi; end
end

function m = analyze_square(t, r, y, u)
    dr = diff(r);
    up_idx = find(dr > 0.5) + 1;     % 上升沿
    down_idx = find(dr < -0.5) + 1;  % 下降沿
    sigmas_up = []; sigmas_down = [];
    for i = 1:length(up_idx)
        seg = y(up_idx(i):min(up_idx(i)+round(5/0.001), length(y)));
        target = r(min(up_idx(i)+2, length(r)));  % 跳变后稳态值
        if abs(target) < 0.01, continue; end
        sigmas_up(end+1) = max(0, (max(seg) - target) / abs(target));
    end
    for i = 1:length(down_idx)
        seg = y(down_idx(i):min(down_idx(i)+round(5/0.001), length(y)));
        target = r(min(down_idx(i)+2, length(r)));
        if abs(target) < 0.01, continue; end
        sigmas_down(end+1) = max(0, (target - min(seg)) / abs(target));
    end
    m.avg_sigma_up = 0; m.avg_sigma_down = 0;
    if ~isempty(sigmas_up), m.avg_sigma_up = mean(sigmas_up); end
    if ~isempty(sigmas_down), m.avg_sigma_down = mean(sigmas_down); end
    m.n_transitions = length(up_idx) + length(down_idx);
end

function [max_dev, rec_time] = disturb_metrics(t, r, y, t_disturb)
    idx_d = find(t >= t_disturb, 1);
    N = length(t);
    y_ref = mean(y(round(N*0.3):idx_d-1));
    max_dev = max(abs(y(idx_d:end) - y_ref));
    tol = 0.05 * abs(y_ref) + 0.02;
    rec_time = NaN;
    % 跳过扰动后 0.5s，等系统响应后再检测恢复
    skip = round(0.5 / (t(2)-t(1)));
    for k = idx_d+skip:N
        if abs(y(k) - y_ref) < tol
            rec_time = t(k) - t_disturb;
            break;
        end
    end
end

%% ==================== 绘图函数 ====================

function plot_step(t, r, y, u, Kp, Ki, Kd, m, fig_dir)
    figure('Name', '阶跃响应', 'NumberTitle', 'off', 'Visible', 'off');
    subplot(2,2,1);
    plot(t, r, 'k--', t, y, 'b', 'LineWidth', 1.2);
    xlabel('t (s)'); ylabel('信号'); legend('r', 'y'); title('跟踪响应'); grid on;
    subplot(2,2,2);
    plot(t, u, 'LineWidth', 1.2);
    xlabel('t (s)'); ylabel('u'); title('控制量'); grid on;
    subplot(2,2,3);
    plot(t, Kp, t, Ki, t, Kd, 'LineWidth', 1.2);
    xlabel('t (s)'); ylabel('参数'); legend('Kp','Ki','Kd'); title('PID 参数自适应'); grid on;
    subplot(2,2,4);
    bar([m.sigma*100, m.ts, m.ess*100]);
    set(gca, 'XTickLabel', {'σ%', 'ts(s)', 'ess×100'});
    title(sprintf('σ=%.1f%%, ts=%.2fs, ess=%.4f', m.sigma*100, m.ts, m.ess)); grid on;
    saveas(gcf, fullfile(fig_dir, 'fig_dyn_step.png')); close;
end

function plot_sine(t, r, y, u, f, fig_dir)
    figure('Name', sprintf('正弦跟踪 f=%.1f', f), 'NumberTitle', 'off', 'Visible', 'off');
    subplot(2,1,1);
    plot(t, r, 'k--', t, y, 'b', 'LineWidth', 1.2);
    xlabel('t (s)'); ylabel('信号'); legend('r', 'y');
    title(sprintf('正弦跟踪 (f=%.1f rad/s)', f)); grid on;
    subplot(2,1,2);
    plot(t, u, 'LineWidth', 1.2);
    xlabel('t (s)'); ylabel('u'); title('控制量'); grid on;
    fname = sprintf('fig_dyn_sine_%d', round(f*10));
    saveas(gcf, fullfile(fig_dir, [fname '.png'])); close;
end

function plot_square(t, r, y, u, fig_dir)
    figure('Name', '方波跟踪', 'NumberTitle', 'off', 'Visible', 'off');
    subplot(2,1,1);
    plot(t, r, 'k--', t, y, 'b', 'LineWidth', 1.2);
    xlabel('t (s)'); ylabel('信号'); legend('r', 'y'); title('方波跟踪'); grid on;
    subplot(2,1,2);
    plot(t, u, 'LineWidth', 1.2);
    xlabel('t (s)'); ylabel('u'); title('控制量'); grid on;
    saveas(gcf, fullfile(fig_dir, 'fig_dyn_square.png')); close;
end

function plot_ramp(t, r, y, u, fig_dir)
    figure('Name', '斜坡跟踪', 'NumberTitle', 'off', 'Visible', 'off');
    subplot(2,1,1);
    plot(t, r, 'k--', t, y, 'b', 'LineWidth', 1.2);
    xlabel('t (s)'); ylabel('信号'); legend('r', 'y'); title('斜坡跟踪'); grid on;
    subplot(2,1,2);
    plot(t, u, 'LineWidth', 1.2);
    xlabel('t (s)'); ylabel('u'); title('控制量'); grid on;
    saveas(gcf, fullfile(fig_dir, 'fig_dyn_ramp.png')); close;
end

function plot_disturb(t, r, y, u, t_d, fig_dir)
    figure('Name', '扰动抑制', 'NumberTitle', 'off', 'Visible', 'off');
    subplot(2,1,1);
    plot(t, r, 'k--', t, y, 'b', 'LineWidth', 1.2);
    xline(t_d, 'r--', 'LineWidth', 1.2);
    xlabel('t (s)'); ylabel('信号'); legend('r', 'y', '扰动时刻'); title('外加扰动抑制'); grid on;
    subplot(2,1,2);
    plot(t, u, 'LineWidth', 1.2);
    xline(t_d, 'r--', 'LineWidth', 1.2);
    xlabel('t (s)'); ylabel('u'); title('控制量'); grid on;
    saveas(gcf, fullfile(fig_dir, 'fig_rob_disturb.png')); close;
end

function plot_noise(t, r, y, u, nv, fig_dir)
    figure('Name', sprintf('测量噪声 σ²=%.2f', nv), 'NumberTitle', 'off', 'Visible', 'off');
    subplot(2,1,1);
    plot(t, r, 'k--', t, y, 'b', 'LineWidth', 1.2);
    xlabel('t (s)'); ylabel('信号'); legend('r', 'y');
    title(sprintf('测量噪声鲁棒性 (σ²=%.2f)', nv)); grid on;
    subplot(2,1,2);
    plot(t, u, 'LineWidth', 1.2);
    xlabel('t (s)'); ylabel('u'); title('控制量'); grid on;
    fname = sprintf('fig_rob_noise_%d', round(nv*100));
    saveas(gcf, fullfile(fig_dir, [fname '.png'])); close;
end
