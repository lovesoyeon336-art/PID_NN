function run_tests()
% BP-PID (rate-limited) vs 固定 PID 对比测试
% 每个场景同时输出两个控制器的 du/dt 指标

model = 'bpnn_claude';
here = fileparts(mfilename('fullpath'));
fig_dir = fullfile(here, 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

%% ===== 初始化 =====
open_system(model);
rt = sfroot;
chart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [model '/MATLAB Function']);
chart.Script = fileread(fullfile(here, 'bp_pid_controller.m'));
save_system(model);

all_results = struct();

try
    %% ========== 测试 1: 阶跃响应 ==========
    fprintf('\n========== 测试 1: 阶跃响应 ==========\n');
    set_param([model '/Step1'], 'Time', '5', 'Before', '0', 'After', '1');

    % BP-PID
    set_param([model '/Manual Switch'], 'sw', '1');
    out_bp = sim(model, 'StopTime', '50');
    m_bp = step_metrics(out_bp.t, out_bp.r, out_bp.y, out_bp.u);
    du_bp = du_metrics(out_bp.t, out_bp.u);
    fprintf('  BP-PID:  σ=%.2f%%, ts=%.2fs, ess=%.4f, max|du/dt|=%.2f, std(du/dt)=%.2f\n', ...
        m_bp.sigma*100, m_bp.ts, m_bp.ess, du_bp.max_du, du_bp.std_du);

    % 固定 PID
    set_param([model '/Manual Switch'], 'sw', '0');
    out_fix = sim(model, 'StopTime', '50');
    m_fix = step_metrics(out_fix.t, out_fix.r, out_fix.y, out_fix.u);
    du_fix = du_metrics(out_fix.t, out_fix.u);
    fprintf('  固定PID: σ=%.2f%%, ts=%.2fs, ess=%.4f, max|du/dt|=%.2f, std(du/dt)=%.2f\n', ...
        m_fix.sigma*100, m_fix.ts, m_fix.ess, du_fix.max_du, du_fix.std_du);

    all_results.step = struct('bp', m_bp, 'fix', m_fix, 'du_bp', du_bp, 'du_fix', du_fix);
    plot_step_compare(out_bp, out_fix, fig_dir);

    %% ========== 测试 2: 正弦跟踪 ==========
    fprintf('\n========== 测试 2: 正弦跟踪 ==========\n');
    freqs = [0.1, 1, 5];
    delete_line(model, 'Step1/1', 'Sum1/1');
    add_line(model, 'Sine Wave1/1', 'Sum1/1');
    for i = 1:length(freqs)
        f = freqs(i);
        set_param([model '/Sine Wave1'], 'Frequency', num2str(f), ...
            'Amplitude', '1', 'Bias', '0');

        set_param([model '/Manual Switch'], 'sw', '1');
        out_bp = sim(model, 'StopTime', '100');
        du_bp = du_metrics(out_bp.t, out_bp.u);

        set_param([model '/Manual Switch'], 'sw', '0');
        out_fix = sim(model, 'StopTime', '100');
        du_fix = du_metrics(out_fix.t, out_fix.u);

        N = length(out_bp.t); idx_ss = round(N*0.5):N;
        rms_bp = rms(out_bp.r(idx_ss) - out_bp.y(idx_ss));
        rms_fix = rms(out_fix.r(idx_ss) - out_fix.y(idx_ss));
        all_results.sine(i) = struct('freq', f, 'rms_bp', rms_bp, 'rms_fix', rms_fix, ...
            'du_bp', du_bp, 'du_fix', du_fix);
        fprintf('  f=%.1f: BP RMS=%.4f(max|du/dt|=%.1f), Fix RMS=%.4f(max|du/dt|=%.1f)\n', ...
            f, rms_bp, du_bp.max_du, rms_fix, du_fix.max_du);
        plot_sine_compare(out_bp, out_fix, f, fig_dir);
    end
    delete_line(model, 'Sine Wave1/1', 'Sum1/1');
    add_line(model, 'Step1/1', 'Sum1/1');

    %% ========== 测试 3: 方波跟踪 ==========
    fprintf('\n========== 测试 3: 方波跟踪 ==========\n');
    add_block('simulink/Sources/Pulse Generator', [model '/SquarePulse']);
    set_param([model '/SquarePulse'], 'Amplitude', '1', 'Period', '20', ...
        'PulseWidth', '50', 'PhaseDelay', '5');
    delete_line(model, 'Step1/1', 'Sum1/1');
    add_line(model, 'SquarePulse/1', 'Sum1/1');

    set_param([model '/Manual Switch'], 'sw', '1');
    out_bp = sim(model, 'StopTime', '100');
    du_bp = du_metrics(out_bp.t, out_bp.u);

    set_param([model '/Manual Switch'], 'sw', '0');
    out_fix = sim(model, 'StopTime', '100');
    du_fix = du_metrics(out_fix.t, out_fix.u);

    sq_bp = analyze_square(out_bp.t, out_bp.r, out_bp.y, out_bp.u);
    sq_fix = analyze_square(out_fix.t, out_fix.r, out_fix.y, out_fix.u);
    all_results.square = struct('bp', sq_bp, 'fix', sq_fix, 'du_bp', du_bp, 'du_fix', du_fix);
    fprintf('  BP: 超调↑=%.1f%% ↓=%.1f%%, max|du/dt|=%.1f | Fix: 超调↑=%.1f%% ↓=%.1f%%, max|du/dt|=%.1f\n', ...
        sq_bp.avg_sigma_up*100, sq_bp.avg_sigma_down*100, du_bp.max_du, ...
        sq_fix.avg_sigma_up*100, sq_fix.avg_sigma_down*100, du_fix.max_du);
    plot_compare(out_bp, out_fix, '方波跟踪', 'fig_dyn_square', fig_dir);

    delete_line(model, 'SquarePulse/1', 'Sum1/1');
    delete_block([model '/SquarePulse']);
    add_line(model, 'Step1/1', 'Sum1/1');

    %% ========== 测试 4: 斜坡跟踪 ==========
    fprintf('\n========== 测试 4: 斜坡跟踪 ==========\n');
    add_block('simulink/Sources/Ramp', [model '/RampSrc']);
    set_param([model '/RampSrc'], 'slope', '0.05', 'start', '5', 'InitialOutput', '0');
    delete_line(model, 'Step1/1', 'Sum1/1');
    add_line(model, 'RampSrc/1', 'Sum1/1');

    set_param([model '/Manual Switch'], 'sw', '1');
    out_bp = sim(model, 'StopTime', '100');
    du_bp = du_metrics(out_bp.t, out_bp.u);

    set_param([model '/Manual Switch'], 'sw', '0');
    out_fix = sim(model, 'StopTime', '100');
    du_fix = du_metrics(out_fix.t, out_fix.u);

    Nb = length(out_bp.t); idx_ss_b = round(Nb*0.5):Nb;
    Nf = length(out_fix.t); idx_ss_f = round(Nf*0.5):Nf;
    err_bp = mean(abs(out_bp.r(idx_ss_b) - out_bp.y(idx_ss_b)));
    err_fix = mean(abs(out_fix.r(idx_ss_f) - out_fix.y(idx_ss_f)));
    all_results.ramp = struct('err_bp', err_bp, 'err_fix', err_fix, 'du_bp', du_bp, 'du_fix', du_fix);
    fprintf('  BP: err=%.4f, max|du/dt|=%.1f | Fix: err=%.4f, max|du/dt|=%.1f\n', ...
        err_bp, du_bp.max_du, err_fix, du_fix.max_du);
    plot_compare(out_bp, out_fix, '斜坡跟踪', 'fig_dyn_ramp', fig_dir);

    delete_line(model, 'RampSrc/1', 'Sum1/1');
    delete_block([model '/RampSrc']);
    add_line(model, 'Step1/1', 'Sum1/1');

    %% ========== 测试 5: 参数摄动 ==========
    fprintf('\n========== 测试 5: 参数摄动 ==========\n');
    tf_cases = {'2', '[1 1.2 1]'; '1', '[1 0.3 1]'; '1', '[1 3 1]'};
    names = {'增益加倍', '欠阻尼', '过阻尼'};
    baseline_num = '[1]'; baseline_den = '[1 1.2 1]';
    set_param([model '/Step1'], 'Time', '5', 'Before', '0', 'After', '1');

    for i = 1:size(tf_cases,1)
        set_param([model '/Transfer Fcn'], 'Numerator', tf_cases{i,1}, ...
            'Denominator', tf_cases{i,2});

        set_param([model '/Manual Switch'], 'sw', '1');
        out_bp = sim(model, 'StopTime', '50');
        du_bp = du_metrics(out_bp.t, out_bp.u);

        set_param([model '/Manual Switch'], 'sw', '0');
        out_fix = sim(model, 'StopTime', '50');
        du_fix = du_metrics(out_fix.t, out_fix.u);

        Nb = length(out_bp.t); Nf = length(out_fix.t);
        ess_bp = mean(abs(out_bp.r(round(Nb*0.7):Nb) - out_bp.y(round(Nb*0.7):Nb)));
        ess_fix = mean(abs(out_fix.r(round(Nf*0.7):Nf) - out_fix.y(round(Nf*0.7):Nf)));
        sigma_bp = max(0, (max(out_bp.y) - 1) / 1);
        sigma_fix = max(0, (max(out_fix.y) - 1) / 1);
        all_results.param(i) = struct('case', names{i}, 'sigma_bp', sigma_bp, ...
            'sigma_fix', sigma_fix, 'ess_bp', ess_bp, 'ess_fix', ess_fix, ...
            'du_bp', du_bp, 'du_fix', du_fix);
        fprintf('  %s: BP σ=%.1f%% ess=%.4f max|du/dt|=%.1f | Fix σ=%.1f%% ess=%.4f max|du/dt|=%.1f\n', ...
            names{i}, sigma_bp*100, ess_bp, du_bp.max_du, ...
            sigma_fix*100, ess_fix, du_fix.max_du);
    end
    set_param([model '/Transfer Fcn'], 'Numerator', baseline_num, ...
        'Denominator', baseline_den);

    %% ========== 测试 6: 外加扰动 ==========
    fprintf('\n========== 测试 6: 外加扰动 ==========\n');
    add_block('simulink/Sources/Pulse Generator', [model '/DisturbPulse']);
    set_param([model '/DisturbPulse'], 'Amplitude', '2', 'Period', '1000', ...
        'PulseWidth', '0.5', 'PhaseDelay', '30');
    add_line(model, 'DisturbPulse/1', 'Sum1/2');

    set_param([model '/Manual Switch'], 'sw', '1');
    out_bp = sim(model, 'StopTime', '50');
    du_bp = du_metrics(out_bp.t, out_bp.u);

    set_param([model '/Manual Switch'], 'sw', '0');
    out_fix = sim(model, 'StopTime', '50');
    du_fix = du_metrics(out_fix.t, out_fix.u);

    [dev_bp, rec_bp] = disturb_metrics(out_bp.t, out_bp.r, out_bp.y, 30);
    [dev_fix, rec_fix] = disturb_metrics(out_fix.t, out_fix.r, out_fix.y, 30);
    all_results.disturb = struct('dev_bp', dev_bp, 'dev_fix', dev_fix, ...
        'rec_bp', rec_bp, 'rec_fix', rec_fix, 'du_bp', du_bp, 'du_fix', du_fix);
    fprintf('  BP: dev=%.4f rec=%.2fs max|du/dt|=%.1f | Fix: dev=%.4f rec=%.2fs max|du/dt|=%.1f\n', ...
        dev_bp, rec_bp, du_bp.max_du, dev_fix, rec_fix, du_fix.max_du);
    plot_compare(out_bp, out_fix, '扰动抑制', 'fig_rob_disturb', fig_dir);

    dplh = get_param([model '/DisturbPulse'], 'LineHandles');
    if dplh.Outport > 0, delete_line(dplh.Outport); end
    delete_block([model '/DisturbPulse']);

    %% ========== 测试 7: 测量噪声 ==========
    fprintf('\n========== 测试 7: 测量噪声 ==========\n');
    noise_vars = [0.01, 0.05];
    for i = 1:length(noise_vars)
        nv = noise_vars(i);
        set_param([model '/Random Number'], 'Mean', '0', 'Variance', num2str(nv));
        add_line(model, 'Random Number/1', 'Sum1/2');

        rng(0);
        set_param([model '/Manual Switch'], 'sw', '1');
        out_bp = sim(model, 'StopTime', '50');
        du_bp = du_metrics(out_bp.t, out_bp.u);

        rng(0);
        set_param([model '/Manual Switch'], 'sw', '0');
        out_fix = sim(model, 'StopTime', '50');
        du_fix = du_metrics(out_fix.t, out_fix.u);

        Nb = length(out_bp.t); idx_b = round(Nb*0.3):Nb;
        Nf = length(out_fix.t); idx_f = round(Nf*0.3):Nf;
        rms_bp = rms(out_bp.r(idx_b) - out_bp.y(idx_b));
        rms_fix = rms(out_fix.r(idx_f) - out_fix.y(idx_f));
        all_results.noise(i) = struct('var', nv, 'rms_bp', rms_bp, 'rms_fix', rms_fix, ...
            'du_bp', du_bp, 'du_fix', du_fix);
        fprintf('  σ²=%.2f: BP RMS=%.4f max|du/dt|=%.1f | Fix RMS=%.4f max|du/dt|=%.1f\n', ...
            nv, rms_bp, du_bp.max_du, rms_fix, du_fix.max_du);
        plot_compare(out_bp, out_fix, sprintf('测量噪声 σ²=%.2f', nv), ...
            sprintf('fig_rob_noise_%d', round(nv*100)), fig_dir);

        rn_lh = get_param([model '/Random Number'], 'LineHandles');
        if rn_lh.Outport > 0, delete_line(rn_lh.Outport); end
    end

    %% ========== 测试 8: 纯滞后 ==========
    fprintf('\n========== 测试 8: 纯滞后 ==========\n');
    delays = [0.1, 0.3, 0.5];
    sw_lh = get_param([model '/Manual Switch'], 'LineHandles');
    tf_ph = get_param([model '/Transfer Fcn'], 'PortHandles');
    delete_line(sw_lh.Outport);
    add_line(model, 'Manual Switch/1', 'Transport Delay/1', 'autorouting', 'on');
    add_line(model, 'Transport Delay/1', 'Transfer Fcn/1', 'autorouting', 'on');

    for i = 1:length(delays)
        tau = delays(i);
        set_param([model '/Transport Delay'], 'DelayTime', num2str(tau));

        set_param([model '/Manual Switch'], 'sw', '1');
        out_bp = sim(model, 'StopTime', '50');
        du_bp = du_metrics(out_bp.t, out_bp.u);

        set_param([model '/Manual Switch'], 'sw', '0');
        out_fix = sim(model, 'StopTime', '50');
        du_fix = du_metrics(out_fix.t, out_fix.u);

        Nb = length(out_bp.t); Nf = length(out_fix.t);
        ess_bp = mean(abs(out_bp.r(round(Nb*0.7):Nb) - out_bp.y(round(Nb*0.7):Nb)));
        ess_fix = mean(abs(out_fix.r(round(Nf*0.7):Nf) - out_fix.y(round(Nf*0.7):Nf)));
        sigma_bp = max(0, (max(out_bp.y) - 1) / 1);
        sigma_fix = max(0, (max(out_fix.y) - 1) / 1);
        all_results.delay(i) = struct('tau', tau, ...
            'sigma_bp', sigma_bp, 'sigma_fix', sigma_fix, ...
            'ess_bp', ess_bp, 'ess_fix', ess_fix, ...
            'du_bp', du_bp, 'du_fix', du_fix);
        fprintf('  τ=%.1f: BP σ=%.1f%% ess=%.4f max|du/dt|=%.1f | Fix σ=%.1f%% ess=%.4f max|du/dt|=%.1f\n', ...
            tau, sigma_bp*100, ess_bp, du_bp.max_du, ...
            sigma_fix*100, ess_fix, du_fix.max_du);
    end

    delete_line(model, 'Transport Delay/1', 'Transfer Fcn/1');
    delete_line(model, 'Manual Switch/1', 'Transport Delay/1');
    add_line(model, 'Manual Switch/1', 'Transfer Fcn/1', 'autorouting', 'on');

    %% ===== 汇总表 =====
    fprintf('\n========== du/dt 对比汇总 ==========\n');
    fprintf('%-18s %16s %16s\n', '测试场景', 'BP max|du/dt|', 'Fix max|du/dt|');
    fprintf('%s\n', repmat('-', 1, 54));
    fprintf('%-18s %16.1f %16.1f\n', '1.阶跃', all_results.step.du_bp.max_du, all_results.step.du_fix.max_du);
    for i = 1:3
        fprintf('%-18s %16.1f %16.1f\n', sprintf('2.正弦 f=%.1f', all_results.sine(i).freq), ...
            all_results.sine(i).du_bp.max_du, all_results.sine(i).du_fix.max_du);
    end
    fprintf('%-18s %16.1f %16.1f\n', '3.方波', all_results.square.du_bp.max_du, all_results.square.du_fix.max_du);
    fprintf('%-18s %16.1f %16.1f\n', '4.斜坡', all_results.ramp.du_bp.max_du, all_results.ramp.du_fix.max_du);
    for i = 1:3
        fprintf('%-18s %16.1f %16.1f\n', sprintf('5.参数(%s)', all_results.param(i).case), ...
            all_results.param(i).du_bp.max_du, all_results.param(i).du_fix.max_du);
    end
    fprintf('%-18s %16.1f %16.1f\n', '6.扰动', all_results.disturb.du_bp.max_du, all_results.disturb.du_fix.max_du);
    for i = 1:2
        fprintf('%-18s %16.1f %16.1f\n', sprintf('7.噪声 v=%.2f', all_results.noise(i).var), ...
            all_results.noise(i).du_bp.max_du, all_results.noise(i).du_fix.max_du);
    end
    for i = 1:3
        fprintf('%-18s %16.1f %16.1f\n', sprintf('8.时滞 τ=%.1f', all_results.delay(i).tau), ...
            all_results.delay(i).du_bp.max_du, all_results.delay(i).du_fix.max_du);
    end

    save(fullfile(here, 'test_results.mat'), 'all_results');
    fprintf('\n结果已保存到 %s\n', fullfile(here, 'test_results.mat'));

catch ME
    fprintf(2, '测试中断: %s\n', ME.message);
end
bdclose('all');
end

%% ==================== 指标计算 ====================

function d = du_metrics(t, u)
    dt = mean(diff(t));
    du_dt = diff(u) / dt;
    d.max_du = max(abs(du_dt));
    d.std_du = std(du_dt);
end

function m = step_metrics(t, r, y, u)
    N = length(t);
    idx0 = find(r >= 0.5, 1);
    if isempty(idx0), idx0 = 1; end
    y_final = mean(y(round(N*0.8):N));
    y_peak = max(y(idx0:end));
    m.sigma = max(0, (y_peak - y_final) / abs(y_final + eps));
    y10 = 0.1 * y_final; y90 = 0.9 * y_final;
    t10 = t(find(y(idx0:end) >= y10, 1) + idx0 - 1);
    t90 = t(find(y(idx0:end) >= y90, 1) + idx0 - 1);
    m.tr = t90 - t10;
    [~, idx_peak] = max(y(idx0:end));
    m.tp = t(idx_peak + idx0 - 1) - t(idx0);
    tol = 0.02 * abs(y_final);
    m.ts = NaN;
    for k = N:-1:idx0
        if abs(y(k) - y_final) > tol
            m.ts = t(k) - t(idx0);
            break;
        end
    end
    m.ess = mean(abs(r(round(N*0.8):N) - y(round(N*0.8):N)));
end

function m = analyze_square(t, r, y, u)
    dr = diff(r);
    up_idx = find(dr > 0.5) + 1;
    down_idx = find(dr < -0.5) + 1;
    sigmas_up = []; sigmas_down = [];
    for i = 1:length(up_idx)
        seg = y(up_idx(i):min(up_idx(i)+round(5/0.001), length(y)));
        target = r(min(up_idx(i)+2, length(r)));
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
    skip = round(0.5 / (t(2)-t(1)));
    for k = idx_d+skip:N
        if abs(y(k) - y_ref) < tol
            rec_time = t(k) - t_disturb;
            break;
        end
    end
end

%% ==================== 对比绘图 ====================

function plot_step_compare(out_bp, out_fix, fig_dir)
    figure('Visible','off','Name','阶跃响应对比');
    subplot(2,2,1);
    plot(out_bp.t, out_bp.r, 'k--', out_bp.t, out_bp.y, 'b', out_fix.t, out_fix.y, 'r', 'LineWidth',1.2);
    xlabel('t (s)'); ylabel('y'); legend('r','BP-PID','固定PID'); title('跟踪响应'); grid on;
    subplot(2,2,2);
    plot(out_bp.t, out_bp.u, 'b', out_fix.t, out_fix.u, 'r', 'LineWidth',1.2);
    xlabel('t (s)'); ylabel('u'); legend('BP-PID','固定PID'); title('控制量'); grid on;
    subplot(2,2,3);
    dt = mean(diff(out_bp.t));
    plot(out_bp.t(1:end-1), abs(diff(out_bp.u))/dt, 'b', out_fix.t(1:end-1), abs(diff(out_fix.u))/dt, 'r', 'LineWidth',1.2);
    xlabel('t (s)'); ylabel('|du/dt|'); legend('BP-PID','固定PID'); title('控制加速度对比'); grid on;
    subplot(2,2,4);
    bar([max(abs(diff(out_bp.u))/dt), max(abs(diff(out_fix.u))/dt)]);
    set(gca,'XTickLabel',{'BP-PID','固定PID'});
    title(sprintf('max|du/dt|: BP=%.1f, Fix=%.1f', max(abs(diff(out_bp.u))/dt), max(abs(diff(out_fix.u))/dt))); grid on;
    saveas(gcf, fullfile(fig_dir, 'fig_dyn_step.png')); close;
end

function plot_sine_compare(out_bp, out_fix, f, fig_dir)
    figure('Visible','off','Name',sprintf('正弦 f=%.1f', f));
    subplot(2,2,1);
    plot(out_bp.t, out_bp.r, 'k--', out_bp.t, out_bp.y, 'b', 'LineWidth',1.2);
    xlabel('t (s)'); ylabel('y'); legend('r','BP-PID'); title(sprintf('BP-PID f=%.1f',f)); grid on;
    subplot(2,2,2);
    plot(out_fix.t, out_fix.r, 'k--', out_fix.t, out_fix.y, 'r', 'LineWidth',1.2);
    xlabel('t (s)'); ylabel('y'); legend('r','固定PID'); title(sprintf('固定PID f=%.1f',f)); grid on;
    subplot(2,2,3);
    plot(out_bp.t, out_bp.u, 'b', out_fix.t, out_fix.u, 'r', 'LineWidth',1.2);
    xlabel('t (s)'); ylabel('u'); legend('BP','Fix'); title('控制量对比'); grid on;
    subplot(2,2,4);
    dt = mean(diff(out_bp.t));
    plot(out_bp.t(1:end-1), abs(diff(out_bp.u))/dt, 'b', out_fix.t(1:end-1), abs(diff(out_fix.u))/dt, 'r', 'LineWidth',1.2);
    xlabel('t (s)'); ylabel('|du/dt|'); legend('BP','Fix'); title('|du/dt| 对比'); grid on;
    fname = sprintf('fig_dyn_sine_%d', round(f*10));
    saveas(gcf, fullfile(fig_dir, [fname '.png'])); close;
end

function plot_compare(out_bp, out_fix, ttl, fname, fig_dir)
    figure('Visible','off','Name',ttl);
    subplot(2,2,1);
    plot(out_bp.t, out_bp.r, 'k--', out_bp.t, out_bp.y, 'b', 'LineWidth',1.2);
    xlabel('t (s)'); ylabel('y'); legend('r','BP-PID'); title([ttl ' - BP-PID']); grid on;
    subplot(2,2,2);
    plot(out_fix.t, out_fix.r, 'k--', out_fix.t, out_fix.y, 'r', 'LineWidth',1.2);
    xlabel('t (s)'); ylabel('y'); legend('r','固定PID'); title([ttl ' - 固定PID']); grid on;
    subplot(2,2,3);
    plot(out_bp.t, out_bp.u, 'b', out_fix.t, out_fix.u, 'r', 'LineWidth',1.2);
    xlabel('t (s)'); ylabel('u'); legend('BP','Fix'); title('控制量对比'); grid on;
    subplot(2,2,4);
    dt = mean(diff(out_bp.t));
    plot(out_bp.t(1:end-1), abs(diff(out_bp.u))/dt, 'b', out_fix.t(1:end-1), abs(diff(out_fix.u))/dt, 'r', 'LineWidth',1.2);
    xlabel('t (s)'); ylabel('|du/dt|'); legend('BP','Fix'); title('|du/dt| 对比'); grid on;
    saveas(gcf, fullfile(fig_dir, [fname '.png'])); close;
end
