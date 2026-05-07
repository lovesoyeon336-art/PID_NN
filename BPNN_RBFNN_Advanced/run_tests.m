function run_tests()
% =========================================================================
% run_tests.m — BP+RBF神经网络控制系统仿真与鲁棒性分析
% =========================================================================
% 5个测试场景:
%   Test 1: 阶跃响应 — 基准动态性能
%   Test 2: 扰动抑制 — 阶跃负载扰动
%   Test 3: 噪声鲁棒 — 测量白噪声
%   Test 4: 正弦跟踪 — 时变信号
%   Test 5: 参数摄动 — 对象增益翻倍
%
% 前提: 先运行 build_model 生成 BP_RBFNN_Sim.slx
% 运行: >> run_tests
% =========================================================================

model = 'BP_RBFNN_Sim';

% 检查模型文件
slx_file = fullfile(pwd, [model '.slx']);
if ~exist(slx_file, 'file')
    error('Model "%s.slx" not found. Run build_model first.', model);
end

% 添加当前路径
if ~contains(path, pwd)
    addpath(pwd);
end

fprintf('========================================\n');
fprintf('  BP+RBF Neural Network Control Tests\n');
fprintf('========================================\n\n');

%% ===== Test 1: 阶跃响应 — 基准动态性能 =====
fprintf('[Test 1/5] Step Response - Baseline Dynamic Performance\n');
load_system(model);
set_param([model '/Ref SW'],  'sw', '0');   % 阶跃
set_param([model '/Dist SW'], 'sw', '1');   % 扰动OFF
set_param([model '/Noise SW'], 'sw', '1');  % 噪声OFF
set_param(model, 'StopTime', '15');

simOut = sim(model, 'ReturnWorkspaceOutputs', 'on');
[t1, r1, y1, u1, Kp1, Ki1, Kd1, yh1] = get_data(simOut);

perf1 = analyze_step(t1, y1, 0.5, 1.0);
fprintf('  Rise time:     %.3f s\n', perf1.tr);
fprintf('  Overshoot:     %.1f %%\n', perf1.Mp);
fprintf('  Settling time: %.3f s\n', perf1.ts);
fprintf('  SSE:           %.4f\n', perf1.sse);
fprintf('  ITAE:          %.2f\n', perf1.itae);
close_system(model, 0);

%% ===== Test 2: 扰动抑制 =====
fprintf('\n[Test 2/5] Disturbance Rejection\n');
load_system(model);
set_param([model '/Ref SW'],  'sw', '0');
set_param([model '/Dist SW'], 'sw', '0');   % 扰动ON
set_param([model '/Noise SW'], 'sw', '1');
set_param(model, 'StopTime', '15');

simOut = sim(model, 'ReturnWorkspaceOutputs', 'on');
[t2, r2, y2, u2] = get_data(simOut);

dp = analyze_dist(t2, y2, u2, 8.0);
fprintf('  Max deviation:  %.4f\n', dp.max_dev);
fprintf('  Recovery time:  %.3f s\n', dp.rec_time);
fprintf('  IAE (post-dist):%.4f\n', dp.iae);
close_system(model, 0);

%% ===== Test 3: 测量噪声鲁棒性 =====
fprintf('\n[Test 3/5] Measurement Noise Robustness\n');
load_system(model);
set_param([model '/Ref SW'],  'sw', '0');
set_param([model '/Dist SW'], 'sw', '1');
set_param([model '/Noise SW'], 'sw', '0');  % 噪声ON
set_param(model, 'StopTime', '15');

simOut = sim(model, 'ReturnWorkspaceOutputs', 'on');
[t3, r3, y3, u3] = get_data(simOut);

np = analyze_noise(t3, y3, u3, 6.0, 15.0);
fprintf('  Output std:     %.4f\n', np.y_std);
fprintf('  Control std:    %.4f\n', np.u_std);
fprintf('  Mean |error|:   %.4f\n', np.mean_abs_err);
close_system(model, 0);

%% ===== Test 4: 正弦跟踪 =====
fprintf('\n[Test 4/5] Sine Wave Tracking\n');
load_system(model);
set_param([model '/Ref SW'],  'sw', '1');   % 正弦
set_param([model '/Dist SW'], 'sw', '1');
set_param([model '/Noise SW'], 'sw', '1');
set_param(model, 'StopTime', '20');

simOut = sim(model, 'ReturnWorkspaceOutputs', 'on');
[t4, r4, y4] = get_data(simOut);

sp = analyze_sine(t4, r4, y4, 4.0, 20.0);
fprintf('  RMSE:           %.4f\n', sp.rmse);
fprintf('  Max error:      %.4f\n', sp.max_err);
if ~isnan(sp.phase_lag)
    fprintf('  Phase lag:      %.1f deg\n', sp.phase_lag);
end
close_system(model, 0);

%% ===== Test 5: 参数摄动鲁棒性 =====
fprintf('\n[Test 5/5] Parameter Perturbation Robustness\n');
load_system(model);
set_param([model '/Ref SW'],  'sw', '0');
set_param([model '/Dist SW'], 'sw', '1');
set_param([model '/Noise SW'], 'sw', '1');

orig_num = get_param([model '/Plant'], 'Numerator');
set_param([model '/Plant'], 'Numerator', '[20]');
set_param(model, 'StopTime', '20');

simOut = sim(model, 'ReturnWorkspaceOutputs', 'on');
[t5, r5, y5, u5, Kp5] = get_data(simOut);

set_param([model '/Plant'], 'Numerator', orig_num);

perf5 = analyze_step(t5, y5, 0.5, 1.0);
fprintf('  (Plant gain: 10 -> 20)\n');
fprintf('  Rise time:     %.3f s\n', perf5.tr);
fprintf('  Overshoot:     %.1f %%\n', perf5.Mp);
fprintf('  Settling time: %.3f s\n', perf5.ts);
fprintf('  SSE:           %.4f\n', perf5.sse);
close_system(model, 0);

%% ===== 综合绘图 =====
fprintf('\nGenerating comprehensive plots...\n');
try
    do_plots(t1,y1,u1,Kp1,Ki1,Kd1,yh1, t2,y2,u2, t3,y3,u3, ...
             t4,r4,y4, t5,y5,u5,Kp5, perf1, dp, np, sp, perf5);
catch ME
    fprintf(2, 'Plot error: %s\n', ME.message);
end

%% ===== 总结 =====
fprintf('\n========== Performance Summary ==========\n');
fprintf('Test 1 (Baseline):    t_r=%.3fs  M_p=%.1f%%  t_s=%.3fs  SSE=%.4f  ITAE=%.2f\n', ...
    perf1.tr, perf1.Mp, perf1.ts, perf1.sse, perf1.itae);
fprintf('Test 2 (Disturbance): max_dev=%.3f  recovery=%.3fs  IAE=%.4f\n', ...
    dp.max_dev, dp.rec_time, dp.iae);
fprintf('Test 3 (Noise):       sigma_y=%.4f  sigma_u=%.4f\n', np.y_std, np.u_std);
fprintf('Test 4 (Sine track):  RMSE=%.4f  max|e|=%.4f\n', sp.rmse, sp.max_err);
fprintf('Test 5 (Perturb 2x):  t_r=%.3fs  M_p=%.1f%%  t_s=%.3fs\n', ...
    perf5.tr, perf5.Mp, perf5.ts);
fprintf('==========================================\n');
fprintf('Done.\n');

end

%% ==================== 数据提取 ====================
function varargout = get_data(simOut)
    names = {'t_out','r_out','y_out','u_out','Kp_out','Ki_out', ...
             'Kd_out','yhat_out','dydu_out'};
    n = min(nargout, length(names));
    varargout = cell(1, n);
    for i = 1:n
        try
            ds = simOut.get(names{i});
            if isa(ds, 'timeseries')
                varargout{i} = ds.Data(:);
            elseif isa(ds, 'Simulink.SimulationData.Dataset')
                varargout{i} = ds{1}.Values.Data(:);
            elseif isnumeric(ds)
                varargout{i} = ds(:);
            else
                varargout{i} = [];
                warning('Unknown data type for: %s', names{i});
            end
        catch
            varargout{i} = [];
            warning('Failed to read: %s', names{i});
        end
    end
end

%% ==================== 阶跃响应分析 ====================
function perf = analyze_step(t, y, step_t, step_amp)
    if isempty(t) || isempty(y)
        perf.tr=NaN; perf.Mp=NaN; perf.ts=NaN; perf.sse=NaN; perf.itae=NaN;
        return;
    end
    idx = find(t >= step_t, 1);
    if isempty(idx), idx = 1; end
    t_s = t(idx:end) - t(idx);
    y_s = y(idx:end);
    n = length(y_s);
    if n < 10
        perf.tr=NaN; perf.Mp=NaN; perf.ts=NaN; perf.sse=NaN; perf.itae=NaN;
        return;
    end

    y_ss = mean(y_s(max(1,round(n*0.75)):end));
    perf.sse = abs(step_amp - y_ss);

    y10 = 0.10 * step_amp;  y90 = 0.90 * step_amp;
    i10 = find(y_s >= y10, 1);  i90 = find(y_s >= y90, 1);
    if ~isempty(i10) && ~isempty(i90)
        perf.tr = t_s(i90) - t_s(i10);
    else
        perf.tr = NaN;
    end

    y_max = max(y_s);
    if y_max > step_amp
        perf.Mp = (y_max - step_amp) / step_amp * 100;
    else
        perf.Mp = 0;
    end

    band = max(0.02 * step_amp, 0.001);
    out = abs(y_s - y_ss) > band;
    last = find(out, 1, 'last');
    if ~isempty(last) && last < n
        perf.ts = t_s(last + 1);
    else
        perf.ts = t_s(end);
    end

    e = abs(step_amp - y_s);
    perf.itae = trapz(t_s, t_s .* e);
end

%% ==================== 扰动分析 ====================
function perf = analyze_dist(t, y, u, dist_t)
    if isempty(t) || isempty(y)
        perf.max_dev=0; perf.rec_time=NaN; perf.iae=NaN; return;
    end
    idx = find(t >= dist_t, 1);
    if isempty(idx), idx = length(t); end

    pre_idx = max(1,idx-1000):max(1,idx-1);
    y_pre = mean(y(pre_idx));

    post_idx = idx:min(idx+2000, length(y));
    [perf.max_dev, max_i] = max(abs(y(post_idx) - y_pre));
    if isempty(perf.max_dev), perf.max_dev = 0; end

    band = max(0.05 * abs(y_pre), 0.01);
    k_start = min(idx + max_i, length(y));
    rec_samp = 0;
    for k = k_start:length(y)
        if abs(y(k) - y_pre) < band
            rec_samp = k - k_start;
            break;
        end
    end
    dt = t(2) - t(1);
    if dt > 0
        perf.rec_time = rec_samp * dt;
    else
        perf.rec_time = NaN;
    end

    e = abs(y_pre - y(idx:end));
    perf.iae = sum(e) * dt;
end

%% ==================== 噪声分析 ====================
function perf = analyze_noise(t, y, u, t_s, t_e)
    if isempty(t) || isempty(y)
        perf.y_std=NaN; perf.u_std=NaN; perf.mean_abs_err=NaN; return;
    end
    idx = find(t >= t_s & t <= t_e);
    if length(idx) < 10
        perf.y_std=std(y); perf.u_std=std(u); perf.mean_abs_err=mean(abs(y-1));
        return;
    end
    perf.y_std = std(y(idx));
    perf.u_std = std(u(idx));
    perf.mean_abs_err = mean(abs(y(idx) - 1));
end

%% ==================== 正弦跟踪分析 ====================
function perf = analyze_sine(t, r, y, t_s, t_e)
    if isempty(t) || isempty(y)
        perf.rmse=NaN; perf.max_err=NaN; perf.phase_lag=NaN; return;
    end
    idx = find(t >= t_s & t <= t_e);
    if length(idx) < 10
        e = r - y;
        perf.rmse = sqrt(mean(e.^2));
        perf.max_err = max(abs(e));
        perf.phase_lag = NaN;
        return;
    end
    e = r(idx) - y(idx);
    perf.rmse = sqrt(mean(e.^2));
    perf.max_err = max(abs(e));

    [c, lags] = xcorr(y(idx)-mean(y(idx)), r(idx)-mean(r(idx)));
    [~, max_i] = max(c);
    lag = lags(max_i);
    [~, peaks] = findpeaks(r(idx));
    if length(peaks) >= 2
        period = mean(diff(peaks));
        perf.phase_lag = 360 * lag / period;
    else
        perf.phase_lag = NaN;
    end
end

%% ==================== 综合绘图 ====================
function do_plots(t1,y1,u1,Kp1,Ki1,Kd1,yh1, ...
                  t2,y2,u2, t3,y3,u3, t4,r4,y4, t5,y5,u5,Kp5, ...
                  perf1, dp, np, sp, perf5)

    scrsz = get(0, 'ScreenSize');
    FW = min(1200, scrsz(3)*0.85);
    FH = min(800, scrsz(4)*0.85);

    %% Figure 1
    figure('Name','Fig1: Step Response & Identification', ...
           'Position',[40, scrsz(4)-FH-60, FW, FH], 'Color','w', ...
           'Visible','off');

    subplot(2,2,1);
    plot(t1,y1,'b-','LineWidth',1.5); hold on;
    yline(1,'r--','LineWidth',1.2);
    xlabel('Time (s)'); ylabel('Output');
    title('(a) Step Response'); grid on;
    legend('y(t)','r=1','Location','southeast');
    if ~isnan(perf1.tr)
        text(0.02,0.95,sprintf('t_r=%.3fs  M_p=%.1f%%  t_s=%.3fs',...
            perf1.tr,perf1.Mp,perf1.ts),'Units','normalized',...
            'FontSize',9,'BackgroundColor','w');
    end

    subplot(2,2,2);
    plot(t1,Kp1,'r-',t1,Ki1,'g-',t1,Kd1,'b-','LineWidth',1.2);
    xlabel('Time (s)'); ylabel('PID Parameters');
    title('(b) Online Self-Tuning PID'); grid on;
    legend('K_p','K_i','K_d','Location','best');
    xlim([0, min(3, max(t1))]);

    subplot(2,2,3);
    plot(t1,u1,'k-','LineWidth',1.2);
    xlabel('Time (s)'); ylabel('Control Signal');
    title('(c) Control Input u(t)'); grid on;

    subplot(2,2,4);
    plot(t1,y1,'b-','LineWidth',1.2); hold on;
    plot(t1,yh1,'r--','LineWidth',1.2);
    xlabel('Time (s)'); ylabel('Output');
    title('(d) RBF Identification: y vs ŷ'); grid on;
    legend('y','ŷ (RBF)','Location','southeast');
    xlim([0, min(3, max(t1))]);

    saveas(gcf, 'fig1_step_response.png');
    fprintf('  fig1_step_response.png saved.\n');

    %% Figure 2
    figure('Name','Fig2: Robustness Tests', ...
           'Position',[40, scrsz(4)-FH-60, FW, FH], 'Color','w', ...
           'Visible','off');

    subplot(2,2,1);
    plot(t2,y2,'b-','LineWidth',1.5); hold on;
    yline(1,'r--','LineWidth',1.2);
    xline(8,'k:','LineWidth',1.5);
    xlabel('Time (s)'); ylabel('Output'); grid on;
    title(sprintf('(a) Disturbance Rejection (max dev=%.3f, rec=%.2fs)',...
        dp.max_dev, dp.rec_time));
    legend('y(t)','Reference','Disturbance','Location','southeast');

    subplot(2,2,2);
    plot(t2,u2,'k-','LineWidth',1.2); hold on;
    xline(8,'k:','LineWidth',1.5);
    xlabel('Time (s)'); ylabel('Control'); grid on;
    title('(b) Control Action During Disturbance');

    subplot(2,2,3);
    plot(t3,y3,'b-','LineWidth',1.0); hold on;
    yline(1,'r--','LineWidth',1.2);
    xlabel('Time (s)'); ylabel('Output'); grid on;
    title(sprintf('(c) Noise Robustness (\\sigma_y=%.4f)',np.y_std));
    legend('y(t)','Reference','Location','southeast');

    subplot(2,2,4);
    plot(t5,y5,'b-','LineWidth',1.5); hold on;
    yline(1,'r--','LineWidth',1.2);
    xlabel('Time (s)'); ylabel('Output'); grid on;
    title(sprintf('(d) Plant Gain x2 (t_r=%.3fs, M_p=%.1f%%)',...
        perf5.tr, perf5.Mp));
    legend('y(t)','Reference','Location','southeast');

    saveas(gcf, 'fig2_robustness.png');
    fprintf('  fig2_robustness.png saved.\n');

    %% Figure 3
    figure('Name','Fig3: Sine Wave Tracking', ...
           'Position',[80, scrsz(4)-400, FW*0.55, FH*0.55], 'Color','w', ...
           'Visible','off');

    subplot(2,1,1);
    plot(t4,r4,'r--','LineWidth',1.5); hold on;
    plot(t4,y4,'b-','LineWidth',1.2);
    xlabel('Time (s)'); ylabel('Output'); grid on;
    title(sprintf('Sine Wave Tracking (RMSE=%.4f, max|e|=%.4f)',sp.rmse,sp.max_err));
    legend('r(t)','y(t)','Location','northeast');

    subplot(2,1,2);
    plot(t4, r4-y4, 'm-', 'LineWidth', 1.0);
    xlabel('Time (s)'); ylabel('Error'); grid on;
    title('Tracking Error e(t) = r(t) - y(t)');

    saveas(gcf, 'fig3_sine_tracking.png');
    fprintf('  fig3_sine_tracking.png saved.\n');

    %% Figure 4
    figure('Name','Fig4: PID Adaptation', ...
           'Position',[80, scrsz(4)-400, FW*0.55, FH*0.55], 'Color','w', ...
           'Visible','off');

    subplot(1,2,1);
    plot(t1,Kp1,'LineWidth',1.2); hold on;
    plot(t1,Ki1,'LineWidth',1.2);
    plot(t1,Kd1,'LineWidth',1.2);
    xlabel('Time (s)'); ylabel('PID Parameters'); xlim([0, 5]); grid on;
    title('(a) Nominal Plant'); legend('K_p','K_i','K_d');

    subplot(1,2,2);
    plot(t5,Kp5,'LineWidth',1.2);
    xlabel('Time (s)'); ylabel('K_p'); xlim([0, 5]); grid on;
    title('(b) Plant Gain x2 — K_p Adaptation');

    saveas(gcf, 'fig4_pid_adaptation.png');
    fprintf('  fig4_pid_adaptation.png saved.\n');
end
