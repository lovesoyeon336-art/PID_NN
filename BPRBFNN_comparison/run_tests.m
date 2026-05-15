function run_tests()
% 综合对比测试：BPRBFNN vs BPNN_1 动态性能与鲁棒性

model_rbf = 'BP_RBFNN';
model_bp  = 'bpnn_claude';
fig_dir = fullfile(fileparts(mfilename('fullpath')), 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

% BPNN_1 路径
bp1_path = fullfile(fileparts(mfilename('fullpath')), '..', 'BPNN');

%% ===== 初始化 BPRBFNN =====
fprintf('初始化 BPRBFNN ...\n');
open_system(model_rbf);
rt = sfroot;
chart1 = rt.find('-isa', 'Stateflow.EMChart', 'Path', [model_rbf '/MATLAB Function1']);
chart2 = rt.find('-isa', 'Stateflow.EMChart', 'Path', [model_rbf '/MATLAB Function2']);
chart1.Script = fileread(fullfile(fileparts(mfilename('fullpath')), 'BP_PID_Controller.m'));
chart2.Script = fileread(fullfile(fileparts(mfilename('fullpath')), 'RBF_Identifier_claude.m'));
set_param([model_rbf '/Manual Switch'], 'sw', '1');
save_system(model_rbf);

all_rbf = struct(); all_bp = struct();

try
    %% ===== 测试 1: 阶跃响应 =====
    fprintf('\n========== 测试 1: 阶跃响应 ==========\n');

    set_param([model_rbf '/Step1'], 'Time', '5', 'Before', '0', 'After', '1');
    out_rbf = sim(model_rbf, 'StopTime', '50');
    m1_rbf = step_metrics(out_rbf.t, out_rbf.r, out_rbf.y, out_rbf.u);
    all_rbf.step = m1_rbf;
    bdclose(model_rbf);

    cd(bp1_path);
    open_system(model_bp);
    rt2 = sfroot;
    ch = rt2.find('-isa', 'Stateflow.EMChart', 'Path', [model_bp '/MATLAB Function']);
    ch.Script = fileread('bp_pid_controller.m');
    set_param([model_bp '/Manual Switch'], 'sw', '1');
    set_param([model_bp '/Step1'], 'Time', '5', 'Before', '0', 'After', '1');
    save_system(model_bp);
    out_bp = sim(model_bp, 'StopTime', '50');
    m1_bp = step_metrics(out_bp.t, out_bp.r, out_bp.y, out_bp.u);
    all_bp.step = m1_bp;
    bdclose(model_bp);

    fprintf('  RBF:  sigma=%.1f%%, ts=%.2fs, ess=%.4f\n', m1_rbf.sigma*100, m1_rbf.ts, m1_rbf.ess);
    fprintf('  BP:   sigma=%.1f%%, ts=%.2fs, ess=%.4f\n', m1_bp.sigma*100, m1_bp.ts, m1_bp.ess);
    plot_step_compare(out_rbf, out_bp, fig_dir);

    %% ===== 测试 2: 正弦跟踪 =====
    fprintf('\n========== 测试 2: 正弦跟踪 ==========\n');
    freqs = [0.1, 1, 5];
    for i = 1:length(freqs)
        f = freqs(i);
        % RBF
        cd(fullfile(fileparts(mfilename('fullpath'))));
        open_system(model_rbf);
        rt = sfroot;
        chart1 = rt.find('-isa', 'Stateflow.EMChart', 'Path', [model_rbf '/MATLAB Function1']);
        chart2 = rt.find('-isa', 'Stateflow.EMChart', 'Path', [model_rbf '/MATLAB Function2']);
        chart1.Script = fileread('BP_PID_Controller.m');
        chart2.Script = fileread('RBF_Identifier_claude.m');
        set_param([model_rbf '/Manual Switch'], 'sw', '1');
        delete_line(model_rbf, 'Step1/1', 'Sum1/1');
        add_line(model_rbf, 'Sine Wave1/1', 'Sum1/1');
        set_param([model_rbf '/Sine Wave1'], 'Frequency', num2str(f), 'Amplitude', '1', 'Bias', '0');
        save_system(model_rbf);
        out1 = sim(model_rbf, 'StopTime', '100');
        t1=out1.t; r1=out1.r; y1=out1.y;
        N=length(t1); idx_ss=round(N*0.5):N;
        rms1 = rms(r1(idx_ss)-y1(idx_ss));
        delete_line(model_rbf, 'Sine Wave1/1', 'Sum1/1');
        add_line(model_rbf, 'Step1/1', 'Sum1/1');
        save_system(model_rbf);
        bdclose(model_rbf);

        % BP
        cd(bp1_path);
        open_system(model_bp);
        rt2 = sfroot;
        ch = rt2.find('-isa', 'Stateflow.EMChart', 'Path', [model_bp '/MATLAB Function']);
        ch.Script = fileread('bp_pid_controller.m');
        set_param([model_bp '/Manual Switch'], 'sw', '1');
        delete_line(model_bp, 'Step1/1', 'Sum1/1');
        add_line(model_bp, 'Sine Wave1/1', 'Sum1/1');
        set_param([model_bp '/Sine Wave1'], 'Frequency', num2str(f), 'Amplitude', '1', 'Bias', '0');
        save_system(model_bp);
        out2 = sim(model_bp, 'StopTime', '100');
        t2=out2.t; r2=out2.r; y2=out2.y;
        rms2 = rms(r2(idx_ss)-y2(idx_ss));
        delete_line(model_bp, 'Sine Wave1/1', 'Sum1/1');
        add_line(model_bp, 'Step1/1', 'Sum1/1');
        save_system(model_bp);
        bdclose(model_bp);

        all_rbf.sine(i) = struct('freq',f,'rms_err',rms1);
        all_bp.sine(i)  = struct('freq',f,'rms_err',rms2);
        fprintf('  f=%.1f rad/s: RBF RMS=%.4f, BP RMS=%.4f\n', f, rms1, rms2);
        plot_sine_compare(t1,r1,y1,t2,r2,y2,f,fig_dir);
    end

    %% ===== 测试 3: 方波跟踪 =====
    fprintf('\n========== 测试 3: 方波跟踪 ==========\n');
    for mdl = {model_rbf, model_bp}
        m = mdl{1};
        if strcmp(m, model_rbf)
            cd(fullfile(fileparts(mfilename('fullpath'))));
        else
            cd(bp1_path);
        end
        open_system(m);
        if strcmp(m, model_rbf)
            rt=sfroot; c1=rt.find('-isa','Stateflow.EMChart','Path',[m '/MATLAB Function1']);
            c2=rt.find('-isa','Stateflow.EMChart','Path',[m '/MATLAB Function2']);
            c1.Script=fileread('BP_PID_Controller.m'); c2.Script=fileread('RBF_Identifier_claude.m');
        else
            rt2=sfroot; c=rt2.find('-isa','Stateflow.EMChart','Path',[m '/MATLAB Function']);
            c.Script=fileread('bp_pid_controller.m');
        end
        set_param([m '/Manual Switch'], 'sw', '1');
        add_block('simulink/Sources/Pulse Generator', [m '/SquarePulse']);
        set_param([m '/SquarePulse'], 'Amplitude','1','Period','20','PulseWidth','50','PhaseDelay','5');
        delete_line(m, 'Step1/1', 'Sum1/1');
        add_line(m, 'SquarePulse/1', 'Sum1/1');
        save_system(m);
        out = sim(m, 'StopTime', '100');
        delete_line(m, 'SquarePulse/1', 'Sum1/1');
        delete_block([m '/SquarePulse']);
        add_line(m, 'Step1/1', 'Sum1/1');
        save_system(m);
        bdclose(m);
        if strcmp(m, model_rbf)
            out_rbf=out;
        else
            out_bp=out;
        end
    end
    sq_rbf = analyze_square(out_rbf.t, out_rbf.r, out_rbf.y);
    sq_bp  = analyze_square(out_bp.t, out_bp.r, out_bp.y);
    all_rbf.square = sq_rbf; all_bp.square = sq_bp;
    fprintf('  RBF: %d transitions, rise_sigma=%.1f%%, fall_sigma=%.1f%%\n', sq_rbf.n_transitions, sq_rbf.avg_sigma_up*100, sq_rbf.avg_sigma_down*100);
    fprintf('  BP:  %d transitions, rise_sigma=%.1f%%, fall_sigma=%.1f%%\n', sq_bp.n_transitions, sq_bp.avg_sigma_up*100, sq_bp.avg_sigma_down*100);
    plot_square_compare(out_rbf.t,out_rbf.r,out_rbf.y,out_bp.t,out_bp.r,out_bp.y,fig_dir);

    %% ===== 测试 4: 斜坡跟踪 =====
    fprintf('\n========== 测试 4: 斜坡跟踪 ==========\n');
    for mdl = {model_rbf, model_bp}
        m = mdl{1};
        if strcmp(m, model_rbf)
            cd(fullfile(fileparts(mfilename('fullpath'))));
        else
            cd(bp1_path);
        end
        open_system(m);
        if strcmp(m, model_rbf)
            rt=sfroot; c1=rt.find('-isa','Stateflow.EMChart','Path',[m '/MATLAB Function1']);
            c2=rt.find('-isa','Stateflow.EMChart','Path',[m '/MATLAB Function2']);
            c1.Script=fileread('BP_PID_Controller.m'); c2.Script=fileread('RBF_Identifier_claude.m');
        else
            rt2=sfroot; c=rt2.find('-isa','Stateflow.EMChart','Path',[m '/MATLAB Function']);
            c.Script=fileread('bp_pid_controller.m');
        end
        set_param([m '/Manual Switch'], 'sw', '1');
        add_block('simulink/Sources/Ramp', [m '/RampSrc']);
        set_param([m '/RampSrc'], 'slope','0.05','start','5','InitialOutput','0');
        delete_line(m, 'Step1/1', 'Sum1/1');
        add_line(m, 'RampSrc/1', 'Sum1/1');
        save_system(m);
        out = sim(m, 'StopTime', '100');
        t=out.t; N=length(t); idx_ss=round(N*0.5):N;
        ramp_err = mean(abs(out.r(idx_ss)-out.y(idx_ss)));
        delete_line(m, 'RampSrc/1', 'Sum1/1');
        delete_block([m '/RampSrc']);
        add_line(m, 'Step1/1', 'Sum1/1');
        save_system(m);
        bdclose(m);
        if strcmp(m, model_rbf)
            all_rbf.ramp=struct('steady_err',ramp_err); out_rbf=out;
        else
            all_bp.ramp=struct('steady_err',ramp_err); out_bp=out;
        end
    end
    fprintf('  RBF: ramp_err=%.4f, BP: ramp_err=%.4f\n', all_rbf.ramp.steady_err, all_bp.ramp.steady_err);
    plot_ramp_compare(out_rbf, out_bp, fig_dir);

    %% ===== 测试 5: 参数摄动 =====
    fprintf('\n========== 测试 5: 参数摄动 ==========\n');
    tf_cases = {'[2]','[1 1.2 1]','增益加倍'; '[1]','[1 0.3 1]','欠阻尼'; '[1]','[1 3 1]','过阻尼'};
    baseline_num='[1]'; baseline_den='[1 1.2 1]';
    for i = 1:size(tf_cases,1)
        for mdl = {model_rbf, model_bp}
            m = mdl{1};
            if strcmp(m, model_rbf)
                cd(fullfile(fileparts(mfilename('fullpath'))));
            else
                cd(bp1_path);
            end
            open_system(m);
            if strcmp(m, model_rbf)
                rt=sfroot; c1=rt.find('-isa','Stateflow.EMChart','Path',[m '/MATLAB Function1']);
                c2=rt.find('-isa','Stateflow.EMChart','Path',[m '/MATLAB Function2']);
                c1.Script=fileread('BP_PID_Controller.m'); c2.Script=fileread('RBF_Identifier_claude.m');
            else
                rt2=sfroot; c=rt2.find('-isa','Stateflow.EMChart','Path',[m '/MATLAB Function']);
                c.Script=fileread('bp_pid_controller.m');
            end
            set_param([m '/Manual Switch'],'sw','1');
            set_param([m '/Step1'],'Time','5','Before','0','After','1');
            set_param([m '/Transfer Fcn'],'Numerator',tf_cases{i,1},'Denominator',tf_cases{i,2});
            save_system(m);
            out = sim(m,'StopTime','50');
            set_param([m '/Transfer Fcn'],'Numerator',baseline_num,'Denominator',baseline_den);
            save_system(m);
            bdclose(m);
            N=length(out.t); idx_ss=round(N*0.7):N;
            ess_val = mean(abs(out.r(idx_ss)-out.y(idx_ss)));
            sigma_val = max(0,(max(out.y)-1)/1);
            if strcmp(m, model_rbf)
                all_rbf.param(i)=struct('case',tf_cases{i,3},'ess',ess_val,'sigma',sigma_val);
            else
                all_bp.param(i)=struct('case',tf_cases{i,3},'ess',ess_val,'sigma',sigma_val);
            end
        end
        fprintf('  %s: RBF sig=%.1f%%, BP sig=%.1f%%\n', tf_cases{i,3}, ...
            all_rbf.param(i).sigma*100, all_bp.param(i).sigma*100);
    end

    %% ===== 测试 6: 外加扰动 =====
    fprintf('\n========== 测试 6: 外加扰动 =====\n');
    for mdl = {model_rbf, model_bp}
        m = mdl{1};
        if strcmp(m, model_rbf)
            cd(fullfile(fileparts(mfilename('fullpath'))));
        else
            cd(bp1_path);
        end
        open_system(m);
        if strcmp(m, model_rbf)
            rt=sfroot; c1=rt.find('-isa','Stateflow.EMChart','Path',[m '/MATLAB Function1']);
            c2=rt.find('-isa','Stateflow.EMChart','Path',[m '/MATLAB Function2']);
            c1.Script=fileread('BP_PID_Controller.m'); c2.Script=fileread('RBF_Identifier_claude.m');
        else
            rt2=sfroot; c=rt2.find('-isa','Stateflow.EMChart','Path',[m '/MATLAB Function']);
            c.Script=fileread('bp_pid_controller.m');
        end
        set_param([m '/Manual Switch'],'sw','1');
        add_block('simulink/Sources/Pulse Generator',[m '/DisturbPulse']);
        set_param([m '/DisturbPulse'],'Amplitude','2','Period','1000','PulseWidth','0.5','PhaseDelay','30');
        add_line(m, 'DisturbPulse/1', 'Sum1/2');
        save_system(m);
        out = sim(m,'StopTime','50');
        [max_dev, rec_time] = disturb_metrics(out.t, out.r, out.y, 30);
        dplh = get_param([m '/DisturbPulse'],'LineHandles');
        if dplh.Outport>0, delete_line(dplh.Outport); end
        delete_block([m '/DisturbPulse']);
        save_system(m);
        bdclose(m);
        if strcmp(m, model_rbf)
            all_rbf.disturb=struct('max_dev',max_dev,'recovery_time',rec_time);
            out_rbf=out;
        else
            all_bp.disturb=struct('max_dev',max_dev,'recovery_time',rec_time);
            out_bp=out;
        end
    end
    fprintf('  RBF: max=%.4f, rec=%.2fs | BP: max=%.4f, rec=%.2fs\n', ...
        all_rbf.disturb.max_dev, all_rbf.disturb.recovery_time, ...
        all_bp.disturb.max_dev, all_bp.disturb.recovery_time);

    %% ===== 测试 7: 测量噪声 =====
    fprintf('\n========== 测试 7: 测量噪声 =====\n');
    noise_vars = [0.01, 0.05];
    for i = 1:length(noise_vars)
        nv = noise_vars(i);
        for mdl = {model_rbf, model_bp}
            m = mdl{1};
            if strcmp(m, model_rbf)
                cd(fullfile(fileparts(mfilename('fullpath'))));
            else
                cd(bp1_path);
            end
            open_system(m);
            if strcmp(m, model_rbf)
                rt=sfroot; c1=rt.find('-isa','Stateflow.EMChart','Path',[m '/MATLAB Function1']);
                c2=rt.find('-isa','Stateflow.EMChart','Path',[m '/MATLAB Function2']);
                c1.Script=fileread('BP_PID_Controller.m'); c2.Script=fileread('RBF_Identifier_claude.m');
            else
                rt2=sfroot; c=rt2.find('-isa','Stateflow.EMChart','Path',[m '/MATLAB Function']);
                c.Script=fileread('bp_pid_controller.m');
            end
            set_param([m '/Manual Switch'],'sw','1');
            rng(0);
            set_param([m '/Random Number'],'Mean','0','Variance',num2str(nv));
            add_line(m, 'Random Number/1', 'Sum1/2');
            save_system(m);
            out = sim(m,'StopTime','50');
            N=length(out.t); idx_ss=round(N*0.3):N;
            rms_val = rms(out.r(idx_ss)-out.y(idx_ss));
            u_std_val = std(out.u(idx_ss));
            rn_lh = get_param([m '/Random Number'],'LineHandles');
            if rn_lh.Outport>0, delete_line(rn_lh.Outport); end
            save_system(m);
            bdclose(m);
            if strcmp(m, model_rbf)
                all_rbf.noise(i)=struct('var',nv,'rms_err',rms_val,'u_std',u_std_val);
            else
                all_bp.noise(i)=struct('var',nv,'rms_err',rms_val,'u_std',u_std_val);
            end
        end
        fprintf('  noise var=%.2f: RBF RMS=%.4f, BP RMS=%.4f\n', nv, ...
            all_rbf.noise(i).rms_err, all_bp.noise(i).rms_err);
    end

    %% ===== 测试 8: 纯滞后 =====
    fprintf('\n========== 测试 8: 纯滞后 =====\n');
    delays = [0.1, 0.3, 0.5];
    for i = 1:length(delays)
        tau = delays(i);
        for mdl = {model_rbf, model_bp}
            m = mdl{1};
            if strcmp(m, model_rbf)
                cd(fullfile(fileparts(mfilename('fullpath'))));
            else
                cd(bp1_path);
            end
            open_system(m);
            if strcmp(m, model_rbf)
                rt=sfroot; c1=rt.find('-isa','Stateflow.EMChart','Path',[m '/MATLAB Function1']);
                c2=rt.find('-isa','Stateflow.EMChart','Path',[m '/MATLAB Function2']);
                c1.Script=fileread('BP_PID_Controller.m'); c2.Script=fileread('RBF_Identifier_claude.m');
            else
                rt2=sfroot; c=rt2.find('-isa','Stateflow.EMChart','Path',[m '/MATLAB Function']);
                c.Script=fileread('bp_pid_controller.m');
            end
            set_param([m '/Manual Switch'],'sw','1');
            sw_lh = get_param([m '/Manual Switch'],'LineHandles');
            delete_line(sw_lh.Outport);
            add_line(m, 'Manual Switch/1', 'Transport Delay/1');
            add_line(m, 'Transport Delay/1', 'Transfer Fcn/1');
            set_param([m '/Transport Delay'],'DelayTime',num2str(tau));
            save_system(m);
            out = sim(m,'StopTime','50');
            N=length(out.t); idx_ss=round(N*0.7):N;
            ess_val = mean(abs(out.r(idx_ss)-out.y(idx_ss)));
            sigma_val = max(0,(max(out.y)-1)/1);
            delete_line(m, 'Transport Delay/1', 'Transfer Fcn/1');
            delete_line(m, 'Manual Switch/1', 'Transport Delay/1');
            add_line(m, 'Manual Switch/1', 'Transfer Fcn/1');
            save_system(m);
            bdclose(m);
            if strcmp(m, model_rbf)
                all_rbf.delay(i)=struct('tau',tau,'ess',ess_val,'sigma',sigma_val);
            else
                all_bp.delay(i)=struct('tau',tau,'ess',ess_val,'sigma',sigma_val);
            end
        end
        fprintf('  tau=%.1f: RBF sig=%.1f%%, BP sig=%.1f%%\n', tau, ...
            all_rbf.delay(i).sigma*100, all_bp.delay(i).sigma*100);
    end

    %% ===== 汇总对比 =====
    fprintf('\n========== 综合对比汇总 ==========\n');
    fprintf('测试项                | RBF           | BPNN_1        \n');
    fprintf('----------------------|---------------|---------------\n');
    fprintf('阶跃: sigma%%          | %6.1f         | %6.1f\n', all_rbf.step.sigma*100, all_bp.step.sigma*100);
    fprintf('阶跃: ts (s)          | %6.2f         | %6.2f\n', all_rbf.step.ts, all_bp.step.ts);
    fprintf('正弦 f=0.1: RMS       | %6.4f         | %6.4f\n', all_rbf.sine(1).rms_err, all_bp.sine(1).rms_err);
    fprintf('正弦 f=1.0: RMS       | %6.4f         | %6.4f\n', all_rbf.sine(2).rms_err, all_bp.sine(2).rms_err);
    fprintf('正弦 f=5.0: RMS       | %6.4f         | %6.4f\n', all_rbf.sine(3).rms_err, all_bp.sine(3).rms_err);
    fprintf('方波: n_transitions   | %6d          | %6d\n', all_rbf.square.n_transitions, all_bp.square.n_transitions);
    fprintf('斜坡: steady_err      | %6.4f         | %6.4f\n', all_rbf.ramp.steady_err, all_bp.ramp.steady_err);
    for i=1:3
        fprintf('参数摄动(%s): sigma%% | %6.1f         | %6.1f\n', ...
            all_rbf.param(i).case, all_rbf.param(i).sigma*100, all_bp.param(i).sigma*100);
    end
    fprintf('扰动: max_dev         | %6.4f         | %6.4f\n', all_rbf.disturb.max_dev, all_bp.disturb.max_dev);
    fprintf('扰动: rec_time        | %6.2f         | %6.2f\n', all_rbf.disturb.recovery_time, all_bp.disturb.recovery_time);
    fprintf('噪声0.01: RMS         | %6.4f         | %6.4f\n', all_rbf.noise(1).rms_err, all_bp.noise(1).rms_err);
    fprintf('噪声0.05: RMS         | %6.4f         | %6.4f\n', all_rbf.noise(2).rms_err, all_bp.noise(2).rms_err);
    for i=1:3
        fprintf('时滞tau=%.1f: sigma%%  | %6.1f         | %6.1f\n', ...
            all_rbf.delay(i).tau, all_rbf.delay(i).sigma*100, all_bp.delay(i).sigma*100);
    end

    cd(fullfile(fileparts(mfilename('fullpath'))));
    save('test_results.mat', 'all_rbf', 'all_bp');

catch ME
    fprintf(2, '测试中断: %s\n', ME.message);
    cd(fullfile(fileparts(mfilename('fullpath'))));
end
end

%% ==================== 指标计算函数 ====================

function m = step_metrics(t, r, y, u)
    N = length(t);
    idx0 = find(r >= 0.5, 1); if isempty(idx0), idx0 = 1; end
    y_final = mean(y(round(N*0.8):N));
    y_peak = max(y(idx0:end));
    m.sigma = max(0, (y_peak - y_final) / abs(y_final + eps));
    y10 = 0.1 * y_final; y90 = 0.9 * y_final;
    t10 = t(find(y(idx0:end) >= y10, 1) + idx0 - 1);
    t90 = t(find(y(idx0:end) >= y90, 1) + idx0 - 1);
    m.tr = t90 - t10;
    [~, idx_peak] = max(y(idx0:end));
    m.tp = t(idx_peak + idx0 - 1) - t(idx0);
    tol = 0.02 * abs(y_final); m.ts = NaN;
    for k = N:-1:idx0
        if abs(y(k) - y_final) > tol, m.ts = t(k) - t(idx0); break; end
    end
    m.ess = mean(abs(r(round(N*0.8):N) - y(round(N*0.8):N)));
end

function m = analyze_square(t, r, y)
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
            rec_time = t(k) - t_disturb; break;
        end
    end
end

%% ==================== 对比绘图函数 ====================

function plot_step_compare(out_rbf, out_bp, fig_dir)
    t1=out_rbf.t; r1=out_rbf.r; y1=out_rbf.y; u1=out_rbf.u;
    t2=out_bp.t; r2=out_bp.r; y2=out_bp.y; u2=out_bp.u;
    figure('Name','阶跃对比','NumberTitle','off','Visible','off');
    subplot(2,1,1);
    plot(t1,r1,'k--',t1,y1,'r',t2,y2,'b','LineWidth',1.2);
    xlabel('t (s)'); ylabel('y'); legend('r','RBF','BPNN_1'); title('阶跃响应对比'); grid on;
    subplot(2,1,2);
    plot(t1,u1,'r',t2,u2,'b','LineWidth',1.2);
    xlabel('t (s)'); ylabel('u'); legend('RBF','BPNN_1'); title('控制量对比'); grid on;
    saveas(gcf, fullfile(fig_dir, 'fig_compare_step.png')); close;
end

function plot_sine_compare(t1,r1,y1,t2,r2,y2,f,fig_dir)
    figure('Name',sprintf('正弦对比 f=%.1f',f),'NumberTitle','off','Visible','off');
    subplot(2,1,1);
    plot(t1,r1,'k--',t1,y1,'r',t2,y2,'b','LineWidth',1.2);
    xlabel('t (s)'); ylabel('y'); legend('r','RBF','BPNN_1');
    title(sprintf('正弦跟踪对比 f=%.1f rad/s',f)); grid on;
    saveas(gcf, fullfile(fig_dir, sprintf('fig_compare_sine_%d.png',round(f*10)))); close;
end

function plot_square_compare(t1,r1,y1,t2,r2,y2,fig_dir)
    figure('Name','方波对比','NumberTitle','off','Visible','off');
    subplot(2,1,1);
    plot(t1,r1,'k--',t1,y1,'r',t2,y2,'b','LineWidth',1.2);
    xlabel('t (s)'); ylabel('y'); legend('r','RBF','BPNN_1'); title('方波跟踪对比'); grid on;
    saveas(gcf, fullfile(fig_dir, 'fig_compare_square.png')); close;
end

function plot_ramp_compare(out_rbf, out_bp, fig_dir)
    t1=out_rbf.t; r1=out_rbf.r; y1=out_rbf.y;
    t2=out_bp.t; r2=out_bp.r; y2=out_bp.y;
    figure('Name','斜坡对比','NumberTitle','off','Visible','off');
    plot(t1,r1,'k--',t1,y1,'r',t2,y2,'b','LineWidth',1.2);
    xlabel('t (s)'); ylabel('y'); legend('r','RBF','BPNN_1'); title('斜坡跟踪对比'); grid on;
    saveas(gcf, fullfile(fig_dir, 'fig_compare_ramp.png')); close;
end
