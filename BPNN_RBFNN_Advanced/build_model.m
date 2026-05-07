function build_model()
% =========================================================================
% build_model.m — 构建 BP+RBF 神经网络自适应控制 Simulink 模型
% =========================================================================
% Phase 1: 创建模型, 添加所有模块, 设置参数
% Phase 2: 嵌入式 MATLAB Function 代码
% Phase 3: 更新框图以自动检测 I/O 端口
% Phase 4: 连线
% Phase 5: 保存
%
% 运行: >> build_model
% =========================================================================

model = 'BP_RBFNN_Sim';
cd(fileparts(mfilename('fullpath')));

%% ===== Phase 0: 清理 =====
% 强制关闭所有已打开的模型实例
bdclose('all');
old_file = fullfile(pwd, [model '.slx']);
if exist(old_file, 'file'), delete(old_file); end

%% ===== Phase 1: 创建模型 & 添加模块 =====
fprintf('Phase 1: Creating model and adding blocks...\n');
new_system(model);
open_system(model);

set_param(model, ...
    'Solver',     'ode4', ...
    'FixedStep',  '0.001', ...
    'StopTime',   '15', ...
    'StartTime',  '0.0');

% ----- 参考信号 -----
add_block('simulink/Sources/Step',                [model '/Step']);
add_block('simulink/Sources/Sine Wave',           [model '/Sine']);
add_block('simulink/Signal Routing/Manual Switch',[model '/Ref SW']);
set_param([model '/Step'], 'Time', '0.5', 'Before', '0', 'After', '1');
set_param([model '/Sine'], 'Amplitude', '1', 'Frequency', '0.5');

% ----- 控制器 (MATLAB Function) -----
add_block('simulink/User-Defined Functions/MATLAB Function', [model '/BP PID']);

% ----- 控制路径 -----
add_block('simulink/Discontinuities/Saturation',  [model '/Saturation']);
set_param([model '/Saturation'], 'UpperLimit', '10', 'LowerLimit', '-10');

% ----- 扰动支路 -----
add_block('simulink/Sources/Step',                [model '/Dist Step']);
add_block('simulink/Sources/Constant',            [model '/Zero D']);
add_block('simulink/Signal Routing/Manual Switch',[model '/Dist SW']);
add_block('simulink/Math Operations/Sum',         [model '/Dist Sum']);
set_param([model '/Dist Step'], 'Time', '8', 'Before', '0', 'After', '1.5');
set_param([model '/Zero D'],   'Value', '0');
set_param([model '/Dist Sum'], 'Inputs', '|++', 'IconShape', 'round');

% ----- 被控对象 (Hammerstein: 死区 + 线性) -----
add_block('simulink/Discontinuities/Dead Zone',   [model '/Dead Zone']);
add_block('simulink/Continuous/Transfer Fcn',     [model '/Plant']);
set_param([model '/Dead Zone'], 'LowerValue', '-0.3', 'UpperValue', '0.3');
set_param([model '/Plant'], 'Numerator', '[10]', 'Denominator', '[1 2 5]');

% ----- 噪声支路 -----
add_block('simulink/Sources/Band-Limited White Noise', [model '/Noise']);
add_block('simulink/Sources/Constant',                 [model '/Zero N']);
add_block('simulink/Signal Routing/Manual Switch',     [model '/Noise SW']);
add_block('simulink/Math Operations/Sum',              [model '/Noise Sum']);
set_param([model '/Noise'], 'Cov', '0.0001', 'Ts', '0.001', 'seed', '23341');
set_param([model '/Zero N'], 'Value', '0');
set_param([model '/Noise Sum'], 'Inputs', '|++', 'IconShape', 'round');

% ----- RBF 辨识器相关 -----
add_block('simulink/Discrete/Unit Delay', [model '/Delay']);
set_param([model '/Delay'], 'SampleTime', '0.001');
add_block('simulink/User-Defined Functions/MATLAB Function', [model '/RBF Ident']);

% ----- 监控 -----
add_block('simulink/Sinks/Scope',          [model '/Scope y-r']);
set_param([model '/Scope y-r'], 'NumInputPorts', '2');
add_block('simulink/Sinks/Scope',          [model '/Scope u']);
add_block('simulink/Sinks/Scope',          [model '/Scope PID']);
set_param([model '/Scope PID'], 'NumInputPorts', '3');
add_block('simulink/Sinks/Scope',          [model '/Scope Ident']);
set_param([model '/Scope Ident'], 'NumInputPorts', '2');

% ----- 数据记录 (Timeseries 格式) -----
tw_names = {'t_out','r_out','y_out','u_out','Kp_out','Ki_out', ...
            'Kd_out','yhat_out','dydu_out'};
for i = 1:length(tw_names)
    add_block('simulink/Sinks/To Workspace', [model '/' tw_names{i}]);
    set_param([model '/' tw_names{i}], ...
        'VariableName', tw_names{i}, 'SaveFormat', 'Timeseries');
end
add_block('simulink/Sources/Clock', [model '/Clock']);

% ----- 粗略布局 -----
blk = @(name, x, y, w, h) set_param([model '/' name], ...
    'Position', [x, y, x+w, y+h]);
blk('Step',       30,  40,  70, 32);
blk('Sine',       30,  100, 70, 32);
blk('Ref SW',     120, 55,  40, 40);
blk('BP PID',     190, 30,  70, 100);
blk('Saturation', 290, 55,  40, 40);
blk('Dist Step',  290, 140, 60, 32);
blk('Zero D',     290, 185, 60, 32);
blk('Dist SW',    370, 145, 40, 40);
blk('Dist Sum',   430, 55,  30, 40);
blk('Dead Zone',  490, 55,  40, 40);
blk('Plant',      560, 55,  60, 40);
blk('Noise Sum',  650, 55,  30, 40);
blk('Noise',      560, 140, 70, 32);
blk('Zero N',     560, 185, 70, 32);
blk('Noise SW',   650, 145, 40, 40);
blk('Delay',      310, 240, 40, 40);
blk('RBF Ident',  400, 230, 70, 90);
blk('Scope y-r',  740, 25,  60, 45);
blk('Scope u',    740, 85,  60, 45);
blk('Scope PID',  740, 145, 60, 60);
blk('Scope Ident',740, 220, 60, 45);
blk('Clock',      30,  240, 40, 30);

for i = 1:length(tw_names)
    set_param([model '/' tw_names{i}], ...
        'Position', [830, 20+(i-1)*28, 880, 20+i*28-2]);
end

fprintf('  %d blocks added.\n', length(find_system(model, 'Type', 'Block')));

%% ===== Phase 2: MATLAB Function 代码 =====
fprintf('Phase 2: Embedding MATLAB Function code...\n');
try
    load_system(model);
    bp_src  = fileread('BP_PID_Controller.m');
    rbf_src = fileread('RBF_Identifier.m');

    sf_root = sfroot;
    c = sf_root.find('Path', [model '/BP PID'], '-isa', 'Stateflow.EMChart');
    if ~isempty(c)
        c.Script = bp_src;
        % Leave SampleTime as inherited (-1); set at subsystem level instead
        fprintf('  BP PID: OK (%d bytes)\n', length(bp_src));
    else, warning('  BP PID EMChart not found'); end

    c = sf_root.find('Path', [model '/RBF Ident'], '-isa', 'Stateflow.EMChart');
    if ~isempty(c)
        c.Script = rbf_src;
        fprintf('  RBF Ident: OK (%d bytes)\n', length(rbf_src));
    else, warning('  RBF Ident EMChart not found'); end

    % 设置为原子单元并指定采样时间 (persistent变量需要离散采样时间)
    set_param([model '/BP PID'], 'TreatAsAtomicUnit', 'on');
    set_param([model '/BP PID'], 'SystemSampleTime', '0.001');
    set_param([model '/RBF Ident'], 'TreatAsAtomicUnit', 'on');
    set_param([model '/RBF Ident'], 'SystemSampleTime', '0.001');
    fprintf('  MATLAB Function blocks set as atomic units (Ts=0.001).\n');
catch ME
    fprintf(2, '  Stateflow API error: %s\n', ME.message);
    fprintf(2, '  After build, manually paste code into MATLAB Function blocks.\n');
end

%% ===== Phase 3: 更新框图 (检测端口) =====
fprintf('Phase 3: Updating diagram (auto-detecting I/O ports)...\n');
try
    set_param(model, 'SimulationCommand', 'update');
    fprintf('  Diagram updated. Ports detected.\n');
catch ME
    fprintf(2, '  Update warning: %s\n', ME.message);
    fprintf(2, '  Please press Ctrl+D after model opens.\n');
end

%% ===== Phase 4: 连线 =====
fprintf('Phase 4: Connecting blocks...\n');
% 辅助函数: 安全连线 (失败时警告而非崩溃)
    function safe_line(src, dst)
        try
            add_line(model, src, dst);
        catch ME
            fprintf(2, '  WARNING: line %s -> %s failed: %s\n', src, dst, ME.message);
        end
    end

% 信号流: Ref → BP PID(in1) → ... → y_meas → BP PID(in2) + RBF(in2)
%          RBF(out1:dy_du) → BP PID(in3)

% 参考信号
safe_line('Step/1',   'Ref SW/1');
safe_line('Sine/1',   'Ref SW/2');
safe_line('Ref SW/1', 'BP PID/1');                  % r → BP PID in1

% 控制路径 (从 BP PID 输出1: u)
safe_line('BP PID/1',     'Saturation/1');          % u → 限幅
safe_line('Saturation/1', 'Dist Sum/1');            % → 扰动求和
safe_line('Saturation/1', 'Delay/1');               % → 延迟 (为RBF提供u(k-1))

% 扰动支路
safe_line('Dist Step/1', 'Dist SW/1');
safe_line('Zero D/1',    'Dist SW/2');
safe_line('Dist SW/1',   'Dist Sum/2');             % 扰动 → Sum

% 对象
safe_line('Dist Sum/1',  'Dead Zone/1');
safe_line('Dead Zone/1', 'Plant/1');

% 噪声支路
safe_line('Plant/1',   'Noise Sum/1');              % 纯对象输出 → Sum
safe_line('Noise/1',   'Noise SW/1');
safe_line('Zero N/1',  'Noise SW/2');
safe_line('Noise SW/1','Noise Sum/2');              % 噪声 → Sum

% 反馈: y_meas → BP PID in2
safe_line('Noise Sum/1', 'BP PID/2');               % y_meas → BP PID in2

% RBF 辨识器
safe_line('Delay/1',     'RBF Ident/1');            % u(k-1) → RBF in1
safe_line('Noise Sum/1', 'RBF Ident/2');            % y(k)   → RBF in2
safe_line('RBF Ident/1', 'BP PID/3');               % dy_du  → BP PID in3

% ---- Scope 连线 ----
safe_line('Ref SW/1',    'Scope y-r/1');            % r → Scope
safe_line('Noise Sum/1', 'Scope y-r/2');            % y → Scope
safe_line('BP PID/1',    'Scope u/1');              % u → Scope
safe_line('BP PID/2',    'Scope PID/1');            % Kp → Scope
safe_line('BP PID/3',    'Scope PID/2');            % Ki → Scope
safe_line('BP PID/4',    'Scope PID/3');            % Kd → Scope
safe_line('Noise Sum/1', 'Scope Ident/1');          % y → Scope
safe_line('RBF Ident/2', 'Scope Ident/2');          % y_hat → Scope

% ---- To Workspace 连线 ----
safe_line('Clock/1',     't_out/1');
safe_line('Ref SW/1',    'r_out/1');
safe_line('Noise Sum/1', 'y_out/1');
safe_line('BP PID/1',    'u_out/1');
safe_line('BP PID/2',    'Kp_out/1');
safe_line('BP PID/3',    'Ki_out/1');
safe_line('BP PID/4',    'Kd_out/1');
safe_line('RBF Ident/2', 'yhat_out/1');
safe_line('RBF Ident/1', 'dydu_out/1');

n_lines = length(find_system(model, 'FindAll', 'on', 'type', 'line'));
fprintf('  %d lines connected.\n', n_lines);

%% ===== Phase 5: 保存 =====
fprintf('Phase 5: Saving model...\n');
save_path = fullfile(pwd, [model '.slx']);
save_system(model, save_path);
fprintf('\n=== Model saved: %s ===\n', save_path);
fprintf('Open:  >> open_system(''%s'')\n', model);
fprintf('Run tests: >> run_tests\n');

end
