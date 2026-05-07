% Build Simulink Model: DC Motor PID Control
% Programmatically builds a Simulink model for PID control simulation

function build_simulink_model()
    % Load controller data
    load('controller_data.mat', 'C', 'G_motor');

    modelName = 'dc_motor_pid_ctrl';

    % Close existing model if open
    if bdIsLoaded(modelName)
        close_system(modelName, 1);
    end

    % Create new model
    new_system(modelName);
    open_system(modelName);

    % ---- Add Blocks ----

    % Step input (reference position)
    add_block('simulink/Sources/Step', [modelName '/Reference'], ...
        'Position', [80, 100, 130, 130], ...
        'Time', '0', ...
        'Before', '0', ...
        'After', '1');

    % Sum block (error = ref - output)
    add_block('simulink/Math Operations/Sum', [modelName '/Error'], ...
        'Position', [200, 100, 230, 130], ...
        'IconShape', 'round', ...
        'Inputs', '|+-');

    % PID Controller
    add_block('simulink/Continuous/PID Controller', [modelName '/PID_Controller'], ...
        'Position', [310, 100, 370, 130]);

    % Configure PID block with designed gains
    set_param([modelName '/PID_Controller'], ...
        'P', num2str(C.Kp, '%.6f'), ...
        'I', num2str(C.Ki, '%.6f'), ...
        'D', num2str(C.Kd, '%.6f'));

    % DC Motor (Transfer Function)
    [num, den] = tfdata(G_motor, 'v');
    numStr = mat2str(num, 6);
    denStr = mat2str(den, 6);

    add_block('simulink/Continuous/Transfer Fcn', [modelName '/DC_Motor'], ...
        'Position', [450, 100, 520, 130], ...
        'Numerator', numStr, ...
        'Denominator', denStr);

    % Saturation (motor voltage limits)
    add_block('simulink/Discontinuities/Saturation', [modelName '/Voltage_Limit'], ...
        'Position', [370, 200, 430, 240], ...
        'UpperLimit', '12', ...
        'LowerLimit', '-12');

    % Scope for output visualization
    add_block('simulink/Sinks/Scope', [modelName '/Scope'], ...
        'Position', [610, 100, 670, 130]);

    % To Workspace blocks for data analysis
    add_block('simulink/Sinks/To Workspace', [modelName '/Output_Data'], ...
        'Position', [610, 200, 670, 230], ...
        'VariableName', 'y_out', ...
        'SaveFormat', 'Array');

    add_block('simulink/Sinks/To Workspace', [modelName '/Control_Signal'], ...
        'Position', [610, 260, 670, 290], ...
        'VariableName', 'u_ctrl', ...
        'SaveFormat', 'Array');

    add_block('simulink/Sinks/To Workspace', [modelName '/Time_Data'], ...
        'Position', [610, 320, 670, 350], ...
        'VariableName', 't_out', ...
        'SaveFormat', 'Array');

    % Clock for time data
    add_block('simulink/Sources/Clock', [modelName '/Clock'], ...
        'Position', [80, 320, 130, 350]);

    % ---- Add Lines ----

    % Reference -> Error(+)
    add_line(modelName, 'Reference/1', 'Error/1', 'autorouting', 'smart');

    % Error -> PID Controller
    add_line(modelName, 'Error/1', 'PID_Controller/1', 'autorouting', 'smart');

    % PID -> Voltage Limit (bypass for now, optional)
    % Actually let's route PID directly to motor

    % PID Controller -> DC Motor
    add_line(modelName, 'PID_Controller/1', 'DC_Motor/1', 'autorouting', 'smart');

    % DC Motor -> Error(-) (feedback)
    add_line(modelName, 'DC_Motor/1', 'Error/2', 'autorouting', 'smart');

    % DC Motor -> Scope
    add_line(modelName, 'DC_Motor/1', 'Scope/1', 'autorouting', 'smart');

    % DC Motor -> Output_Data
    add_line(modelName, 'DC_Motor/1', 'Output_Data/1', 'autorouting', 'smart');

    % PID Controller -> Control_Signal
    add_line(modelName, 'PID_Controller/1', 'Control_Signal/1', 'autorouting', 'smart');

    % Clock -> Time_Data
    add_line(modelName, 'Clock/1', 'Time_Data/1', 'autorouting', 'smart');

    % ---- Configure Model ----
    set_param(modelName, ...
        'StopTime', '10', ...
        'Solver', 'ode45', ...
        'MaxStep', '0.01', ...
        'SolverType', 'Variable-step');

    % Save model
    save_system(modelName);

    fprintf('Simulink model "%s" built successfully.\n', modelName);
    fprintf('Model saved to: %s\n', fullfile(pwd, [modelName '.slx']));
end
