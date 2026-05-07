%% DC Motor PID Control - Full Simulation Demo
% 1. Define motor parameters
% 2. Design PID controller
% 3. Build Simulink model
% 4. Run simulation
% 5. Plot and analyze results

clear; clc; close all;
cd(fileparts(mfilename('fullpath')));

fprintf('========================================\n');
fprintf('  DC Motor PID Control Demo\n');
fprintf('========================================\n\n');

%% Step 1: Load motor model
fprintf('[1/5] Loading DC motor parameters...\n');
run('dc_motor_params.m');

%% Step 2: Design PID controller
fprintf('[2/5] Designing PID controller...\n');
run('design_pid.m');

%% Step 3: Build Simulink model
fprintf('[3/5] Building Simulink model...\n');
build_simulink_model();

%% Step 4: Run simulation
fprintf('[4/5] Running simulation...\n');
simOut = sim('dc_motor_pid_ctrl', 'ReturnWorkspaceOutputs', 'on');
fprintf('Simulation completed.\n');

%% Step 5: Analyze and plot results
fprintf('[5/5] Plotting results...\n');

figure('Name', 'DC Motor PID Control Results', ...
       'NumberTitle', 'off', 'Position', [100, 100, 900, 600]);

% Extract data from simulation output
t_sim = simOut.t_out;
y_sim = simOut.y_out;
u_sim = simOut.u_ctrl;

% Time response
subplot(2,2,1);
plot(t_sim, y_sim, 'b-', 'LineWidth', 2);
hold on;
yline(1, 'r--', 'Target', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Position');
title('Step Response');
grid on;

% Control signal
subplot(2,2,2);
plot(t_sim, u_sim, 'r-', 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Control Signal (V)');
title('PID Controller Output');
grid on;

% Bode plot
subplot(2,2,3);
[Gm, Pm, Wcg, Wcp] = margin(C * G_motor);
margin(C * G_motor);
title(sprintf('Open-Loop Bode (GM=%.1f dB, PM=%.1f deg)', ...
    20*log10(Gm), Pm));

% Pole-Zero map
subplot(2,2,4);
pzmap(T);
title('Closed-Loop Pole-Zero Map');
grid on;

sgtitle('DC Motor PID Control - Simulation Results');

%% Performance summary
fprintf('\n========================================\n');
fprintf('  Simulation Results Summary\n');
fprintf('========================================\n');
fprintf('PID Gains:   Kp=%.4f, Ki=%.4f, Kd=%.4f\n', C.Kp, C.Ki, C.Kd);
fprintf('Rise time:   %.4f s\n', S.RiseTime);
fprintf('Settling:    %.4f s\n', S.SettlingTime);
fprintf('Overshoot:   %.2f %%\n', S.Overshoot);
if isfinite(Gm)
    fprintf('Gain margin: %.2f dB @ %.2f rad/s\n', 20*log10(Gm), Wcg);
end
fprintf('Phase margin: %.2f deg @ %.2f rad/s\n', Pm, Wcp);
fprintf('========================================\n');

fprintf('\nDemo completed. Check Simulink model and figures.\n');
