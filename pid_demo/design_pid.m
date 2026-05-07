% PID Controller Design Algorithm
% Uses pidtune to design an optimal PID controller for the DC motor

% Load motor parameters
run('dc_motor_params.m');

% Design PID controller using pidtune
% Target bandwidth: 10 rad/s, Phase margin: 60 degrees
opts = pidtuneOptions('PhaseMargin', 60, 'DesignFocus', 'balanced');
[C, info] = pidtune(G_motor, 'PID', 10, opts);

fprintf('\n=== PID Controller Design Results ===\n');
fprintf('Kp = %.4f\n', C.Kp);
fprintf('Ki = %.4f\n', C.Ki);
fprintf('Kd = %.4f\n', C.Kd);
[Gm, Pm] = margin(C * G_motor);
fprintf('Gain margin = %.2f dB\n', 20*log10(Gm));
fprintf('Phase margin = %.2f deg\n', Pm);

% Closed-loop system
T = feedback(C * G_motor, 1);

% Step response analysis
S = stepinfo(T);
fprintf('\n=== Closed-Loop Performance ===\n');
fprintf('Rise time = %.4f s\n', S.RiseTime);
fprintf('Settling time = %.4f s\n', S.SettlingTime);
fprintf('Overshoot = %.2f %%\n', S.Overshoot);
fprintf('Steady-state gain = %.4f\n', dcgain(T));

save('controller_data.mat', 'C', 'G_motor', 'T', 'S', 'info');
disp('Controller data saved to controller_data.mat');
