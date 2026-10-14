% DC Motor Parameters
% G(s) = K / (J*s^2 + B*s)

J = 0.01;    % Moment of inertia (kg*m^2)
B = 0.1;     % Damping coefficient (N*m*s)
K = 1.0;     % Motor torque constant (N*m/A)

% Transfer function: theta(s)/V(s) = K / (J*s^2 + B*s)
num = K;
den = [J, B, 0];
G_motor = tf(num, den);

disp('DC Motor Transfer Function:');
disp(G_motor);
