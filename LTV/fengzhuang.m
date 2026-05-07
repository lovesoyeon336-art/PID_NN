num = [1];
den = [1 25 0];
G = tf(num, den);
% 自动生成可编辑的 Simulink 子系统
open_system(new_system);
realizeiz(G, 'Block', 'Subsystem');