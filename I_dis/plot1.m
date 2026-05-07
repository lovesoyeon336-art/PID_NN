close all;

% 1. 提取时间轴（直接从 y 里面拿，最准）
t_val = out.t;

% 2. 提取数值矩阵（Data 包含了两列：理想信号和跟踪信号）
y_val = out.y;

% 3. 绘图
% 注意：y_val 现在是一个矩阵，第一列是 y_val(:,1)，第二列是 y_val(:,2)
plot(t_val, y_val(:,1), 'r', t_val, y_val(:,2), 'k', 'linewidth', 2);

xlabel('time(s)'); 
ylabel('yd, y');
legend('Ideal position signal', 'Position tracking');