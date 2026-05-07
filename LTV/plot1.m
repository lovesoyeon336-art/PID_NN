close all;
clc;

% 1. 从结构体提取时间和信号数据
t = out.tout;                  % 提取时间向量（1363×1）
y_data = out.y.Data;           % 从timeseries中提取信号矩阵（N×2）

% 2. 绘制跟踪对比曲线
plot(t, y_data(:,1), 'r', t, y_data(:,2), 'k:', 'linewidth', 2);
xlabel('time(s)');
ylabel('yd,y');
legend('Ideal position signal', 'Position tracking');
grid on;
set(gca, 'FontSize', 12);      % 优化字体，适合论文/报告
