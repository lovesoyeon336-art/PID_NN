function plot_result(out)
% 从 out 结构体读取仿真数据并绘图
% 用法：仿真结束后在命令行输入 plot_result(out)
% out.t, out.r, out.p, out.i, out.d, out.u, out.y

    if nargin < 1
        out = evalin('base', 'out');
    end

    t = out.t;  r = out.r;
    p = out.p;  i = out.i;  d = out.d;
    u = out.u;  y = out.y;

    [here, ~, ~] = fileparts(mfilename('fullpath'));
    fig_dir = fullfile(here, 'figures');
    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir);
    end

    figure('Name', 'BPNN PID 仿真结果', 'NumberTitle', 'off');

    subplot(3,1,1);
    plot(t, p, 'LineWidth', 1.2);  hold on;
    plot(t, i, 'LineWidth', 1.2);
    plot(t, d, 'LineWidth', 1.2);
    ylabel('PID 参数');  legend('Kp', 'Ki', 'Kd');
    title('BP 神经网络 PID 参数自适应');  grid on;

    subplot(3,1,2);
    plot(t, u, 'LineWidth', 1.2);
    ylabel('控制量 u');  grid on;

    subplot(3,1,3);
    plot(t, r, 'LineWidth', 1.2);  hold on;
    plot(t, y, 'LineWidth', 1.2);
    xlabel('时间 t (s)');  ylabel('信号值');
    legend('r (设定值)', 'y (输出)');
    grid on;

    saveas(gcf, fullfile(fig_dir, 'pid_result.png'));

    save(fullfile(fig_dir, 'sim_data.mat'), 't', 'r', 'y', 'u', 'p', 'i', 'd');
end
