function plot_results()
% PLOT_RESULTS 运行 BP_RBFNN.slx 仿真并绘制结果图
%   仿真后从 out 结构体读取变量，生成 5 张图：
%     图1: P, I, D 参数自适应曲线
%     图2: 控制量 u
%     图3: 设定值 r 与输出 y 跟踪对比
%     图4: 雅可比 ∂y/∂u
%     图5: RBF 辨识器预测输出 y_hat

    %% 仿真
    model = 'BP_RBFNN';
    fprintf('正在仿真 %s ...\n', model);
    out = sim(model, 'StopTime', '1000');
    fprintf('仿真完成\n');

    t = out.t;

    here = fileparts(mfilename('fullpath'));
    fig_dir = fullfile(here, 'figures');
    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir);
    end

    %% 图1: PID 参数 (P, I, D)
    figure('Name', 'PID 参数自适应', 'NumberTitle', 'off');
    plot(t, out.p, 'LineWidth', 1.2); hold on;
    plot(t, out.i, 'LineWidth', 1.2);
    plot(t, out.d, 'LineWidth', 1.2);
    xlabel('时间 t (s)');
    ylabel('参数值');
    legend('Kp', 'Ki', 'Kd', 'Location', 'best');
    title('BP 神经网络 PID 参数自适应');
    grid on;
    saveas(gcf, fullfile(fig_dir, 'fig1_pid_params.png'));

    %% 图2: 控制量 u
    figure('Name', '控制量', 'NumberTitle', 'off');
    plot(t, out.u, 'LineWidth', 1.2);
    xlabel('时间 t (s)');
    ylabel('u');
    title('控制量');
    grid on;
    saveas(gcf, fullfile(fig_dir, 'fig2_control_u.png'));

    %% 图3: 设定值 r 与输出 y
    figure('Name', '跟踪性能', 'NumberTitle', 'off');
    plot(t, out.r, 'LineWidth', 1.2); hold on;
    plot(t, out.y, 'LineWidth', 1.2);
    xlabel('时间 t (s)');
    ylabel('信号值');
    legend('r (设定值)', 'y (输出)', 'Location', 'best');
    title('系统输出跟踪');
    grid on;
    saveas(gcf, fullfile(fig_dir, 'fig3_tracking.png'));

    %% 图4: 雅可比 dy/du
    figure('Name', '雅可比估计', 'NumberTitle', 'off');
    plot(t, out.dydu, 'LineWidth', 1.2);
    xlabel('时间 t (s)');
    ylabel('\partial y / \partial u');
    title('RBF 辨识器雅可比估计');
    grid on;
    saveas(gcf, fullfile(fig_dir, 'fig4_dydu.png'));

    %% 图5: RBF 预测输出 y_hat
    figure('Name', 'RBF 预测输出', 'NumberTitle', 'off');
    plot(t, out.y_hat, 'LineWidth', 1.2);
    xlabel('时间 t (s)');
    ylabel('y_{hat}');
    title('RBF 神经网络预测输出');
    grid on;
    saveas(gcf, fullfile(fig_dir, 'fig5_yhat.png'));

    save(fullfile(here, 'simulation_output.mat'), 'out');
    fprintf('仿真数据已保存到 %s\n', fullfile(here, 'simulation_output.mat'));
    fprintf('5 张图已保存到 %s\n', fig_dir);
end
