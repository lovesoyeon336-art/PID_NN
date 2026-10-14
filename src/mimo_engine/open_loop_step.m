% open_loop_step.m
% Engine Model 4: H=18km, Ma=0.8 开环单位阶跃响应（无控制器）
% 两路输入分别阶跃，观察 n_H (y1) 和 π_T (y2) 的响应

clear; clc;

Ts = 0.01; N_sim = 800;  % 8 秒，匹配参考图
t = (0:N_sim-1) * Ts;

all_y = cell(2, 1);
legends_text = {'n_H (y_1)', '\pi_T (y_2)'};
case_names = {'u_1 阶跃, u_2=0'; 'u_1=0, u_2 阶跃'};

for case_idx = 1:2
    clear plant_engine_mimo;  % 重置持久变量

    y1_1=0; y2_1=0; u1_1=0; u1_2=0; u1_3=0; u2_1=0; u2_2=0; u2_3=0;
    y1 = zeros(1, N_sim); y2 = zeros(1, N_sim);

    for k = 1:N_sim
        if case_idx == 1
            u1_k = 1; u2_k = 0;
        else
            u1_k = 0; u2_k = 1;
        end

        [y1(k), y2(k)] = plant_engine_mimo(...
            y1_1, y2_1, u1_1, u1_2, u1_3, u2_1, u2_2, u2_3);

        y1_1 = y1(k); y2_1 = y2(k);
        u1_3 = u1_2; u1_2 = u1_1; u1_1 = u1_k;
        u2_3 = u2_2; u2_2 = u2_1; u2_1 = u2_k;
    end

    all_y{case_idx} = struct('y1', y1, 'y2', y2);
    fprintf('Case %d (%s):  y1_ss=%.4f  y2_ss=%.4f\n', ...
        case_idx, case_names{case_idx}, mean(y1(end-100:end)), mean(y2(end-100:end)));
end

%% 绘图
figure('Position', [50, 50, 1000, 650]);

for case_idx = 1:2
    subplot(2,1,case_idx);
    yy = all_y{case_idx};
    plot(t, yy.y1, 'r-', 'LineWidth', 1.5); hold on;
    plot(t, yy.y2, 'r--', 'LineWidth', 1.5);
    ylabel('输出'); xlabel('t / s');
    legend(legends_text, 'Location', 'southeast');
    title(sprintf('(%s) %s', char('a'+case_idx-1), case_names{case_idx}));
    grid on; xlim([0, 8]);
end

sgtitle('Engine Model 4 — 开环单位阶跃响应  (H=18km, Ma=0.8)');

% 中式字体
ft = 'SimHei';
set(findall(gcf,'-property','FontName'), 'FontName', ft);

out_dir = fileparts(mfilename('fullpath'));
saveas(gcf, fullfile(out_dir, 'open_loop_step.png'));
fprintf('\nFigure saved: %s\n', fullfile(out_dir, 'open_loop_step.png'));
