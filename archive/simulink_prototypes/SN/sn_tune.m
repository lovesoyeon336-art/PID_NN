clear; close all;

%% ==================== 单神经元 PID 参数网格搜索 ====================

N_tune = 2000;
r_target = 1;

K_grid = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0];
eta_grid = [0.01, 0.03, 0.05, 0.08, 0.1, 0.2];

fprintf('===== 单神经元 PID 参数寻优 =====\n');

for plant_idx = 1:3
    pid = sprintf('plant%d', plant_idx);
    best_cost = inf;
    best_K = 0; best_eta = 0;

    for ki = 1:length(K_grid)
        for ei = 1:length(eta_grid)
            cost = sn_cost(K_grid(ki), eta_grid(ei), N_tune, pid, r_target);
            if cost < best_cost
                best_cost = cost;
                best_K = K_grid(ki);
                best_eta = eta_grid(ei);
            end
        end
    end

    fprintf('\n对象%d 最优参数: K=%.2f  eta=%.2f  MAE=%.6f\n', plant_idx, best_K, best_eta, best_cost);

    % 保存
    [script_dir, ~, ~] = fileparts(mfilename('fullpath'));
    K_opt = best_K; eta_opt = best_eta;
    save(fullfile(script_dir, sprintf('sn_tuned_params_plant%d.mat', plant_idx)), 'K_opt', 'eta_opt');
end

fprintf('\n参数已保存\n');

%% ==================== 局部函数 ====================

function mae = sn_cost(K_neuron, eta, N_sim, plant_id, r_target)
    w = [0.3, 0.3, 0.3];
    y_1 = 0; y_2 = 0; u_1 = 0; error_1 = 0; error_2 = 0;
    mae_sum = 0;

    for k = 1:N_sim
        e_cur = r_target - y_1;

        x1 = e_cur;
        x2 = e_cur - error_1;
        x3 = e_cur - 2*error_1 + error_2;

        w_sum = abs(w(1)) + abs(w(2)) + abs(w(3)) + 1e-6;
        w1_n = w(1)/w_sum; w2_n = w(2)/w_sum; w3_n = w(3)/w_sum;

        delta_u = K_neuron * (w1_n*x1 + w2_n*x2 + w3_n*x3);
        delta_u = max(-0.5, min(0.5, delta_u));
        u_k = u_1 + delta_u;

        y_k = plant_dynamics(plant_id, y_1, y_2, u_k, u_1, k);

        w(1) = w(1) + eta * e_cur * x1;
        w(2) = w(2) + eta * e_cur * x2;
        w(3) = w(3) + eta * e_cur * x3;

        mae_sum = mae_sum + abs(e_cur);

        u_1 = u_k; y_2 = y_1; y_1 = y_k;
        error_2 = error_1; error_1 = r_target - y_k;
    end
    mae = mae_sum / N_sim;
end
