function y_k = plant_dynamics(plant_id, y_1, y_2, u_k, u_1, k, a_k_override)
    % 统一被控对象模型
    % y_1: y(k-1),  y_2: y(k-2) (一阶对象传0)
    % u_k: u(k),   u_1: u(k-1)
    % a_k_override: 用于参数摄动等场景 (仅 plant1)

    switch plant_id
        case 'plant1'
            % 一阶非线性时变: y(k) = a(k)/(1+y(k-1)^2) * y(k-1) + u(k-1)
            if nargin >= 7 && ~isempty(a_k_override)
                a_k = a_k_override;
            else
                a_k = 1.2 * (1 - 0.8 * exp(-0.1 * k));
            end
            y_k = a_k / (1 + y_1^2) * y_1 + u_1;

        case 'plant2'
            % 二阶因果: y(k) = a1*y(k-1) + a2*y(k-2) + b*u(k-1)
            a1 = 1.7;  a2 = -0.72;
            b = 0.03;
            y_k = a1*y_1 + a2*y_2 + b*u_1;

        case 'plant3'
            % Hammerstein: 静态输入非线性 + 二阶欠阻尼线性动态
            % v = u + 0.15*u^3   (增强立方非线性: 大信号有效增益变化更大)
            % y(k) = 1.6*y(k-1) - 0.68*y(k-2) + 0.06*v(k-1)
            v = u_1 + 0.15 * u_1^3;
            y_k = 1.6*y_1 - 0.68*y_2 + 0.06*v;

        otherwise
            error('Unknown plant_id: %s', plant_id);
    end
end
