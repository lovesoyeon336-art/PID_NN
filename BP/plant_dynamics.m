function y_k = plant_dynamics(plant_id, y_1, y_2, u_k, u_1, k, a_k_override)
    % 统一被控对象模型
    % y_1: y(k-1),  y_2: y(k-2) (一阶对象传0)
    % u_k: u(k),   u_1: u(k-1)
    % a_k_override: 用于参数摄动等场景 (仅 plant1)

    switch plant_id
        case 'plant1'
            % 一阶非线性时变: y(k) = a(k)/(1+y(k-1)^2) * y(k-1) + u(k)
            if nargin >= 7 && ~isempty(a_k_override)
                a_k = a_k_override;
            else
                a_k = 1.2 * (1 - 0.8 * exp(-0.1 * k));
            end
            y_k = a_k / (1 + y_1^2) * y_1 + u_k;

        case 'plant2'
            % 二阶无死区: y(k) = a1*y(k-1) + a2*y(k-2) + b1*u(k) + b2*u(k-1)
            a1 = 1.7;  a2 = -0.72;
            b1 = 0.02; b2 = 0.01;
            y_k = a1*y_1 + a2*y_2 + b1*u_k + b2*u_1;

        case 'plant3'
            % Hammerstein: 输入非线性 + 线性动态
            v_k = u_k + 0.3*u_k^2 - 0.1*u_k^3;
            v_1 = u_1 + 0.3*u_1^2 - 0.1*u_1^3;
            a1 = 1.5;  a2 = -0.56;
            b0 = 0.05; b1 = 0.03;
            y_k = a1*y_1 + a2*y_2 + b0*v_k + b1*v_1;

        otherwise
            error('Unknown plant_id: %s', plant_id);
    end
end
