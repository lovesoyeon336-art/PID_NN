function [y1_k, y2_k] = plant_engine_mimo(y1_1, y2_1, u1_1, u1_2, u1_3, u2_1, u2_2, u2_3)
% Engine Model 4: H=18km, Ma=0.8 线性化状态空间模型
% dx/dt = A x + B u,  y = C x + D u
% 欧拉离散化 Ts=0.01s，接口与原 plant_dynamics_mimo 兼容
% 仅使用 u1_1, u2_1（即 u(k-1)），其余历史输入保留为接口兼容

persistent x A B C D Ts Ad Bd initialized

if isempty(initialized)
    Ts = 0.01;
    A = [-0.301, -0.050; 0.325, -0.375];
    B = [0.079, 0.099; 0.133, 0.302];
    C = [1, 0; 0.038, 0.67];
    D = [0, 0; 0.19, -1.31];
    Ad = eye(2) + Ts * A;
    Bd = Ts * B;
    x = [0; 0];
    initialized = true;
end

u_k = [u1_1; u2_1];
y = C * x + D * u_k;
x = Ad * x + Bd * u_k;

y1_k = y(1);
y2_k = y(2);
end
