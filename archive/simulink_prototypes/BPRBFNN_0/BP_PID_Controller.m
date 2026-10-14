function [u, Kp, Ki, Kd] = BP_PID_Controller(r, y, dydu)
% =========================================================================
% BP Neural Network PID Controller — Simulink MATLAB Function Block
% =========================================================================
%
% 【网络结构】 3输入 → M隐层(tanh) → 3输出(sigmoid→有界)
%
% 【输入端口】
%   r    : 设定值 (reference)
%   y    : 被控对象当前输出
%   dydu : RBF辨识器提供的雅可比 ∂y/∂u（标量）
%
% 【输出端口】
%   u         : 控制量（增量式PID累加输出）
%   Kp, Ki, Kd: PID参数（用于Scope监控）
%
% 【梯度链（在线BP）】
%   E = 0.5*e^2
%   dE/dW = dE/dy · dy/du · du/dKj · dKj/doj · doj/dnet_oj · dnet_oj/dW
%            (-e)   (dydu)   (PID式)  (缩放)     (sigmoid')     (tanh'·x/h)
%
% 【使用说明】
%   1. 在Simulink中添加 MATLAB Function 模块，粘贴本代码
%   2. 超参数区域按被控对象调整 M / lr / Ts / 各参数范围 / 限幅
%   3. dydu 接口连接 RBF 辨识器的雅可比输出（暂无辨识器时可接常数 1 或 sign）
%   4. 首次运行前无需预训练，权值随机初始化后全程在线学习
% =========================================================================

%% ========== 超参数（按需修改）==========
M      = 5;       % 隐层神经元数量
lr     = 0.01;    % 学习率 η（建议范围 0.001~0.05）
Ts     = 0.01;    % 采样周期 (s)，需与Simulink步长一致

% PID 参数输出范围（sigmoid 映射目标区间）
Kp_min = 0;    Kp_max = 15;
Ki_min = 0;    Ki_max = 5;
Kd_min = 0;    Kd_max = 5;

% 控制量限幅（防止执行器饱和）
u_min  = -10;
u_max  =  10;

% 积分限幅（防止 sum_e 发散导致 tanh 饱和）
sum_max = 3.0;

% 数值稳定：梯度裁剪阈值（防止梯度爆炸）
grad_clip = 1.0;


%% ========== 持久变量（跨步长保留状态）==========
persistent W1 W2 b1 b2       % 网络权值与偏置
persistent e1 e2              % e(k-1), e(k-2)
persistent u_prev             % u(k-1)，增量式累加用
persistent sum_e              % 误差积分（∑e·Ts）
persistent is_init

nin  = 3;   % 输入维度
nout = 3;   % 输出维度 [Kp, Ki, Kd]

%% ========== 初始化（仅第一步执行）==========
if isempty(is_init)
    % Xavier 初始化：方差 = 1/fan_in，有助于 tanh 激活稳定
    W1 = (2*rand(M, nin)  - 1) * sqrt(1/nin);   % M×3
    W2 = (2*rand(nout, M) - 1) * sqrt(1/M);     % 3×M
    b1 = zeros(M,    1);
    b2 = zeros(nout, 1);
    
    e1     = 0;
    e2     = 0;
    u_prev = 0;
    sum_e  = 0;
    is_init = true;
end

%% ========== 误差信号计算 ==========
e    = r - y;                       % 当前误差 e(k)
de   = e - e1;                      % 误差一阶差分 Δe(k) = e(k)-e(k-1)
d2e  = e - 2*e1 + e2;              % 误差二阶差分（Kd项分子）
% 抗积分饱和：仅在 u 未饱和且误差方向不加剧饱和时累加
if ~((u_prev >= u_max && e > 0) || (u_prev <= u_min && e < 0))
    sum_e = sum_e + e * Ts;
end
sum_e = max(-sum_max, min(sum_max, sum_e));   % 积分限幅

%% ========== 前向传播 ==========
% --- 输入向量 ---
x = [e; de; sum_e];                 % 3×1

% --- 隐层 ---
net_h = W1 * x + b1;               % M×1，加权求和
h     = tanh(net_h);                % M×1，tanh激活

% --- 输出层 ---
net_o = W2 * h + b2;               % 3×1，加权求和
o     = 1 ./ (1 + exp(-net_o));    % 3×1，sigmoid激活，输出∈(0,1)

% --- 映射到物理 PID 参数范围 ---
Kp = Kp_min + (Kp_max - Kp_min) * o(1);
Ki = Ki_min + (Ki_max - Ki_min) * o(2);
Kd = Kd_min + (Kd_max - Kd_min) * o(3);

%% ========== 增量式 PID 计算控制量 ==========
% Δu(k) = Kp·Δe + Ki·e + Kd·(e - 2e₁ + e₂)
delta_u = Kp * de + Ki * e + Kd * d2e;
u = u_prev + delta_u;
u = max(u_min, min(u_max, u));      % 限幅

%% ========== 反向传播（在线权值更新）==========
%
% 性能指标：E = 0.5·e(k)²   （最小化当前时刻误差的平方）
%
% Step 1: dE/du
%   dE/dy = -e(k)               [∵ e=r-y, dE/de=e, de/dy=-1]
%   dE/du = dE/dy · dy/du = -e · dydu     ← dydu来自RBF辨识器
%
dE_du = -e * dydu;

%
% Step 2: du/dKj  (增量PID对各参数偏导)
%   ∂(Δu)/∂Kp = Δe
%   ∂(Δu)/∂Ki = e
%   ∂(Δu)/∂Kd = e-2e₁+e₂
%
du_dK = [de; e; d2e];              % 3×1

%
% Step 3: dKj/doj  (线性映射的斜率，即范围宽度)
%
scale = [Kp_max - Kp_min; ...
         Ki_max - Ki_min; ...
         Kd_max - Kd_min];         % 3×1

%
% Step 4: doj/dnet_oj  (sigmoid 导数)
%
dsigmoid = o .* (1 - o);           % 3×1

%
% 输出层误差信号 δo（3×1）
%   δo_j = dE/du · du/dKj · dKj/doj · doj/dnet_oj
%
delta_o = dE_du * (du_dK .* scale .* dsigmoid);  % 3×1

%
% 隐层误差信号 δh（M×1）
%   反传通过 W2，再乘 tanh 导数 (1 - h²)
%
delta_h = (W2' * delta_o) .* (1 - h.^2);         % M×1

% --- 梯度裁剪（防止梯度爆炸）---
delta_o = max(-grad_clip, min(grad_clip, delta_o));
delta_h = max(-grad_clip, min(grad_clip, delta_h));

%
% 梯度下降更新（E 最小化：W ← W - lr·∂E/∂W）
%
W2 = W2 - lr * (delta_o * h');    % 3×M
b2 = b2 - lr * delta_o;
W1 = W1 - lr * (delta_h * x');    % M×3
b1 = b1 - lr * delta_h;

%% ========== 更新历史状态（为下一步长准备）==========
e2     = e1;
e1     = e;
u_prev = u;

end
% =========================================================================
% 注：RBF辨识器尚未完成时的临时替代方案
%   dydu = 1;          % 假设正方向，方向正确但幅值不准
%   dydu = sign(de);   % 用误差变化方向近似，比常数稍好
% 两种近似均会降低控制性能，建议尽快接入 RBF 辨识器。
% =========================================================================