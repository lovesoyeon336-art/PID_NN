function [u, Kp, Ki, Kd] = BP_PID_Controller(r, y, dydu)
% =========================================================================
% BP神经网络PID参数自整定控制器 (Simulink MATLAB Function Block)
% Jacobian ∂y/∂u 由外部 RBF 辨识器提供
% =========================================================================
%
% 网络结构：
%   输入层  (3个神经元)：误差 e(k)、误差变化率 de(k)、误差积分 ei(k)
%   隐含层  (5个神经元)：激活函数 Leaky ReLU
%   输出层  (3个神经元)：激活函数 sigmoid，对应 Kp、Ki、Kd
%
% 端口说明：
%   输入端口：
%     r    - 参考指令（设定值）
%     y    - 被控对象输出（测量值/反馈量）
%     dydu - RBF辨识器提供的雅可比 ∂y/∂u（标量）
%
%   输出端口：
%     u  - PID控制量（输出给执行器/被控对象）
%     Kp - 比例系数（实时监控）
%     Ki - 积分系数（实时监控）
%     Kd - 微分系数（实时监控）
%
% =========================================================================

%% ---------- 持久变量声明（网络权值 & 状态量）----------
persistent w_ih dw_ih   % 输入层→隐含层 权值矩阵 & 上次增量（动量项）
persistent w_ho dw_ho   % 隐含层→输出层 权值矩阵 & 上次增量（动量项）
persistent e_1 e_2      % e(k-1), e(k-2)：计算增量式PID所需历史误差
persistent u_1          % u(k-1)：上一时刻控制量
persistent Oh           % 隐含层输出（供反向传播复用）
persistent Oo           % 输出层输出（供反向传播复用）
persistent ei_acc       % 误差积分累加器（真积分）
persistent step_count     % 步数计数器，学习率衰减用
persistent u_smooth       % 控制量指数平滑值

%% ---------- 超参数 ----------
n_in  = 3;   % 输入层神经元数
n_h   = 5;   % 隐含层神经元数
n_out = 3;   % 输出层神经元数（固定为3，对应 Kp/Ki/Kd）

lr0      = 0.01;    % 初始学习率
lr_decay = 0.0005;  % 指数衰减速率: lr = lr0 * exp(-step*ts*lr_decay)
alpha = 0.05;   % 动量系数 α
smooth_factor = 0.3;  % 控制量指数平滑因子 (0~1，越小越平滑)

ts = 0.01;   % 采样周期（与 Simulink 步长一致）

% PID 参数取值范围
Kp_min = 0.0;  Kp_max = 15.0;
Ki_min = 0.0;  Ki_max =  5.0;
Kd_min = 0.0;  Kd_max =  5.0;

% 控制量限幅
u_max =  10.0;
u_min = -10.0;

%% ---------- 首次调用：初始化权值和状态 ----------
if isempty(w_ih)
    rng(1);
    % Xavier 初始化：N(0, sqrt(2/(fan_in+fan_out)))
    w_ih  = sqrt(2/(n_in + n_h)) * randn(n_h, n_in + 1);
    dw_ih = zeros(n_h,  n_in + 1);
    w_ho  = sqrt(2/(n_h + n_out)) * randn(n_out, n_h + 1);
    dw_ho = zeros(n_out, n_h  + 1);
    e_1 = 0;   e_2 = 0;
    u_1 = 0;
    Oh     = zeros(n_h,   1);
    Oo     = 0.5 * ones(n_out, 1);
    ei_acc = 0;
    step_count = 0;
    u_smooth = 0;
end

% 学习率指数衰减
lr = lr0 * exp(-step_count * ts * lr_decay);

%% ---------- Step 1：计算当前误差 ----------
e  = r - y;                    % 跟踪误差 e(k)
de = e - e_1;                  % 误差变化量

% 真积分累加 + 抗饱和
if ~((u_1 >= u_max && e > 0) || (u_1 <= u_min && e < 0))
    ei_acc = ei_acc + e * ts;
end
ei_acc = max(-3.0, min(3.0, ei_acc));
ei = ei_acc;

Xi = [e; de; ei];              % 3×1

%% ---------- Step 2：前向传播 ----------
% 隐含层（Leaky ReLU 激活，负斜率 0.01）
net_h = w_ih * [Xi; 1];       % [n_h × 1]，+1 为偏置输入
Oh    = max(0.01*net_h, net_h);  % [n_h × 1]

% 输出层（sigmoid 激活，输出范围 (0, 1)）
net_o = w_ho * [Oh; 1];       % [n_out × 1]
Oo    = 1 ./ (1 + exp(-net_o)); % [n_out × 1]

%% ---------- Step 3：映射到 PID 参数 ----------
Kp = Kp_min + (Kp_max - Kp_min) * Oo(1);
Ki = Ki_min + (Ki_max - Ki_min) * Oo(2);
Kd = Kd_min + (Kd_max - Kd_min) * Oo(3);

%% ---------- Step 4：增量式PID控制律 ----------
delta_u = Kp * de + Ki * e + Kd * (e - 2*e_1 + e_2);

u = u_1 + delta_u;

% 控制量限幅
u = max(u_min, min(u_max, u));

% 输出指数平滑（抑制高频振荡）
u_smooth = smooth_factor * u + (1 - smooth_factor) * u_smooth;
u = u_smooth;

%% ---------- Step 5：反向传播在线更新权值 ----------
%
% 代价函数：J = 0.5 * e(k)^2
% Jacobian ∂J/∂u = -e * dydu   ← dydu 由外部 RBF 辨识器提供
dJ_du = -e * dydu;

% 各 K 对控制量的偏导（增量式 PID）
dU_dKp = de;
dU_dKi = e;
dU_dKd = e - 2*e_1 + e_2;
dU_dK  = [dU_dKp; dU_dKi; dU_dKd];   % [3 × 1]

% Kp/Ki/Kd 对 Oo 的偏导（线性映射）
dK_dOo = [Kp_max - Kp_min;
           Ki_max - Ki_min;
           Kd_max - Kd_min];            % [3 × 1]

% sigmoid 导数：dOo/dnet_o = Oo*(1-Oo)
d_sigmoid = Oo .* (1 - Oo);            % [3 × 1]

% 输出层误差信号 δ_o（[3 × 1]）
delta_o = dJ_du * dU_dK .* dK_dOo .* d_sigmoid;

% 输出层权值更新（含动量项）
grad_w_ho  = delta_o * [Oh; 1]';       % [3 × (n_h+1)]
new_dw_ho  = -lr * grad_w_ho + alpha * dw_ho;
w_ho       = w_ho + new_dw_ho;
dw_ho      = new_dw_ho;

% 隐含层误差信号 δ_h（[n_h × 1]）
d_leaky = ones(n_h, 1); d_leaky(net_h <= 0) = 0.01;  % LeakyReLU 导数
delta_h    = (w_ho(:, 1:n_h)' * delta_o) .* d_leaky;  % [n_h × 1]

% 梯度裁剪
delta_o = max(-1.0, min(1.0, delta_o));
delta_h = max(-1.0, min(1.0, delta_h));

% 输入层→隐含层权值更新（含动量项）
grad_w_ih  = delta_h * [Xi; 1]';      % [n_h × (n_in+1)]
new_dw_ih  = -lr * grad_w_ih + alpha * dw_ih;
w_ih       = w_ih + new_dw_ih;
dw_ih      = new_dw_ih;

%% ---------- Step 6：状态更新 ----------
e_2 = e_1;
e_1 = e;
u_1 = u;
step_count = step_count + 1;

end   % function BP_PID_Controller
