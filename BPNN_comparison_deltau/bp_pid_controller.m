function [u, Kp, Ki, Kd] = bp_pid_controller(r, y, ts)
% =========================================================================
% BP神经网络PID参数自整定控制器 (Simulink MATLAB Function Block)
% =========================================================================
%
% 功能说明：
%   采用3层BP神经网络在线实时整定PID三个参数（Kp, Ki, Kd），
%   结合增量式PID算法输出控制量u。网络权值通过梯度下降法在每个
%   采样时刻进行在线更新，实现参数的自适应调整。
%
% 端口说明：
%   输入端口：
%     r  - 参考指令（设定值）
%     y  - 被控对象输出（测量值/反馈量）
%     ts - 采样时间 (s)，建议范围 [0.001, 0.1]
%
%   输出端口：
%     u  - PID控制量（输出给执行器/被控对象）
%     Kp - 比例系数（实时监控）
%     Ki - 积分系数（实时监控）
%     Kd - 微分系数（实时监控）
%
% 网络结构：
%   输入层  (3个神经元)：误差 e(k)、误差变化率 ec(k)、误差积分 ei(k)
%   隐含层  (5个神经元)：激活函数 tanh
%   输出层  (3个神经元)：激活函数 sigmoid，对应 Kp、Ki、Kd
%
% 使用方法（Simulink）：
%   1. 拖入 "MATLAB Function" 模块
%   2. 粘贴本函数内容（或通过 Edit 引用本文件）
%   3. 连接 r（Constant/信号源）、y（被控对象输出）、ts（Constant）
%   4. 输出 u 接被控对象，Kp/Ki/Kd 接 Scope 或 Display 模块
%   5. 在 Simulink 仿真参数中设置固定步长，与 ts 保持一致
%
% 参数整定建议：
%   - 学习率 lr：过大导致振荡，过小收敛慢，建议 [0.05, 0.5]
%   - 惯性系数 alpha：动量项，防止陷入局部极值，建议 [0.01, 0.1]
%   - Kp/Ki/Kd 上限：根据实际对象动态特性设定，防止过激控制
%
% =========================================================================
% 作者注：工厂增益方向未知时，使用 sign(Δy) 替代下方 dJdu 中的符号项
% =========================================================================

%% ---------- 持久变量声明（网络权值 & 状态量）----------
persistent w_ih dw_ih   % 输入层→隐含层 权值矩阵 & 上次增量（动量项）
persistent w_ho dw_ho   % 隐含层→输出层 权值矩阵 & 上次增量（动量项）
persistent e_1 e_2      % e(k-1), e(k-2)：计算增量式PID所需历史误差
persistent u_1          % u(k-1)：上一时刻控制量
persistent Oh           % 隐含层输出（供反向传播复用）
persistent Oo           % 输出层输出（供反向传播复用）
persistent ei_acc       % 误差积分累加器（真积分）
persistent y_1 u_2       % y(k-1) 和 u(k-2)，用于 Jacobian 有限差分估算
persistent step_count     % 步数计数器，学习率衰减用
persistent u_smooth       % 控制量指数平滑值

%% ---------- 超参数（可根据对象特性修改）----------
n_in  = 3;   % 输入层神经元数
n_h   = 5;   % 隐含层神经元数（增加可提升拟合能力，但计算量增大）
n_out = 3;   % 输出层神经元数（固定为3，对应 Kp/Ki/Kd）

lr0      = 0.01;    % 初始学习率（论文推荐）
lr_decay = 0.0005;  % 指数衰减速率: lr = lr0 * exp(-step*ts*lr_decay)
alpha = 0.05;   % 动量系数 α
smooth_factor = 0.3;  % 控制量指数平滑因子 (0~1，越小越平滑)

% PID 参数取值范围
Kp_min = 0.0;  Kp_max = 15.0;
Ki_min = 0.0;  Ki_max =  5.0;
Kd_min = 0.0;  Kd_max =  5.0;

% 控制量限幅（防止执行器饱和）
u_max =  10.0;
u_min = -10.0;

%% ---------- 首次调用：初始化权值和状态 ----------
if isempty(w_ih)
    % 权值初始化（使用固定种子保证可重复性）
    rng(1);
    % Xavier 初始化：N(0, sqrt(2/(fan_in+fan_out)))
    w_ih  = sqrt(2/(n_in + n_h)) * randn(n_h, n_in + 1);
    dw_ih = zeros(n_h,  n_in + 1);
    w_ho  = sqrt(2/(n_h + n_out)) * randn(n_out, n_h + 1);
    dw_ho = zeros(n_out, n_h  + 1);
    % 状态变量每轮仿真重置
    e_1 = 0;   e_2 = 0;
    u_1 = 0;
    Oh     = zeros(n_h,   1);
    Oo     = 0.5 * ones(n_out, 1);
    ei_acc = 0;
    y_1 = 0;  u_2 = 0;
    step_count = 0;
    u_smooth = 0;
end

% 学习率指数衰减（方案2A）
lr = lr0 * exp(-step_count * ts * lr_decay);

%% ---------- Step 1：计算当前误差 ----------
e  = r - y;                    % 跟踪误差 e(k)
ec = e - e_1;                  % 误差变化量（原始差分，与PID公式一致）

% 真积分累加 + 抗饱和：仅在 u 未饱和且误差方向不加剧饱和时累加
if ~((u_1 >= u_max && e > 0) || (u_1 <= u_min && e < 0))
    ei_acc = ei_acc + e * ts;
end
ei_acc = max(-3.0, min(3.0, ei_acc));
ei = ei_acc;

Xi = [e; ec; ei];              % 3×1，三输入同量级

%% ---------- Step 2：前向传播 ----------
% 隐含层（Leaky ReLU 激活，负斜率 0.01）
net_h = w_ih * [Xi; 1];       % [n_h × 1]，+1 为偏置输入
Oh    = max(0.01*net_h, net_h);  % [n_h × 1]

% 输出层（sigmoid 激活，输出范围 (0, 1)，保证 Kp/Ki/Kd ≥ 0）
net_o = w_ho * [Oh; 1];       % [n_out × 1]
Oo    = 1 ./ (1 + exp(-net_o)); % [n_out × 1]

%% ---------- Step 3：映射到 PID 参数 ----------
Kp = Kp_min + (Kp_max - Kp_min) * Oo(1);
Ki = Ki_min + (Ki_max - Ki_min) * Oo(2);
Kd = Kd_min + (Kd_max - Kd_min) * Oo(3);

%% ---------- Step 4：增量式PID控制律 ----------
%   Δu(k) = Kp*[e(k)-e(k-1)] + Ki*e(k) + Kd*[e(k)-2e(k-1)+e(k-2)]
delta_u = Kp * (e - e_1) ...
        + Ki *  e         ...
        + Kd * (e - 2*e_1 + e_2);

u = u_1 + delta_u;

% 控制量限幅（硬饱和）
u = max(u_min, min(u_max, u));

% 输出指数平滑（抑制高频振荡）
u_smooth = smooth_factor * u + (1 - smooth_factor) * u_smooth;
u = u_smooth;

%% ---------- Step 5：反向传播在线更新权值 ----------
%
% 代价函数：J = 0.5 * e(k)^2
% 关键链式法则：
%   ∂J/∂w_ho = ∂J/∂u * ∂u/∂K * ∂K/∂Oo * ∂Oo/∂net_o * [Oh;1]'
%   ∂J/∂w_ih = ∂J/∂w_ho 逐层回传
%
% ∂J/∂u：有限差分估算被控对象 Jacobian
%   ∂y/∂u ≈ (y(k)-y(k-1)) / (u(k-1)-u(k-2))   ← 离散差分
%   dJ/du = ∂J/∂y · ∂y/∂u = -e · dydu
dy_raw = (y - y_1) / (u_1 - u_2 + 1e-6);
dydu   = max(-1.0, min(1.0, dy_raw));   % 限幅，防除零异常
dJ_du  = -e * dydu;

% 各 K 对控制量的偏导（增量式 PID）
dU_dKp = e - e_1;
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
% 注：w_ho(:,1:n_h) 排除偏置列的反传
d_leaky = ones(n_h, 1); d_leaky(net_h <= 0) = 0.01;  % LeakyReLU 导数
delta_h    = (w_ho(:, 1:n_h)' * delta_o) .* d_leaky;  % [n_h × 1]

% 梯度裁剪（防止梯度爆炸）
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
y_1 = y;
u_2 = u_1;
u_1 = u;
step_count = step_count + 1;

end   % function bp_pid_controller