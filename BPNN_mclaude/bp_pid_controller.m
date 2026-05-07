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

%% ---------- 超参数（可根据对象特性修改）----------
n_in  = 3;   % 输入层神经元数
n_h   = 5;   % 隐含层神经元数（增加可提升拟合能力，但计算量增大）
n_out = 3;   % 输出层神经元数（固定为3，对应 Kp/Ki/Kd）

lr    = 0.15;   % 学习率 η（Learning Rate）
alpha = 0.05;   % 动量系数 α（Momentum，防止陷入局部极值）

% PID 参数取值范围（根据实际对象整定，必须 >= 0）
Kp_min = 0.0;  Kp_max = 2.0;
Ki_min = 0.0;  Ki_max = 1.0;
Kd_min = 0.0;  Kd_max = 1.0;

% 控制量限幅（防止执行器饱和）
u_max =  10.0;
u_min = -10.0;

%% ---------- 首次调用：初始化权值和状态 ----------
if isempty(w_ih)
    % 使用固定种子保证可重复性；如需随机初始化可删除 rng(1)
    rng(1);
    % 含偏置项：输入层 (n_in+1) → 隐含层 (n_h)
    w_ih  = 0.5 * (2 * rand(n_h,  n_in + 1) - 1);
    dw_ih = zeros(n_h,  n_in + 1);
    % 含偏置项：隐含层 (n_h+1) → 输出层 (n_out)
    w_ho  = 0.5 * (2 * rand(n_out, n_h  + 1) - 1);
    dw_ho = zeros(n_out, n_h  + 1);

    e_1 = 0;   e_2 = 0;
    u_1 = 0;
    Oh  = zeros(n_h,   1);
    Oo  = 0.5 * ones(n_out, 1);   % 初始输出取中值
end

%% ---------- Step 1：计算当前误差 ----------
e = r - y;                     % 跟踪误差 e(k)
ec = (e - e_1) / ts;           % 误差变化率（近似微分，单位：/s）
ei =  e_1 + e * ts;            % 误差累积（近似积分，防止 persistent ei 累计初值问题）
% 注：如需真积分累计，可改为 persistent ei; ei = ei + e*ts;
%     但需注意积分饱和，此处采用单步梯形近似避免饱和问题

% 输入向量归一化（有助于网络训练稳定性；如对象信号幅值差异大可调整 scale）
scale_e  = 1.0;
scale_ec = 1.0;
scale_ei = 1.0;
Xi = [e  / scale_e;
      ec / scale_ec;
      ei / scale_ei];

%% ---------- Step 2：前向传播 ----------
% 隐含层（tanh 激活，输出范围 (-1, 1)）
net_h = w_ih * [Xi; 1];       % [n_h × 1]，+1 为偏置输入
Oh    = tanh(net_h);           % [n_h × 1]

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

%% ---------- Step 5：反向传播在线更新权值 ----------
%
% 代价函数：J = 0.5 * e(k)^2
% 关键链式法则：
%   ∂J/∂w_ho = ∂J/∂u * ∂u/∂K * ∂K/∂Oo * ∂Oo/∂net_o * [Oh;1]'
%   ∂J/∂w_ih = ∂J/∂w_ho 逐层回传
%
% ∂J/∂u 依赖对象雅可比 ∂y/∂u，此处用符号近似（方向已知时可替换）：
%   dJ_du = -e（负号因为 ∂J/∂e = e, ∂e/∂y = -1，再近似 ∂y/∂u > 0）
%
dJ_du = -e;   % 若对象增益为负，改为 +e 或乘以 sign(delta_u) 判断

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
d_tanh     = 1 - Oh.^2;               % tanh 导数
delta_h    = (w_ho(:, 1:n_h)' * delta_o) .* d_tanh;  % [n_h × 1]

% 输入层→隐含层权值更新（含动量项）
grad_w_ih  = delta_h * [Xi; 1]';      % [n_h × (n_in+1)]
new_dw_ih  = -lr * grad_w_ih + alpha * dw_ih;
w_ih       = w_ih + new_dw_ih;
dw_ih      = new_dw_ih;

%% ---------- Step 6：状态更新 ----------
e_2 = e_1;
e_1 = e;
u_1 = u;

end   % function bp_pid_controller