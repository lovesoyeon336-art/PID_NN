# BP + RBF 神经网络自适应 PID 控制器 — 算法与仿真数据

> **目标读者：另一个 AI / 控制算法工程师**
> 本文档包含完整算法代码、仿真数据、以及已知问题诊断，供独立分析和算法改进使用。

---

## 1. 系统架构

```
Ref ─→ Sum(–) ─→ BP PID ─→ Saturation(±10) ─→ Plant ─→ y (输出)
  ↑               ↑  ↑                           |
  |    Jacobian ∂y/∂u |                           |
  |        ┌──────────┘                           |
  |        |                                      |
  └────────┴──── RBF Identifier ←─────────────────┘
```

- **BP PID**：3-5-3 结构 BP 神经网络，在线输出 Kp, Ki, Kd，采用增量式 PID 计算控制量 u
- **RBF Identifier**：4-8-1 结构 RBF 神经网络，在线辨识被控对象，提供 Jacobian ∂y/∂u 供 BP 反传
- **Plant**：Hammerstein 模型 — 死区非线性 (±0.3) + 线性传递函数 10/(s²+2s+5)
- **Solver**：ode4 (Runge-Kutta)，固定步长 0.001s
- **控制限幅**：±10

---

## 2. BP_PID_Controller.m — BP 神经网络 PID 控制器

```matlab
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
```

### BP_PID_Controller 关键设计点

| 设计点 | 当前实现 | 说明 |
|--------|---------|------|
| 网络结构 | 3-5-3 | 输入=[e, Δe, ∑e]，输出=[Kp, Ki, Kd] |
| 隐层激活 | tanh | 对称饱和，输出∈(-1,1) |
| 输出层激活 | sigmoid | 输出∈(0,1)，映射到 PID 参数物理范围 |
| 权值初始化 | Xavier (uniform) | 方差=1/fan_in，适配 tanh |
| 优化器 | 纯 SGD，无动量 | W = W - lr * grad |
| 损失函数 | E = 0.5·e(k)² | 瞬时误差，无批量/无平均 |
| 梯度裁剪 | ±1.0 | 在 δ_o 和 δ_h 上裁剪 |
| 积分抗饱和 | 条件累积 + ±3.0 限幅 | 仅 u 未饱和时累加 |
| PID 形式 | 增量式 | Δu = Kp·Δe + Ki·e + Kd·Δ²e |
| 采样周期 | Ts = 0.01s | 注意：实际步长是 0.001s，但 Ts=0.01 |

---

## 3. RBF_Identifier_claude.m — RBF 神经网络在线辨识器

```matlab
function [dy_du, y_hat] = rbf_online_identification(u_prev, y, reset)
% RBF 神经网络在线辨识 —— Simulink MATLAB Function Block
%
% 输入:
%   u_prev —— u(k-1)，控制量经 Simulink Unit Delay 后接入，避免代数环
%   y      —— y(k)，被控对象当前输出
%   reset  —— 复位信号，接常数 0 即可
%
% 输出:
%   dy_du  —— 雅可比 ∂y/∂u，接 BP-PID 反传入口
%   y_hat  —— 预测输出，接 Scope 监控或 Terminator

persistent W C Sigma dW_prev dC_prev dSig_prev y1 y2 u2 init_flag

%% ===== 超参数 =====
M       = 8;      % RBF 隐层节点数
n       = 4;      % 输入维度
eta_w   = 0.30;   % W 学习率
eta_c   = 0.05;   % C 学习率
eta_s   = 0.05;   % σ 学习率
mu_w    = 0.05;   % W 动量系数
mu_c    = 0.02;   % C 动量系数
mu_s    = 0.02;   % σ 动量系数
sig_min = 0.01;   % σ 下限

%% ===== 初始化 =====
if isempty(init_flag) || reset > 0.5
    W         = 0.1 * randn(1, M);
    C         = repmat(linspace(-1, 1, M)', 1, n) + 0.1 * randn(M, n);
    Sigma     = 2.0 * ones(M, 1);
    dW_prev   = zeros(1, M);
    dC_prev   = zeros(M, n);
    dSig_prev = zeros(M, 1);
    y1 = 0; y2 = 0; u2 = 0;
    init_flag = true;
end

%% ===== 构造输入向量 =====
% x(k) = [y(k-1), y(k-2), u(k-1), u(k-2)]
% u(k-1) 直接来自输入端口，u(k-2) 用 persistent 变量 u2 存储
x = [y1; y2; double(u_prev); u2];   % 4×1

%% ===== 前向传播 =====
D     = bsxfun(@minus, x', C);              % M×n, 输入与中心的距离
dist2 = sum(D .^ 2, 2);                     % M×1, 欧氏距离平方
H     = exp(-dist2 ./ (2 * Sigma .^ 2));    % M×1, 高斯径向基函数
y_hat = W * H;                              % 标量, 线性加权输出

%% ===== 辨识误差 =====
e = double(y) - y_hat;

%% ===== 梯度计算 =====
grad_W = -e * H';
scale  = (W .* H') ./ (Sigma .^ 2)';
grad_C = -e * bsxfun(@times, D, scale');
grad_S = -e * (W' .* H .* dist2 ./ (Sigma .^ 3));

%% ===== 带动量的参数更新 =====
dW = -eta_w * grad_W + mu_w * dW_prev;
dC = -eta_c * grad_C + mu_c * dC_prev;
dS = -eta_s * grad_S + mu_s * dSig_prev;

W         = W + dW;
C         = C + dC;
Sigma     = max(Sigma + dS, sig_min);

dW_prev   = dW;
dC_prev   = dC;
dSig_prev = dS;

%% ===== 计算雅可比 ∂ŷ/∂u(k-1) =====
% u(k-1) 在 x 中是第 3 个分量，对应 D 的第 3 列
dy_du = sum(W' .* H .* (-D(:,3) ./ Sigma .^ 2));

if abs(dy_du) < 1e-4
    dy_du = sign(dy_du + eps) * 1e-4;
end

%% ===== 更新延迟寄存器 =====
y2 = y1;           y1 = double(y);
u2 = double(u_prev);   % u(k-1) 下一拍变成 u(k-2)
```

### RBF_Identifier 关键设计点

| 设计点 | 当前实现 | 说明 |
|--------|---------|------|
| 网络结构 | 4-8-1 | 输入=[y(k-1), y(k-2), u(k-1), u(k-2)]，输出=ŷ(k) |
| 基函数 | 高斯 RBF | H_j = exp(-||x-C_j||² / 2σ_j²) |
| 训练参数 | W, C, Sigma 三组 | 全部在线学习 |
| 优化器 | 带动量 SGD | 三组参数各有独立学习率和动量系数 |
| Jacobian | ∂ŷ/∂u(k-1) | 解析求导，直接输出给 BP PID |
| Jacobian 下限 | 1e-4 | 防止梯度消失 |
| 输入结构 | NARX 模型 | y(k)=f(y(k-1),y(k-2),u(k-1),u(k-2)) |

---

## 4. 仿真配置

| 参数 | 值 |
|------|-----|
| Simulink 模型 | `BP_RBFNN.slx` |
| 求解器 | ode4 (Runge-Kutta)，固定步长 |
| 步长 | 0.001 s |
| 仿真时长 | 7000 s |
| 样本数 | 7,000,001 |
| 被控对象 | Hammerstein: 死区(±0.3) + 10/(s²+2s+5) |
| 控制限幅 | ±10 |
| 参考信号 | r = 1（恒定） |

---

## 5. 仿真结果 — 关键指标

### 5.1 跟踪性能

| 指标 | 值 |
|------|-----|
| RMS 跟踪误差 | 0.5869 |
| 稳态 MAE (t>5600s) | 0.9653 |
| 最大绝对误差 | 2.4141 |
| 平均绝对误差 | 0.4322 |
| 参考值 | r = 1（恒定） |

### 5.2 控制信号

| 指标 | 值 |
|------|-----|
| u 范围 | [-10.00, 10.00] |
| u 均值 | 0.9031 |
| **饱和率 (>|9.9|)** | **88.73%** |
| 近饱和率 (>|9.5|) | 91.18% |
| u 标准差 | 9.6933 |

> **关键问题：88.73% 的采样点控制量处于饱和状态 (±10)，系统表现为典型的 bang-bang 振荡。**

### 5.3 PID 参数

| 参数 | 终值 | 均值 | 标准差 | 范围 |
|------|------|------|--------|------|
| Kp | 1.1893 | 4.3062 | 1.9778 | [0.4517, 9.4189] |
| Ki | 4.9989 | 4.2873 | 1.5373 | [0.0000, 4.9997] |
| Kd | 2.4701 | 2.1317 | 0.1517 | [1.2662, 2.6052] |

**观察**：
- Ki 频繁触及上限 5.0 和下限 0.0（饱和），说明积分通道不稳定
- Kp 波动范围大（0.45→9.4），说明比例通道在剧烈调整但无法收敛
- Kd 相对稳定，波动最小

### 5.4 RBF 辨识器

| 指标 | 值 |
|------|-----|
| 辨识 RMSE | 0.0504 |
| dy/du 均值 | -0.0006 |
| dy/du 标准差 | 0.0496 |
| dy/du 范围 | [-0.4505, 0.6248] |

### 5.5 PID 参数随时间演化

| 时间 | Kp | Ki | Kd | y | u |
|------|-----|-----|-----|-----|-----|
| t=1s | 7.08 | 2.37 | 2.45 | 1.46 | -10.0 |
| t=10s | 7.10 | 2.38 | 2.45 | 1.44 | -10.0 |
| t=50s | 7.81 | 2.60 | 2.50 | 0.91 | +10.0 |
| t=100s | 8.94 | 0.10 | 2.45 | 0.43 | +10.0 |
| t=500s | 8.76 | 4.47 | 1.76 | 0.87 | +10.0 |
| t=1000s | 6.20 | 5.00 | 2.14 | 1.15 | -10.0 |
| t=2000s | 4.78 | 5.00 | 2.11 | 1.36 | -10.0 |
| t=3000s | 4.05 | 5.00 | 2.10 | 0.62 | +10.0 |
| t=4000s | 3.13 | 5.00 | 2.09 | 1.04 | -10.0 |
| t=5000s | 5.07 | 4.92 | 1.49 | 1.32 | -10.0 |
| t=6000s | 0.99 | 0.00 | 2.29 | 0.27 | +9.9 |
| t=7000s | 1.19 | 5.00 | 2.47 | 1.45 | -10.0 |

> **系统从未收敛** — y 在 0.27~1.46 之间持续振荡，u 在 ±10 之间跳变，PID 参数无收敛趋势。

---

## 6. 已知问题诊断

### 问题 1：控制量几乎完全饱和 (88.73%)
**根因**：增量式 PID 在 0.001s 步长下，每次 Δu 很小但累加 7000 秒后极容易碰到 ±10 边界。一旦饱和，积分抗饱和逻辑虽然阻止了 sum_e 继续累积，但 u 已经卡在边界上，系统失去调节能力。

**可能方向**：
- 减小 PID 参数范围（尤其是 Kp_max）
- 改增量式为位置式 PID（直接输出 u 而非累加）
- 在饱和时对 u_prev 做"回拉"处理
- 降低学习率使参数变化更平滑

### 问题 2：Ki 在 0 和 5 之间剧烈跳变
**根因**：Ki 的梯度 ∝ e·(Ki_max-Ki_min)，当误差持续存在时，Ki 会被快速推到上限。而当 u 饱和方向改变时，梯度反转又把 Ki 拉下来。

**可能方向**：
- 对 Ki 输出做低通滤波
- 减小 Ki 范围至 [0, 2] 或更小
- 给 Ki 单独设置更小的学习率

### 问题 3：dy/du 均值为 ~0
Jacobian 均值 -0.0006 接近于零，说明 RBF 辨识的 ∂y/∂u 在正负之间对称摆动。对于稳定系统，真实的 ∂y/∂u 应为正值。dy/du 接近零意味着 BP 网络的梯度链 `dE_du = -e * dydu` 非常弱，权值更新几乎无效。

**可能方向**：
- RBF 输入加入更多历史信息（如 y(k-3), u(k-3)）
- 初始化 C 中心点范围扩大
- 对 dy/du 加更大下限（如 0.01）

### 问题 4：Ts = 0.01s 但步长 = 0.001s
BP_PID 的 Ts 参数与 Simulink 实际步长不一致，导致积分累加量 sum_e 仅为实际值的 1/10。积分通道弱化使得稳态误差难以消除。

---

## 7. 文件结构

```
BPNN_RBFNN/
├── BP_RBFNN.slx              # Simulink 模型
├── BP_PID_Controller.m       # BP PID 控制器（MATLAB Function Block 代码）
├── RBF_Identifier_claude.m   # RBF 在线辨识器（MATLAB Function Block 代码）
├── plot_result.m             # 仿真 + 绘图脚本
├── simulation_output.mat     # 仿真输出数据 (395MB, 700万样本 × 10通道)
└── figures/                  # 仿真结果图
```

---

## 8. 给接收方 AI 的建议

1. **首要目标**：降低控制量饱和率（从 88.73% 降到 <20%），这是所有其他问题的根源
2. **约束**：单次只改一个地方，改完立刻仿真验证（使用 `plot_result.m`，StopTime=7000s）
3. **对比基准**：当前 RMS error = 0.587, sat = 88.73%, steady MAE = 0.965
4. **可改的杠杆**：超参数（lr, M, 各范围）、网络结构、PID 形式（增量→位置）、优化器（加动量/自适应学习率）、积分处理
5. **RBF 辨识器**：当前 RMSE=0.05 尚可，Jacobian 质量是主要问题
