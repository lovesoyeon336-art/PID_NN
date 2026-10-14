## Project Overview

This workspace targets a single folder: `BPNN_mclaude/` — a **self-contained BP neural network adaptive PID controller** in MATLAB/Simulink. Uses **MATLAB R2022a** with Simulink. Unlike `BPNN_RBFNN/`, this project does NOT use an external RBF identifier — the Jacobian ∂y/∂u is approximated internally as `dJ_du = -e` (assumes positive plant gain). Designed to be simpler and more self-reliant.

## MATLAB MCP Server

A MATLAB MCP server is configured (see `.vscode/mcp.json`):
- Executable: `D:\MATLAB\matlab-mcp-core-server-win64.exe`
- MATLAB root: `D:\MATLAB\R2022a`
- Working folder: `d:\MATLAB\MATLAB WORKSPACE`
- Mode: `nodesktop`, auto-initialize on startup

Use the MCP server for programmatic MATLAB operations (running scripts, querying toolboxes, Simulink model manipulation).

## Directory Contents

The working directory is `BPNN_mclaude/`, containing:

| File | Role |
|------|------|
| `bp_pid_controller.m` | BP neural network PID controller (MATLAB Function block). 3-5-3 structure with bias neurons, tanh hidden / sigmoid output, **momentum SGD**, fixed rng seed, Jacobian approximated as `dJ_du = -e`. |
| `bpnn_claude.slx` | Simulink model containing the MATLAB Function block, plant, reference input, scopes, and To Workspace blocks. |

## Common Workflow

**重要：每次仿真必须在单个 MATLAB 会话中完成全部操作（更新代码 → 同步模型 → 仿真 → 分析 → 绘图），不要反复开关 MATLAB。**

```matlab
% === 标准工作流程（单次会话）===
cd('BPNN_mclaude');

% 1. 打开模型并同步最新代码到 Simulink 内部的 MATLAB Function block
open_system('bpnn_claude');
rt = sfroot;
chart = rt.find('-isa', 'Stateflow.EMChart', 'Path', 'bpnn_claude/MATLAB Function');
chart.Script = fileread('bp_pid_controller.m');
save_system('bpnn_claude');

% 2. 仿真（快速迭代建议 200-500s，完整验证用更长时间）
out = sim('bpnn_claude', 'StopTime', '500');

% 3. 提取数据（To Workspace 输出为普通 double 数组，非 timeseries）
t = out.tout;   r = out.r;   y = out.y;   u = out.u;
% PID 参数: out.p (Kp), out.i (Ki), out.d (Kd)

% 4. 自动分析指标
err_rms = rms(r-y);
ess = mean(abs(r(round(0.8*end):end) - y(round(0.8*end):end)));
sat_rate = sum(abs(u) > 9.9) / length(u) * 100;
fprintf('RMS误差: %.4f, 稳态MAE: %.4f, u饱和率: %.1f%%\n', err_rms, ess, sat_rate);

% 5. 绘图并保存到 figures/
figure('visible','off'); plot(t,r,t,y); legend('r','y'); grid on;
saveas(gcf, 'figures/fig3_tracking.png');
figure('visible','off'); plot(t,u); xlabel('t'); ylabel('u'); grid on;
saveas(gcf, 'figures/fig2_control_u.png');
figure('visible','off'); plot(t,out.p,t,out.i,t,out.d); legend('Kp','Ki','Kd'); grid on;
saveas(gcf, 'figures/fig1_pid_params.png');

% 6. 关闭模型
bdclose('all');
```

## Algorithm Improvement Rules

修改 `bp_pid_controller.m` 时必须遵守以下规则：

1. **一次只改一处。** 每次迭代只做一个算法修改，不要同时改多个。

2. **每次修改后必须仿真并绘图。** 在单个 MATLAB 会话中完成：同步代码 → 保存模型 → 仿真 → 分析指标 → 生成图表。

3. **Simulink 缓存机制：** 编辑 `.m` 文件本身不会自动更新 Simulink 模型内部缓存的代码。必须通过 `chart.Script = fileread(...)` 显式同步后保存。

4. **自动分析指标：** 每次仿真后立即提取数值指标（RMS 误差、稳态 MAE、u 饱和率、超调量），与上一次仿真结果对比，判断改进方向是否正确。

5. **仿真时长策略：** 快速迭代用 200-500s，最终验证用 7000s。每次仿真覆盖上一次的 `simulation_output.mat` 和 `figures/`。

## Model Architecture (BPNN_mclaude)

```
Ref → Sum(–) → BP PID (MATLAB Function) → Plant → y
  ↑               ↑
  |    Jacobian (internal approximation)
  |        dJ_du = -e  (assumes ∂y/∂u > 0)
  |               |
  └───────────────┘
```

Single MATLAB Function block:
- **3 inputs**: r (reference), y (measured output), ts (sampling time)
- **4 outputs**: u (control signal), Kp, Ki, Kd
- **No external identifier** — gradient chain replaces ∂y/∂u with the sign assumption `dJ_du = -e`
- **Bias neurons**: weight matrices include an extra column for bias (n_in+1 → n_h, n_h+1 → n_out), eliminating separate bias vectors

## Key Technical Conventions

- **Solver:** Fixed-step, step size `0.001` (set in Simulink, ts input must match)
- **Plant model:** Not specified in code — generic second-order system (configured in Simulink model)
- **Control signal limits:** ±10
- **Network structure:** 3-5-3 with bias neurons in both layers
- **Hidden activation:** tanh, output range (-1, 1)
- **Output activation:** sigmoid, output range (0, 1), mapped to [K_min, K_max]
- **Weight initialization:** Uniform random `0.5*(2*rand-1)` with `rng(1)` for reproducibility
- **State persistence:** `persistent` variables for weights (w_ih, w_ho), weight deltas (dw_ih, dw_ho), error history (e_1, e_2), previous control (u_1), cached activations (Oh, Oo)
- **Gradient chain:** `∂J/∂w = dJ_du · dU/dK · dK/dOo · dOo/dnet_o · [prev_layer;1]'`
- **Momentum:** `new_dw = -lr * grad + alpha * dw_prev`, then `w = w + new_dw`
- **Integral approximation:** `ei = e_1 + e*ts` (single-step trapezoidal, resets each call — no windup risk but limited steady-state error elimination)

## File Patterns

- `.slx` — Simulink models (binary ZIP format)
- `.m` — MATLAB scripts/functions (two types: standalone scripts and MATLAB Function block code meant for embedding)
- `.mat` — MATLAB data files
- `.slxc` / `slprj/` / `_sfprj/` / `_jitprj/` — Simulink cache/build artifacts (do not edit)
- `*.asv` — MATLAB editor auto-save backups

## Configuration Files

- `CLAUDE.md` — This file, in `BPNN_mclaude/`
- `../CLAUDE.md` — Root-level behavioral guidelines (loaded globally)
- `../.claude/settings.json` — Root-level permissions and hooks
- `../.claude/settings.local.json` — Local overrides (git-ignored)
- `../.vscode/mcp.json` — MATLAB MCP server connection config
