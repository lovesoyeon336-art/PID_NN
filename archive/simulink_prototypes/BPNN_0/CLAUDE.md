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

```matlab
% Run simulation
open_system('BPNN_mclaude/bpnn_claude.slx')
sim('BPNN_mclaude/bpnn_claude.slx')

% Access results from workspace
% Block outputs: u (control signal), Kp, Ki, Kd (PID gains)
```

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
