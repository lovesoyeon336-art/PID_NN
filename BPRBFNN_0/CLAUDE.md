# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Behavioral Guidelines

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Project Overview

This workspace targets a single folder: `BPNN_RBFNN/` — a **BP + RBF combined neural network adaptive PID controller** in MATLAB/Simulink. Uses **MATLAB R2022a** with Simulink.

## MATLAB MCP Server

A MATLAB MCP server is configured (see `.vscode/mcp.json`):
- Executable: `D:\MATLAB\matlab-mcp-core-server-win64.exe`
- MATLAB root: `D:\MATLAB\R2022a`
- Working folder: `d:\MATLAB\MATLAB WORKSPACE`
- Mode: `nodesktop`, auto-initialize on startup

Use the MCP server for programmatic MATLAB operations (running scripts, querying toolboxes, Simulink model manipulation).

## Directory Contents

The working directory is `BPNN_RBFNN/`, containing:

| File | Role |
|------|------|
| `BP_PID_Controller.m` | BP neural network PID controller (MATLAB Function block). 3-5-3 structure, Xavier init, tanh hidden / sigmoid output, plain SGD, gradient clipping, incremental PID. |
| `RBF_Identifier_claude.m` | RBF neural network online identifier (MATLAB Function block). 4-8-1 structure with momentum on all three parameter groups (W, C, Sigma). Provides Jacobian ∂y/∂u for BP backpropagation. |
| `BP_RBFNN.slx` | Simulink model wiring the two blocks together with plant, reference input, scopes, and To Workspace blocks. |
| `plot_result.m` | Standalone script. Runs `sim('BP_RBFNN', 'StopTime', '1000')` then plots 5 figures (PID params, control u, tracking, Jacobian, y_hat) and saves to `figures/`. |

## Common Workflow

```matlab
% Quick: run simulation and auto-generate 5 diagnostic plots
plot_results()

% Or manually:
open_system('BPNN_RBFNN/BP_RBFNN.slx')
sim('BPNN_RBFNN/BP_RBFNN.slx')

% Access results from the 'out' structure in workspace
out.y_meas   % measured output
out.u        % control signal
out.Kp, out.Ki, out.Kd  % PID gains
```

## Algorithm Improvement Rules

**These rules are mandatory when modifying `BP_PID_Controller.m` or `RBF_Identifier_claude.m`:**

1. **One change at a time.** Only make a single algorithm modification per iteration. Do not batch multiple changes together.

2. **Simulate and plot after every change.** After each modification:
   - Run `sim('BP_RBFNN', 'StopTime', '1000')` to simulate 1000s
   - Call `plot_results()` (or the equivalent plotting code) to generate diagnostic figures
   - Save all figures to `BPNN_RBFNN/figures/` for visual review
   - Each run must overwrite the previous figures (fixed filenames: `fig1_pid_params.png`, `fig2_control_u.png`, etc.) so the folder always reflects the latest simulation
   - Do not proceed to the next change until the simulation completes and plots are generated

3. **Update Simulink blocks + simulate in a single MATLAB session.** Editing the `.m` file alone is insufficient — the Simulink model caches block code internally. Open Simulink once, update the block, save, then immediately run the simulation — do not open/close twice.

   **Update BP PID + simulate (single session):**
   ```matlab
   open_system('BP_RBFNN');
   rt = sfroot;
   chart = rt.find('-isa', 'Stateflow.EMChart', 'Path', 'BP_RBFNN/MATLAB Function1');
   chart.Script = fileread('BP_PID_Controller.m');
   save_system('BP_RBFNN');
   out = sim('BP_RBFNN', 'StopTime', '1000');
   plot_results();
   bdclose('all');
   ```

   **Update RBF Identifier + simulate (single session):**
   ```matlab
   open_system('BP_RBFNN');
   rt = sfroot;
   chart = rt.find('-isa', 'Stateflow.EMChart', 'Path', 'BP_RBFNN/MATLAB Function2');
   chart.Script = fileread('RBF_Identifier_claude.m');
   save_system('BP_RBFNN');
   out = sim('BP_RBFNN', 'StopTime', '1000');
   plot_results();
   bdclose('all');
   ```

4. **Consent model: plan approval = simulation consent.** Ask the user for permission before creating new files (`.m`, `.slx`, `.mat`, etc.) or when proposing an algorithm change plan. Once the user approves the plan, proceed with code change + block update + simulation + analysis without re-asking for each step.

5. **Each simulation .mat data must overwrite the previous.** The file `simulation_output.mat` uses a fixed filename — every new simulation replaces the old data. The `figures/` directory likewise only holds the latest run's PNG files.

6. **Auto-analyze after each simulation.** After the simulation completes and plots are saved, load `simulation_output.mat` and extract key numerical metrics:
   - Control saturation rate (`|u| > 9.9` percentage)
   - PID parameter means, final values, and ranges (Kp, Ki, Kd)
   - Tracking performance (max overshoot, steady-state MAE, RMS error)
   - RBF ident quality (RMS ident error, dy/du statistics)
   - Present the analysis together with algorithmic interpretation — what changed, what broke, what improved.

## Model Architecture (BPNN_RBFNN)

```
Ref → Sum(–) → BP PID → Saturation(±10) → Plant → y_meas
       ↑                                         |
       |            ┌── RBF Ident ←──────────────┤
       |            │ (Jacobian ∂y/∂u)            │
       └────────────┴─────────────────────────────┘
```

Two MATLAB Function blocks connected through signal lines:
- **BP_PID_Controller** (3 inputs): r (reference), y (measured output), dy_du (Jacobian from RBF)
- **RBF_Identifier_claude** (2 inputs): u_prev (delayed control), y (measured output)

The RBF identifies the plant Jacobian ∂y/∂u online; the BP controller uses it in the gradient descent chain rule to tune Kp, Ki, Kd.

## Key Technical Conventions

- **Solver:** Fixed-step `ode4` (Runge-Kutta), step size `0.001`
- **Plant model:** Hammerstein structure — static dead zone nonlinearity (±0.3) + linear transfer function (default: `10/(s²+2s+5)`)
- **Control signal limits:** ±10
- **Sample time:** All MATLAB Function blocks set as atomic units with `Ts=0.001`
- **State persistence:** MATLAB Function blocks use `persistent` variables for network weights, error history, delay registers, and momentum buffers — NOT Simulink discrete states
- **Jacobian floor:** Minimum `abs(dy_du) >= 1e-4` to prevent gradient vanishing
- **To Workspace:** Timeseries format used for post-simulation analysis
- **BP weight update:** Plain SGD `W = W - lr * grad` (no momentum). RBF uses momentum on all three parameter groups (W, C, Sigma).

## File Patterns

- `.slx` — Simulink models (binary ZIP format)
- `.m` — MATLAB scripts/functions (two types: standalone scripts and MATLAB Function block code meant for embedding)
- `.mat` — MATLAB data files
- `.slxc` / `slprj/` / `_sfprj/` / `_jitprj/` — Simulink cache/build artifacts (do not edit)
- `*.asv` — MATLAB editor auto-save backups

## Configuration Files

- `CLAUDE.md` — This file, in repo root
- `.claude/settings.json` — Root-level permissions and hooks
- `.claude/settings.local.json` — Local overrides (git-ignored)
- `.vscode/mcp.json` — MATLAB MCP server connection config
