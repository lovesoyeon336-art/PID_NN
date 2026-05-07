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

This is a MATLAB/Simulink workspace for **adaptive neural network control systems** research, specifically BP (Back-Propagation) and RBF (Radial Basis Function) neural network self-tuning PID controllers. The workspace uses **MATLAB R2022a** with Simulink.

## MATLAB MCP Server

A MATLAB MCP server is configured (see `.vscode/mcp.json`):
- Executable: `D:\MATLAB\matlab-mcp-core-server-win64.exe`
- MATLAB root: `D:\MATLAB\R2022a`
- Working folder: `d:\MATLAB\MATLAB WORKSPACE`
- Mode: `nodesktop`, auto-initialize on startup

Use the MCP server for programmatic MATLAB operations (running scripts, querying toolboxes, Simulink model manipulation).

## Directory Architecture

The project shows progressive development of neural-adaptive controllers, from basic exercises to a polished simulation framework:

### Linear Control Fundamentals (early explorations)
- `LTI/`, `LTV/` — Linear time-invariant/varying systems, basic Simulink modeling
- `discrete/` — Discrete-time control with `plant.m`, `ctrl.m`, `plot1.m`
- `PID_pa/` — PID parameter analysis, Ziegler-Nichols tuning (`Z_N.slx`)
- `filter_and_pidjifenfenli/` — PID with integral separation and filtering
- `smith/` — Smith predictor for time-delay systems
- `observerofslowd/` — State observer + PID combination

### Single Neuron / Early Neural PID
- `single_neuron PID/` — Single-neuron adaptive PID (`karina.m`, `improve.m`, `forcomparison.m`)
- `BPNN/` — BP neural network PID as a **Simulink S-Function** (`BPNN.m`). Uses 3-5-3 network structure. Has incomplete gradient code in `mdlOutputs`.
- `New bpnn/` — Refactored BP PID S-Function (`bp_pid_sf.m`). Improved: self-managed persistent state, algebraic-loop-free 2-input interface (ek, yk), Jacobian estimation via sign(Δy/Δu) filtering.

### RBF Neural Network PID
- `RBFNN/` — RBF neural network PID (`RBF_PID.m`). Combined identifier + controller in one MATLAB Function block. RBF identifies Jacobian ∂y/∂u online; PID gains updated via gradient descent.
- `BPNN_mclaude/` — BP PID in MATLAB Function block form (`bp_pid_controller.m`).

### Combined BP+RBF Architecture (most advanced)
- `BPNN_RBFNN/` — Two-block architecture: `BP_PID_Controller.m` (BP NN tunes PID gains) + `RBF_Identifier_claude.m` (RBF NN provides Jacobian ∂y/∂u estimate). Simulink model at `BP_RBFNN.slx`.
- `BPNN_RBFNN_Advanced/` — **Current canonical implementation.** Refined version with:
  - `build_model.m` — Programmatically constructs the complete Simulink model from scratch (blocks, MATLAB Function embedding, layout, wiring, save).
  - `BP_PID_Controller.m` — BP network: 3-6-3 structure, He initialization, tanh hidden / sigmoid output, gradient clipping, momentum, incremental PID.
  - `RBF_Identifier.m` — RBF network: 4-10-1 structure, 4-input vector [y(k-1), y(k-2), u(k-1), u(k-2)], centers/widths/weights all online-updated with momentum.
  - `run_tests.m` — Automated 5-scenario test suite (step response, disturbance rejection, noise robustness, sine tracking, parameter perturbation) with performance metrics and 4-figure output.

### Demo
- `pid_demo/` — DC motor PID control demo (`dc_motor_pid_ctrl.slx`). Separate from the neural control research.

## Common Workflow

**Building and simulating the canonical BP+RBF model:**
```matlab
cd('d:\MATLAB\MATLAB WORKSPACE\BPNN_RBFNN_Advanced')
build_model          % Creates BP_RBFNN_Sim.slx programmatically
run_tests            % Runs all 5 test scenarios, generates figures
```

**Batch execution from command line:**
```powershell
& "D:\MATLAB\R2022a\bin\matlab.exe" -batch "cd('BPNN_RBFNN_Advanced'); build_model; run_tests"
```

## Key Technical Conventions

- **Solver:** Fixed-step `ode4` (Runge-Kutta), step size `0.001` throughout all models
- **Plant model:** Hammerstein structure — static dead zone nonlinearity (`Dead Zone` block, ±0.3) followed by linear transfer function (default: `10/(s²+2s+5)`)
- **Control signal limits:** ±10
- **Sample time:** All MATLAB Function blocks set as atomic units with `Ts=0.001`
- **State persistence:** MATLAB Function blocks use `persistent` variables for network weights, error history, and delay registers — NOT Simulink discrete states
- **Fixed random seeds:** `rng(42)` or `rng(0)` for reproducible initialization
- **Jacobian floor:** Minimum `abs(dy_du) >= 1e-4` to prevent gradient vanishing
- **To Workspace:** Timeseries format used for post-simulation analysis

## Model Architecture (BPNN_RBFNN_Advanced)

```
Ref (Step/Sine) → Manual Switch → BP PID → Saturation → Dist Sum → Dead Zone → Plant → Noise Sum → y_meas
                    ↓                                                                        ↑
                    r → BP PID(in1)                                                          |
                    y_meas → BP PID(in2) ←──────────────────────────────────────────────────┘
                                       ↑
                    RBF Ident(out1: dy_du) → BP PID(in3)
                    RBF Ident(in1: u(k-1) from Delay)
                    RBF Ident(in2: y_meas)
```

BP and RBF blocks are independent MATLAB Function blocks connected through signal lines. The RBF estimates ∂y/∂u which the BP controller uses in its gradient descent chain rule.

## File Patterns

- `.slx` — Simulink models (binary ZIP format)
- `.m` — MATLAB scripts/functions (two types: standalone scripts and MATLAB Function block code meant for embedding)
- `.mat` — MATLAB data files
- `.slxc` / `slprj/` / `_sfprj/` / `_jitprj/` — Simulink cache/build artifacts (do not edit)
- `*.asv` — MATLAB editor auto-save backups
- `s*.l` / `s*.mat` — Simulink JIT accelerator cache files in the root (auto-generated, can be deleted)
- `EMLReport/`, `precompile/` — Simulink code generation artifacts

## Subdirectory .claude/ Configurations

Some subdirectories have their own `.claude/` settings (e.g., `BPNN_RBFNN/.claude/`). The root `.claude/` contains the canonical permissions and MCP configuration. When working in a specific subdirectory, root-level `.claude/` settings apply unless overridden.
