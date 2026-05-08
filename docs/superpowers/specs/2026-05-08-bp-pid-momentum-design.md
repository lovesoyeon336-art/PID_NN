# BP Neural Network PID Controller — Add Momentum

**Date:** 2026-05-08
**Scope:** Single-file optimization of `BPNN_RBFNN/BP_PID_Controller.m`
**Type:** Algorithm improvement (optimizer upgrade)

## Motivation

- Current BP controller uses plain SGD for online weight updates
- RBF identifier (`RBF_Identifier_claude.m`) already uses momentum — the two blocks are inconsistent
- Online learning with single-sample gradients per step produces high-noise gradient estimates
- Momentum smooths the update trajectory, reducing PID gain jitter and control signal oscillation

## Design

### Momentum Rule

```
v(t) = lr * grad  +  mu * v(t-1)
θ(t) = θ(t-1)  -  v(t)
```

This matches the formulation used in the RBF identifier block for consistency.

### Hyperparameter

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `mu` (momentum coefficient) | 0.3 | Small network (35 params), high-noise online setting; moderate momentum avoids overshoot |

Placed alongside existing `lr`, `grad_clip` in the hyperparameter section.

### New Persistent State

Four velocity buffers matching the four weight/bias tensors:

| Variable | Shape | Matches |
|----------|-------|---------|
| `vW1` | M × 3 | W1 |
| `vW2` | 3 × M | W2 |
| `vb1` | M × 1 | b1 |
| `vb2` | 3 × 1 | b2 |

All initialized to zeros. Standard practice — first step behaves identically to plain SGD.

### Modified Update Section (lines 149–152 → replacement)

Before:
```matlab
W2 = W2 - lr * (delta_o * h');
b2 = b2 - lr * delta_o;
W1 = W1 - lr * (delta_h * x');
b1 = b1 - lr * delta_h;
```

After:
```matlab
vW2 = lr * (delta_o * h') + mu * vW2;
vb2 = lr * delta_o         + mu * vb2;
vW1 = lr * (delta_h * x') + mu * vW1;
vb1 = lr * delta_h         + mu * vb1;

W2 = W2 - vW2;
b2 = b2 - vb2;
W1 = W1 - vW1;
b1 = b1 - vb1;
```

## Changes

Three insertion points in one file:

1. **persistent declaration** — add `vW1 vW2 vb1 vb2`  
2. **init block** — zero-initialize all four velocity buffers  
3. **weight update** — replace 4 plain-SGD lines with 8 momentum lines  

No changes to forward pass, PID computation, error signal, gradient clipping, or state update logic.

## Verification

- Run `BP_RBFNN.slx` with a step input
- Observe Kp/Ki/Kd scope: traces should be visibly smoother than baseline
- Observe control signal u: reduced high-frequency jitter
- Step response: equal or better settling time, equal or lower overshoot
