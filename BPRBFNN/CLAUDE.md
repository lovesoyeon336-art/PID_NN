## BPRBFNN — BP + RBF 神经网络自适应 PID 控制器

MATLAB/Simulink，R2022a。

## 快速运行

```matlab
plot_results()
```

仿真完成后自动生成 5 张诊断图，保存到 `figures/`，每次运行覆盖旧图（固定文件名：`fig1_pid_params.png`、`fig2_control_u.png`、`fig3_tracking.png`、`fig4_dydu.png`、`fig5_yhat.png`）。

## 架构

```
Ref → Sum(–) → BP PID → Saturation(±10) → Plant → y
      ↑                                      |
      |         ┌── RBF Ident ←──────────────┤
      |         │ (Jacobian ∂y/∂u)            |
      └─────────┴─────────────────────────────┘
```

## 命令行运行

通过 COM 桥接向已运行的 MATLAB 桌面实例发送命令（需先在 MATLAB 中执行一次 `enableservice('AutomationServer', true)`）：

```powershell
powershell -File "D:\MATLAB\MATLAB WORKSPACE\matlab-run.ps1" "cd('D:\MATLAB\MATLAB WORKSPACE\BPRBFNN'); plot_results()"
```

## 技术要点

- **求解器:** Fixed-step ode4, 步长 0.001
