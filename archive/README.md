# 历史版本说明

本目录保留毕业设计开发过程中的中间版本与工具配置，**不是最终成果**。最终成果见仓库根目录 `README.md` 指向的 `src/`、`results/`、`docs/`。

## simulink_prototypes/

早期基于 Simulink 的控制器原型与对照版本，多数以 `.slx` 模型 + 配套 `.m` 脚本形式存在：

| 目录 | 内容 |
|---|---|
| `BPNN`、`BPNN_0`、`BPNN_1` | BP 神经网络 PID 的早期 Simulink 实现 |
| `BPNN_comparison_deltau`、`BPNN_comparsion_nodeltau` | 控制增量有无的对照版本（目录名中的拼写错误保留原样） |
| `BPRBFNN`、`BPRBFNN_0`、`BPRBFNN_1` | BP + RBF 辨识组合结构的早期实现 |
| `BPRBFNN_comparison` | 上述结构的对照版本 |
| `LTV`、`LTI`、`discrete`、`I_dis` | 被控对象与线性系统基础验证模型 |
| `smith`、`observerofslowd`、`filter_and_pidjifenfenli` | 史密斯预估、慢扰观测器、滤波与 PID 分离等探索尝试 |
| `single_neuron PID`、`PID_pa`、`RBFNN`、`pid_demo` | 单神经元 PID、参数自整定、纯 RBF 等对照尝试 |
| `SN`、`LTV` | 其他中间实验 |

这些目录中包含大量重复文件（例如四个 `BPNN*` 目录下的 `plot_result.m` 内容完全一致），保留原样以便追溯当时的思路。

## tooling/

| 文件 | 说明 |
|---|---|
| `CLAUDE.md`、`.claude/` | 开发时使用的 AI 助手行为约定，非项目内容 |
| `.vscode/mcp.json` | 编辑器配置 |
| `matlab-run.ps1` | MATLAB 调用脚本 |
| `test.m` | 临时验证脚本 |
| `bp_pid_result.mat` | 早期结果数据 |
| `Z-N法.fig` | Ziegler-Nichols 整定的早期尝试 |
| `2026-05-08-bp-pid-momentum-design.md` | 开发过程中的设计说明 |

## 说明

历史版本在 git 提交历史中同样可以追溯；如需进一步精简，可删除本目录后单独提交，历史记录不受影响。
