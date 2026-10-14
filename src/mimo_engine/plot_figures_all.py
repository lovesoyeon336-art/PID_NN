#!/usr/bin/env python3
"""Nature-figure 全部重绘: 参考 Pretrain_plant3.pdf 简洁风格"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from pathlib import Path

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "Arial", "sans-serif"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.5,
    "legend.frameon": True,
    "legend.edgecolor": "#CCCCCC",
    "legend.fontsize": 6,
    "lines.linewidth": 0.7,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})

DATA_DIR = Path(r"D:\MATLAB\engine")
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

PAL = {
    "act1": "#2166AC",   # y1实际 深蓝
    "act2": "#B2182B",   # y2实际 深红
    "pred": "#4DAF4A",   # 预测 绿
    "ref":  "#888888",   # 参考 灰
    "grid": "#E8E8E8",
}

def save_fig(fig, name):
    fig.savefig(str(FIG_DIR/f"{name}.svg"), bbox_inches="tight")
    print(f"  -> {FIG_DIR}/{name}.svg")
    plt.close(fig)

def style_ax(ax, xl="", yl="", xticks=True, yticks=True):
    if xl: ax.set_xlabel(xl, fontsize=7, labelpad=1)
    if yl: ax.set_ylabel(yl, fontsize=7, labelpad=1)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(labelsize=6, width=0.5, length=2.5, pad=1.5)
    if not xticks: ax.tick_params(labelbottom=False)
    if not yticks: ax.tick_params(labelleft=False)
    ax.grid(True, color=PAL["grid"], linewidth=0.3, alpha=0.7)
    ax.set_axisbelow(True)

# ============================================================
# Fig 1: RBF 离线辨识验证 (2×2, 每面板独立清晰)
# ============================================================
print("Fig 1: RBF 辨识验证")
rbf = sio.loadmat(str(DATA_DIR/"figdata_rbf.mat"))
t_test = rbf["t_test"].flatten()
y1_t = rbf["y1_t"].flatten(); y1_h = rbf["y1_h"].flatten()
y2_t = rbf["y2_t"].flatten(); y2_h = rbf["y2_h"].flatten()
J21_a = rbf["J21"].flatten(); J22_a = rbf["J22"].flatten()
mae1 = float(np.asarray(rbf["mae1"]).flatten()[0])
mae2 = float(np.asarray(rbf["mae2"]).flatten()[0])
r2_1 = float(np.asarray(rbf["r2_1"]).flatten()[0])
r2_2 = float(np.asarray(rbf["r2_2"]).flatten()[0])
J21_err = float(np.asarray(rbf["J21_err"]).flatten()[0])
J22_err = float(np.asarray(rbf["J22_err"]).flatten()[0])

fig1 = plt.figure(figsize=(170/25.4, 70/25.4), dpi=300)
gs = fig1.add_gridspec(1, 2, hspace=0.48, wspace=0.28)

# (a) y1 预测 vs 实际
ax = fig1.add_subplot(gs[0])
style_ax(ax, "时间 (s)", "$y_1$")
ax.plot(t_test[:200], y1_t[:200], color=PAL["act1"], lw=0.7, label="实际输出")
ax.plot(t_test[:200], y1_h[:200], color=PAL["pred"], lw=0.7, ls="--", label="RBF 预测")
ax.legend(fontsize=6, loc="upper right")
ax.set_title(f"$y_1$ 辨识  (MAE = {mae1:.4f})", fontsize=7.5, loc="center", pad=4)
ax.text(0.02, 0.94, "(a)", transform=ax.transAxes, fontsize=8, fontweight="bold")

# (b) y2 预测 vs 实际
ax = fig1.add_subplot(gs[1])
style_ax(ax, "时间 (s)", "$y_2$")
ax.plot(t_test[:200], y2_t[:200], color=PAL["act2"], lw=0.7, label="实际输出")
ax.plot(t_test[:200], y2_h[:200], color=PAL["pred"], lw=0.7, ls="--", label="RBF 预测")
ax.legend(fontsize=6, loc="upper right")
ax.set_title(f"$y_2$ 辨识  (MAE = {mae2:.4f})", fontsize=7.5, loc="center", pad=4)
ax.text(0.02, 0.94, "(b)", transform=ax.transAxes, fontsize=8, fontweight="bold")

fig1.suptitle("RBF 离线辨识验证  (独立测试集, $u\\in[-8,8]$, 滤波噪声激励)",
              fontsize=8.5, y=1.06)
save_fig(fig1, "fig1_rbf_validation")

# Fig 2 由独立脚本生成 (累积步数跟踪图), 此处跳过
print("Fig 2: skip (generated separately)")

# ============================================================
# Fig 3: 控制量汇总 — 7×2, 简洁
# ============================================================
print("Fig 3: 控制量汇总")
etd = sio.loadmat(str(DATA_DIR/"engine_test_data.mat"))
all_res = etd["all_results"][0]

SCENARIO_CN_TITLE = {
    "Step r1 Only":  "通道1 单通道阶跃",
    "Step r2 Only":  "通道2 单通道阶跃",
    "r1 Ramp":       "通道1 单通道斜坡",
    "r2 Ramp":       "通道2 单通道斜坡",
    "r1 Sine (30s)": "通道1 单通道正弦",
    "r2 Sine (30s)": "通道2 单通道正弦",
    "Dual Step":     "双通道阶跃",
}

fig3 = plt.figure(figsize=(170/25.4, 210/25.4), dpi=300)
gs = fig3.add_gridspec(7, 2, hspace=0.38, wspace=0.25,
                       left=0.10, right=0.97, top=0.97, bottom=0.04)

for i in range(7):
    r = all_res[i]
    name = SCENARIO_CN_TITLE.get(str(r["scenario_name"][0,0][0]), str(r["scenario_name"][0,0][0]))
    t = np.asarray(r["t"][0,0]).flatten()
    u1 = np.asarray(r["u1"][0,0]).flatten()
    u2 = np.asarray(r["u2"][0,0]).flatten()
    du1 = np.diff(u1, prepend=u1[0])
    du2 = np.diff(u2, prepend=u2[0])
    is_bottom = (i == 6)

    # 左: 控制量
    ax = fig3.add_subplot(gs[i,0])
    style_ax(ax, "时间 (s)" if is_bottom else "", "控制量",
             xticks=is_bottom, yticks=True)
    ax.plot(t, u1, color=PAL["act1"], lw=0.5, label="$u_1$ (燃油)")
    ax.plot(t, u2, color=PAL["act2"], lw=0.5, ls="--", label="$u_2$ (喷口)")
    if i == 0: ax.legend(fontsize=5.5, ncol=2, loc="upper right")
    ax.set_title(name, fontsize=6.5, loc="left", pad=2)

    # 右: 控制增量
    ax = fig3.add_subplot(gs[i,1])
    style_ax(ax, "时间 (s)" if is_bottom else "", "控制增量",
             xticks=is_bottom, yticks=True)
    ax.plot(t, du1, color=PAL["act1"], lw=0.4, alpha=0.8, label=r"$\Delta u_1$")
    ax.plot(t, du2, color=PAL["act2"], lw=0.4, alpha=0.8, ls="--", label=r"$\Delta u_2$")
    if i == 0: ax.legend(fontsize=5.5, ncol=2, loc="upper right")
    ax.set_title(name, fontsize=6.5, loc="left", pad=2)

fig3.suptitle("控制信号 — 全部 7 测试场景", fontsize=8.5, y=0.995)
save_fig(fig3, "fig3_control_summary")

# ============================================================
# Fig 4: 输出跟踪 — 7×2, 参考信号 + 实际输出
# ============================================================
print("Fig 4: 输出跟踪汇总")
fig4 = plt.figure(figsize=(170/25.4, 210/25.4), dpi=300)
gs = fig4.add_gridspec(7, 2, hspace=0.38, wspace=0.25,
                       left=0.10, right=0.97, top=0.97, bottom=0.04)

for i in range(7):
    r = all_res[i]
    name = SCENARIO_CN_TITLE.get(str(r["scenario_name"][0,0][0]), str(r["scenario_name"][0,0][0]))
    t = np.asarray(r["t"][0,0]).flatten()
    y1 = np.asarray(r["y1"][0,0]).flatten()
    y2 = np.asarray(r["y2"][0,0]).flatten()
    r1 = np.asarray(r["r1"][0,0]).flatten()
    r2 = np.asarray(r["r2"][0,0]).flatten()
    is_bottom = (i == 6)

    ch1_act = np.max(np.abs(r1)) > 0.01
    ch2_act = np.max(np.abs(r2)) > 0.01
    yl1 = "通道1 跟踪情况" if ch1_act else ("通道1 耦合情况" if ch2_act else "通道 1")
    yl2 = "通道2 跟踪情况" if ch2_act else ("通道2 耦合情况" if ch1_act else "通道 2")

    # 左: 通道1
    ax = fig4.add_subplot(gs[i,0])
    style_ax(ax, "时间 (s)" if is_bottom else "", yl1,
             xticks=is_bottom, yticks=True)
    ax.plot(t, r1, color=PAL["ref"], lw=0.5, ls="--", label="$r_1$ (参考)")
    ax.plot(t, y1, color=PAL["act1"], lw=0.7, label="$y_1$ (输出)")
    if i == 0: ax.legend(fontsize=5.5, ncol=2, loc="best")
    ax.set_title(name, fontsize=6.5, loc="left", pad=2)

    # 右: 通道2
    ax = fig4.add_subplot(gs[i,1])
    style_ax(ax, "时间 (s)" if is_bottom else "", yl2,
             xticks=is_bottom, yticks=True)
    ax.plot(t, r2, color=PAL["ref"], lw=0.5, ls="--", label="$r_2$ (参考)")
    ax.plot(t, y2, color=PAL["act2"], lw=0.7, label="$y_2$ (输出)")
    if i == 0: ax.legend(fontsize=5.5, ncol=2, loc="best")
    ax.set_title(name, fontsize=6.5, loc="left", pad=2)

fig4.suptitle("输出跟踪 — 全部 7 测试场景", fontsize=8.5, y=0.995)
save_fig(fig4, "fig4_tracking_summary")

# ============================================================
# Fig 5: PI 增益时变曲线
# ============================================================
print("Fig 5: PI gains")
PAL_g = {"kp": "#2E9E44", "ki": "#E28E2C"}

fig5 = plt.figure(figsize=(170/25.4, 215/25.4), dpi=300)
gs = fig5.add_gridspec(7, 2, hspace=0.38, wspace=0.25,
                       left=0.12, right=0.97, top=0.97, bottom=0.04)

for i in range(7):
    r = all_res[i]
    name = SCENARIO_CN_TITLE.get(str(r["scenario_name"][0,0][0]), str(r["scenario_name"][0,0][0]))
    t = np.asarray(r["t"][0,0]).flatten()
    kp11 = np.asarray(r["kp11"][0,0]).flatten()
    ki11 = np.asarray(r["ki11"][0,0]).flatten()
    kp22 = np.asarray(r["kp22"][0,0]).flatten()
    ki22 = np.asarray(r["ki22"][0,0]).flatten()
    is_bottom = (i == 6)

    ax = fig5.add_subplot(gs[i,0])
    style_ax(ax, "时间 (s)" if is_bottom else "", "通道 1 增益")
    ax.plot(t, kp11, color=PAL_g["kp"], lw=0.5, label="$K_{p1}$")
    ax.plot(t, ki11, color=PAL_g["ki"], lw=0.5, ls="--", label="$K_{i1}$")
    if i == 0: ax.legend(fontsize=5.5, ncol=2, loc="upper right")
    ax.set_title(name, fontsize=6.5, loc="left", pad=2)

    ax = fig5.add_subplot(gs[i,1])
    style_ax(ax, "时间 (s)" if is_bottom else "", "通道 2 增益")
    ax.plot(t, kp22, color=PAL_g["kp"], lw=0.5, label="$K_{p2}$")
    ax.plot(t, ki22, color=PAL_g["ki"], lw=0.5, ls="--", label="$K_{i2}$")
    if i == 0: ax.legend(fontsize=5.5, ncol=2, loc="upper right")
    ax.set_title(name, fontsize=6.5, loc="left", pad=2)

fig5.suptitle("BP 输出 PI 增益 — 全部 7 场景", fontsize=8.5, y=0.995)
save_fig(fig5, "fig5_pi_gains")

# ============================================================
# Fig 6: 性能指标汇总表
# ============================================================
print("Fig 6: 性能指标汇总表")

SCENARIO_CN = {
    "Step r1 Only":    "通道1 单通道阶跃",
    "Step r2 Only":    "通道2 单通道阶跃",
    "r1 Ramp":         "通道1 单通道斜坡",
    "r2 Ramp":         "通道2 单通道斜坡",
    "r1 Sine (30s)":   "通道1 单通道正弦",
    "r2 Sine (30s)":   "通道2 单通道正弦",
    "Dual Step":       "双通道阶跃",
}


def _is_active(r):
    return np.max(np.abs(r)) > 0.01


def compute_metrics(r):
    t = np.asarray(r["t"][0, 0]).flatten()
    y1 = np.asarray(r["y1"][0, 0]).flatten()
    y2 = np.asarray(r["y2"][0, 0]).flatten()
    r1 = np.asarray(r["r1"][0, 0]).flatten()
    r2 = np.asarray(r["r2"][0, 0]).flatten()
    N = len(t)
    ss = range(int(0.7 * N), N)
    r1_ss = np.mean(r1[ss])
    r2_ss = np.mean(r2[ss])
    r1_const = np.max(r1[ss]) - np.min(r1[ss]) < 0.02
    r2_const = np.max(r2[ss]) - np.min(r2[ss]) < 0.02
    r1_act = _is_active(r1)
    r2_act = _is_active(r2)

    # Overshoot
    os1_str, os2_str = "—", "—"
    if r1_const and r1_act:
        os1 = max(0, (np.max(y1) - r1_ss) / abs(r1_ss) * 100)
        os1_str = f"{os1:.1f}%"
    if r2_const and r2_act:
        os2 = max(0, (np.max(y2) - r2_ss) / abs(r2_ss) * 100)
        os2_str = f"{os2:.1f}%"

    # Coupling: max|y| of the zero-reference channel
    coup_str = "—"
    if r1_act and not r2_act:
        coup_str = f"y2: {np.max(np.abs(y2)):.4f}"
    elif r2_act and not r1_act:
        coup_str = f"y1: {np.max(np.abs(y1)):.4f}"

    # Steady-state error: only active channels
    ess1_str, ess2_str = "—", "—"
    if r1_act:
        ess1_str = f"{np.mean(np.abs(r1[ss] - y1[ss])):.4f}"
    if r2_act:
        ess2_str = f"{np.mean(np.abs(r2[ss] - y2[ss])):.4f}"

    return os1_str, os2_str, coup_str, ess1_str, ess2_str


# Build table
col_labels = ["场景", "超调 通道1", "超调 通道2", "耦合 max|y|",
              "稳态误差 通道1", "稳态误差 通道2"]
rows = []
for i in range(7):
    r = all_res[i]
    name_en = str(r["scenario_name"][0, 0][0])
    name_cn = SCENARIO_CN.get(name_en, name_en)
    os1, os2, coup, ess1, ess2 = compute_metrics(r)
    rows.append([name_cn, os1, os2, coup, ess1, ess2])

# Figure
fig6 = plt.figure(figsize=(170 / 25.4, 72 / 25.4), dpi=300)
ax = fig6.add_axes([0.03, 0.20, 0.94, 0.72])
ax.axis("off")

col_widths = [0.14, 0.13, 0.13, 0.18, 0.21, 0.21]
tbl = ax.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    colWidths=col_widths,
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(6.5)

HEADER_FG = "#1A1A1A"
CELL_FG = "#2A2A2A"
RULE_COLOR = "#AAAAAA"

for j in range(len(col_labels)):
    cell = tbl[0, j]
    cell.set_facecolor("white")
    cell.set_text_props(fontweight="bold", fontsize=6, color=HEADER_FG)
    cell.set_edgecolor(RULE_COLOR)
    cell.set_linewidth(0.6)

for i in range(7):
    for j in range(len(col_labels)):
        cell = tbl[i + 1, j]
        cell.set_facecolor("white")
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.3)
        cell.set_text_props(fontsize=6, color=CELL_FG)

        if j == 0:
            cell.set_text_props(fontsize=6, color=CELL_FG, ha="left")
            cell._loc = "left"

fig6.suptitle("性能指标汇总", fontsize=8.5, y=1.005)

# Footnote: metric definitions
footnote_lines = [
    "超调: OS = (max(y) - r_ss) / |r_ss| x 100%，适用于阶跃和斜坡信号，零参考通道不适用；",
    "耦合: max|y_零参考通道|，即当一通道有非零参考、另一通道为零时，零参考通道输出的最大绝对值；",
    "稳态误差: ess = mean(|r - y|)，取仿真最后 30% 时段，适用于所有信号。",
]
for i, line in enumerate(footnote_lines):
    fig6.text(0.03, 0.13 - i * 0.04, line, fontsize=5.5, color="#555555",
              va="top", ha="left")

save_fig(fig6, "fig6_metrics_table")
print("全部 6 张图已生成。")
