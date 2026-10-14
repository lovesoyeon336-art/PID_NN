#!/usr/bin/env python3
"""将综合大图拆成单张小图，按类别放入子文件夹。

输出结构 (共 46 张):
  figures/rbf/        — RBF 辨识验证 (y1, y2)                        =  2 张
  figures/pretrain/   — BP 预训练 12 轮拼接 (通道1, 通道2 各 1 张)     =  2 张
  figures/control/    — 每场景控制量 + 控制增量 (7 场景 × 2)           = 14 张
  figures/tracking/   — 每场景通道1 + 通道2 跟踪 (7 场景 × 2)          = 14 张
  figures/pi_gains/   — 每场景通道1 + 通道2 PI 增益 (7 场景 × 2)       = 14 张
"""

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

PAL = {
    "act1": "#2166AC",
    "act2": "#B2182B",
    "pred": "#4DAF4A",
    "ref":  "#888888",
    "grid": "#E8E8E8",
}
PAL_g = {"kp": "#2E9E44", "ki": "#E28E2C"}


def save_fig(fig, subdir, name, tight=False):
    d = FIG_DIR / subdir
    d.mkdir(exist_ok=True)
    kwargs = dict(bbox_inches="tight", pad_inches=0.02) if tight else dict()
    fig.savefig(str(d / f"{name}.svg"), **kwargs)
    fig.savefig(str(d / f"{name}.pdf"), **kwargs)
    plt.close(fig)


def style_ax(ax, xl="", yl=""):
    if xl: ax.set_xlabel(xl, fontsize=7, labelpad=2)
    if yl: ax.set_ylabel(yl, fontsize=7, labelpad=2)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(labelsize=6, width=0.5, length=2.5, pad=1.5)
    ax.grid(True, color=PAL["grid"], linewidth=0.3, alpha=0.7)
    ax.set_axisbelow(True)


def legend_above(ax, ncol=2):
    ax.legend(
        fontsize=6,
        ncol=ncol,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0,
        frameon=True,
        edgecolor="#CCCCCC",
        handlelength=1.4,
        columnspacing=0.8,
    )


def make_fig(w=4.0, h=3.0):
    """Create a single-panel 4:3 figure.  w,h in inches.
    Uses subplots_adjust for fixed margins; no tight bbox needed."""
    fig = plt.figure(figsize=(w, h), dpi=300)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.18, top=0.74)
    ax = fig.add_subplot(111)
    return fig, ax


# ============================================================
# 1. RBF validation figures
# ============================================================
print("=== RBF validation ===")
rbf = sio.loadmat(str(DATA_DIR / "figdata_rbf.mat"))
t_test = rbf["t_test"].flatten()
y1_t = rbf["y1_t"].flatten()
y1_h = rbf["y1_h"].flatten()
y2_t = rbf["y2_t"].flatten()
y2_h = rbf["y2_h"].flatten()
mae1 = float(np.asarray(rbf["mae1"]).flatten()[0])
mae2 = float(np.asarray(rbf["mae2"]).flatten()[0])

# y1  (宽图，保留 tight)
fig, ax = make_fig(w=5.0, h=3.3)
style_ax(ax, "时间 (s)", "$y_1$")
ax.plot(t_test[:200], y1_t[:200], color=PAL["act1"], lw=0.7, label="实际输出")
ax.plot(t_test[:200], y1_h[:200], color=PAL["pred"], lw=0.7, ls="--", label="RBF预测")
legend_above(ax)
ax.set_title(f"$y_1$ 辨识验证  (MAE = {mae1:.4f})", fontsize=7, loc="center", pad=22)
save_fig(fig, "rbf", "y1_validation", tight=True)

# y2  (宽图，保留 tight)
fig, ax = make_fig(w=5.0, h=3.3)
style_ax(ax, "时间 (s)", "$y_2$")
ax.plot(t_test[:200], y2_t[:200], color=PAL["act2"], lw=0.7, label="实际输出")
ax.plot(t_test[:200], y2_h[:200], color=PAL["pred"], lw=0.7, ls="--", label="RBF预测")
legend_above(ax)
ax.set_title(f"$y_2$ 辨识验证  (MAE = {mae2:.4f})", fontsize=7, loc="center", pad=22)
save_fig(fig, "rbf", "y2_validation", tight=True)

# ============================================================
# 2--4. 按场景拆控制/跟踪/增益
# ============================================================
SCENARIO_CN_FNAME = {
    "Step r1 Only":  "通道1_单通道阶跃",
    "Step r2 Only":  "通道2_单通道阶跃",
    "r1 Ramp":       "通道1_单通道斜坡",
    "r2 Ramp":       "通道2_单通道斜坡",
    "r1 Sine (30s)": "通道1_单通道正弦",
    "r2 Sine (30s)": "通道2_单通道正弦",
    "Dual Step":     "双通道_阶跃",
}
SCENARIO_CN_TITLE = {k: v.replace("_", " ") for k, v in SCENARIO_CN_FNAME.items()}

etd = sio.loadmat(str(DATA_DIR / "engine_test_data.mat"))
all_res = etd["all_results"][0]

for i in range(7):
    r = all_res[i]
    name_en = str(r["scenario_name"][0, 0][0])
    name_title = SCENARIO_CN_TITLE.get(name_en, name_en)
    name_fname = SCENARIO_CN_FNAME.get(name_en, name_en)
    t = np.asarray(r["t"][0, 0]).flatten()
    u1 = np.asarray(r["u1"][0, 0]).flatten()
    u2 = np.asarray(r["u2"][0, 0]).flatten()
    du1 = np.diff(u1, prepend=u1[0])
    du2 = np.diff(u2, prepend=u2[0])
    y1 = np.asarray(r["y1"][0, 0]).flatten()
    y2 = np.asarray(r["y2"][0, 0]).flatten()
    r1 = np.asarray(r["r1"][0, 0]).flatten()
    r2 = np.asarray(r["r2"][0, 0]).flatten()
    kp11 = np.asarray(r["kp11"][0, 0]).flatten()
    ki11 = np.asarray(r["ki11"][0, 0]).flatten()
    kp22 = np.asarray(r["kp22"][0, 0]).flatten()
    ki22 = np.asarray(r["ki22"][0, 0]).flatten()

    print(f"  {name_title}")

    # -- 控制量 --
    fig, ax = make_fig()
    style_ax(ax, "时间 (s)", "控制量")
    ax.plot(t, u1, color=PAL["act1"], lw=0.5, label="$u_1$ (燃油)")
    ax.plot(t, u2, color=PAL["act2"], lw=0.5, ls="--", label="$u_2$ (喷口)")
    legend_above(ax)
    ax.set_title(name_title, fontsize=7, loc="center", pad=22)
    save_fig(fig, "control", f"{name_fname}_u")

    # -- 控制增量 --
    fig, ax = make_fig()
    style_ax(ax, "时间 (s)", "控制增量")
    ax.plot(t, du1, color=PAL["act1"], lw=0.4, alpha=0.8, label=r"$\Delta u_1$")
    ax.plot(t, du2, color=PAL["act2"], lw=0.4, alpha=0.8, ls="--", label=r"$\Delta u_2$")
    legend_above(ax)
    ax.set_title(name_title, fontsize=7, loc="center", pad=22)
    save_fig(fig, "control", f"{name_fname}_du")

    # 判断各通道角色
    ch1_act = np.max(np.abs(r1)) > 0.01
    ch2_act = np.max(np.abs(r2)) > 0.01
    yl1 = "通道1 跟踪情况" if ch1_act else ("通道1 耦合情况" if ch2_act else "通道 1")
    yl2 = "通道2 跟踪情况" if ch2_act else ("通道2 耦合情况" if ch1_act else "通道 2")

    # -- 通道1 跟踪 --
    fig, ax = make_fig()
    style_ax(ax, "时间 (s)", yl1)
    ax.plot(t, r1, color=PAL["ref"], lw=0.5, ls="--", label="$r_1$ (参考)")
    ax.plot(t, y1, color=PAL["act1"], lw=0.7, label="$y_1$ (输出)")
    legend_above(ax)
    ax.set_title(name_title, fontsize=7, loc="center", pad=22)
    save_fig(fig, "tracking", f"{name_fname}_ch1")

    # -- 通道2 跟踪 --
    fig, ax = make_fig()
    style_ax(ax, "时间 (s)", yl2)
    ax.plot(t, r2, color=PAL["ref"], lw=0.5, ls="--", label="$r_2$ (参考)")
    ax.plot(t, y2, color=PAL["act2"], lw=0.7, label="$y_2$ (输出)")
    legend_above(ax)
    ax.set_title(name_title, fontsize=7, loc="center", pad=22)
    save_fig(fig, "tracking", f"{name_fname}_ch2")

    # -- 通道1 PI 增益 --
    fig, ax = make_fig()
    style_ax(ax, "时间 (s)", "通道 1 增益")
    ax.plot(t, kp11, color=PAL_g["kp"], lw=0.5, label="$K_{p1}$")
    ax.plot(t, ki11, color=PAL_g["ki"], lw=0.5, ls="--", label="$K_{i1}$")
    legend_above(ax)
    ax.set_title(name_title, fontsize=7, loc="center", pad=22)
    save_fig(fig, "pi_gains", f"{name_fname}_ch1")

    # -- 通道2 PI 增益 --
    fig, ax = make_fig()
    style_ax(ax, "时间 (s)", "通道 2 增益")
    ax.plot(t, kp22, color=PAL_g["kp"], lw=0.5, label="$K_{p2}$")
    ax.plot(t, ki22, color=PAL_g["ki"], lw=0.5, ls="--", label="$K_{i2}$")
    legend_above(ax)
    ax.set_title(name_title, fontsize=7, loc="center", pad=22)
    save_fig(fig, "pi_gains", f"{name_fname}_ch2")

# ============================================================
# 5. BP 预训练跟踪 — 12 轮首尾拼接 (累积步数)
# ============================================================
print("=== BP pretrain tracking ===")
bp = sio.loadmat(str(DATA_DIR / "figdata_bp.mat"))
N_sim_bp = int(bp["N_sim"][0, 0])
epochs = int(bp["epochs"][0, 0])
snap_y1 = bp["snap_y1"][0]
snap_y2 = bp["snap_y2"][0]
snap_r1 = bp["snap_r1"][0]
snap_r2 = bp["snap_r2"][0]

total_steps = epochs * N_sim_bp
t_cum = np.arange(total_steps) * 0.01
k_cum = np.arange(total_steps)

# 拼接所有 epoch
r1_cat = np.concatenate([np.asarray(snap_r1[ep]).flatten() for ep in range(epochs)])
y1_cat = np.concatenate([np.asarray(snap_y1[ep]).flatten() for ep in range(epochs)])
r2_cat = np.concatenate([np.asarray(snap_r2[ep]).flatten() for ep in range(epochs)])
y2_cat = np.concatenate([np.asarray(snap_y2[ep]).flatten() for ep in range(epochs)])

# Epoch 分隔线位置
ep_divs = [ep * N_sim_bp for ep in range(1, epochs)]

for ch, r_cat, y_cat, ylabel, ch_title in [
    (1, r1_cat, y1_cat, "通道 1", "通道1"),
    (2, r2_cat, y2_cat, "通道 2", "通道2"),
]:
    fig, ax = make_fig(6.7, 2.8)
    style_ax(ax, "累积步数 k", ylabel)
    ax.plot(k_cum, r_cat, color=PAL["ref"], lw=0.5, ls="--", label="参考信号")
    ax.plot(k_cum, y_cat, color=PAL["act1"] if ch == 1 else PAL["act2"],
            lw=0.4, label=f"$y_{ch}$ 被控输出")
    # Epoch 分隔虚线
    for d in ep_divs:
        ax.axvline(d, color="#CCCCCC", lw=0.3, ls=":")
    legend_above(ax)
    ax.set_title(f"BP 预训练 {ch_title}  ({epochs} epochs, 权重继承, 幅值 0.3~1.5)",
                 fontsize=7, loc="center", pad=22)
    save_fig(fig, "pretrain", f"ch{ch}_concatenated", tight=True)

print("\n全部拆分完成。")
