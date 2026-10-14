#!/usr/bin/env python3
"""MIMO BP-PID+RBF engine controller -- publication figures.

Reads engine_test_data.mat, generates two multi-panel time-series figures:
  - figures/Step_r1_Only.{svg,png}
  - figures/Step_r2_Only.{svg,png}

Usage:  D:/Python312/python.exe plot_figures.py [data_dir]
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# ==== 中文字体 + Nature-figure rcParams ====
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["legend.frameon"] = False
plt.rcParams["lines.linewidth"] = 0.8

# ==== Palette ====
PALETTE = {
    "ch1": "#0F4D92",
    "ch2": "#B64342",
    "ref": "#4D4D4D",
    "kp":  "#2E9E44",
    "ki":  "#E28E2C",
    "kd":  "#9A4D8E",
    "grid": "#E0E0E0",
}


def load_data(data_dir):
    mat_path = Path(data_dir) / "engine_test_data.mat"
    data = sio.loadmat(str(mat_path))
    return data["all_results"][0]


def style_ax(ax, xlabel="时间 (s)"):
    ax.set_xlabel(xlabel, fontsize=7)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(labelsize=6, width=0.5, length=2.5)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.3, alpha=0.7)
    ax.set_axisbelow(True)


def add_panel_label(ax, label, x=-0.08, y=1.02):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", color="#272727", ha="left", va="bottom")


def _get1d(obj, key):
    arr = obj[key][0, 0]
    return np.asarray(arr).flatten()


def draw_one_figure(results, fig_dir, fname_stem):
    """Draw a single 3x2 multi-panel figure from one scenario's results."""
    t  = _get1d(results, "t")
    y1 = _get1d(results, "y1")
    y2 = _get1d(results, "y2")
    u1 = _get1d(results, "u1")
    u2 = _get1d(results, "u2")
    r1 = _get1d(results, "r1")
    r2 = _get1d(results, "r2")
    du1 = np.diff(u1, prepend=u1[0])
    du2 = np.diff(u2, prepend=u2[0])
    kp11 = _get1d(results, "kp11")
    ki11 = _get1d(results, "ki11")
    kp22 = _get1d(results, "kp22")
    ki22 = _get1d(results, "ki22")

    fig = plt.figure(figsize=(180 / 25.4, 200 / 25.4), dpi=300)
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30)

    # -- (a) 控制量 --
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a)
    ax_a.plot(t, u1, color=PALETTE["ch1"], lw=0.8, label="$u_1$")
    ax_a.plot(t, u2, color=PALETTE["ch2"], lw=0.8, ls="--", label="$u_2$")
    ax_a.set_ylabel("控制量", fontsize=7)
    ax_a.legend(fontsize=6, loc="best", handlelength=1.5)
    add_panel_label(ax_a, "(a)")

    # -- (b) 控制增量 --
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b)
    ax_b.plot(t, du1, color=PALETTE["ch1"], lw=0.5, alpha=0.8, label=r"$\Delta u_1$")
    ax_b.plot(t, du2, color=PALETTE["ch2"], lw=0.5, alpha=0.8, ls="--", label=r"$\Delta u_2$")
    ax_b.set_ylabel("控制增量", fontsize=7)
    ax_b.legend(fontsize=6, loc="best", handlelength=1.5)
    add_panel_label(ax_b, "(b)")

    # -- (c) 通道1 跟踪 --
    ax_c = fig.add_subplot(gs[1, 0])
    style_ax(ax_c)
    ax_c.plot(t, r1, color=PALETTE["ref"], lw=0.8, ls="--", dashes=(3, 2), label="$r_1$")
    ax_c.plot(t, y1, color=PALETTE["ch1"], lw=0.8, label="$y_1$")
    ax_c.set_ylabel("通道1 输出", fontsize=7)
    ax_c.legend(fontsize=6, loc="best", handlelength=1.5)
    add_panel_label(ax_c, "(c)")

    # -- (d) 通道2 跟踪 --
    ax_d = fig.add_subplot(gs[1, 1])
    style_ax(ax_d)
    ax_d.plot(t, r2, color=PALETTE["ref"], lw=0.8, ls="--", dashes=(3, 2), label="$r_2$")
    ax_d.plot(t, y2, color=PALETTE["ch2"], lw=0.8, label="$y_2$")
    peak_idx = np.argmax(y2)
    if y2[peak_idx] > r2[peak_idx] * 1.02:
        ax_d.annotate(f"{y2[peak_idx]:.3f}", xy=(t[peak_idx], y2[peak_idx]),
                      xytext=(t[peak_idx] + 0.3, y2[peak_idx] + 0.05),
                      fontsize=5.5, color=PALETTE["ch2"],
                      arrowprops=dict(arrowstyle="->", color=PALETTE["ch2"], lw=0.5))
    min_idx = np.argmin(y2)
    if y2[min_idx] < -0.05:
        ax_d.annotate(f"{y2[min_idx]:.3f}", xy=(t[min_idx], y2[min_idx]),
                      xytext=(t[min_idx] + 0.3, y2[min_idx] - 0.08),
                      fontsize=5.5, color=PALETTE["ch2"],
                      arrowprops=dict(arrowstyle="->", color=PALETTE["ch2"], lw=0.5))
    ax_d.set_ylabel("通道2 输出", fontsize=7)
    ax_d.legend(fontsize=6, loc="best", handlelength=1.5)
    add_panel_label(ax_d, "(d)")

    # -- (e) 通道1 PID 增益 --
    ax_e = fig.add_subplot(gs[2, 0])
    style_ax(ax_e)
    ax_e.plot(t, kp11, color=PALETTE["kp"], lw=0.6, label="$K_{p11}$")
    ax_e.plot(t, ki11, color=PALETTE["ki"], lw=0.6, ls="--", label="$K_{i11}$")
    ax_e.set_ylabel("通道1 PI 增益", fontsize=7)
    ax_e.legend(fontsize=5.5, loc="best", ncol=2, handlelength=1.5)
    add_panel_label(ax_e, "(e)")

    # -- (f) 通道2 PID 增益 --
    ax_f = fig.add_subplot(gs[2, 1])
    style_ax(ax_f)
    ax_f.plot(t, kp22, color=PALETTE["kp"], lw=0.6, label="$K_{p22}$")
    ax_f.plot(t, ki22, color=PALETTE["ki"], lw=0.6, ls="--", label="$K_{i22}$")
    ax_f.set_ylabel("通道2 PI 增益", fontsize=7)
    ax_f.legend(fontsize=5.5, loc="best", ncol=2, handlelength=1.5)
    add_panel_label(ax_f, "(f)")

    fig.subplots_adjust(hspace=0.45, wspace=0.30, left=0.10, right=0.95, top=0.95, bottom=0.08)

    out_png = fig_dir / f"{fname_stem}.png"
    fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_png}")


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    fig_dir = data_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    print(f"Loading: {data_dir / 'engine_test_data.mat'}")
    all_results = load_data(data_dir)
    n = len(all_results)
    print(f"  {n} scenarios found")

    for i in range(n):
        r = all_results[i]
        raw = r["scenario_name"][0, 0]
        name = str(raw[0]) if hasattr(raw, "__iter__") and not isinstance(raw, str) else str(raw)
        fname = name.replace(" ", "_")
        print(f"  Drawing: {name}")
        draw_one_figure(r, fig_dir, fname)

    print("Done.")


if __name__ == "__main__":
    main()
