"""
Nature-style figure generation for BP-PID paper.
Reads MATLAB-exported metrics.csv and timeseries_export.mat.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.io
import os

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
})

C_RBF  = '#2166AC'
C_BPFD = '#B2182B'
C_FIX  = '#4DAF4A'
C_REF  = '#333333'

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_pub(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name + '.pdf'), bbox_inches='tight')
    print(f'  Saved: {name}.pdf')


# ===== Load data =====
with open(os.path.join(OUT_DIR, 'metrics.csv'), 'r', encoding='utf-8') as f:
    f.readline()
    lines = [l.strip().split(',') for l in f]

scenarios = [l[0] for l in lines]
nS = len(scenarios)
R_MAE = np.array([float(l[1])  for l in lines])
B_MAE = np.array([float(l[2])  for l in lines])
F_MAE = np.array([float(l[3])  for l in lines])
R_Ov  = np.array([float(l[4])  for l in lines])
B_Ov  = np.array([float(l[5])  for l in lines])
F_Ov  = np.array([float(l[6])  for l in lines])
R_SS  = np.array([float(l[7])  for l in lines])
B_SS  = np.array([float(l[8])  for l in lines])
F_SS  = np.array([float(l[9])  for l in lines])

md = scipy.io.loadmat(os.path.join(OUT_DIR, 'timeseries_export.mat'))
Y_r = md['Y_r_arr'].T
Y_b = md['Y_b_arr'].T
Y_f = md['Y_f_arr'].T
R   = md['R_arr'].T
N   = Y_r.shape[0]
Ts  = 0.01
t   = np.arange(1, N+1) * Ts

print(f'Loaded {nS} scenarios, N={N}')

# Plant definitions
PLANTS = {
    'plant1': ('Plant 1:  y(k) = a(k)/(1+y(k-1)^2)*y(k-1) + u(k-1)\n'
               'a(k) = 1.2*(1-0.8*exp(-0.1k)),  Ts=0.01s',
               range(0, 7)),
    'plant2': ('Plant 2:  y(k) = 1.7*y(k-1) - 0.72*y(k-2) + 0.03*u(k-1)\n'
               'Ts=0.01s',
               range(7, 14)),
    'plant3': ('Plant 3:  v = u + 0.15*u^3,  '
               'y(k) = 1.6*y(k-1) - 0.68*y(k-2) + 0.06*v\nTs=0.01s',
               range(14, 21)),
}

# ===== Per-plant metric tables =====
def draw_plant_table(plant_id, title_text, row_range):
    rows = list(row_range)
    nP = len(rows)
    fig, ax = plt.subplots(figsize=(11, 0.43 * nP + 1.1))
    ax.axis('off')

    col_labels = ['场景',
                  'BPRBF\nMAE', 'BPPID\nMAE', 'FixPID\nMAE',
                  'BPRBF\n超调%', 'BPPID\n超调%', 'FixPID\n超调%',
                  'BPRBF\n稳态误差', 'BPPID\n稳态误差', 'FixPID\n稳态误差']

    cell_text = []
    for i in rows:
        cell_text.append([scenarios[i],
                          f'{R_MAE[i]:.4f}', f'{B_MAE[i]:.4f}', f'{F_MAE[i]:.4f}',
                          f'{R_Ov[i]:.1f}%', f'{B_Ov[i]:.1f}%', f'{F_Ov[i]:.1f}%',
                          f'{R_SS[i]:.4f}', f'{B_SS[i]:.4f}', f'{F_SS[i]:.4f}'])

    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.5)

    col_w = [0.28] + [0.08]*9
    for i in range(nP + 1):
        for j in range(10):
            tbl[i, j].set_width(col_w[0] if j == 0 else col_w[1])

    for j in range(10):
        c = tbl[0, j]
        c.set_facecolor('#333333')
        c.get_text().set_color('white')
        c.get_text().set_fontweight('bold')
        c.get_text().set_fontsize(7)

    for ii, i in enumerate(rows):
        bg = '#F5F5F5' if ii % 2 == 0 else 'white'
        for j in range(10):
            c = tbl[ii+1, j]
            c.set_facecolor(bg)
            c.set_edgecolor('#CCCCCC')
            c.get_text().set_fontsize(7)

    ax.set_title(f'{title_text}\nBPPID与BPRBF控制器指标对比',
                 fontsize=9, fontweight='bold', pad=10)
    save_pub(fig, f'Table_metrics_{plant_id}')


# ===== Timeseries plots =====
def draw_timeseries(idx):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(t, R[:, idx], color=C_REF, linewidth=1.0, label='目标')
    ax.plot(t, Y_f[:, idx], color=C_FIX, linewidth=0.8, label='FixPID')
    ax.plot(t, Y_b[:, idx], color=C_BPFD, linewidth=0.8, linestyle='--', label='BPPID')
    ax.plot(t, Y_r[:, idx], color=C_RBF, linewidth=0.8, linestyle='--', label='BPRBF')

    ax.set_xlim(0, N * Ts)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('y')
    ax.legend(fontsize=6, loc='best')
    ax.set_title(scenarios[idx], fontsize=8, fontweight='bold')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    save_pub(fig, f'Timeseries_{idx+1:02d}')
    plt.close(fig)


# ===== Generate =====
print('\nGenerating per-plant metric tables...')
for plant_id, (title_text, row_range) in PLANTS.items():
    draw_plant_table(plant_id, title_text, row_range)

print(f'\nGenerating {nS} timeseries plots...')
for i in range(nS):
    draw_timeseries(i)

print(f'\nAll figures saved to {OUT_DIR}')
plt.close('all')
