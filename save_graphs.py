import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display needed
import matplotlib.pyplot as plt
import os

run_labels = [
    "T=30 S0=15 OLR=1",
    "T=35 S0=35 OLR=3",
    "T=42 S0=55 OLR=6",
    "T=30 S0=55 OLR=6",
]

online = {
    'ANN':  {'S': [0.07, 0.07, 0.07, 0.07], 'M': [0.06, 0.06, 0.06, 0.06], 'pH': [0.00, 0.00, 0.00, 0.00]},
    'RNN':  {'S': [0.47, 0.33, 0.17, 1.25], 'M': [1.81, 2.31, 2.14, 4.78], 'pH': [0.01, 0.04, 0.10, 0.07]},
    'LSTM': {'S': [0.63, 0.94, 0.67, 2.01], 'M': [2.57, 4.83, 5.67, 6.48], 'pH': [0.01, 0.08, 0.12, 0.13]},
}

offline = {
    'ANN':  {'S': [0.07, 0.45, 0.21, 0.73], 'M': [0.06, 0.36, 0.19, 0.56], 'pH': [0.00, 0.02, 0.02, 0.03]},
    'RNN':  {'S': [13.67, 9.81, 2.96, 69.62], 'M': [8.80, 13.38, 10.91, 35.66], 'pH': [0.09, 0.78, 2.23, 0.97]},
    'LSTM': {'S': [26.21, 52.65, 32.13, 193.36], 'M': [13.72, 38.29, 49.21, 63.27], 'pH': [0.18, 1.69, 2.48, 2.85]},
}

models  = ['ANN', 'RNN', 'LSTM']
colors  = {'ANN': '#1F4E79', 'RNN': '#C00000', 'LSTM': '#70AD47'}
var_keys  = ['S', 'M', 'pH']
var_names = ['Volatile Solids (S)', 'Methane Rate (M)', 'pH']


# graph 1: mean MAPE online vs offline grid
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
frameworks     = [online, offline]
frame_labels   = ['Online Framework', 'Offline Framework']

for fi, (framework, flabel) in enumerate(zip(frameworks, frame_labels)):
    for vi, (vkey, vname) in enumerate(zip(var_keys, var_names)):
        ax    = axes[fi, vi]
        means = [np.mean(framework[m][vkey]) for m in models]
        bars  = ax.bar(models, means,
                       color=[colors[m] for m in models],
                       alpha=0.85, edgecolor='white', width=0.5)
        ax.set_title(f"{flabel}\n{vname}", fontweight='bold', fontsize=11)
        ax.set_ylabel('Mean MAPE (%)')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(means) * 0.02,
                    f'{val:.2f}%', ha='center', fontsize=10, fontweight='bold')

plt.suptitle('Mean MAPE: Online vs Offline Frameworks — ANN vs RNN vs LSTM',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('comparison_online_offline_grid.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved: comparison_online_offline_grid.png")


# graph 2: per-run offline MAPE 
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
x     = np.arange(len(run_labels))
width = 0.25

for vi, (vkey, vname) in enumerate(zip(var_keys, var_names)):
    ax = axes[vi]
    for mi, m in enumerate(models):
        ax.bar(x + (mi - 1) * width, offline[m][vkey], width,
               label=m, color=colors[m], alpha=0.85, edgecolor='white')
    ax.set_title(f'{vname}', fontweight='bold', fontsize=12)
    ax.set_ylabel('Offline MAPE (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=15, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Offline MAPE per Test Run — ANN vs RNN vs LSTM',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('comparison_per_run_offline.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved: comparison_per_run_offline.png")


# graph 3: training loss vs offline MAPE scatter 
training_losses = {'ANN': 0.001549, 'RNN': 0.000138, 'LSTM': 0.000114}
offline_S_mean  = {m: np.mean(offline[m]['S']) for m in models}
markers         = {'ANN': 'o', 'RNN': 's', 'LSTM': '^'}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for m in models:
    axes[0].scatter(training_losses[m], offline_S_mean[m],
                    color=colors[m], s=250, marker=markers[m],
                    label=m, zorder=5)
    axes[0].annotate(f"  {m}",
                     (training_losses[m], offline_S_mean[m]),
                     fontsize=12, fontweight='bold', color=colors[m])

axes[0].set_xlabel('Training Loss (MSE)', fontsize=12)
axes[0].set_ylabel('Offline Mean MAPE for S (%)', fontsize=12)
axes[0].set_title('Training Loss vs Offline MAPE\nlower training loss does not mean better offline performance',
                  fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)

for m in models:
    axes[1].scatter(np.mean(online[m]['S']), offline_S_mean[m],
                    color=colors[m], s=250, marker=markers[m],
                    label=m, zorder=5)
    axes[1].annotate(f"  {m}",
                     (np.mean(online[m]['S']), offline_S_mean[m]),
                     fontsize=12, fontweight='bold', color=colors[m])

axes[1].set_xlabel('Online Mean MAPE for S (%)', fontsize=12)
axes[1].set_ylabel('Offline Mean MAPE for S (%)', fontsize=12)
axes[1].set_title('Online vs Offline MAPE\ndegradation factor shows error accumulation',
                  fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

plt.suptitle('Why Training Metrics Do Not Predict Offline Performance',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('comparison_training_vs_offline.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved: comparison_training_vs_offline.png")


# graph 4: ANN offline MAPE summary bar chart 
ann_offline_per_run = {
    'S':   [0.07, 0.45, 0.21, 0.73],
    'M':   [0.06, 0.36, 0.19, 0.56],
    'pH':  [0.00, 0.02, 0.02, 0.03],
}
ann_online_per_run = {
    'S':   [0.07, 0.07, 0.07, 0.07],
    'M':   [0.06, 0.06, 0.06, 0.06],
    'pH':  [0.00, 0.00, 0.00, 0.00],
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
x     = np.arange(len(run_labels))
width = 0.35
on_color  = '#1F4E79'
off_color = '#2E75B6'

for vi, (vkey, vname) in enumerate(zip(var_keys, var_names)):
    ax = axes[vi]
    b1 = ax.bar(x - width/2, ann_online_per_run[vkey], width,
                label='Online', color=on_color, alpha=0.85, edgecolor='white')
    b2 = ax.bar(x + width/2, ann_offline_per_run[vkey], width,
                label='Offline', color=off_color, alpha=0.85, edgecolor='white')
    ax.set_title(f'{vname}', fontweight='bold', fontsize=11)
    ax.set_ylabel('MAPE (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=15, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0.001:
            ax.text(bar.get_x() + bar.get_width()/2,
                    h + 0.01, f'{h:.2f}%', ha='center', fontsize=8)

plt.suptitle('ANN — Online vs Offline MAPE per Test Run',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('ann_mape_bar_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved: ann_mape_bar_chart.png")


print()
print("all graphs saved. add them to the olive mill folder and push to GitHub.")
