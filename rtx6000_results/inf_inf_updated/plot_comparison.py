import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

models = ['ResNet50', 'MobileNetV2', 'ResNet101', 'BERT']
colors = ["royalblue", "darkorange"]

def get_data(csv_file, error=False):
    df = pd.read_csv(csv_file)
    df = df.drop(df.columns[0], axis=1)
    df.index = models

    for model_row in models:
        for model_col in models:
            cell = df.at[model_row, model_col]
            df.at[model_row, model_col] = float(cell.split('/')[0])
    
    if error:
        return df.std()
    else:
        return df.mean()

# Latency comparison
print("Creating latency comparison plot...")
method2file_lat = {
    'MPS': 'results/mps_latency.csv',
    'Ideal': 'results/ideal_latency.csv'
}

method2data_lat = {}
method2err_lat = {}

for method, file in method2file_lat.items():
    method2data_lat[method] = get_data(file)
    method2err_lat[method] = get_data(file, error=True)

width = 0.35
fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(models))

for method_id, method in enumerate(['MPS', 'Ideal']):
    ax.bar(
        x + width * method_id, method2data_lat[method], width,
        label=method, yerr=method2err_lat[method],
        align='edge', color=colors[method_id], alpha=0.8
    )

x_tick_positions = x + width
ax.set_xticks(ticks=x_tick_positions, labels=models, fontsize=18)
plt.yticks(fontsize=16)
ax.set_ylabel('Average P95 Latency (ms)', fontsize=18)
ax.set_xlabel('Model', fontsize=18)
ax.set_title('MPS vs Ideal: P95 Latency Comparison (RTX 6000)', fontsize=20)
plt.legend(loc='upper left', fontsize=16)
plt.tight_layout()
plt.savefig("results/mps_vs_ideal_latency.png", bbox_inches="tight", dpi=300)
print("✓ Saved: results/mps_vs_ideal_latency.png")

# Throughput comparison
print("Creating throughput comparison plot...")
method2file_thr = {
    'MPS': ['results/mps_be_throughput.csv', 'results/mps_hp_throughput.csv'],
    'Ideal': ['results/ideal_be_throughput.csv', 'results/ideal_hp_throughput.csv']
}

def get_throughput_data(csv_files, error=False):
    df_be = pd.read_csv(csv_files[0])
    df_be = df_be.drop(df_be.columns[0], axis=1)
    df_be.index = models

    df_hp = pd.read_csv(csv_files[1])
    df_hp = df_hp.drop(df_hp.columns[0], axis=1)
    df_hp.index = models

    df_be_new = pd.DataFrame()
    df_hp_new = pd.DataFrame()

    for model_row in models:
        for model_col in models:
            cell_be = df_be.at[model_row, model_col]
            cell_hp = df_hp.at[model_row, model_col]
            df_be_new.at[model_row, model_col] = float(cell_be.split('/')[0])
            df_hp_new.at[model_row, model_col] = float(cell_hp.split('/')[0])
    
    if error:
        return df_be_new.std(), df_hp_new.std()
    else:
        return df_be_new.mean(), df_hp_new.mean()

method2data_thr = {}
method2err_thr = {}

for method, files in method2file_thr.items():
    method2data_thr[method] = get_throughput_data(files)
    method2err_thr[method] = get_throughput_data(files, error=True)

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(models))
width = 0.35

for method_id, method in enumerate(['MPS', 'Ideal']):
    # Stack BE and HP throughput
    ax.bar(
        x + width * method_id, method2data_thr[method][1], width,
        yerr=method2err_thr[method][1],
        align='edge', hatch="\\", color=colors[method_id], alpha=0.6,
    )
    ax.bar(
        x + width * method_id, method2data_thr[method][0], width,
        label=method, yerr=method2err_thr[method][0],
        bottom=method2data_thr[method][1],
        align='edge', hatch="/", color=colors[method_id], alpha=0.9
    )

x_tick_positions = x + width
ax.set_xticks(ticks=x_tick_positions, labels=models, fontsize=18)
plt.yticks(fontsize=16)
ax.set_ylabel('Total Throughput (req/s)', fontsize=18)
ax.set_xlabel('High-Priority Model', fontsize=18)
ax.set_title('MPS vs Ideal: Total Throughput Comparison (RTX 6000)', fontsize=20)
plt.legend(loc='upper left', fontsize=16)
plt.tight_layout()
plt.savefig("results/mps_vs_ideal_throughput.png", bbox_inches="tight", dpi=300)
print("✓ Saved: results/mps_vs_ideal_throughput.png")

# Slowdown visualization
print("Creating slowdown heatmap...")
df_mps_lat = pd.read_csv('results/mps_latency.csv', index_col=0)
df_ideal_lat = pd.read_csv('results/ideal_latency.csv', index_col=0)

slowdown = pd.DataFrame(index=models, columns=models)
for be in models:
    for hp in models:
        mps_val = float(str(df_mps_lat.at[be, hp]).split('/')[0])
        ideal_val = float(str(df_ideal_lat.at[be, hp]).split('/')[0])
        slowdown.at[be, hp] = mps_val / ideal_val if ideal_val > 0 else 0

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(slowdown.astype(float), cmap='YlOrRd', aspect='auto')

ax.set_xticks(np.arange(len(models)))
ax.set_yticks(np.arange(len(models)))
ax.set_xticklabels(models, fontsize=14)
ax.set_yticklabels(models, fontsize=14)
ax.set_xlabel('High-Priority Model', fontsize=16)
ax.set_ylabel('Best-Effort Model', fontsize=16)
ax.set_title('MPS Latency Slowdown vs Ideal (RTX 6000)', fontsize=18)

# Add text annotations
for i in range(len(models)):
    for j in range(len(models)):
        text = ax.text(j, i, f'{float(slowdown.iloc[i, j]):.2f}x',
                      ha="center", va="center", color="black", fontsize=12, weight='bold')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Slowdown Factor', fontsize=14)
plt.tight_layout()
plt.savefig("results/mps_slowdown_heatmap.png", bbox_inches="tight", dpi=300)
print("✓ Saved: results/mps_slowdown_heatmap.png")

print("\n" + "=" * 80)
print("All visualizations created successfully!")
print("=" * 80)
