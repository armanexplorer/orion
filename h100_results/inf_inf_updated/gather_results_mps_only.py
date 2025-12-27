import pandas as pd
import numpy as np
import json
import itertools
import os

models = ['ResNet50', 'MobileNetV2', 'ResNet101', 'BERT']

hp_list = ['ResNet50', 'MobileNetV2', 'ResNet101', 'BERT']
be_list = ['ResNet50', 'MobileNetV2', 'ResNet101', 'BERT']
num_runs = 1

print("=" * 80)
print("Processing IDEAL results...")
print("=" * 80)

df_lat = pd.DataFrame("0", index=models, columns=models)
df_hp_ideal_throughput = pd.DataFrame("0", index=models, columns=models)
df_be_ideal_throughput = pd.DataFrame("0", index=models, columns=models)

for hp in hp_list:
    res_hp = []
    res_lat = []
    for run in range(num_runs):
        input_file_hp = f"results/ideal/{hp}_{run}_hp.json"
        with open(input_file_hp, 'r') as f:
            data = json.load(f)
            res_hp.append(float(data['throughput']))
            res_lat.append(float(data['p95_latency']))
    for be in be_list:
        df_hp_ideal_throughput.at[be, hp] = f"{round(np.average(res_hp),2)}/{round(np.std(res_hp),2)}"
        df_lat.at[be, hp] = f"{round(np.average(res_lat)/1000,2)}/{round(np.std(res_lat)/1000,2)}"  # Convert to ms

for be in be_list:
    res_be = []
    for run in range(num_runs):
        input_file_be = f"results/ideal/{be}_{run}_hp.json"
        with open(input_file_be, 'r') as f:
            data = json.load(f)
            res_be.append(float(data['throughput']))
    for hp in hp_list:
        df_be_ideal_throughput.at[be, hp] = f"{round(np.average(res_be),2)}/{round(np.std(res_be),2)}"

df_hp_ideal_throughput.to_csv(f'results/ideal_hp_throughput.csv')
df_be_ideal_throughput.to_csv(f'results/ideal_be_throughput.csv')
df_lat.to_csv(f'results/ideal_latency.csv')

print("\nIdeal HP Throughput:")
print(df_hp_ideal_throughput)
print("\nIdeal BE Throughput:")
print(df_be_ideal_throughput)
print("\nIdeal Latency (ms):")
print(df_lat)

print("\n" + "=" * 80)
print("Processing MPS results...")
print("=" * 80)

df_mps = pd.DataFrame("0", index=models, columns=models)
for be, hp in itertools.product(be_list, hp_list):
    results = []
    for run in range(num_runs):
        input_file = f"results/mps/{hp}_{be}_{run}.json"
        with open(input_file, 'r') as f:
            data = json.load(f)
            if data:
                results.append(float(data['p95-latency-0']) / 1000)  # Convert to ms
    df_mps.at[be, hp] = f"{round(np.average(results),2)}/{round(np.std(results),2)}"

df_mps.to_csv(f'results/mps_latency.csv')
print("\nMPS Latency (ms):")
print(df_mps)

df_hp_mps_throughput = pd.DataFrame("0", index=models, columns=models)
df_be_mps_throughput = pd.DataFrame("0", index=models, columns=models)

for be, hp in itertools.product(be_list, hp_list):
    res_hp = []
    res_be = []
    for run in range(num_runs):
        input_file_hp = f"results/mps/{hp}_{be}_{run}.json"
        with open(input_file_hp, 'r') as f:
            data = json.load(f)
            if data:
                res_be.append(float(data['throughput-1']))
                res_hp.append(float(data['throughput-0']))

    df_hp_mps_throughput.at[be, hp] = f"{round(np.average(res_hp),2)}/{round(np.std(res_hp),2)}"
    df_be_mps_throughput.at[be, hp] = f"{round(np.average(res_be),2)}/{round(np.std(res_be),2)}"

df_hp_mps_throughput.to_csv(f'results/mps_hp_throughput.csv')
df_be_mps_throughput.to_csv(f'results/mps_be_throughput.csv')

print("\nMPS HP Throughput:")
print(df_hp_mps_throughput)
print("\nMPS BE Throughput:")
print(df_be_mps_throughput)

print("\n" + "=" * 80)
print("✓ CSV files generated successfully!")
print("=" * 80)
print("Created files:")
print("  - results/ideal_latency.csv")
print("  - results/ideal_hp_throughput.csv")
print("  - results/ideal_be_throughput.csv")
print("  - results/mps_latency.csv")
print("  - results/mps_hp_throughput.csv")
print("  - results/mps_be_throughput.csv")
