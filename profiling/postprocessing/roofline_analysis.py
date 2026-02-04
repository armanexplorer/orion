import pandas as pd
import argparse
from pathlib import Path


def _read_ncu_csv(path: str) -> pd.DataFrame:
    """Read an NCU --csv export that may contain non-CSV preamble lines."""
    p = Path(path)
    header_idx = 0
    with p.open("r", errors="replace") as f:
        for idx, line in enumerate(f):
            if line.startswith('"ID"') or line.startswith('ID,'):
                header_idx = idx
                break
    return pd.read_csv(p, skiprows=header_idx)

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, required=True,
                        help='path to directory containing the profiling files')
parser.add_argument('--ai_threshold', type=float, default=9.2,
                        help='arithmetic intensity that seperates compute from memory bound kernels')
parser.add_argument('--verbose', action='store_true', help='print per-kernel debug information')
args = parser.parse_args()

df_raw = _read_ncu_csv(f'{args.results_dir}/raw_ncu.csv')

# Check if this is long format (NCU default CSV) and pivot it
if 'Metric Name' in df_raw.columns:
    if args.verbose:
        print("Converting long format to wide format for roofline analysis...")
    # Combine Metric Name + Unit to match expected format (e.g., "metric_name [unit]")
    df_raw['Metric_Full'] = df_raw['Metric Name'] + ' ' + df_raw['Metric Unit']
    # Create unique kernel identifier using ID column (each kernel invocation has unique ID)
    df_pivoted = df_raw.pivot_table(
        index='Metric_Full',
        columns='ID', 
        values='Metric Value',
        aggfunc='first'
    )
    # Reset index to make Metric_Full a column (column 0)
    df_raw = df_pivoted.reset_index()
    if args.verbose:
        print(f"Pivoted to {len(df_raw)} metrics x {len(df_raw.columns)-1} kernels")

startp = 0
df_raw = df_raw.iloc[startp:]

if args.verbose:
    # Debug: Print all available metric names
    print("\n=== Available metrics in pivoted dataframe ===")
    print(df_raw.iloc[:, 0].tolist())
    print("=" * 50 + "\n")

df_basic = pd.read_csv(f'{args.results_dir}/output_ncu_sms.csv', index_col=0)
dram_throughput = df_basic['DRAM_Throughput(%)']
comp_throughput = df_basic['Compute(SM)(%)']

# Search for metrics by partial name match (NCU units may vary)
def find_metric_row(df, metric_substring):
    """Find metric row by searching for substring in metric name"""
    for idx, metric_name in enumerate(df.iloc[:, 0]):
        if metric_substring in str(metric_name):
            if args.verbose:
                print(f"Found metric: {metric_name}")
            return list(df.iloc[idx, 1:])
    print(f"WARNING: Metric containing '{metric_substring}' not found!")
    return None

if args.verbose:
    print("\nSearching for required metrics...")
df_add = find_metric_row(df_raw, 'fadd_pred_on')
df_mul = find_metric_row(df_raw, 'fmul_pred_on')
df_fma = find_metric_row(df_raw, 'ffma_pred_on')
df_cycles = find_metric_row(df_raw, 'cycles_elapsed')
df_bytes = find_metric_row(df_raw, 'dram__bytes')

# Verify all metrics were found
if None in [df_add, df_mul, df_fma, df_cycles, df_bytes]:
    print("\nERROR: Some required metrics are missing!")
    print("Please check the NCU profiling output and ensure all 12 metrics were collected.")
    exit(1)

ai_list = []
roofline_prof = [] # 1: comp, 0: mem, -1: invalid

comp_bound = 0
mem_bound = 0
rest = 0
num_kernels = len(df_add)

for i in range(num_kernels):
    add = df_add[i]
    mul = df_mul[i]
    fma = df_fma[i]

    # Convert to float, handling both numeric and string types
    def to_float(val):
        if isinstance(val, (int, float)):
            return float(val)
        elif isinstance(val, str):
            return float(val.replace("'", ''))
        else:
            return float(val)
    
    add = to_float(add)
    mul = to_float(mul)
    fma = to_float(fma)
    cycles = to_float(df_cycles[i])
    bytes_val = to_float(df_bytes[i])
    
    # Avoid division by zero
    if bytes_val == 0.0 or bytes_val < 0.00001:
        bytes_val = 0.00001
    
    # Convert bytes/s to Tbyte/s (divide by 10^12) then back for calculation
    bytes_val = bytes_val / 1e12  # Now in Tbyte/s
    cycles_ghz = cycles / 1e9  # Convert Hz to GHz

    if args.verbose:
        print(i, add, mul, fma, cycles_ghz, bytes_val)

    if add or mul or fma:
        flops_cycle = add + mul + fma * 2
        flops_sec = flops_cycle * cycles_ghz  # GFLOPs/s
        ai = flops_sec / bytes_val  # arithmetic intensity
        ai_list.append(ai)
        if ai > args.ai_threshold:
            roofline_prof.append(1)
            comp_bound += 1
        else:
            roofline_prof.append(0)
            mem_bound += 1
    else:
        ai_list.append(0.0)
        if comp_throughput[i] >= 60.0:
            roofline_prof.append(1)
        elif dram_throughput[i] >= 60.0:
            roofline_prof.append(0)
        else:
            roofline_prof.append(-1)
        rest += 1

# older NCU version
# for index, row in df_raw.iterrows():
#     add = str(row[fadd])
#     mul = str(row[fmul])
#     fma = row[ffma]
#     cycles = row[cycles_sec]
#     bytes = row[bytes_sec]
#     #print(add, mul, fma, cycles, bytes)

#     if not isinstance(fma, float):
#         fma = float(fma.replace("'", ''))
#     add = float(add.replace("'", ''))
#     mul = float(mul.replace("'", ''))


#     if add or mul or fma:
#         flops_cycle = add+mul+fma*2
#         flops_sec = flops_cycle * cycles
#         ai = flops_sec/bytes
#         ai_list.append(ai)
#         print(index, ai)
#         if ai > args.ai_threshold:
#             roofline_prof.append(1)
#             comp_bound += 1
#         else:
#             roofline_prof.append(0)
#             mem_bound += 1
#     else:
#         ai_list.append(0.0)
#         if comp_throughput[index-startp] >= 60.0:
#             roofline_prof.append(1)
#         elif dram_throughput[index-startp] >= 60.0:
#             roofline_prof.append(0)
#         else:
#             roofline_prof.append(-1)
#         rest += 1


if args.verbose:
    print(df_basic)
df_basic['AI(flops/bytes)'] = ai_list
df_basic['Roofline_prof'] = roofline_prof
df_basic.to_csv(f'{args.results_dir}/output_ncu_sms_roofline.csv')

print(f"comp bound: {comp_bound}, mem bound: {mem_bound}, rest: {rest}, total: {comp_bound+mem_bound+rest}")
