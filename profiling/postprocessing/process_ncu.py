import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, required=True,
                        help='path to directory containing the profiling files')
args = parser.parse_args()

df = pd.read_csv(f'{args.results_dir}/output_ncu.csv')

# Check if this is wide format (from --page raw) or long format (default)
if 'Metric Name' in df.columns:
    # Long format - NCU default CSV format
    print("Detected long format CSV (NCU default)")
    
    # Create metric name mapping from NCU names to expected names
    metric_mapping = {
        'gpu__time_duration.sum': 'Duration',
        'launch__block_size': 'Block Size',
        'launch__grid_size': 'Grid Size',
        'sm__throughput.avg.pct_of_peak_sustained_elapsed': 'Compute (SM) Throughput',
        'dram__throughput.avg.pct_of_peak_sustained_elapsed': 'DRAM Throughput',
        'launch__registers_per_thread': 'Registers Per Thread',
        'launch__shared_mem_per_block_static': 'Static Shared Memory Per Block'
    }
    
    # Track all kernel invocations, not just unique names
    # Use (ID, Kernel Name) as unique key for each invocation
    kernels = {}
    kernel_order = []
    
    for index, row in df.iterrows():
        kernel_id = row['ID']
        kernel_name = row['Kernel Name']
        metric_name = row['Metric Name']
        metric_value = row['Metric Value']
        
        # Create unique key for this kernel invocation
        kernel_key = (kernel_id, kernel_name)
        
        # Map NCU metric name to expected name
        if metric_name in metric_mapping:
            expected_name = metric_mapping[metric_name]
            
            # Create kernel entry if not exists
            if kernel_key not in kernels:
                kernels[kernel_key] = {'Name': kernel_name}
                kernel_order.append(kernel_key)
            
            kernels[kernel_key][expected_name] = metric_value
    
    print(f"Found {len(kernel_order)} kernel invocations")
    
    kernels_list = []
    metrics_to_get = ['Duration', 'Block Size', 'Grid Size', 'Compute (SM) Throughput', 'DRAM Throughput', 'Registers Per Thread', 'Static Shared Memory Per Block']
    
    for kernel_key in kernel_order:
        kernel = kernels[kernel_key]
        
        # Check if all metrics are present
        if all(metric in kernel for metric in metrics_to_get):
            num_threads = int(float(kernel['Block Size'])) * int(float(kernel['Grid Size']))
            num_registers = num_threads * int(float(kernel['Registers Per Thread']))
            kernel_list = [kernel['Name']]
            for metric in metrics_to_get:
                kernel_list.append(kernel[metric])
            kernel_list += [num_threads, num_registers]
            kernels_list.append(kernel_list)
        else:
            print(f"WARNING: Kernel {kernel['Name']} (ID {kernel_key[0]}) missing some metrics, skipping")
    
    print(f"Successfully processed {len(kernels_list)} kernels")

elif 'gpu__time_duration.sum' in df.columns:
    # Wide format (from --page raw)
    print("Detected wide format CSV (from --page raw)")
    
    # Map raw metric names to expected names
    kernels_list = []
    for index, row in df.iterrows():
        kernel_name = row['Kernel Name']
        
        # Skip rows with NaN kernel names (empty header rows)
        if pd.isna(kernel_name):
            continue
            
        print(kernel_name)
        print("------------------------------------")
        
        # Extract metrics with proper column names
        duration = row['gpu__time_duration.sum']
        block_size = row['launch__block_size']
        grid_size = row['launch__grid_size']
        sm_throughput = row['sm__throughput.avg.pct_of_peak_sustained_elapsed']
        dram_throughput = row['dram__throughput.avg.pct_of_peak_sustained_elapsed']
        registers_per_thread = row['launch__registers_per_thread']
        static_shmem = row['launch__shared_mem_per_block_static']
        
        num_threads = int(block_size) * int(grid_size)
        num_registers = num_threads * int(registers_per_thread)
        
        kernel_list = [kernel_name, duration, block_size, grid_size, sm_throughput, 
                      dram_throughput, registers_per_thread, static_shmem, 
                      num_threads, num_registers]
        kernels_list.append(kernel_list)
        print(kernel_list)
else:
    raise ValueError("Unknown CSV format - missing expected columns")

print(len(kernels_list))
labels = ['Kernel_Name', 'Duration(ns)', 'Block', 'Grid',  'Compute(SM)(%)', 'DRAM_Throughput(%)', 'Registers_Per_Thread', 'Static_shmem_per_block', 'Number_of_threads', 'Number_of_registers']
df_new = pd.DataFrame(kernels_list, columns=labels)
print(df_new)
df_new.to_csv(f'{args.results_dir}/output_ncu_processed.csv')
