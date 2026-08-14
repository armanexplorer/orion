# YOLOv5 Integration Plan for Orion

**Target**: Add YOLOv5s support to run with `run_ideal.py` and `run_mps.py`  
**Model Variant**: YOLOv5s (small - good balance of speed and accuracy)  
**Estimated Time**: 2-4 hours (mostly profiling)

---

## PHASE 1: Setup YOLOv5 Model Files

### Step 1.1: Create Directory Structure
```bash
cd ~/orion
mkdir -p models/yolov5
```

### Step 1.2: Download YOLOv5 Repository (v7.0)
```bash
wget -O /tmp/yolov5.zip https://github.com/ultralytics/yolov5/zipball/v7.0
unzip -q /tmp/yolov5.zip -d /tmp
mv /tmp/ultralytics-yolov5-* ~/orion/models/yolov5/repo
rm /tmp/yolov5.zip
```

**Verify**:
```bash
ls ~/orion/models/yolov5/repo/
# Should see: hubconf.py, models/, utils/, etc.
```

**Step 1.2.1: Patch YOLOv5 for PyTorch 2.6+ Compatibility**

PyTorch 2.6+ changed the default `weights_only` parameter in `torch.load` from `False` to `True`. We need to patch the YOLOv5 code:

```bash
# Patch models/experimental.py to add weights_only=False
sed -i 's/torch.load(attempt_download(w), map_location=/torch.load(attempt_download(w), map_location=/' ~/orion/models/yolov5/repo/models/experimental.py
sed -i "s/ckpt = torch.load(attempt_download(w), map_location='cpu')/ckpt = torch.load(attempt_download(w), map_location='cpu', weights_only=False)/" ~/orion/models/yolov5/repo/models/experimental.py
```

**Verify the patch**:
```bash
grep "weights_only=False" ~/orion/models/yolov5/repo/models/experimental.py
# Should show the patched line with weights_only=False
```

### Step 1.3: Download YOLOv5s Model Weights
```bash
cd ~/orion/models/yolov5
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```

**Verify**:
```bash
ls -lh ~/orion/models/yolov5/yolov5s.pt
# Should be ~14MB
```

### Step 1.4: Install Required Dependencies
```bash
# Downgrade numpy to fix compatibility with opencv-python
pip install "numpy<2"
pip install opencv-python==4.8.0.74 PyYAML>=5.3.1 scipy>=1.4.1 psutil matplotlib>=3.2.2 seaborn>=0.11.0 gitpython>=3.1.30
```

---

## PHASE 2: Create YOLOv5 Profiling Benchmark

### Step 2.1: Create `profiling/benchmarks/yolov5.py`

**File**: `~/orion/profiling/benchmarks/yolov5.py`

```python
import os
import torch
import time
import sys

def yolov5(model_variant='yolov5s', batchsize=4, local_rank=0, do_eval=True, profile=None):
    """
    YOLOv5 inference benchmark for profiling with Orion
    
    Args:
        model_variant: 'yolov5s' (small variant)
        batchsize: batch size for inference
        local_rank: GPU device id (default: 0)
        do_eval: True for inference mode
        profile: 'ncu' or 'nsys' for profiling
    """
    
    # Set model paths
    orion_root = os.path.expanduser('~') + '/orion'
    repo_path = f'{orion_root}/models/yolov5/repo'
    weights_path = f'{orion_root}/models/yolov5/{model_variant}.pt'
    
    print(f"Loading YOLOv5 model: {model_variant}")
    print(f"Repo path: {repo_path}")
    print(f"Weights path: {weights_path}")
    
    # Load model using torch.hub
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model='custom',
        source='local',
        path=weights_path,
        device=f'cuda:{local_rank}'
    )
    
    # Create dummy input images (640x640 is YOLOv5 default input size)
    # YOLOv5 expects a batch tensor, not a list of tensors
    images = torch.rand(batchsize, 3, 640, 640).to(torch.float32).cuda(local_rank)
    
    if do_eval:
        model.eval()
    else:
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    batch_idx = 0
    torch.cuda.synchronize()
    start_all = time.time()
    
    # Run 10 iterations (warm up + profiled iteration)
    while batch_idx < 10:
        print(f"Iteration {batch_idx}")
        start = time.time()
        
        # Start profiling on iteration 9 (after warmup)
        if batch_idx == 9:
            if profile == 'ncu':
                torch.cuda.nvtx.range_push("start")
            elif profile == 'nsys':
                torch.cuda.profiler.cudart().cudaProfilerStart()
        
        # Run inference
        if do_eval:
            with torch.no_grad():
                output = model(images)
        else:
            optimizer.zero_grad()
            output = model(images)
            if isinstance(output, dict) and 'loss' in output:
                loss = output['loss']
                loss.backward()
                optimizer.step()
        
        torch.cuda.synchronize()
        
        # Stop profiling after iteration 9
        if batch_idx == 9:
            if profile == 'ncu':
                torch.cuda.nvtx.range_pop()
            elif profile == 'nsys':
                torch.cuda.profiler.cudart().cudaProfilerStop()
        
        print(f"Iteration took {time.time()-start} sec")
        batch_idx += 1
    
    print(f"Done! Total time: {time.time()-start_all} sec")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_variant', type=str, default='yolov5s')
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--profile', type=str, default='nsys', choices=['ncu', 'nsys', None])
    args = parser.parse_args()
    
    yolov5(args.model_variant, args.batchsize, 0, True, args.profile)
```

**Create the file**:
```bash
cat > ~/orion/profiling/benchmarks/yolov5.py << 'EOF'
[PASTE THE ENTIRE PYTHON CODE ABOVE]
EOF
```

### Step 2.2: Test the Profiling Script
```bash
cd ~/orion
python3 profiling/benchmarks/yolov5.py --model_variant yolov5s --batchsize 4
```

**Expected output**: Should see 10 iterations complete without errors.

---

## PHASE 3: Profile YOLOv5 to Generate Kernel Files

### Step 3.1: Create Profiles Directory
```bash
mkdir -p ~/orion/profiling/profiles
cd ~/orion/profiling
```

### Step 3.2: Profile with NSYS
```bash
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none \
  -o profiles/yolov5s_4_fwd_nsys \
  --cudabacktrace=true \
  --capture-range=cudaProfilerApi \
  -f true -x true \
  python3 benchmarks/yolov5.py --model_variant yolov5s --batchsize 4 --profile nsys
```

**Expected output**: `profiles/yolov5s_4_fwd_nsys.nsys-rep` file created.

### Step 3.3: Convert NSYS to CSV
```bash
nsys stats --report cuda_gpu_trace --format csv,column \
  --output .,- profiles/yolov5s_4_fwd_nsys.nsys-rep
```

**Expected output**: `profiles/yolov5s_4_fwd_nsys_cuda_gpu_trace.csv` file created.

### Step 3.4: Profile with NCU (CSV mode with Roofline Metrics)

**First, clean up any old CSV file:**
```bash
rm -f profiles/raw_ncu.csv
```

**Then run the NCU command with all required metrics:**
```bash
cd ~/orion/profiling
ncu --csv --profile-from-start off \
  --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,launch__shared_mem_per_block_static,launch__block_size,launch__grid_size,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed,smsp__cycles_elapsed.avg.per_second,dram__bytes.sum.per_second \
  python3 benchmarks/yolov5.py --model_variant yolov5s --batchsize 4 --profile nsys \
  > profiles/raw_ncu.csv 2>&1
```

**IMPORTANT**: 
- **Do NOT use `--page raw`** - the default CSV format creates long format (metrics as rows) which is required
- Collects 12 metrics total: 7 for basic profiling + 5 for roofline analysis
- `--profile-from-start off` with `cudaProfilerStart()`/`Stop()` profiles only iteration 9
- Shell redirection (`>`) creates CSV-only output (no `.ncu-rep` binary file)
- This will take **6-10 minutes** to complete (more metrics = longer profiling time)
- The `--profile nsys` argument activates the profiling hooks in the script

**Metrics collected**:

*Basic profiling (for process_ncu.py):*
- `gpu__time_duration.sum` → Duration
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` → Compute (SM) Throughput
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` → DRAM Throughput  
- `launch__registers_per_thread` → Registers Per Thread
- `launch__shared_mem_per_block_static` → Static Shared Memory Per Block
- `launch__block_size` → Block Size
- `launch__grid_size` → Grid Size

*Roofline analysis (for roofline_analysis.py):*
- `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed` → Floating-point add operations
- `smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed` → Floating-point multiply operations
- `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed` → Fused multiply-add operations
- `smsp__cycles_elapsed.avg.per_second` → SM cycles per second (GHz)
- `dram__bytes.sum.per_second` → DRAM bytes per second (TByte/s)

**Expected output**: 
- `profiles/raw_ncu.csv` - kernel profiling data in long format CSV (required by both process_ncu.py and roofline_analysis.py)

### Step 3.5: ~~Profile with NCU (Full mode)~~ [SKIPPED - Not needed]

**This step is not necessary** - the CSV from Step 3.4 contains all the data needed for postprocessing. The binary `.ncu-rep` file is only useful for viewing in the GUI, which is not required for Orion integration.

### Step 3.6: ~~Export NCU Raw CSV~~ [SKIPPED - Not needed]

**This step is not necessary** - Step 3.4 already produces the CSV file directly. No GUI export needed.

### Step 3.7: Process Profiling Files

**3.7.1: Extract Clean NCU Data**

The NCU CSV file contains metadata headers before the actual data. Find where the CSV data starts:
```bash
cd ~/orion
grep -n "\"ID\",\"Process ID\"" profiling/profiles/raw_ncu.csv | head -1
```

This will show the line number (e.g., `51:"ID","Process ID",...`). Extract from that line:
```bash
# Replace 51 with the line number from previous command
tail -n +51 profiling/profiles/raw_ncu.csv > profiling/profiles/raw_ncu_clean.csv
```

**3.7.2: Process NCU Data with process_ncu.py**

The script needs to be updated to read from `raw_ncu_clean.csv` instead of `output_ncu.csv`:

```bash
cd ~/orion/profiling
# First, create a symlink or copy the file to expected location
cp profiles/raw_ncu_clean.csv profiles/output_ncu.csv

# Then run the processing script
python3 postprocessing/process_ncu.py --results_dir profiles
```

**Expected output**: `profiles/output_ncu_processed.csv` with 284 kernels × 10 columns

**3.7.3: Get Number of Blocks**

**For Quadro RTX 6000 / V100 GPU**:
```bash
cd ~/orion/profiling
python3 postprocessing/get_num_blocks.py \
  --results_dir profiles/ \
  --max_threads_sm 2048 \
  --max_shmem_sm 98304 \
  --max_regs_sm 65536
```

**For H100 GPU**:
```bash
python3 postprocessing/get_num_blocks.py \
  --results_dir profiles/ \
  --max_threads_sm 2048 \
  --max_shmem_sm 228096 \
  --max_regs_sm 65536
```

**Expected output**: `profiles/output_ncu_sms.csv` with SM_needed column added

**3.7.4: Roofline Analysis**

**For Quadro RTX 6000 / V100** (ai_threshold ~13.4):
```bash
cd ~/orion/profiling
python3 postprocessing/roofline_analysis.py \
  --results_dir profiles/ \
  --ai_threshold 13.4
```

**For H100** (ai_threshold ~20):
```bash
python3 postprocessing/roofline_analysis.py \
  --results_dir profiles/ \
  --ai_threshold 20
```

**Expected output**: `profiles/output_ncu_sms_roofline.csv` file created.

### Step 3.8: Generate Kernel Info File
```bash
cd ~/orion/profiling
python3 postprocessing/generate_file_updated.py \
  --input_file_name profiles/output_ncu_sms_roofline.csv \
  --output_file_name ../benchmarking/model_kernels/v100/yolov5s_4_fwd \
  --model_type vision
```

**IMPORTANT**: Note the number of kernels printed (e.g., "Number of kernels: 145"). You'll need this for config files.

**Expected output**: `benchmarking/model_kernels/v100/yolov5s_4_fwd` file created (or h100/ if using H100 GPU).

---

## PHASE 4: Create YOLOv5 Benchmark Inference Script

### Step 4.1: Create `benchmarking/benchmark_suite/yolov5_inference.py`

**File**: `~/orion/benchmarking/benchmark_suite/yolov5_inference.py`

```python
import os
import sys
import torch
import time
import argparse
import threading
import json
from ctypes import *
import numpy as np

def seed_everything(seed: int):
    import random, os
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def block(backend_lib, it):
    """Block client until request served"""
    backend_lib.block(it)

def check_stop(backend_lib):
    """Check if scheduler signals stop"""
    return backend_lib.stop()

def set_stream(backend_lib, idx):
    """Set CUDA stream for this client"""
    backend_lib.get_stream_ptr.restype = c_void_p
    stream_id = backend_lib.get_stream_ptr(idx)
    print("Setting stream to ", stream_id)
    new_stream = torch.cuda.get_stream_from_external(stream_id)
    torch.cuda.set_stream(new_stream)

def yolov5_loop(
    model_name,
    batchsize,
    train,
    num_iters,
    rps,
    uniform,
    dummy_data,
    local_rank,
    barriers,
    client_barrier,
    tid,
    input_file=''
):
    """
    YOLOv5 inference loop for Orion scheduler
    
    Args:
        model_name: YOLOv5 variant (e.g., 'yolov5s')
        batchsize: batch size for inference
        train: False for inference (YOLOv5 training not supported yet)
        num_iters: number of inference iterations
        rps: requests per second
        uniform: uniform vs exponential distribution
        dummy_data: always True (use random tensors)
        local_rank: GPU device id
        barriers: synchronization barriers
        client_barrier: client synchronization barrier
        tid: thread/client id
        input_file: path to inter-arrival times JSON (optional)
    """
    
    seed_everything(42)
    print(f"YOLOv5 Inference - Model: {model_name}, Batch: {batchsize}, Rank: {local_rank}, TID: {tid}")
    
    # Load Orion backend library
    backend_lib = cdll.LoadLibrary(os.path.expanduser('~') + "/orion/src/cuda_capture/libinttemp.so")
    
    # Setup request arrival times
    if rps > 0 and input_file == '':
        if uniform:
            sleep_times = [1/rps] * num_iters
        else:
            sleep_times = np.random.exponential(scale=1/rps, size=num_iters)
    elif input_file != '':
        with open(input_file) as f:
            sleep_times = json.load(f)
    else:
        sleep_times = [0] * num_iters
    
    print(f"Number of requests: {len(sleep_times)}")
    barriers[0].wait()
    
    print(f"Thread native ID: {threading.get_native_id()}")
    
    # Training not supported for YOLOv5 in this script
    if train and tid == 1:
        time.sleep(5)
    
    set_stream(backend_lib, tid)
    
    # Load YOLOv5 model
    orion_root = os.path.expanduser('~') + '/orion'
    repo_path = f'{orion_root}/models/yolov5/repo'
    weights_path = f'{orion_root}/models/yolov5/{model_name}.pt'
    
    print(f"Loading YOLOv5 from: {weights_path}")
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model='custom',
        source='local',
        path=weights_path,
        device=f'cuda:{local_rank}'
    )
    model.eval()
    
    # Prepare dummy data (640x640 images for YOLOv5)
    # YOLOv5 expects a batch tensor, not a list of tensors
    images = torch.rand(batchsize, 3, 640, 640).to(torch.float32).cuda(local_rank)
    
    barriers[1].wait()
    client_barrier.wait()
    
    # Warmup phase
    print("YOLOv5 warmup started")
    for _ in range(10):
        with torch.no_grad():
            _ = model(images)
    
    barriers[2].wait()
    print("YOLOv5 benchmarking started")
    
    start_time = time.time()
    
    # Main inference loop
    for it in range(len(sleep_times)):
        if check_stop(backend_lib):
            print(f"Stopped at iteration {it}")
            break
        
        time.sleep(sleep_times[it])
        
        with torch.no_grad():
            output = model(images)
        
        block(backend_lib, it)
        
        if it % 100 == 0:
            print(f"YOLOv5 iteration {it}/{len(sleep_times)}")
    
    end_time = time.time()
    print(f"YOLOv5 inference completed in {end_time - start_time:.2f}s")
    barriers[3].wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='yolov5s')
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--num_iters', type=int, default=100)
    parser.add_argument('--rps', type=float, default=0)
    parser.add_argument('--uniform', action='store_true')
    parser.add_argument('--dummy_data', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    
    args = parser.parse_args()
    
    # For standalone testing
    from threading import Barrier
    barriers = [Barrier(1) for _ in range(4)]
    client_barrier = Barrier(1)
    
    yolov5_loop(
        args.model_name,
        args.batchsize,
        False,  # train=False
        args.num_iters,
        args.rps,
        args.uniform,
        args.dummy_data,
        args.local_rank,
        barriers,
        client_barrier,
        0
    )
```

**Create the file**:
```bash
cat > ~/orion/benchmarking/benchmark_suite/yolov5_inference.py << 'EOF'
[PASTE THE ENTIRE PYTHON CODE ABOVE]
EOF
```

### Step 4.2: Test the Inference Script Standalone
```bash
cd ~/orion
python3 benchmarking/benchmark_suite/yolov5_inference.py \
  --model_name yolov5s \
  --batchsize 4 \
  --num_iters 10
```

**Expected output**: Should complete 10 iterations without errors.

---

## PHASE 5: Update Orion Launch Scripts

### Step 5.1: Update `benchmarking/launch_jobs.py`

**Add import at the top** (around line 16, after other imports):
```python
from benchmark_suite.yolov5_inference import yolov5_loop
```

**Add to function_dict** (around line 18, after existing entries):
```python
function_dict = {
    "resnet50": imagenet_loop,
    "resnet101": imagenet_loop,
    "mobilenet_v2": imagenet_loop,
    "bert": bert_loop,
    "transformer": transformer_loop,
    "yolov5s": yolov5_loop,  # ADD THIS LINE
}
```

**Manual edit required**:
```bash
nano ~/orion/benchmarking/launch_jobs.py
# Or use your preferred editor: vim, emacs, etc.
```

1. After line `from benchmark_suite.train_imagenet import imagenet_loop` (around line 16), add:
   ```python
   from benchmark_suite.yolov5_inference import yolov5_loop
   ```

2. In the `function_dict` dictionary (around line 18-24), add `"yolov5s": yolov5_loop,` after the last entry.

---

## PHASE 6: Create Configuration Files

### Step 6.1: Create Ideal (Orion) Config for YOLOv5s

**File**: `~/orion/h100_results/inf_inf_updated/config_files/ideal/yolov5s_inf.json`

**Replace `<NUMBER_OF_KERNELS>` with the value from Step 3.8**

**IMPORTANT**: Update the `kernel_file` path based on your GPU:
- **Quadro RTX 6000 / V100**: Use `/home/cc/orion/benchmarking/model_kernels/v100/yolov5s_4_fwd`
- **H100**: Use `/home/cc/orion/benchmarking/model_kernels/h100/yolov5s_4_fwd`

```json
[
    {
        "arch": "yolov5s",
        "kernel_file": "/home/cc/orion/benchmarking/model_kernels/v100/yolov5s_4_fwd",
        "num_kernels": <NUMBER_OF_KERNELS>,
        "num_iters": 9200,
        "args": {
            "model_name": "yolov5s",
            "batchsize": 4,
            "rps": 200,
            "uniform": false,
            "dummy_data": true,
            "train": false
        }
    }
]
```

**Create the file**:
```bash
cat > ~/orion/h100_results/inf_inf_updated/config_files/ideal/yolov5s_inf.json << 'EOF'
[
    {
        "arch": "yolov5s",
        "kernel_file": "/home/cc/orion/benchmarking/model_kernels/v100/yolov5s_4_fwd",
        "num_kernels": 145,
        "num_iters": 9200,
        "args": {
            "model_name": "yolov5s",
            "batchsize": 4,
            "rps": 200,
            "uniform": false,
            "dummy_data": true,
            "train": false
        }
    }
]
EOF
```

**IMPORTANT**: 
- Replace `145` with your actual kernel count from Step 3.8
- The path uses **v100** since you're using Quadro RTX 6000 (change to h100 only if you're on H100 GPU)
- The directory name `h100_results` is just the experiment folder name and doesn't affect which kernel file you use

### Step 6.2: Update MPS Config File

**File**: `~/orion/h100_results/inf_inf_updated/config_files/mps/config.yaml`

**Add YOLOv5s configuration section** at the end of the file:

```yaml
yolov5s:
  arch: yolov5s
  batch_size: 4
  num_iterations: 9200
  request_rate: 200  # measured in 1/seconds
```

**Manual edit**:
```bash
nano ~/orion/h100_results/inf_inf_updated/config_files/mps/config.yaml
# Add the yolov5s section at the end
```

---

## PHASE 7: Update Baseline MPS Scripts

### Step 7.1: Update `related/baselines/main.py`

**Add YOLOv5 to model_to_wrapper dictionary** (around line 14-31).

Since YOLOv5 uses the same wrapper as vision models (ResNet, MobileNet), add:

```python
model_to_wrapper = {
    'resnet50': {
        'train': vision_train_wrapper,
        'eval': vision_eval_wrapper,
    },
    'resnet101': {
        'train': vision_train_wrapper,
        'eval': vision_eval_wrapper,
    },
    'mobilenet_v2': {
        'train': vision_train_wrapper,
        'eval': vision_eval_wrapper,
    },
    'bert': {
        'train': bert_train_wrapper,
        'eval': bert_eval_wrapper,
    },
    'transformer': {
        'train': transformer_train_wrapper,
        'eval': transformer_eval_wrapper,
    },
    'yolov5s': {  # ADD THIS ENTRY
        'train': vision_train_wrapper,
        'eval': vision_eval_wrapper,
    }
}
```

**Manual edit**:
```bash
nano ~/orion/related/baselines/main.py
# Add the yolov5s entry to model_to_wrapper dictionary
```

### Step 7.2: Update `related/baselines/vision/train_imagenet.py`

**Add YOLOv5 model loading support** in the `setup` function.

Find the `setup` function and add YOLOv5 handling. The function likely creates models using `models.__dict__[arch]()`.

**Add this logic** (around where models are instantiated):

```python
def setup(model_config, shared_config, device):
    arch = model_config['arch']
    batch_size = model_config['batch_size']
    
    # YOLOv5 handling
    if arch.startswith('yolov5'):
        orion_root = os.path.expanduser('~') + '/orion'
        repo_path = f'{orion_root}/models/yolov5/repo'
        weights_path = f'{orion_root}/models/yolov5/{arch}.pt'
        
        model = torch.hub.load(
            repo_or_dir=repo_path,
            model='custom',
            source='local',
            path=weights_path,
            device='cuda:0'
        ).to(device)
    else:
        # Existing torchvision models
        model = models.__dict__[arch](num_classes=1000).to(device)
    
    # Rest of setup...
```

**Find and edit the setup function**:
```bash
nano ~/orion/related/baselines/vision/train_imagenet.py
# Locate the setup() function and add YOLOv5 handling as shown above
```

---

## PHASE 8: Update Run Scripts

### Step 8.1: Update `run_ideal.py`

**File**: `~/orion/h100_results/inf_inf_updated/run_ideal.py`

**Add YOLOv5s to trace_files_hp list** (around line 5-10):

```python
trace_files_hp = [
    ("ResNet50", "rnet"),
    ("MobileNetV2", "mnet"),
    ("ResNet101", "rnet101"),
    ("BERT", "bert"),
    ("YOLOv5s", "yolov5s"),  # ADD THIS LINE
]
```

**Manual edit**:
```bash
nano ~/orion/h100_results/inf_inf_updated/run_ideal.py
# Add ("YOLOv5s", "yolov5s") to the trace_files_hp list
```

### Step 8.2: Update `run_mps.py`

**File**: `~/orion/h100_results/inf_inf_updated/run_mps.py`

**Two edits needed**:

1. **Add to mnames dictionary** (around line 5-10):
```python
mnames = {
    'resnet50': "ResNet50",
    'mobilenet_v2': "MobileNetV2",
    'resnet101': 'ResNet101',
    'bert': 'BERT',
    'yolov5s': 'YOLOv5s',  # ADD THIS LINE
}
```

2. **Add to models list** (around line 36):
```python
models = ['resnet50', 'mobilenet_v2', 'resnet101', 'bert', 'yolov5s']  # ADD 'yolov5s'
```

**Manual edit**:
```bash
nano ~/orion/h100_results/inf_inf_updated/run_mps.py
# 1. Add 'yolov5s': 'YOLOv5s' to mnames dictionary
# 2. Add 'yolov5s' to models list
```

---

## PHASE 9: Testing and Validation

### Step 9.1: Test Single YOLOv5 with Orion Scheduler
```bash
cd ~/orion/h100_results/inf_inf_updated

# Set LD_PRELOAD based on your Python site-packages location
# Adjust paths if using conda or different Python version
HOME_DIR=$(echo ~)
export LD_PRELOAD="$HOME_DIR/orion/src/cuda_capture/libinttemp.so:$HOME_DIR/.local/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.9:$HOME_DIR/.local/lib/python3.10/site-packages/nvidia/cublas/lib/libcublasLt.so.12:$HOME_DIR/.local/lib/python3.10/site-packages/nvidia/cublas/lib/libcublas.so.12"

python3 ../../benchmarking/launch_jobs.py \
  --algo orion \
  --config_file config_files/ideal/yolov5s_inf.json
```

**Expected output**: 
- YOLOv5 model loads successfully
- Completes warmup and benchmarking phases
- Creates `client_0.json` with results

### Step 9.2: Create Results Directories
```bash
cd ~/orion/h100_results/inf_inf_updated
mkdir -p results/ideal
mkdir -p results/mps
```

### Step 9.3: Run Full Ideal Test
```bash
cd ~/orion/h100_results/inf_inf_updated
python3 run_ideal.py
```

**Expected output**:
- Runs YOLOv5s (and other models) individually
- Creates `results/ideal/YOLOv5s_0_hp.json`
- Shows latency and throughput metrics

### Step 9.4: Run Full MPS Test
```bash
cd ~/orion/h100_results/inf_inf_updated
python3 run_mps.py
```

**Expected output**:
- Runs all model combinations including YOLOv5s
- Creates files like `results/mps/YOLOv5s_ResNet50_0.json`
- Tests all pairwise combinations

---

## PHASE 10: Verification Checklist

**Before declaring success, verify**:

- [ ] YOLOv5 repo and weights downloaded to `~/orion/models/yolov5/`
- [ ] Profiling script runs: `python3 profiling/benchmarks/yolov5.py`
- [ ] Kernel file exists: `benchmarking/model_kernels/h100/yolov5s_4_fwd`
- [ ] Inference script runs: `python3 benchmarking/benchmark_suite/yolov5_inference.py`
- [ ] `launch_jobs.py` updated with yolov5_loop import and function_dict entry
- [ ] Config files created: `config_files/ideal/yolov5s_inf.json` and MPS config updated
- [ ] `main.py` has yolov5s in model_to_wrapper
- [ ] `train_imagenet.py` setup() handles YOLOv5 loading
- [ ] `run_ideal.py` has YOLOv5s entry
- [ ] `run_mps.py` has yolov5s in mnames and models list
- [ ] Single client test passes (Step 9.1)
- [ ] `run_ideal.py` completes successfully
- [ ] `run_mps.py` completes successfully

---

## Troubleshooting

### Issue: "transpose() received an invalid combination of arguments - got (tuple)"
**Solution**: YOLOv5 expects batched tensor input, not a list of tensors. Update the input creation:
```python
# WRONG: images = [torch.rand(3, 640, 640).cuda() for _ in range(batchsize)]
# CORRECT:
images = torch.rand(batchsize, 3, 640, 640).to(torch.float32).cuda(local_rank)
```

### Issue: "Weights only load failed" or "weights_only argument in torch.load"
**Solution**: PyTorch 2.6+ compatibility issue. Patch the YOLOv5 repo:
```bash
sed -i "s/ckpt = torch.load(attempt_download(w), map_location='cpu')/ckpt = torch.load(attempt_download(w), map_location='cpu', weights_only=False)/" ~/orion/models/yolov5/repo/models/experimental.py
```
Verify:
```bash
grep "weights_only=False" ~/orion/models/yolov5/repo/models/experimental.py
```

### Issue: "AttributeError: _ARRAY_API not found" or "numpy.core.multiarray failed to import"
**Solution**: NumPy 2.x incompatibility with opencv-python. Downgrade NumPy:
```bash
pip install "numpy<2"
```
Then reinstall opencv-python if needed:
```bash
pip install --force-reinstall opencv-python==4.8.0.74
```

### Issue: "Module not found: yolov5_inference"
**Solution**: Verify the import path in `launch_jobs.py`:
```python
from benchmark_suite.yolov5_inference import yolov5_loop
```

### Issue: "No such file: yolov5s.pt"
**Solution**: Re-download weights:
```bash
cd ~/orion/models/yolov5
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```

### Issue: "hubconf.py not found"
**Solution**: Verify repo structure:
```bash
ls ~/orion/models/yolov5/repo/hubconf.py
```

### Issue: Profiling produces no kernels
**Solution**: 
- Check CUDA/GPU availability
- Verify profiling range is correct (iteration 9)
- Make sure you're using `--profile nsys` or `--profile ncu` argument to activate profiling hooks
- Try increasing iterations or profiling different iteration

### Issue: NCU creating .ncu-rep file instead of CSV-only
**Solution**: 
- Remove the `-o` flag from NCU command
- Use shell redirection (`>`) instead: `ncu --csv ... python3 script.py > output.csv`
- The `-o` flag creates binary .ncu-rep files even when `--csv` is specified
- According to NCU documentation, `--csv` only affects console output format, not file format

### Issue: LD_PRELOAD libraries not found
**Solution**: Find your Python site-packages:
```bash
python3 -c "import torch; print(torch.__file__)"
# Adjust LD_PRELOAD paths accordingly
```

### Issue: Different GPU than H100
**Solution**: Adjust parameters in Step 3.7.2 and 3.7.3 for your GPU architecture.

---

## Summary of Files Created/Modified

### ✨ Created (New Files):
1. `profiling/benchmarks/yolov5.py`
2. `benchmarking/benchmark_suite/yolov5_inference.py`
3. `benchmarking/model_kernels/h100/yolov5s_4_fwd`
4. `h100_results/inf_inf_updated/config_files/ideal/yolov5s_inf.json`

### 📝 Modified (Existing Files):
1. `benchmarking/launch_jobs.py` (add import and function_dict entry)
2. `h100_results/inf_inf_updated/config_files/mps/config.yaml` (add yolov5s section)
3. `related/baselines/main.py` (add yolov5s to model_to_wrapper)
4. `related/baselines/vision/train_imagenet.py` (add YOLOv5 model loading in setup())
5. `h100_results/inf_inf_updated/run_ideal.py` (add YOLOv5s to trace_files_hp)
6. `h100_results/inf_inf_updated/run_mps.py` (add to mnames and models)

---

## Final Notes

1. **Model Variant**: This plan uses YOLOv5s (small) for best balance of speed and accuracy
2. **Batch Size**: Configured for batch size 4 (matching other models in Orion)
3. **No Real Images Needed**: Orion uses synthetic random tensors for benchmarking
4. **Request Patterns**: Controlled via RPS and distribution settings, not image traces
5. **Total Time**: Expect 2-4 hours, mostly for profiling steps

**Ready to execute!** This plan is complete and can be run on your server.
