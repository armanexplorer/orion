## Plan (reproducible): Re-generate RTX 6000 (sm75) kernels + all results from scratch

This plan assumes you **delete** these generated artifacts:
- `benchmarking/model_kernels/rtx6000/`
- `profiling/profiles/rtx6000/` (or any RTX6000 profiling outputs)
- `rtx6000_results/` (the experiment folder + notebook)

Goal: reproduce the same *inputs* (kernel info files) and *outputs* (JSON results, CSV matrices, and paper-style PNG figures) for RTX 6000.

Terminology used throughout this repo/paper-style plots:
- **HP** = high-priority client/job
- **BE** = best-effort client/job

---

### 0) Prerequisites (one-time)

**Hardware/driver/tools**
- NVIDIA RTX 6000 (Turing, sm75) visible to CUDA.
- Working `ncu` (Nsight Compute) and `nsys` (Nsight Systems) installed.
- A CUDA-enabled PyTorch install.

**Repo + Python env**
```bash
git clone https://github.com/eth-easl/orion.git
cd orion
bash compile.sh
pip install -e .
```
This must produce:
- `src/cuda_capture/libinttemp.so`
- `src/scheduler/scheduler_eval.so`

**Tip: use tmux for long-running commands**
Profiling (`ncu`) and full experiment sweeps can take a long time. If you are on SSH, run them inside `tmux` so they keep running after disconnects:
```bash
tmux new -s rtx6000
# run your long command(s)
# detach: Ctrl-b then d
tmux a -t rtx6000
```

**External benchmark dependency for Orion’s BERT path**
`benchmarking/launch_jobs.py` imports BERT from NVIDIA’s `DeepLearningExamples` by default.
Create it at exactly this location:
```bash
git clone https://github.com/NVIDIA/DeepLearningExamples.git ~/DeepLearningExamples
```
If you put it elsewhere, you must update the `sys.path.append(...)` lines in `benchmarking/launch_jobs.py`.

---

### 1) Recreate the RTX6000 experiment folder (if `rtx6000_results/` is removed)

The simplest reproducible approach is to clone the “inf_inf_updated” harness from the H100 folder and then set it to RTX6000.

```bash
mkdir -p rtx6000_results
cp -r h100_results/inf_inf_updated rtx6000_results/inf_inf_updated
```

Now ensure the RTX6000 harness contains these scripts (either by copying them from your previous working tree, or by editing the copied ones):
- `run_ideal.py`, `run_orion.py`, `run_temporal.py`, `run_reef.py` should call:
  - `python3 ../../benchmarking/launch_jobs.py ... --kernel_gpu rtx6000`
- `run_mps.py` should write outputs to `results/mps/{HP}_{BE}_{run}.json`
- `run_streams.py` should write outputs to `results/streams/{HP}_{BE}_{run}.json`
- `prep_dirs.sh` should create: `results/{ideal,orion,temporal,reef,mps,streams}`

Notes:
- The `--kernel_gpu rtx6000` flag is important: it rewrites any `kernel_file` paths inside JSON configs to `~/orion/benchmarking/model_kernels/rtx6000/<basename>`.
- Your config JSONs can still point at `.../model_kernels/h100/...` as long as the **basenames** exist under `benchmarking/model_kernels/rtx6000/`.

---

### 2) Re-generate RTX6000 kernel info files (the critical input)

You need to create these kernel files under:
- `benchmarking/model_kernels/rtx6000/`

For the inference configs used in `rtx6000_results/inf_inf_updated/config_files/*.json`, the expected basenames are typically:
- `resnet50_8_fwd`
- `resnet101_8_fwd`
- `mobilenet_16_fwd`
- `bert_2_fwd`

#### 2.1) Create profiling output folders
```bash
mkdir -p profiling/profiles/rtx6000/{resnet50_8_fwd,resnet101_8_fwd,mobilenet_16_fwd,bert_2_fwd}
mkdir -p benchmarking/model_kernels/rtx6000
```

#### 2.2) Collect NCU CSVs (both “normal” + “raw”)

Each profile directory must contain:
- `output_ncu.csv` (for `process_ncu.py`)
- `raw_ncu.csv` (for `roofline_analysis.py`)

Practical note: the default `--set detailed` collection can be slow. In practice, we got robust results using a **metrics-only** capture containing exactly what the postprocessing scripts need.

Define the metrics list once:
```bash
METRICS="dram__bytes.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,launch__block_size,launch__grid_size,launch__registers_per_thread,launch__shared_mem_per_block_static,sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__cycles_elapsed.avg.per_second,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed"
```

Then, for each model, produce BOTH `output_ncu.csv` and `raw_ncu.csv`. One convenient approach is to collect once into `raw_ncu.csv` and then copy it to `output_ncu.csv` (since the scripts can handle long-format CSV).

Example “one-liner” (this is the style that worked reliably for long runs):
```bash
name=resnet101_8_fwd \
  && outdir=profiling/profiles/rtx6000/$name \
  && mkdir -p "$outdir" \
  && echo "=== Profiling $name ===" \
  && ncu --csv --profile-from-start off --target-processes all --replay-mode kernel \
       --metrics "$METRICS" \
       python3 profiling/benchmarks/profile_vision_silent.py --model resnet101 --batchsize 8 --device 0 \
       > "$outdir/raw_ncu.csv" 2> "$outdir/ncu.stderr.log" \
  && cp "$outdir/raw_ncu.csv" "$outdir/output_ncu.csv"
```

If you prefer capturing `output_ncu.csv` directly (and keeping stderr logs), this command pattern also worked:
```bash
mkdir -p profiling/profiles/rtx6000/resnet50_8_fwd \
  && /usr/local/cuda/bin/ncu --csv --metrics "$METRICS" \
       --profile-from-start off -f --target-processes application-only \
       /home/cc/orion/.venv/bin/python profiling/benchmarks/profile_vision_silent.py --model resnet50 --batchsize 8 --device 0 \
       > profiling/profiles/rtx6000/resnet50_8_fwd/output_ncu.csv \
       2> profiling/profiles/rtx6000/resnet50_8_fwd/output_ncu.stderr.log
```
If you use the second pattern, still create `raw_ncu.csv` too (either by re-running with `> raw_ncu.csv`, or by copying `output_ncu.csv` to `raw_ncu.csv` if it contains the required roofline metrics).

##### Legacy fallback: `--set detailed` (slower, but sometimes simpler)

If you want the “classic” NCU collection (what the original docs in this repo describe), here’s the full block in one place. It collects two CSVs per model:
- `output_ncu.csv` (standard)
- `raw_ncu.csv` (required by `roofline_analysis.py`)

```bash
# ResNet50 (batch=8)
outdir=profiling/profiles/rtx6000/resnet50_8_fwd && mkdir -p "$outdir" && cd "$outdir" \
  && ncu --csv --set detailed --profile-from-start off -f \
    python3 ../../../benchmarks/profile_vision_silent.py --model resnet50 --batchsize 8 --device 0 \
    > output_ncu.csv 2> ncu_output.stderr.log \
  && ncu --page raw --csv --set detailed --profile-from-start off -f \
    python3 ../../../benchmarks/profile_vision_silent.py --model resnet50 --batchsize 8 --device 0 \
    > raw_ncu.csv 2> ncu_raw.stderr.log

# ResNet101 (batch=8)
outdir=profiling/profiles/rtx6000/resnet101_8_fwd && mkdir -p "$outdir" && cd "$outdir" \
  && ncu --csv --set detailed --profile-from-start off -f \
    python3 ../../../benchmarks/profile_vision_silent.py --model resnet101 --batchsize 8 --device 0 \
    > output_ncu.csv 2> ncu_output.stderr.log \
  && ncu --page raw --csv --set detailed --profile-from-start off -f \
    python3 ../../../benchmarks/profile_vision_silent.py --model resnet101 --batchsize 8 --device 0 \
    > raw_ncu.csv 2> ncu_raw.stderr.log

# MobileNetV2 (batch=16)
outdir=profiling/profiles/rtx6000/mobilenet_16_fwd && mkdir -p "$outdir" && cd "$outdir" \
  && ncu --csv --set detailed --profile-from-start off -f \
    python3 ../../../benchmarks/profile_vision_silent.py --model mobilenet_v2 --batchsize 16 --device 0 \
    > output_ncu.csv 2> ncu_output.stderr.log \
  && ncu --page raw --csv --set detailed --profile-from-start off -f \
    python3 ../../../benchmarks/profile_vision_silent.py --model mobilenet_v2 --batchsize 16 --device 0 \
    > raw_ncu.csv 2> ncu_raw.stderr.log

# BERT (batch=2)
outdir=profiling/profiles/rtx6000/bert_2_fwd && mkdir -p "$outdir" && cd "$outdir" \
  && ncu --csv --set detailed --profile-from-start off -f \
    python3 ../../../benchmarks/profile_bert_silent.py --batchsize 2 --device 0 \
    > output_ncu.csv 2> ncu_output.stderr.log \
  && ncu --page raw --csv --set detailed --profile-from-start off -f \
    python3 ../../../benchmarks/profile_bert_silent.py --batchsize 2 --device 0 \
    > raw_ncu.csv 2> ncu_raw.stderr.log
```

Optional (nsys): only needed if you want timeline traces; Orion’s kernel file generation does not require nsys.

#### 2.3) Postprocess into Orion kernel files

For each directory above, run:

```bash
python3 ../../../postprocessing/process_ncu.py --results_dir .

# RTX 6000 (Turing) SM limits are compatible with these defaults; override here for explicitness:
python3 ../../../postprocessing/get_num_blocks.py --results_dir . \
  --max_threads_sm 1024 --max_shmem_sm 65536 --max_regs_sm 65536

# Uses raw_ncu.csv, ai_threshold default=9.2
python3 ../../../postprocessing/roofline_analysis.py --results_dir . --ai_threshold 9.2
```

Note on `--max_threads_sm`: use the value that matches your GPU architecture / what you used previously for RTX6000.
If you’re unsure, prefer keeping this consistent across re-runs and record it alongside the profiling artifacts.

Then convert to the final kernel info file:

- For vision models:
```bash
python3 ../../../postprocessing/generate_file_updated.py \
  --input_file_name output_ncu_sms_roofline.csv \
  --output_file_name ../../../../benchmarking/model_kernels/rtx6000/<NAME> \
  --model_type vision
```
- For BERT:
```bash
python3 ../../../postprocessing/generate_file_updated.py \
  --input_file_name output_ncu_sms_roofline.csv \
  --output_file_name ../../../../benchmarking/model_kernels/rtx6000/bert_2_fwd \
  --model_type bert
```

Replace `<NAME>` with:
- `resnet50_8_fwd`
- `resnet101_8_fwd`
- `mobilenet_16_fwd`

Sanity checks:
```bash
ls -lah ../../../../benchmarking/model_kernels/rtx6000/
head -n 2 ../../../../benchmarking/model_kernels/rtx6000/resnet50_8_fwd
```

---

### 3) Run RTX6000 experiments (generate `results/*/*.json`)

All runs below are done from:
```bash
cd rtx6000_results/inf_inf_updated
bash prep_dirs.sh
mkdir -p logs
```

#### 3.1) Orion-family runs (Ideal / Orion / Temporal / REEF)
These use `benchmarking/launch_jobs.py` + `LD_PRELOAD` interception and will produce `client_0.json`/`client_1.json`, which the wrapper scripts copy into `results/<baseline>/...`.

Run:
```bash
python3 run_ideal.py    2>&1 | tee logs/ideal.log
python3 run_orion.py    2>&1 | tee logs/orion.log
python3 run_temporal.py 2>&1 | tee logs/temporal.log
python3 run_reef.py     2>&1 | tee logs/reef.log
```

Expected file counts:
- `results/ideal/`: 4 files (`{Model}_0_hp.json`)
- `results/orion/`, `results/temporal/`, `results/reef/`: 32 files each (`{BE}_{HP}_0_{be,hp}.json`)

#### 3.2) CUDA MPS baseline (required for the “MPS” policy runs)
Start the MPS daemon (once per machine boot/session):
```bash
bash ../../related/baselines/start_MPS_control_daemon.sh
```

Stop the daemon when you’re done (optional, but avoids surprises in later runs):
```bash
echo quit | nvidia-cuda-mps-control
```

Run MPS baseline:
```bash
python3 run_mps.py 2>&1 | tee logs/mps.log
```

Expected file count:
- `results/mps/`: 16 files (`{HP}_{BE}_0.json`)

#### 3.3) Streams baseline
Run:
```bash
python3 run_streams.py 2>&1 | tee logs/streams.log
```

Expected file count:
- `results/streams/`: 16 files (`{HP}_{BE}_0.json`)

---

### 4) Produce matrices + figures

#### 4.1) CSV matrices (script-style)
From `rtx6000_results/inf_inf_updated`:
```bash
python3 gather_results.py
```
This writes matrices like:
- `results/{baseline}_latency.csv`
- `results/{baseline}_{be,hp}_throughput.csv`

Note: the notebook workflow is more robust for combining **all** baselines (including Streams) and for the latency unit conversion.

#### 4.2) Notebook: paper-style figures
Open and run:
- `rtx6000_results/inf_inf_updated/compare_orion_mps.ipynb`

Expected saved figures (in that folder):
- `inf_inf_poisson_total_throughput.png`
- `inf_inf_poisson_hp_latency.png`
- `inf_inf_poisson_hp_p99_latency.png`
- `inf_inf_poisson_hp_throughput.png`

Important note on units:
- The JSON outputs record latency values that behave like **microseconds**.
- The paper-style plots label **milliseconds**, so plotting should convert µs → ms (divide by 1000).

---

### 5) Quick verification checklist

From `rtx6000_results/inf_inf_updated`:
```bash
# counts
ls results/ideal/*.json | wc -l
ls results/orion/*.json | wc -l
ls results/temporal/*.json | wc -l
ls results/reef/*.json | wc -l
ls results/mps/*.json | wc -l
ls results/streams/*.json | wc -l

# spot-check keys exist
python3 - << 'PY'
import json, glob
p = glob.glob('results/orion/*_hp.json')[0]
print(p)
d = json.load(open(p))
print(sorted([k for k in d.keys() if 'lat' in k or 'throughput' in k])[:20])

p2 = glob.glob('results/mps/*.json')[0]
print(p2)
d2 = json.load(open(p2))
print([k for k in d2.keys() if 'latency' in k or 'throughput' in k])
PY
```

If you see missing files or empty JSONs, check:
- `logs/*.log`
- your `LD_PRELOAD` library paths
- that `benchmarking/model_kernels/rtx6000/*` basenames match what configs reference

---

### Practical recommendation (for maximal reproducibility)

If you plan to delete `rtx6000_results/` from the repo, keep at least these lightweight artifacts tracked:
- The RTX6000 runner scripts (`run_orion.py`, `run_mps.py`, `run_streams.py`, `run_temporal.py`, `run_reef.py`, `run_ideal.py`)
- The `config_files/` directory
- The notebook (or a script) that produces paper-style plots with correct unit conversion

And keep the heavyweight/generated artifacts untracked:
- `benchmarking/model_kernels/rtx6000/`
- `profiling/profiles/rtx6000/`
- `rtx6000_results/**/results/`
- `rtx6000_results/**/logs/`
