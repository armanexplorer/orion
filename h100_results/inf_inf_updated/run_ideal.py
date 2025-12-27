import os
import time

num_runs = 1
trace_files_hp = [
    ("ResNet50", "rnet"),
    ("MobileNetV2", "mnet"),
    ("ResNet101", "rnet101"),
    ("BERT", "bert")
]

HOME = os.path.expanduser("~")
ld_preload_cmd = f"LD_PRELOAD={HOME}/orion/src/cuda_capture/libinttemp.so:{HOME}/.local/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.9:{HOME}/.local/lib/python3.10/site-packages/nvidia/cublas/lib/libcublasLt.so.12:{HOME}/.local/lib/python3.10/site-packages/nvidia/cublas/lib/libcublas.so.12"

for (model, f) in trace_files_hp:
    for run in range(num_runs):
        print(model, run, flush=True)
        # run
        file_path = f"config_files/ideal/{f}_inf.json"
        os.system(f"{ld_preload_cmd} python3 ../../benchmarking/launch_jobs.py --algo orion --config_file {file_path}")

        # copy results
        os.system(f"cp client_0.json results/ideal/{model}_{run}_hp.json")
        os.system("rm -f client_0.json")