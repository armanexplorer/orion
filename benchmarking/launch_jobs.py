import argparse
import glob
import json
import os
import sys
import threading
import time
from ctypes import *

def _candidate_preload_libs() -> list[str]:
    """Return libs to prepend to LD_PRELOAD for Orion runs.

    Important: LD_PRELOAD must be set before importing torch/initializing CUDA.
    """

    home = os.path.expanduser('~')
    libs: list[str] = []

    # Orion interposer.
    libs.append(os.path.join(home, 'orion', 'src', 'cuda_capture', 'libinttemp.so'))

    # Packaged NVIDIA libs (pip nvidia-* layout). These are what our runners preload.
    py_mm = f"python{sys.version_info.major}.{sys.version_info.minor}"
    base = os.path.join(home, '.local', 'lib', py_mm, 'site-packages', 'nvidia')
    patterns = [
        os.path.join(base, 'cudnn', 'lib', 'libcudnn.so.9'),
        os.path.join(base, 'cudnn', 'lib', 'libcudnn.so'),
        os.path.join(base, 'cublas', 'lib', 'libcublasLt.so.12'),
        os.path.join(base, 'cublas', 'lib', 'libcublasLt.so'),
        os.path.join(base, 'cublas', 'lib', 'libcublas.so.12'),
        os.path.join(base, 'cublas', 'lib', 'libcublas.so'),
    ]
    for pat in patterns:
        for candidate in glob.glob(pat):
            libs.append(candidate)

    # Filter to existing, de-dup while preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for lib in libs:
        if lib and os.path.exists(lib) and lib not in seen:
            seen.add(lib)
            out.append(lib)
    return out


def _maybe_reexec_with_ld_preload(orion_preload: bool) -> None:
    """If requested, re-exec the current process with LD_PRELOAD set.

    This must run before importing torch/torchvision.
    """

    if not orion_preload:
        return

    if os.environ.get('ORION_LAUNCH_JOBS_PRELOADED', '') == '1':
        return

    libs = _candidate_preload_libs()
    if not libs:
        raise RuntimeError('No preload libraries found for Orion (libinttemp.so missing?)')

    existing = os.environ.get('LD_PRELOAD', '')
    existing_parts = [p for p in existing.split(':') if p]

    merged: list[str] = []
    seen: set[str] = set()
    for p in libs + existing_parts:
        if p and p not in seen:
            seen.add(p)
            merged.append(p)

    env = os.environ.copy()
    env['LD_PRELOAD'] = ':'.join(merged)
    env['ORION_LAUNCH_JOBS_PRELOADED'] = '1'

    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)


def _lazy_imports(need_transformer: bool, need_bert: bool):
    """Import torch/torchvision and training loops after preload bootstrap.

    Important: avoid polluting sys.path unless the requested model needs it.
    YOLOv5's repo has top-level packages named `utils` and `models` that can
    collide with other projects (e.g., NVIDIA DeepLearningExamples).
    """

    import torch
    from torchvision import models

    from benchmark_suite.train_imagenet import imagenet_loop
    from benchmark_suite.yolov5_trainer import yolov5_loop
    from src.scheduler_frontend import PyScheduler

    transformer_loop = None
    bert_loop = None

    home_directory = os.path.expanduser('~')
    if need_transformer:
        sys.path.append(f"{home_directory}/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch")
        sys.path.append(f"{home_directory}/DeepLearningExamples/PyTorch/LanguageModeling/Transformer-XL/pytorch/utils")
        from benchmark_suite.transformer_trainer import transformer_loop as _transformer_loop

        transformer_loop = _transformer_loop

    if need_bert:
        sys.path.append(f"{home_directory}/DeepLearningExamples/PyTorch/LanguageModeling/BERT")
        from bert_trainer import bert_loop as _bert_loop

        bert_loop = _bert_loop

    return torch, models, transformer_loop, bert_loop, imagenet_loop, yolov5_loop, PyScheduler

function_dict = {
    # Populated lazily in main() after torch import.
}


def _rewrite_kernel_path(original_path: str | None, kernel_root: str) -> str | None:
    if not original_path:
        return original_path

    marker = "/benchmarking/model_kernels/"
    try:
        idx = original_path.index(marker)
    except ValueError:
        return os.path.join(kernel_root, os.path.basename(original_path))

    suffix = original_path[idx + len(marker) :]
    # suffix is typically "<gpu>/<file_name>"; drop the gpu directory.
    parts = suffix.split("/", 1)
    relative = parts[1] if len(parts) == 2 else suffix
    return os.path.join(kernel_root, relative)


def _apply_kernel_root_overrides(config_dict_list: list[dict], kernel_root: str) -> None:
    for config in config_dict_list:
        if "kernel_file" in config:
            config["kernel_file"] = _rewrite_kernel_path(config["kernel_file"], kernel_root)
        if "additional_kernel_file" in config and config["additional_kernel_file"] is not None:
            config["additional_kernel_file"] = _rewrite_kernel_path(config["additional_kernel_file"], kernel_root)

def seed_everything(seed: int):
    import random, os
    import numpy as np

    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def launch_jobs(config_dict_list, input_args, run_eval):

    archs = {str(cfg.get('arch', '')).lower() for cfg in config_dict_list}
    need_transformer = 'transformer' in archs
    need_bert = 'bert' in archs

    torch, models, transformer_loop, bert_loop, imagenet_loop, yolov5_loop, PyScheduler = _lazy_imports(
        need_transformer=need_transformer,
        need_bert=need_bert,
    )
    global function_dict
    function_dict = {
        "resnet50": imagenet_loop,
        "resnet101": imagenet_loop,
        "mobilenet_v2": imagenet_loop,
        "yolov5n": yolov5_loop,
        "yolov5s": yolov5_loop,
        "bert": bert_loop,
        "transformer": transformer_loop,
    }

    for arch, fn in list(function_dict.items()):
        if fn is None and arch in archs:
            raise ImportError(
                f"Requested arch '{arch}' but its training loop could not be imported. "
                f"If you're running BERT/Transformer, ensure ~/DeepLearningExamples exists as described in INSTALL.md/README."
            )

    seed_everything(42)

    print(config_dict_list)
    num_clients = len(config_dict_list)
    print(num_clients)

    s = torch.cuda.Stream()

    # init

    num_barriers = num_clients+1
    barriers = [threading.Barrier(num_barriers) for i in range(num_clients)]
    client_barrier = threading.Barrier(num_clients)
    home_directory = os.path.expanduser( '~' )
    if run_eval:
        sched_lib = cdll.LoadLibrary(home_directory + "/orion/src/scheduler/scheduler_eval.so")
    else:
        sched_lib = cdll.LoadLibrary(home_directory + "/orion/src/scheduler/scheduler.so")
    py_scheduler = PyScheduler(sched_lib, num_clients)

    print(torch.__version__)

    model_names = [config_dict['arch'] for config_dict in config_dict_list]
    model_files = [config_dict['kernel_file'] for config_dict in config_dict_list]

    additional_model_files = [config_dict['additional_kernel_file'] if 'additional_kernel_file' in config_dict else None for config_dict in config_dict_list]
    num_kernels = [config_dict['num_kernels'] for config_dict in config_dict_list]
    num_iters = [config_dict['num_iters'] for config_dict in config_dict_list]
    train_list = [config_dict['args']['train'] for config_dict in config_dict_list]
    additional_num_kernels = [config_dict['additional_num_kernels'] if 'additional_num_kernels' in config_dict else None  for config_dict in config_dict_list]
    tids = []
    threads = []
    for i, config_dict in enumerate(config_dict_list):
        func = function_dict[config_dict['arch']]
        model_args = config_dict['args']
        model_args.update({"num_iters":num_iters[i], "local_rank": 0, "barriers": barriers, "client_barrier": client_barrier, "tid": i})

        thread = threading.Thread(target=func, kwargs=model_args)
        thread.start()
        tids.append(thread.native_id)
        threads.append(thread)

    print(tids)

    sched_thread = threading.Thread(
        target=py_scheduler.run_scheduler,
        args=(
            barriers,
            tids,
            model_names,
            model_files,
            additional_model_files,
            num_kernels,
            additional_num_kernels,
            num_iters,
            True,
            run_eval,
            input_args.algo=='reef',
            input_args.algo=='sequential',
            input_args.reef_depth if input_args.algo=='reef' else input_args.orion_max_be_duration,
            input_args.orion_hp_limit,
            input_args.orion_start_update,
            train_list
        )
    )

    sched_thread.start()

    for thread in threads:
        thread.join()

    print("train joined!")

    sched_thread.join()
    print("sched joined!")

    print("--------- all threads joined!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True,
                        help='choose one of orion | reef | sequential')
    parser.add_argument('--config_file', type=str, required=True,
                        help='path to the experiment configuration file')
    parser.add_argument('--reef_depth', type=int, default=1,
                        help='If reef is used, this stands for the queue depth')
    parser.add_argument('--orion_max_be_duration', type=int, default=1,
                        help='If orion is used, the maximum aggregate duration of on-the-fly best-effort kernels')
    parser.add_argument('--orion_start_update', type=int, default=1,
                        help='If orion is used, and the high priority job is training, this is the kernel id after which the update phase starts')
    parser.add_argument('--orion_hp_limit', type=int, default=1,
                        help='If orion is used, and the high priority job is training, this shows the maximum tolerated training iteration time')

    parser.add_argument(
        '--orion_preload',
        action='store_true',
        default=True,
        help='Re-exec this script with LD_PRELOAD configured like the experiment runners (libinttemp + packaged cuDNN/cuBLAS). Enabled by default.'
    )
    parser.add_argument(
        '--no_orion_preload',
        action='store_false',
        dest='orion_preload',
        help='Disable the LD_PRELOAD bootstrap/re-exec behavior.'
    )

    parser.add_argument(
        '--kernel_gpu',
        type=str,
        default=None,
        help='If set, rewrites kernel_file paths to $HOME/orion/benchmarking/model_kernels/<kernel_gpu>/*'
    )
    parser.add_argument(
        '--kernel_root',
        type=str,
        default=None,
        help='If set, rewrites kernel_file paths to <kernel_root>/* (overrides --kernel_gpu)'
    )

    args = parser.parse_args()

    # Must happen before importing torch/torchvision.
    _maybe_reexec_with_ld_preload(args.orion_preload)

    import torch

    torch.cuda.set_device(0)
    # affinity_mask = {0,1,2,3}
    # os.sched_setaffinity(0, affinity_mask)
    profile = True
    with open(args.config_file) as f:
        config_dict = json.load(f)

    if args.kernel_root or args.kernel_gpu:
        home_directory = os.path.expanduser('~')
        kernel_root = args.kernel_root
        if not kernel_root:
            kernel_root = os.path.join(home_directory, 'orion', 'benchmarking', 'model_kernels', args.kernel_gpu)
        _apply_kernel_root_overrides(config_dict, kernel_root)

    launch_jobs(config_dict, args, True)
