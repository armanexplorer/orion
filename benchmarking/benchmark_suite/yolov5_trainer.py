import argparse
import json
import os
import threading
import time
from ctypes import *

import numpy as np


def _log(msg: str) -> None:
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] [yolov5_trainer] {msg}", flush=True)


def seed_everything(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def block(backend_lib, it: int) -> None:
    backend_lib.block(it)


def check_stop(backend_lib) -> bool:
    return bool(backend_lib.stop())


def set_stream(backend_lib, idx: int) -> None:
    import torch

    backend_lib.get_stream_ptr.restype = c_void_p
    stream_id = backend_lib.get_stream_ptr(idx)
    print("Setting stream to ", stream_id)
    new_stream = torch.cuda.get_stream_from_external(stream_id)
    torch.cuda.set_stream(new_stream)


def _load_yolov5_model(
    model_name: str,
    device: str,
    autoshape: bool = False,
    load_device: str | None = None,
):
    import torch
    import sys

    if not model_name.lower().startswith("yolov5"):
        raise ValueError(f"model_name must start with yolov5*, got: {model_name}")

    orion_root = os.path.join(os.path.expanduser("~"), "orion")
    repo_path = os.path.join(orion_root, "models", "yolov5", "repo")
    weights_path = os.path.join(orion_root, "models", "yolov5", f"{model_name}.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Missing YOLO weights at {weights_path}. Download {model_name}.pt into ~/orion/models/yolov5/"
        )

    # Ensure YOLOv5's top-level packages (utils/, models/) resolve to this repo.
    # Some other benchmarks (e.g. DeepLearningExamples) also install a top-level
    # `utils` package, which would break YOLO imports.
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    def _purge_if_shadowed(mod_name: str) -> None:
        m = sys.modules.get(mod_name)
        mod_file = getattr(m, "__file__", None)
        if m is None or mod_file is None:
            return
        if not os.path.abspath(mod_file).startswith(os.path.abspath(repo_path) + os.sep):
            for k in list(sys.modules.keys()):
                if k == mod_name or k.startswith(mod_name + "."):
                    del sys.modules[k]

    _purge_if_shadowed("utils")
    _purge_if_shadowed("models")

    if load_device is None:
        load_device = device

    # If we load on CPU first, YOLOv5 may set CUDA_VISIBLE_DEVICES='-1'.
    # Restore it afterwards so later CUDA init / model.to('cuda') keeps working.
    prev_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    model = torch.hub.load(
        repo_or_dir=repo_path,
        model="custom",
        source="local",
        path=weights_path,
        device=load_device,
        autoshape=autoshape,
        verbose=False,
    )
    cur_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if load_device == 'cpu' and cur_cuda_visible_devices != prev_cuda_visible_devices:
        if prev_cuda_visible_devices is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = prev_cuda_visible_devices
    model.eval()
    return model


def _start_stall_watchdog(progress_ref: dict, stall_s: float) -> None:
    """Dump thread stacks if progress stalls for too long."""

    import faulthandler
    import sys

    def _watch() -> None:
        last_idx = -1
        last_change = time.time()
        while True:
            time.sleep(1.0)
            idx = int(progress_ref.get('batch_idx', -1))
            if idx != last_idx:
                last_idx = idx
                last_change = time.time()
                continue
            if stall_s > 0 and (time.time() - last_change) >= stall_s:
                _log(f"WATCHDOG: no batch_idx progress for {stall_s}s (batch_idx={idx}); dumping stacks")
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                last_change = time.time()

    t = threading.Thread(target=_watch, name='yolo-stall-watchdog', daemon=True)
    t.start()


def yolov5_loop(
    model_name: str,
    batchsize: int,
    train: bool,
    num_iters: int,
    rps: float,
    uniform: bool,
    dummy_data: bool,
    local_rank: int,
    barriers,
    client_barrier,
    tid: int,
    input_file: str = "",
    img_size: int = 640,
):
    """Evaluation loop for YOLOv5n/YOLOv5s.

    Matches the call signature expected by `benchmarking/launch_jobs.py`.
    """

    import torch

    if train:
        raise NotImplementedError("YOLOv5 training is not implemented in this repo's benchmarking harness")

    seed_everything(42)
    _log(f"start loop model={model_name} batch={batchsize} rank={local_rank} tid={tid} num_iters={num_iters}")

    backend_lib = cdll.LoadLibrary(os.path.expanduser("~") + "/orion/src/cuda_capture/libinttemp.so")

    if rps > 0 and input_file == "":
        if uniform:
            sleep_times = [1 / rps] * num_iters
        else:
            sleep_times = np.random.exponential(scale=1 / rps, size=num_iters)
    elif input_file != "":
        with open(input_file) as f:
            sleep_times = json.load(f)
    else:
        sleep_times = [0] * num_iters

    _log(f"sleep_times size={len(sleep_times)}")
    barriers[0].wait()

    _log(f"native_thread_id={threading.get_native_id()}")

    set_stream(backend_lib, tid)

    device_str = f"cuda:{int(local_rank)}"
    device = torch.device(device_str)

    watchdog_s = float(os.environ.get('ORION_YOLO_STALL_SEC', '0') or 0)
    progress_ref = {'batch_idx': -1}
    if watchdog_s > 0:
        _log(f"enabling stall watchdog: ORION_YOLO_STALL_SEC={watchdog_s}")
        _start_stall_watchdog(progress_ref, watchdog_s)

    # Important for Orion: per-iteration intercepted record count must be stable.
    # cuDNN benchmark can introduce autotuning kernels in early iterations that
    # change the record count and cause scheduler/client deadlocks.
    torch.backends.cudnn.benchmark = False
    _log("loading model (on cuda)")
    model = _load_yolov5_model(model_name, device=device_str, autoshape=False, load_device=device_str)
    model.eval()
    _log("model ready")

    if dummy_data:
        x = torch.ones([batchsize, 3, img_size, img_size], pin_memory=True).to(device)
    else:
        x = torch.rand([batchsize, 3, img_size, img_size], pin_memory=True).to(device)

    torch.cuda.synchronize()
    _log("Enter loop!")

    next_startup = time.time()
    open_loop = True
    timings: list[float] = []

    warmup_reqs = 10
    batch_idx = 0
    start = time.time()
    while batch_idx < num_iters:
        progress_ref['batch_idx'] = batch_idx
        if check_stop(backend_lib):
            _log("STOP requested by backend")
            break

        with torch.no_grad():
            cur_time = time.time()
            if open_loop:
                if cur_time >= next_startup:
                    if batch_idx == 0:
                        _log("iter0 forward start")
                    _ = model(x)
                    if batch_idx == 0:
                        _log("iter0 forward done; calling block(0)")
                    block(backend_lib, batch_idx)
                    if batch_idx == 0:
                        _log("iter0 block returned")

                    req_time = time.time() - next_startup
                    timings.append(req_time)

                    if batch_idx >= warmup_reqs:
                        next_startup += float(sleep_times[batch_idx])
                    else:
                        next_startup = time.time()

                    batch_idx += 1

                    # Orion eval harness uses an intermediate barrier at batch_idx==1
                    # ("after warmup setup"). For YOLO, this can deadlock if model
                    # initialization/first forward overlaps with the scheduler pausing
                    # scheduling at that barrier. We skip the batch_idx==1 barrier for
                    # YOLO and only synchronize at warmup_reqs (10) like other loops.
                    if batch_idx == warmup_reqs:
                        _log(f"barrier wait at batch_idx={batch_idx}")
                        barriers[0].wait()
                        _log(f"barrier passed at batch_idx={batch_idx}")
                        next_startup = time.time()
                        start = time.time()

                    dur = next_startup - time.time()
                    if dur > 0:
                        while time.time() < next_startup:
                            time.sleep(0.001)
            else:
                _ = model(x)
                block(backend_lib, batch_idx)
                batch_idx += 1
                if batch_idx == 1 or batch_idx == warmup_reqs:
                    barriers[0].wait()

    print(f"Client {tid} at barrier!")
    barriers[0].wait()
    total_time = time.time() - start

    timings = timings[warmup_reqs:]
    timings = sorted(timings)

    if len(timings) > 0:
        p50 = float(np.percentile(timings, 50))
        p95 = float(np.percentile(timings, 95))
        p99 = float(np.percentile(timings, 99))
        print(f"Client {tid} finished! p50: {p50} sec, p95: {p95} sec, p99: {p99} sec")
        data = {
            "p50_latency": p50 * 1000,
            "p95_latency": p95 * 1000,
            "p99_latency": p99 * 1000,
            "throughput": (batch_idx - warmup_reqs) / total_time if total_time > 0 else 0.0,
        }
    else:
        data = {
            "throughput": (batch_idx - warmup_reqs) / total_time if total_time > 0 else 0.0,
        }

    with open(f"client_{tid}.json", "w") as f:
        json.dump(data, f)

    print("Finished! Ready to join!")


if __name__ == "__main__":
    # Minimal local smoke test (does not use Orion scheduler).
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov5n")
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    import torch

    torch.backends.cudnn.benchmark = False
    device_str = f"cuda:{args.device}"
    _log("smoke: loading model (on cuda)")
    model = _load_yolov5_model(args.model, device=device_str, autoshape=False, load_device=device_str)
    model = model.eval()
    x = torch.ones([args.batchsize, 3, args.img_size, args.img_size]).to(torch.device(device_str))
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    print("OK")
