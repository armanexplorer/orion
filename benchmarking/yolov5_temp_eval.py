import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch


def _load_yolov5(repo_path: str, weights_path: str):
    original_cwd = os.getcwd()
    original_path = sys.path.copy()

    modules_to_remove = [k for k in sys.modules.keys() if k.startswith("utils") and "yolov5" not in k]
    for mod in modules_to_remove:
        sys.modules.pop(mod, None)

    try:
        os.chdir(repo_path)
        sys.path.insert(0, repo_path)

        print(f"Loading YOLOv5 from: {weights_path}")
        model = torch.hub.load(
            repo_or_dir=repo_path,
            model="custom",
            source="local",
            path=weights_path,
            device="cpu",
            autoshape=False,
        )
        model.eval()
        return model
    finally:
        os.chdir(original_cwd)
        sys.path = original_path


def _sleep_times(num_iters: int, rps: float, uniform: bool) -> List[float]:
    if rps <= 0:
        return [0.0] * num_iters
    if uniform:
        return [1.0 / rps] * num_iters
    # exponential inter-arrival
    return list(np.random.exponential(scale=1.0 / rps, size=num_iters))


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporary standalone YOLOv5 eval (no Orion scheduler, no LD_PRELOAD).")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    with open(args.config_file) as f:
        config_list: List[Dict[str, Any]] = json.load(f)

    if len(config_list) != 1 or config_list[0].get("arch") != "yolov5s":
        raise ValueError("This temp runner only supports a single-client yolov5s config.")

    cfg = config_list[0]
    model_name = cfg["args"]["model_name"]
    batchsize = int(cfg["args"]["batchsize"])
    num_iters = int(cfg["num_iters"])
    rps = float(cfg["args"].get("rps", 0))
    uniform = bool(cfg["args"].get("uniform", True))

    sleep_times = _sleep_times(num_iters=num_iters, rps=rps, uniform=uniform)

    device = torch.device(args.device)
    torch.cuda.set_device(device.index or 0)

    orion_root = os.path.expanduser("~") + "/orion"
    repo_path = f"{orion_root}/models/yolov5/repo"
    weights_path = f"{orion_root}/models/yolov5/{model_name}.pt"

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing YOLO weights: {weights_path}")

    model = _load_yolov5(repo_path, weights_path).to(device)

    images = torch.rand(batchsize, 3, 640, 640, device=device, dtype=torch.float32)

    # Warmup
    with torch.no_grad():
        for _ in range(max(0, args.warmup)):
            _ = model(images)
    torch.cuda.synchronize()

    timings: List[float] = []

    start_total = time.time()
    next_start = time.time()

    with torch.no_grad():
        for it in range(num_iters):
            # Open-loop pacing
            now = time.time()
            if now < next_start:
                while time.time() < next_start:
                    time.sleep(0.001)

            t0 = time.time()
            _ = model(images)
            torch.cuda.synchronize()
            t1 = time.time()
            timings.append(t1 - t0)

            next_start = max(time.time(), next_start) + float(sleep_times[it])

    end_total = time.time()

    timings_sorted = sorted(timings)
    p50 = float(np.percentile(timings_sorted, 50)) if timings_sorted else 0.0
    p95 = float(np.percentile(timings_sorted, 95)) if timings_sorted else 0.0
    p99 = float(np.percentile(timings_sorted, 99)) if timings_sorted else 0.0

    total_time = end_total - start_total
    throughput = (num_iters / total_time) if total_time > 0 else 0.0

    data = {
        "p50_latency": p50 * 1000.0,
        "p95_latency": p95 * 1000.0,
        "p99_latency": p99 * 1000.0,
        "throughput": throughput,
        "note": "TEMP_STANDALONE_NO_ORION",
    }

    out_path = "client_0.json"
    with open(out_path, "w") as f:
        json.dump(data, f)

    print(f"Wrote {out_path}: p50={data['p50_latency']:.3f}ms p95={data['p95_latency']:.3f}ms thr={data['throughput']:.3f} req/s")


if __name__ == "__main__":
    main()
