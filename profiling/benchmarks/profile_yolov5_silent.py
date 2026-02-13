import argparse
import os

import torch


def run(model_name: str, batchsize: int, img_size: int = 640, device: int = 0) -> None:
    if not model_name.lower().startswith("yolov5"):
        raise ValueError(f"model_name must start with yolov5*, got: {model_name}")

    orion_root = os.path.join(os.path.expanduser("~"), "orion")
    repo_path = os.path.join(orion_root, "models", "yolov5", "repo")
    weights_path = os.path.join(orion_root, "models", "yolov5", f"{model_name}.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Missing YOLO weights at {weights_path}. Download {model_name}.pt into ~/orion/models/yolov5/"
        )

    x = torch.ones([batchsize, 3, img_size, img_size], pin_memory=True).to(device)

    model = torch.hub.load(
        repo_or_dir=repo_path,
        model="custom",
        source="local",
        path=weights_path,
        device=f"cuda:{device}",
    )
    model.eval()

    torch.cuda.synchronize()

    # Single profiled iteration, wrapped in cudaProfilerStart/Stop.
    torch.cuda.profiler.cudart().cudaProfilerStart()
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    torch.cuda.profiler.cudart().cudaProfilerStop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="e.g. yolov5n or yolov5s")
    parser.add_argument("--batchsize", required=True, type=int)
    parser.add_argument("--img_size", default=640, type=int)
    parser.add_argument("--device", default=0, type=int)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    run(args.model, args.batchsize, img_size=args.img_size, device=args.device)
