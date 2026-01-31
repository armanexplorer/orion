import os
import sys
import time
import argparse

import torch


def load_yolov5(repo_path: str, weights_path: str):
    original_cwd = os.getcwd()
    original_path = sys.path.copy()

    # YOLOv5 repo has a top-level `utils` package that can conflict with other repos.
    modules_to_remove = [k for k in sys.modules.keys() if k.startswith('utils') and 'yolov5' not in k]
    for mod in modules_to_remove:
        sys.modules.pop(mod, None)

    try:
        os.chdir(repo_path)
        sys.path.insert(0, repo_path)
        model = torch.hub.load(
            repo_or_dir=repo_path,
            model='custom',
            source='local',
            path=weights_path,
            device='cpu',
            autoshape=False,
        )
        model.eval()
        return model
    finally:
        os.chdir(original_cwd)
        sys.path = original_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov5s')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--iters', type=int, default=12, help='Total iters; profiles only a small window')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    orion_root = os.path.expanduser('~') + '/orion'
    repo_path = f'{orion_root}/models/yolov5/repo'
    weights_path = f'{orion_root}/models/yolov5/{args.model}.pt'

    torch.cuda.set_device(int(args.device.split(':')[-1]))

    model = load_yolov5(repo_path, weights_path)
    model = model.to(args.device)

    images = torch.rand(args.batch, 3, 640, 640, device=args.device, dtype=torch.float32)

    # Warmup a bit (outside profiling)
    with torch.no_grad():
        for _ in range(2):
            _ = model(images)
    torch.cuda.synchronize()

    # Profile a short window inside the loop.
    # This matches the guidance in PROFILE.md: use cudaProfilerStart/Stop.
    with torch.no_grad():
        for i in range(args.iters):
            if i == 2:
                torch.cuda.profiler.cudart().cudaProfilerStart()
            _ = model(images)
            if i == 3:
                torch.cuda.profiler.cudart().cudaProfilerStop()

    torch.cuda.synchronize()
    print('Done')


if __name__ == '__main__':
    main()
