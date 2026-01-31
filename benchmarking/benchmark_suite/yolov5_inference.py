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
    if backend_lib is not None:
        backend_lib.block(it)

def check_stop(backend_lib):
    """Check if scheduler signals stop"""
    if backend_lib is not None:
        return backend_lib.stop()
    return False

def set_stream(backend_lib, idx):
    """Set CUDA stream for this client"""
    if backend_lib is None:
        return
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
    input_file='',
    use_backend=True
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
        use_backend: whether to use Orion backend library (default: True)
    """
    
    seed_everything(42)
    print(f"YOLOv5 Inference - Model: {model_name}, Batch: {batchsize}, Rank: {local_rank}, TID: {tid}")
    
    # Load Orion backend library only if use_backend is True
    backend_lib = None
    if use_backend:
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
    
    # First barrier sync - before set_stream
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
    
    # Aggressively isolate YOLOv5 imports from conflicting modules
    original_cwd = os.getcwd()
    original_path = sys.path.copy()
    
    # Clear any cached 'utils' imports that might conflict
    modules_to_remove = [k for k in sys.modules.keys() if k.startswith('utils') and 'yolov5' not in k]
    for mod in modules_to_remove:
        sys.modules.pop(mod, None)
    
    try:
        # Change to YOLOv5 repo and put it first in path
        os.chdir(repo_path)
        sys.path.insert(0, repo_path)
        
        print(f"Loading YOLOv5 from: {weights_path}")
        # Load on CPU first to avoid kernel launches before scheduler is ready
        model = torch.hub.load(
            repo_or_dir=repo_path,
            model='custom',
            source='local',
            path=weights_path,
            device='cpu',
            autoshape=False
        )
        model.eval()
        print("YOLOv5 model loaded on CPU")
    finally:
        # Restore original state
        os.chdir(original_cwd)
        sys.path = original_path
    
    # Move model to GPU after scheduler is initialized
    print("Moving model to GPU...")
    model = model.to(f'cuda:{local_rank}')
    
    # Prepare dummy data (640x640 images for YOLOv5)
    # YOLOv5 expects a batch tensor, not a list of tensors
    images = torch.rand(batchsize, 3, 640, 640).to(torch.float32).cuda(local_rank)
    
    print("YOLOv5 model and data ready")
    
    print("YOLOv5 starting inference loop (scheduler will handle warmup)")
    
    start_time = time.time()
    next_startup = time.time()
    batch_idx = 0
    
    # Main inference loop
    for it in range(len(sleep_times)):
        if check_stop(backend_lib):
            print(f"Stopped at iteration {it}")
            break
        
        # Keep the same semantics as imagenet_loop:
        # - Submit request for index `it`
        # - After submit, use `next_batch_idx = it + 1` for barrier points
        batch_idx = it
        
        with torch.no_grad():
            cur_time = time.time()
            # Open loop execution matching ResNet inference pattern
            if cur_time >= next_startup:
                if it < 5:
                    print(f"DEBUG: Executing iteration {it}, batch_idx={batch_idx}")
                output = model(images)

                if it == 0:
                    print("YOLOv5 DEBUG: Calling backend_lib.block(0) (first request)")
                block(backend_lib, batch_idx)
                if it == 0:
                    print("YOLOv5 DEBUG: backend_lib.block(0) returned")
                
                if batch_idx >= 10:
                    next_startup += sleep_times[batch_idx]
                else:
                    next_startup = time.time()
                
                # Barrier sync points to match imagenet_loop behavior:
                # - after the first submitted request (next_batch_idx == 1)
                # - after warmup requests (next_batch_idx == 10)
                next_batch_idx = batch_idx + 1
                if next_batch_idx == 1 or next_batch_idx == 10:
                    barriers[0].wait()
                    if next_batch_idx == 10:
                        next_startup = time.time()
                        start_time = time.time()
                        print("Warmup completed, starting evaluation")
                
                if it % 100 == 0 or it < 20:
                    print(f"YOLOv5 iteration {batch_idx}/{len(sleep_times)}")
            
            # Sleep to maintain request rate
            dur = next_startup - time.time()
            if dur > 0:
                while time.time() < next_startup:
                    time.sleep(0.001)
    
    print(f"YOLOv5 at final barrier")
    barriers[0].wait()
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"YOLOv5 inference completed in {total_time:.2f}s")
    
    # Calculate and save results (matching ResNet format)
    if batch_idx > 10:
        data = {
            'throughput': (batch_idx - 10) / total_time
        }
        with open(f'client_{tid}.json', 'w') as f:
            json.dump(data, f)
        print(f"Results saved to client_{tid}.json")
    
    print("YOLOv5 finished! Ready to join!")

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
    
    # For standalone testing - NO backend library
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
        0,
        input_file='',
        use_backend=False  # CRITICAL: Disable backend for standalone testing
    )
