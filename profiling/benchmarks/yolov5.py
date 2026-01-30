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