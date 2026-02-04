import argparse
import time

import torch
from torchvision import models


def run(model_name: str, batchsize: int, device: int = 0, do_eval: bool = True) -> None:
    data = torch.ones([batchsize, 3, 224, 224], pin_memory=True).to(device)
    target = torch.ones([batchsize], pin_memory=True).to(torch.long).to(device)

    model = models.__dict__[model_name](num_classes=1000).to(device)
    if do_eval:
        model.eval()
    else:
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.CrossEntropyLoss().to(device)

    torch.cuda.synchronize()

    # Single profiled iteration, wrapped in cudaProfilerStart/Stop.
    torch.cuda.profiler.cudart().cudaProfilerStart()
    if do_eval:
        with torch.no_grad():
            _ = model(data)
    else:
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.profiler.cudart().cudaProfilerStop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="torchvision model name, e.g. resnet50")
    parser.add_argument("--batchsize", required=True, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--train", action="store_true", help="profile training step instead of eval")
    args = parser.parse_args()

    # Reduce incidental overhead/variability.
    torch.backends.cudnn.benchmark = True

    run(args.model, args.batchsize, device=args.device, do_eval=not args.train)
