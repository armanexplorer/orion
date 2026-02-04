import argparse

import torch

from related.baselines.bert import modeling
from related.baselines.bert.optimization import BertAdam


def run(batchsize: int, device: int = 0, do_eval: bool = True) -> None:
    # Keep behavior consistent with the existing benchmark (profiling/benchmarks/bert.py):
    # eval -> large config, train -> base config.
    model_config_base = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 2,
        "vocab_size": 30522,
    }

    model_config_large = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "type_vocab_size": 2,
        "vocab_size": 30522,
    }

    config_dict = model_config_large if do_eval else model_config_base
    config = modeling.BertConfig.from_dict(config_dict)
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    input_ids = torch.ones((batchsize, 384), dtype=torch.int64, device=device)
    segment_ids = torch.ones((batchsize, 384), dtype=torch.int64, device=device)
    input_mask = torch.ones((batchsize, 384), dtype=torch.int64, device=device)
    start_positions = torch.zeros((batchsize), dtype=torch.int64, device=device)
    end_positions = torch.ones((batchsize), dtype=torch.int64, device=device)

    model = modeling.BertForQuestionAnswering(config).to(device)

    if do_eval:
        model.eval()
    else:
        model.train()
        param_optimizer = [n for n in list(model.named_parameters()) if "pooler" not in n[0]]
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=5e-5, warmup=0.1, t_total=100)

    torch.cuda.synchronize()

    # Warmup + a single profiled iteration at the end.
    for step in range(10):
        if step == 9:
            torch.cuda.profiler.cudart().cudaProfilerStart()

        if do_eval:
            with torch.no_grad():
                _ = model(input_ids, segment_ids, input_mask)
        else:
            optimizer.zero_grad(set_to_none=True)
            start_logits, end_logits = model(input_ids, segment_ids, input_mask)
            ignored_index = start_logits.size(1)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
            loss.backward()
            optimizer.step()

        if step == 9:
            torch.cuda.profiler.cudart().cudaProfilerStop()

    torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", required=True, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--train", action="store_true", help="profile training step instead of eval")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    run(args.batchsize, device=args.device, do_eval=not args.train)
