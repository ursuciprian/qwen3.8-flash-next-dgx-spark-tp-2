"""Small real vLLM TP head test. No checkpoint is loaded; timings are head-only."""
import datetime
import json
import os
import statistics
import time
from types import SimpleNamespace

import torch
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMTP
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


def main():
    rank, world = int(os.environ['RANK']), int(os.environ.get('WORLD_SIZE', '2'))
    mode = os.environ.get('HEAD_MODE', 'eager')
    assert mode in ('eager', 'graph')
    torch.cuda.set_device(0)
    init_distributed_environment(world, rank, 'env://', 0, timeout=datetime.timedelta(seconds=90))
    results = []
    config = VllmConfig(parallel_config=ParallelConfig(
        tensor_parallel_size=world, nnodes=world, node_rank=rank,
        distributed_executor_backend='mp'))
    with set_current_vllm_config(config), torch.inference_mode(), torch.device('cuda'):
        initialize_model_parallel(world)
        # Bypass the transformer constructor; exercise its real methods and real head.
        model = Qwen4ExpMTP.__new__(Qwen4ExpMTP)
        torch.nn.Module.__init__(model)
        probe = SimpleNamespace(use_local_argmax_reduction=True, model=model,
                                speculative_config=SimpleNamespace(draft_sample_method='greedy'))
        DraftModelSpeculator._validate_local_argmax_reduction(probe)
        for vocab, hidden in ((248320, 2560), (259, 32)):
            model.lm_head = ParallelLMHead(vocab, hidden, params_dtype=torch.bfloat16)
            model.logits_processor = LogitsProcessor(vocab)
            # Draft workers require a result on each rank; assert the baseline overlay.
            assert model.logits_processor.use_all_gather
            torch.manual_seed(19 + rank)
            model.lm_head.weight.normal_(std=0.02)
            for batch in (1, 4, 8):
                torch.manual_seed(71)
                x = torch.randn(batch, hidden, dtype=torch.bfloat16)
                full = lambda: model.compute_logits(x).argmax(dim=-1)
                local = lambda: model.get_top_tokens(x)
                assert torch.equal(full(), local()), (rank, vocab, batch)
                if vocab != 248320:
                    continue
                for _ in range(4):
                    full(); local()
                if mode == 'graph':
                    torch.cuda.synchronize()
                    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
                    with torch.cuda.graph(graphs[0]):
                        full_output = full()
                    with torch.cuda.graph(graphs[1]):
                        local_output = local()
                    full, local = graphs[0].replay, graphs[1].replay
                    full(); local()
                    assert torch.equal(full_output, local_output)
                samples = {'full': [], 'local': []}
                for iteration in range(16):
                    # Alternate order to reduce warm-cache/order bias.
                    order = [('full', full), ('local', local)]
                    if iteration % 2:
                        order.reverse()
                    for name, fn in order:
                        torch.distributed.barrier()
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        fn()
                        torch.cuda.synchronize()
                        elapsed = torch.tensor((time.perf_counter() - start) * 1000)
                        torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
                        samples[name].append(elapsed.item())
                results.append({'batch': batch, 'vocab': vocab, 'hidden': hidden,
                                'median_ms': {k: statistics.median(v) for k, v in samples.items()},
                                'samples_ms': samples})
            # Equal positive maxima across ranks, high IDs, and padded rows.
            head = model.lm_head
            for winning_ids in ((0, vocab - 1), (vocab - 1,)):
                head.weight.zero_()
                for token in winning_ids:
                    index = token - head.shard_indices.org_vocab_start_index
                    if 0 <= index < head.shard_indices.num_org_elements:
                        head.weight[index, 0] = 2
                padding = head.shard_indices.num_org_vocab_padding
                if padding:
                    head.weight[-padding:, 0] = 100
                x = torch.zeros(3, hidden, dtype=torch.bfloat16)
                x[:, 0] = 1
                expected = min(winning_ids)
                assert (model.compute_logits(x).argmax(-1) == expected).all()
                assert (model.get_top_tokens(x) == expected).all()
    if rank == 0:
        print('HEAD_RESULT=' + json.dumps({'world_size': world, 'parity': 'passed', 'timing': mode + ' head only', 'results': results}), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
