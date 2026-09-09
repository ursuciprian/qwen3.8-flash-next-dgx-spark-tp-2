"""GPU-only draft-head feasibility probe; synthetic weights, never a serving patch.

Measures the actual TP2 shard shape. Approximate FP8 rankings are not asserted
equal, and this cannot measure target acceptance or whole-model throughput.
"""
import json
import statistics
import time

import torch
import torch.nn.functional as F


def main():
    torch.manual_seed(19)
    with torch.inference_mode(), torch.device('cuda'):
        weight = torch.randn(124160, 2560, dtype=torch.bfloat16) * 0.02
        # Offline quantization. A serving experiment must own this copy separately
        # from the shared target head and account for its extra capacity.
        weight_scale = weight.float().abs().amax().clamp_min(1e-12) / 448
        packed = (weight.float() / weight_scale).to(torch.float8_e4m3fn).t()

        def fp8(x):
            scale = x.float().abs().amax().clamp_min(1e-12) / 448
            quantized = (x.float() / scale).to(torch.float8_e4m3fn)
            return torch._scaled_mm(quantized, packed, scale_a=scale,
                                    scale_b=weight_scale, out_dtype=torch.bfloat16,
                                    use_fast_accum=True)

        def refine8(x):
            # Approximate shortlist, original BF16 weights for the final ranking.
            # This can still miss a winner outside the shortlist; target verification
            # remains mandatory in any eventual speculative-decoding integration.
            ids = fp8(x).topk(8, dim=-1).indices
            rows = F.embedding(ids, weight)
            scores = torch.bmm(rows, x.unsqueeze(-1)).squeeze(-1)
            # Match argmax's lowest-ID tie rule within the shortlist.
            return torch.where(scores == scores.amax(-1, keepdim=True), ids,
                               weight.shape[0]).amin(-1)

        results = []
        for batch in (1, 4, 8):
            x = torch.randn(batch, 2560, dtype=torch.bfloat16)
            functions = [lambda: F.linear(x, weight).argmax(-1),
                         lambda: fp8(x).argmax(-1), lambda: refine8(x)]
            graphs = [torch.cuda.CUDAGraph() for _ in functions]
            for i, fn in enumerate(functions):
                for _ in range(4):
                    fn()
                torch.cuda.synchronize()
                with torch.cuda.graph(graphs[i]):
                    fn()
            samples = {'bf16': [], 'fp8': [], 'fp8_refine8': []}
            for iteration in range(16):
                order = list(zip(samples, graphs))
                if iteration % 2:
                    order.reverse()
                for name, graph in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    graph.replay()
                    torch.cuda.synchronize()
                    samples[name].append((time.perf_counter() - start) * 1000)
            # Independent hidden rows, preserving the timed batch-dependent scale.
            agreed = {'fp8': 0, 'fp8_refine8': 0}
            total = 0
            for _ in range(32):
                x = torch.randn(batch, 2560, dtype=torch.bfloat16)
                reference = functions[0]()
                for name, fn in zip(agreed, functions[1:]):
                    agreed[name] += (reference == fn()).sum().item()
                total += batch
            results.append({'batch': batch, 'agreement': {k: [v, total] for k, v in agreed.items()},
                            'median_ms': {k: statistics.median(v) for k, v in samples.items()},
                            'samples_ms': samples})
        print('FP8_HEAD_RESULT=' + json.dumps({
            'shape': [124160, 2560], 'packed_bytes': packed.numel(),
            'bf16_bytes': weight.numel() * 2,
            'scope': 'single-rank head only; synthetic weights and hidden states',
            'results': results}), flush=True)


if __name__ == '__main__':
    main()
