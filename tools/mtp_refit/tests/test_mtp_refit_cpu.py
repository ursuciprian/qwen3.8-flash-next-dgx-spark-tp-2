"""CPU tests for tools/mtp_refit on a tiny random checkpoint (hidden 64, 8 experts, 4 HC streams).

    PYTHONPATH=<b12x checkout> python -m pytest tools/mtp_refit/tests -q
"""
import gzip
import json
import os

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from tools.mtp_refit import gen, mtp_ref, parity, splice
from tools.mtp_refit.assemble import assemble, iter_windows
from tools.mtp_refit.eval_offline import evaluate, load_refit
from tools.mtp_refit.mtp_ref import (
    _attention,
    depth_targets,
    full_to_draft,
    load_mtp_ref,
    nvfp4_dequant,
    nvfp4_round_trip,
    trainable_names,
    window_loss,
)
from tools.mtp_refit.train import main as train_main

H, S, E, I, V, R = 64, 4, 8, 16, 300, 8
CFG = {
    "vocab_size": V, "hidden_size": H, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
    "head_dim": 16, "hidden_act": "silu", "rms_norm_eps": 1e-6, "moe_intermediate_size": I,
    "shared_expert_intermediate_size": I, "num_experts": E, "num_experts_per_tok": 2, "hc_count": S, "hc_lowrank": R,
    "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 16, "indexer_budget": 64,
    "indexer_compress_ratio": 4, "output_gate_type": "sigmoid", "ple_layer_ids": [2], "tie_word_embeddings": False,
    "layer_types": ["linear_attention", "full_attention"], "max_position_embeddings": 4096,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                        "mrope_section": [1, 1, 0], "mrope_interleaved": True},
    "mtp": {"num_hidden_layers": 1}, "model_type": "qwen4_exp_text",
}


def dense_shapes():
    hc = {"hc_norm.weight": [S * H], "input_mix_weight_down.weight": [R, S * H],
          "input_mix_weight_up.weight": [S * H, R]}
    out = {"mtp.fc_embedding.weight": [H, H], "mtp.fc_hidden.weight": [H, H],
           "mtp.pre_fc_norm_embedding.weight": [H], "mtp.pre_fc_norm_hidden.weight": [S * H]}
    out |= {f"mtp.hyper_connection_mixer.{k}": v for k, v in hc.items()}
    for blk in ("attn_hyper_connection", "mlp_hyper_connection"):
        out |= {f"mtp.layers.0.{blk}.{k}": v for k, v in hc.items()}
        out[f"mtp.layers.0.{blk}.block_inject_weight.weight"] = [S, S * H]
    a = "mtp.layers.0.self_attn."
    out |= {a + "q_proj.weight": [2 * 4 * 16, H], a + "k_proj.weight": [32, H], a + "v_proj.weight": [32, H],
            a + "o_proj.weight": [H, 64], a + "q_norm.weight": [16], a + "k_norm.weight": [16],
            a + "indexer.index_qk_proj.weight": [48, H], a + "indexer.q_layernorm.weight": [16],
            a + "indexer.k_layernorm.weight": [16]}
    m = "mtp.layers.0.mlp."
    out |= {m + "gate.weight": [E, H], m + "shared_expert.gate_proj.weight": [I, H],
            m + "shared_expert.up_proj.weight": [I, H], m + "shared_expert.down_proj.weight": [H, I],
            m + "shared_expert_gate.weight": [1, H]}
    return out


@pytest.fixture(scope="module")
def snap(tmp_path_factory):
    d = tmp_path_factory.mktemp("snap")
    g = torch.Generator().manual_seed(0)
    t = {"model.language_model.embed_tokens.weight": torch.randn(V, H, generator=g).bfloat16(),
         "lm_head.weight": (torch.randn(V, H, generator=g) * 0.2).bfloat16()}
    for n, shp in dense_shapes().items():
        scale = 0.02 if "norm" in n else 0.15
        t[n] = (torch.randn(*shp, generator=g) * scale).bfloat16()
    for e in range(E):
        for p, (o, i) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            pre = f"mtp.layers.0.mlp.experts.{e}.{p}."
            t[pre + "weight"] = torch.randint(0, 256, (o, i // 2), generator=g, dtype=torch.uint8)
            t[pre + "weight_scale"] = (torch.rand(o, i // 16, generator=g) + 0.5).to(torch.float8_e4m3fn)
            t[pre + "weight_scale_2"] = torch.tensor(0.05)
    t["model.language_model.layers.0.other.weight"] = torch.zeros(4).bfloat16()  # a non-MTP tensor
    names = sorted(t)
    files = {"model-00001-of-00002.safetensors": names[: len(names) // 2],
             "model-00002-of-00002.safetensors": names[len(names) // 2 :]}
    for f, ns in files.items():
        save_file({n: t[n] for n in ns}, d / f)
    json.dump({"weight_map": {n: f for f, ns in files.items() for n in ns}},
              open(d / "model.safetensors.index.json", "w"))
    json.dump({"text_config": CFG, "model_type": "qwen4_exp"}, open(d / "config.json", "w"))
    with gzip.open(d / "ids.txt.gz", "wt") as fh:
        fh.write("# test subset\n" + "\n".join(str(i) for i in range(0, V, 3)) + "\n")
    return d, t


def test_nvfp4_known_values():
    # byte 0x21: low nibble 1 (0.5) first, high nibble 2 (1.0); 0xF8: -0.0 then -6
    packed = torch.tensor([[0x21, 0xF8] + [0] * 6], dtype=torch.uint8)
    scale = torch.tensor([[2.0]]).to(torch.float8_e4m3fn)
    w = nvfp4_dequant(packed, scale, torch.tensor(0.5))
    assert w[0, :4].tolist() == [0.5, 1.0, 0.0, -6.0]
    # values on the E2M1 grid of one block survive the round trip exactly
    x = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 1.0]])
    assert torch.equal(nvfp4_round_trip(x), x)
    y = nvfp4_round_trip(torch.tensor([[0.74, 0.76, 2.5, 5.1] + [6.0] + [0.0] * 11]))
    assert y[0, :4].tolist() == [0.5, 1.0, 2.0, 6.0]  # ties as vLLM's cast_to_fp4


def test_load_and_export_identity(snap, tmp_path):
    d, t = snap
    out = tmp_path / "run"
    train_main(["--snapshot", str(d), "--draft-vocab", str(d / "ids.txt.gz"), "--data", str(tmp_path),
                "--epochs", "0", "--eval-windows", "0", "--device", "cpu", "--out", str(out)])
    r = load_file(out / "mtp_refit.safetensors")
    dense = {n for n in dense_shapes() if ".indexer." not in n}
    assert set(r) == dense
    for n in r:
        assert r[n].dtype == torch.bfloat16 and torch.equal(r[n].view(torch.int16), t[n].view(torch.int16)), n
    # the frozen parts hold the served values
    m = load_mtp_ref(str(d), str(d / "ids.txt.gz"))
    pre = "mtp.layers.0.mlp.experts.3.up_proj."
    up = nvfp4_dequant(t[pre + "weight"], t[pre + "weight_scale"], t[pre + "weight_scale_2"])
    assert torch.equal(m.layers[0].mlp.experts.gate_up_proj[3, I:], up.bfloat16())
    assert m.head.shape == (100, H) and m.draft_ids[:3].tolist() == [0, 3, 6]
    assert not any(p.requires_grad for n, p in m.named_parameters() if ".experts." in n or n == "embed_tokens.weight")
    # splice into a copy: headers unchanged, tensors replaced, other files hardlinked
    r = {n: (v.float() + 1).bfloat16() for n, v in r.items()}
    save_file(r, tmp_path / "refit.safetensors")
    rep = splice.splice(str(d), str(tmp_path / "refit.safetensors"), str(tmp_path / "out"))
    assert sum(s["tensors"] for s in rep["shards"].values()) == len(r)
    for f in rep["shards"]:
        assert splice.header(str(d / f)) == splice.header(str(tmp_path / "out" / f))
        with safe_open(tmp_path / "out" / f, "pt") as fh:
            for n in fh.keys():
                want = r[n] if n in r else t[n]
                assert torch.equal(fh.get_tensor(n).reshape(-1).view(torch.uint8), want.reshape(-1).view(torch.uint8)), n
    assert os.path.samefile(d / "config.json", tmp_path / "out" / "config.json")


def test_depth_targets_indexing():
    mask = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.uint8)  # response starts at position 3
    rows, valid = zip(*depth_targets(8, 3, mask))
    # depth 0: anchor t uses x[t+1], target row t+1, label x[t+2]; first drafting anchor is t=2 (x[3] generated)
    assert rows[0].tolist()[:7] == [1, 2, 3, 4, 5, 6, 7]
    assert valid[0].nonzero().flatten().tolist() == [2, 3, 4, 5]
    assert valid[2].nonzero().flatten().tolist() == [2, 3]
    assert rows[2][2] == 5  # anchor 2, depth 2 -> distribution of x[6] at row 5


def _feedback_f64(emb, multi, tok_w, state_w, emb_fc, hid_fc, eps=1e-6):
    """Float64 twin of b12x mtp_feedback.reference.feedback (the oracle itself is BF16-only)."""
    def norm(x, w):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * (1 + w)

    n, s, h = multi.shape
    return norm(multi.flatten(-2), state_w).view(n, s, h) @ hid_fc.T + (norm(emb, tok_w) @ emb_fc.T)[:, None]


@pytest.mark.parametrize("chain_kv", [False, True])
def test_unroll_matches_sequential_chain(snap, monkeypatch, chain_kv):
    """Parallel TTT unroll == a draft loop with an explicit KV cache, as vLLM drafts (float64, so any
    mask, position or state-carry error shows far above rounding). Served (chain_kv False): draft steps
    write their K/V but attend only the prefill rows 0..t."""
    d, _ = snap
    m = load_mtp_ref(str(d), None)
    m.chain_kv = chain_kv
    monkeypatch.setattr(mtp_ref, "feedback", _feedback_f64)
    m.double()
    m.compute_dtype = torch.float64
    with torch.no_grad():  # sharper attention, so the softmax weights depend on every key
        m.layers[0].self_attn.q_norm.weight.fill_(3.0)
        m.layers[0].self_attn.k_norm.weight.fill_(3.0)
    g = torch.Generator().manual_seed(1)
    n, depth = 12, 4
    tokens = torch.randint(0, V, (n,), generator=g)
    hidden = torch.randn(n, S * H, generator=g, dtype=torch.float64)
    pos = torch.arange(100, 100 + n)
    with torch.no_grad():
        par = m.unroll(tokens, hidden, pos, depth)
        for t in (0, 5, n - depth - 1):
            cache = []

            def attend(q, k, v):
                seen = cache if chain_kv or not cache else cache[:1]
                ks = torch.cat([c[0] for c in seen] + ([k] if chain_kv or not cache else []))
                vs = torch.cat([c[1] for c in seen] + ([v] if chain_kv or not cache else []))
                mask = torch.ones(q.shape[0], ks.shape[0], dtype=torch.bool).tril(ks.shape[0] - q.shape[0])
                return _attention(q, ks, vs, mask, [])

            state, sample, k, v = m.block(tokens[1 : t + 2], hidden[: t + 1], pos[: t + 1], attend)
            cache.append((k, v))
            torch.testing.assert_close(sample[-1], par[0][t], rtol=1e-9, atol=1e-9)
            state = state[-1:]
            for j in range(1, depth):
                state, sample, k, v = m.block(tokens[t + 1 + j : t + 2 + j], state, pos[t : t + 1] + j, attend)
                cache.append((k, v))
                torch.testing.assert_close(sample[0], par[j][t], rtol=1e-9, atol=1e-9)
        # the check has teeth: relative positions off by one move the drafts far beyond rounding
        assert not torch.allclose(m.unroll(tokens, hidden, pos + torch.arange(n) % 2, depth)[1][5], par[1][5])


def _capture_shards(tmp, docs, split_at):
    """Write hook-format shards: doc rows, the first doc split across shards 0 and 1."""
    tail = max(d["stored"] for d in docs)
    assert all(d["stored"] == min(d["tokens"].shape[0], tail) for d in docs)
    rows = {k: [] for k in ("hidden", "tokens", "positions", "topk_ids", "topk_logprobs", "req")}
    shards, reqs = [], []
    for i, doc in enumerate(docs):
        n = doc["tokens"].shape[0]
        lo = n - doc["stored"]
        for p in range(lo, n):
            if (i, p) == split_at:
                shards.append((rows, reqs))
                rows = {k: [] for k in rows}
                reqs = []
            if not reqs or reqs[-1]["id"] != f"r{i}":
                reqs.append({"id": f"r{i}", "prefill_len": n, "sha1": None})
            if p == n - 1:
                reqs[-1]["sha1"] = gen.token_sha1(doc["tokens"].tolist())
                if "drafts" in doc:
                    reqs[-1]["drafts"] = doc["drafts"]
                if "draft_topk" in doc:
                    reqs[-1]["draft_topk"] = doc["draft_topk"]
            rows["hidden"].append(doc["hidden"][p : p + 1])
            rows["tokens"].append(doc["tokens"][p : p + 1].int())
            rows["positions"].append(torch.tensor([p], dtype=torch.int32))
            rows["topk_ids"].append(doc["topk_ids"][p : p + 1])
            rows["topk_logprobs"].append(doc["topk_lp"][p : p + 1])
            rows["req"].append(torch.tensor([len(reqs) - 1], dtype=torch.int32))
    shards.append((rows, reqs))
    os.makedirs(tmp, exist_ok=True)
    for s, (r, q) in enumerate(shards):
        save_file({k: torch.cat(v) for k, v in r.items()}, os.path.join(tmp, f"shard-{s:05d}.safetensors"),
                  metadata={"format": "mtp-capture-v1", "requests": json.dumps(q), "tail": str(tail)})


def _docs(g, lens, stored, resp):
    docs = []
    for n, st, rs in zip(lens, stored, resp):
        lp = torch.randn(n, 5, generator=g).log_softmax(-1).sort(descending=True).values
        docs.append({"tokens": torch.randint(0, V, (n,), generator=g), "hidden": torch.randn(n, S * H, generator=g).bfloat16(),
                     "topk_ids": torch.randint(0, V, (n, 5), generator=g, dtype=torch.int32),
                     "topk_lp": lp.half(), "stored": st, "resp": rs})
    return docs


def test_shard_round_trip(tmp_path):
    g = torch.Generator().manual_seed(2)
    docs = _docs(g, [20, 9], [15, 9], [12, 4])
    _capture_shards(tmp_path / "cap", docs, split_at=(0, 11))
    with open(tmp_path / "manifest.jsonl", "w") as f:
        for i, d in enumerate(docs):
            f.write(json.dumps({"id": f"p{i}", "category": "code", "split": "train" if i == 0 else "heldout",
                                "sha1": gen.token_sha1(d["tokens"].tolist()), "response_start": d["resp"],
                                "length": d["tokens"].shape[0]}) + "\n")
    with open(tmp_path / "manifest.jsonl", "a") as f:  # an entry no shard has (rows lost before a flush)
        f.write(json.dumps({"id": "lost", "category": "code", "split": "train", "sha1": "0" * 40,
                            "response_start": 1, "length": 5}) + "\n")
    stats = assemble([str(tmp_path / "cap")], str(tmp_path / "manifest.jsonl"), str(tmp_path / "data"))
    assert stats["docs"] == 2 and stats["incomplete"] == 0 and stats["no_manifest"] == 0
    assert stats["manifest_unseen"] == 1
    for split, i in (("train", 0), ("heldout", 1)):
        (w,) = list(iter_windows(str(tmp_path / "data" / split), window=64))
        d, lo = docs[i], docs[i]["tokens"].shape[0] - docs[i]["stored"]
        assert w["id"] == f"p{i}" and w["positions"].tolist() == list(range(lo, d["tokens"].shape[0]))
        assert torch.equal(w["tokens"], d["tokens"][lo:].int())
        assert torch.equal(w["hidden"], d["hidden"][lo:])
        assert torch.equal(w["topk_ids"], d["topk_ids"][lo:]) and torch.equal(w["topk_logprobs"], d["topk_lp"][lo:])
        assert w["loss_mask"].tolist() == [int(p >= d["resp"]) for p in range(lo, d["tokens"].shape[0])]
    ws = list(iter_windows(str(tmp_path / "data" / "train"), window=8, stride=4))
    assert [w["positions"][0].item() for w in ws] == [5, 9, 13]
    assert ws[1]["loss_mask"][:4].tolist() == [0, 0, 0, 0]  # rows covered by the previous window


def test_loss_decreases_on_toy_batch(snap):
    d, _ = snap
    m = load_mtp_ref(str(d), str(d / "ids.txt.gz"))
    g = torch.Generator().manual_seed(3)
    n = 24
    w = {"tokens": torch.randint(0, V, (n,), generator=g), "hidden": torch.randn(n, S * H, generator=g).bfloat16(),
         "positions": torch.arange(n), "loss_mask": torch.tensor([0] * 4 + [1] * (n - 4), dtype=torch.uint8),
         "topk_ids": m.draft_ids[torch.randint(0, 100, (n, 5), generator=g)].int(),
         "topk_logprobs": torch.randn(n, 5, generator=g).log_softmax(-1).half()}
    names = trainable_names(m, "dense")
    params = [p for k, p in m.named_parameters() if k in names]
    assert all(p.requires_grad and p.dtype == torch.float32 for p in params)
    opt = torch.optim.AdamW(params, lr=3e-3)
    lookup = full_to_draft(m.draft_ids, V)
    losses = []
    for _ in range(25):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, counts, _ = window_loss(m, w, 3, [1.0, 1.0, 1.0], lookup)
        (loss / sum(counts)).backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss) / sum(counts))
    assert counts == [19, 18, 17]
    assert losses[-1] < 0.8 * losses[0], losses
    assert m.layers[0].mlp.experts.gate_up_proj.grad is None and m.head.grad is None


def test_eval_and_refit_overlay(snap, tmp_path):
    d, _ = snap
    g = torch.Generator().manual_seed(4)
    docs = _docs(g, [16], [16], [6])
    _capture_shards(tmp_path / "cap", docs, split_at=None)
    sha = gen.token_sha1(docs[0]["tokens"].tolist())
    (tmp_path / "m.jsonl").write_text(json.dumps({"id": "x", "category": "math", "split": "heldout", "sha1": sha,
                                                  "response_start": 6, "length": 16}) + "\n")
    assemble([str(tmp_path / "cap")], str(tmp_path / "m.jsonl"), str(tmp_path / "data"))
    m = load_mtp_ref(str(d), str(d / "ids.txt.gz"))
    res = evaluate(m, str(tmp_path / "data" / "heldout"), depth=3, window=64)
    assert set(res) == {"math", "all"} and res["all"]["anchors"] == [9, 8, 7]
    t1 = res["all"]["t1"]
    assert all(0 <= x <= 1 for x in t1["per_draft"]) and t1["per_position"][0] == t1["per_draft"][0]
    rep = res["all"]["replay"]  # one whole document: the step replay runs
    assert set(rep) == {"t0", "t1", "t1_hi"} and rep["t0"]["steps"] >= 1
    assert all(lo <= hi + 1e-6 for lo, hi in zip(rep["t1"]["per_position"], rep["t1_hi"]["per_position"]))
    p = tmp_path / "r.safetensors"
    save_file({"mtp.fc_hidden.weight": torch.zeros(H, H).bfloat16()}, p)
    assert load_refit(m, str(p)) == 1 and not m.fc_hidden.weight.any()


def test_gen_adapters_and_exclusion(tmp_path):
    row = {"trajectory": [
        {"role": "system", "system_prompt": "SYS", "text": None},
        {"role": "user", "text": "fix the bug"},
        {"role": "ai", "text": "Look first.\n\n```\nls -la\n```"},
        {"role": "user", "text": "a.py b.py"},
        {"role": "ai", "text": "Open it.\n\n```\nopen a.py\n```"},
        {"role": "user", "text": "1: x = 1"}]}
    import random

    p = gen.swe_agent(row, random.Random(0), 10_000)
    msgs = p["messages"]
    assert msgs[0] == {"role": "system", "content": "SYS"} and msgs[-1]["role"] in ("user", "tool")
    full = gen.swe_agent(row, type("R", (), {"choice": staticmethod(max)})(), 10_000)["messages"]
    assert [m["role"] for m in full] == ["system", "user", "assistant", "tool"]
    assert json.loads(full[2]["tool_calls"][0]["function"]["arguments"]) == {"command": "ls -la"}
    sys_text = ('Rules.\nHere is a list of functions in JSON format that you can invoke:\n[{"name": "f", '
                '"description": "d", "parameters": {"type": "dict", "properties": {}}}]. Trailer.')
    p = gen.toolace({"system": sys_text, "conversations": [{"from": "user", "value": "hi"}]}, None, 10_000)
    assert p["tools"][0]["function"]["parameters"]["type"] == "object"
    bench = tmp_path / "bench.py"
    bench.write_text('PROMPT = "please count the number of words in this sentence and then reply with the total count only"\n')
    excl = gen.exclusion_set([str(bench)])
    assert gen.shingles("Hi! Please count the number of words in this sentence and then reply with the total") & excl
    assert not gen.shingles("an unrelated prompt about sorting a list of integers in python quickly please") & excl
    with pytest.raises(SystemExit):
        gen.exclusion_set([str(tmp_path / "missing*")])


def test_parity_drafts_and_live(snap, tmp_path):
    d, _ = snap
    m = load_mtp_ref(str(d), str(d / "ids.txt.gz"))
    g = torch.Generator().manual_seed(5)
    (doc,) = _docs(g, [10], [10], [4])
    L, D = 10, 3
    chain = [int(m.draft_ids[7])]  # the "sampled" token, then the model's own greedy drafts
    for k in range(D):
        ext = torch.tensor(chain + [0] * (D - len(chain)))
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            sm = m.unroll(torch.cat([doc["tokens"], ext]), torch.cat([doc["hidden"], torch.zeros(D, S * H).bfloat16()]),
                          torch.arange(L + D), D)
        chain.append(int(m.draft_ids[m.logits(sm[k][L - 1 : L]).argmax(-1)]))
    doc["drafts"] = chain
    _capture_shards(tmp_path / "cap", [doc], split_at=None)
    (tmp_path / "m.jsonl").write_text(json.dumps({"id": "x", "category": "code", "split": "heldout", "length": L,
                                                  "sha1": gen.token_sha1(doc["tokens"].tolist()), "response_start": 4}) + "\n")
    assemble([str(tmp_path / "cap")], str(tmp_path / "m.jsonl"), str(tmp_path / "data"))
    args = type("A", (), dict(snapshot=str(d), draft_vocab=str(d / "ids.txt.gz"), device="cpu", experts_impl=None,
                              data=[str(tmp_path / "data" / "heldout")], max_len=64, max_docs=10, min_agree=0.995,
                              out=str(tmp_path / "p.json")))()
    assert parity.drafts(args) == 0
    assert json.load(open(tmp_path / "p.json"))["per_depth_agree"] == [1.0, 1.0, 1.0]
    doc["drafts"] = chain[:2] + [(chain[2] + 1) % V] + chain[3:]  # vLLM disagrees at depth 1
    _capture_shards(tmp_path / "cap2", [doc], split_at=None)
    assemble([str(tmp_path / "cap2")], str(tmp_path / "m.jsonl"), str(tmp_path / "data2"))
    args.data, args.out = [str(tmp_path / "data2" / "heldout")], str(tmp_path / "p2.json")
    assert parity.drafts(args) == 1
    mdir = tmp_path / "live"
    mdir.mkdir()
    def prom(acc, drafted):
        return "".join(f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{position="{i}"}} {a}\n'
                       f'vllm:spec_decode_num_draft_tokens_per_pos_total{{position="{i}"}} {b}\n'
                       for i, (a, b) in enumerate(zip(acc, drafted)))
    (mdir / "metrics-before.txt").write_text(prom([10, 5], [20, 20]))
    (mdir / "metrics-after.txt").write_text(prom([90, 69], [120, 120]))
    assert [round(x, 3) for x in parity.live_rates(str(mdir))] == [0.8, 0.64]


def test_stitch_dedupes_recomputed_rows():
    from tools.mtp_refit.assemble import ROW_KEYS, _complete, _stitch

    def piece(pos, tag):
        k = len(pos)
        return {"positions": torch.tensor(pos, dtype=torch.int32), "tokens": torch.full((k,), tag),
                **{x: torch.full((k, 2), tag) for x in ROW_KEYS if x not in ("positions", "tokens")}}

    pieces = [piece([2, 3, 4], 1), piece([4, 5], 2)]  # position 4 recomputed after a preemption
    assert _complete(pieces, length=6, start=2) and not _complete(pieces, length=6, start=1)
    assert not _complete([piece([2, 3, 5], 1)], length=6, start=2)
    doc = _stitch(pieces)
    assert doc["positions"].tolist() == [2, 3, 4, 5] and doc["tokens"].tolist() == [1, 1, 2, 2]


def test_validate_prompts(tmp_path):
    mix = {"total": 4, "categories": {"a": {"share": 0.5}, "b": {"share": 0.5}}}
    rows = {c: [{"id": f"{c}{i}", "category": c, "split": "train", "messages": [{"role": "user", "content": "x"}]}
                for i in range(2)] for c in "ab"}
    for c, ps in rows.items():
        gen._write_atomic(str(tmp_path / f"{c}.jsonl"), "".join(json.dumps(p) + "\n" for p in ps))
    assert gen.validate_prompts(str(tmp_path), mix) == ["MANIFEST.json: [Errno 2] No such file or directory: '"
                                                        + str(tmp_path / "MANIFEST.json") + "'"]
    man = {"categories": {c: {"count": 2, "sha256": gen._sha256(str(tmp_path / f"{c}.jsonl"))} for c in "ab"}}
    gen._write_atomic(str(tmp_path / "MANIFEST.json"), json.dumps(man))
    assert gen.validate_prompts(str(tmp_path), mix) == []
    with open(tmp_path / "b.jsonl", "a") as f:  # a torn write
        f.write('{"id": "b9", "categ')
    errs = gen.validate_prompts(str(tmp_path), mix)
    assert any("sha256" in e for e in errs) and any("b:3" in e for e in errs)


def _prom(path):
    out = {}
    for line in open(path):
        name, _, v = line.rpartition(" ")
        out[name] = float(v)
    return out


def test_metrics_files(snap, tmp_path):
    from tools.mtp_refit import p2_metrics

    d, _ = snap
    g = torch.Generator().manual_seed(5)
    docs = _docs(g, [16], [16], [6])
    _capture_shards(tmp_path / "cap", docs, split_at=None)
    sha = gen.token_sha1(docs[0]["tokens"].tolist())
    (tmp_path / "m.jsonl").write_text(json.dumps({"id": "x", "category": "code", "split": "train", "sha1": sha,
                                                  "response_start": 6, "length": 16}) + "\n")
    assemble([str(tmp_path / "cap")], str(tmp_path / "m.jsonl"), str(tmp_path / "data"))
    data, mf = str(tmp_path / "data" / "train"), tmp_path / "train.prom"
    train_main(["--snapshot", str(d), "--draft-vocab", str(d / "ids.txt.gz"), "--data", data, "--heldout", data,
                "--depth", "3", "--topk", "5", "--window", "64", "--epochs", "2", "--tokens-per-step", "4", "--eval-windows", "1",
                "--device", "cpu", "--out", str(tmp_path / "run1"), "--metrics-file", str(mf), "--save-every", "1"])
    from safetensors.torch import load_file
    ck = [load_file(tmp_path / "run1" / f"ckpt-step{i:06d}.safetensors") for i in (1, 2)]
    final = load_file(tmp_path / "run1" / "mtp_refit.safetensors")
    assert ck[1].keys() == final.keys() and all(torch.equal(ck[1][k], final[k]) for k in final)
    m = _prom(mf)
    lab = 'run="run1",phase="train"'
    assert m[f"mtp_refit_train_step{{{lab}}}"] == 2 and m[f"mtp_refit_train_epoch{{{lab}}}"] == 2
    rec = [json.loads(x) for x in open(tmp_path / "run1" / "train.jsonl")][-1]
    for k in range(3):
        kl = m[f'mtp_refit_train_kl{{{lab},position="{k + 1}"}}']
        assert kl >= -1e-4 and kl < m[f'mtp_refit_train_ce{{{lab},position="{k + 1}"}}']
        assert abs(kl - rec["kl"][k]) < 1e-4
    for when in ("before", "after"):
        assert 0 <= m[f'mtp_refit_acceptance{{{lab},drafter="{when}",category="all",mode="t1",kind="per_position",'
                      f'position="1"}}'] <= 1
    assert not list(tmp_path.glob("*.tmp.*"))

    # phase 2 sidecar on a fake job layout: partial last line ignored, results picked up
    res, dd = tmp_path / "res", tmp_path / "p2"
    for sub in ("prompts", "gen", "capture/shards", "capture/main"):
        (dd / sub).mkdir(parents=True)
    res.mkdir()
    (dd / "prompts" / "code.jsonl").write_text("{}\n{}\n{}\n")
    (dd / "gen" / "code.jsonl").write_text(json.dumps({"output_token_ids": [1, 2, 3]}) + "\n" + '{"output_tok')
    (dd / "capture" / "shards" / "s0.safetensors").write_bytes(b"x" * 10)
    (dd / "capture" / "main" / "manifest.jsonl").write_text("{}\n")
    (tmp_path / "STATE").write_text("DONE: parity drafts=PASS live: t0=PASS t1=FAIL (x)\n")
    json.dump(json.load(open(tmp_path / "run1" / "eval.json"))["after"], open(res / "eval-baseline.json", "w"))
    out = tmp_path / "p2.prom"
    p2_metrics.main(["--state", str(tmp_path / "STATE"), "--res", str(res), "--data", str(dd), "--out", str(out)])
    m = _prom(out)
    lab = 'run="res",phase="p2"'
    assert m[f'mtp_refit_p2_response_tokens{{{lab},set="main"}}'] == 3
    assert m[f'mtp_refit_p2_generated_docs{{{lab},set="main"}}'] == 1
    assert m[f"mtp_refit_p2_prompts_total{{{lab}}}"] == 3 and m[f"mtp_refit_p2_capture_bytes{{{lab}}}"] == 10
    assert m[f'mtp_refit_p2_captured_docs{{{lab},set="main"}}'] == 1 and m[f"mtp_refit_p2_done{{{lab}}}"] == 1
    assert m[f'mtp_refit_parity_pass{{{lab},check="live_t1"}}'] == 0
    assert any(k.startswith('mtp_refit_acceptance{run="res",phase="p2",drafter="shipped"') for k in m)


def test_replay_counts_steps_not_anchors():
    from tools.mtp_refit.eval_offline import replay

    # T=0: anchor 0 accepts 2 of 3 drafts -> next step at anchor 3 accepts all 3 -> next at 7 (past the end: done)
    acc = torch.zeros(7, 3)
    acc[0, :2] = 1
    acc[3] = 1
    acc[1:3] = 1  # anchors the steps skip over never count
    steps, hits = replay(acc, 0)
    assert steps == 2 and hits == [2, 2, 1]
    # expectation over a stochastic step: anchor 0 accepts depth 0 w.p. 0.5, then 1 or 2 continue
    acc = torch.tensor([[0.5, 0.0], [1.0, 0.0], [0.0, 0.0]])
    steps, hits = replay(acc, 0)
    # 0 -> 1 (rejected, p .5: accepts depth 0) or 2 (accepted 1, p .5: accepts none) -> 3 ends
    assert steps == 2.0 and hits == [1.0, 0.0]
    assert replay(torch.zeros(2, 3), 5) == (0.0, [0.0] * 3)


def test_qsa_mask_matches_transformers_indexer(snap):
    d, _ = snap
    m = load_mtp_ref(str(d), None).float()
    g = torch.Generator().manual_seed(7)
    n = 150  # budget 64 = 16 blocks of 4: rows past 66 select
    ix_mod = m.layers[0].self_attn.indexer
    with torch.no_grad():  # positive q.k, so relu never ties blocks at 0 and the top-k order is unique
        ix_mod.index_qk_proj.weight.abs_()
        ix_mod.q_layernorm.weight.fill_(1.0)
        ix_mod.k_layernorm.weight.fill_(1.0)
    x = torch.randn(n, H, generator=g).abs()
    pos = torch.arange(n)
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    with torch.no_grad():
        ix = m._index(x, pos)
        got = m.qsa_mask(ix[0], ix, causal)
        cos, sin = m.rotary(x, pos.view(1, 1, -1).expand(3, 1, -1))
        ref = m.layers[0].self_attn.indexer(x[None], (cos, sin), causal[None, None], None)[0, 0] & causal
    assert torch.equal(got, ref)
    assert torch.equal(got[:67], causal[:67]) and not torch.equal(got, causal)
    assert int(got[-1].sum()) == 64 + (n % 4)  # 16 blocks + the incomplete tail
    tokens = torch.randint(0, V, (40,), generator=g)
    hidden = torch.randn(40, S * H, generator=g)
    m = load_mtp_ref(str(d), None)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        off = m.unroll(tokens, hidden, torch.arange(40), 3)
        m.qsa_select = True
        on = m.unroll(tokens, hidden, torch.arange(40), 3)
    assert all(torch.equal(a, b) for a, b in zip(off, on))  # short window: dense = selection
    tokens = torch.randint(0, V, (100,), generator=g)
    hidden = torch.randn(100, S * H, generator=g)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        on = m.unroll(tokens, hidden, torch.arange(100), 3)
        m.qsa_select = False
        off = m.unroll(tokens, hidden, torch.arange(100), 3)
    assert all(torch.equal(a[:67], b[:67]) for a, b in zip(off, on)) and not torch.equal(off[2][-1], on[2][-1])


def test_capture_cut_records():
    from tools.mtp_refit.capture_client import cut_records

    recs = [{"id": "a", "split": "train", "prompt_token_ids": [1] * 10, "output_token_ids": list(range(50))},
            {"id": "b", "split": "train", "prompt_token_ids": [1] * 100, "output_token_ids": list(range(5))},
            {"id": "c", "split": "train", "prompt_token_ids": [1] * 10, "output_token_ids": [7]}]
    out = cut_records(recs, 40, 0)
    assert [r["id"].split("#")[0] for r in out] == ["a"]  # b: prompt too long, c: nothing to cut
    r = out[0]
    c = int(r["id"].split("#cut")[1])
    assert 1 <= c <= 30 and r["output_token_ids"] == list(range(c)) and r["split"] == "heldout"
    assert cut_records(recs, 40, 0) == out  # seeded


def test_parity_logit_criterion_and_noise(snap, tmp_path):
    d, _ = snap
    m = load_mtp_ref(str(d), str(d / "ids.txt.gz"))
    g = torch.Generator().manual_seed(5)
    (doc,) = _docs(g, [10], [10], [4])
    L, D = 10, 3
    chain, served = [int(m.draft_ids[7])], [[], []]
    for k in range(D):
        ext = torch.tensor(chain + [0] * (D - len(chain)))
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            sm = m.unroll(torch.cat([doc["tokens"], ext]), torch.cat([doc["hidden"], torch.zeros(D, S * H).bfloat16()]),
                          torch.arange(L + D), D)
        lg = m.logits(sm[k][L - 1 : L])[0]
        v, i = lg.topk(20)
        served[0].append(m.draft_ids[i].tolist())
        served[1].append(v.tolist())
        chain.append(int(m.draft_ids[i[0]]))
    args = dict(snapshot=str(d), draft_vocab=str(d / "ids.txt.gz"), device="cpu", experts_impl=None, max_len=64,
                max_docs=10, min_agree=0.995, kv_fp8="on", hc_mxfp8="off")
    (tmp_path / "m.jsonl").write_text(json.dumps({"id": "x", "category": "code", "split": "heldout", "length": L,
                                                  "sha1": gen.token_sha1(doc["tokens"].tolist()), "response_start": 4}) + "\n")
    json.dump({"delta": 1e-3}, open(tmp_path / "noise.json", "w"))

    def run(topk, name):
        doc["drafts"], doc["draft_topk"] = chain, topk
        _capture_shards(tmp_path / name, [doc], split_at=None)
        assemble([str(tmp_path / name)], str(tmp_path / "m.jsonl"), str(tmp_path / f"data-{name}"))
        a = type("A", (), dict(args, data=[str(tmp_path / f"data-{name}" / "heldout")], noise=str(tmp_path / "noise.json"),
                               dump=str(tmp_path / f"dump-{name}.json"), out=str(tmp_path / f"p-{name}.json")))()
        return parity.drafts(a), json.load(open(a.out))

    rc, res = run(served, "same")  # the served logits are the reference's own: error ~0, every anchor decisive
    assert rc == 0 and max(res["logit_rel_err"]) < 1e-3 and res["decisive_agree"] == [1.0] * D
    off = [served[0], [[x * 1.05 for x in row] for row in served[1]]]
    rc, res = run(off, "off")  # 5% logit error fails the 1e-2 bound even though every argmax agrees
    assert rc == 1 and min(res["logit_rel_err"]) > 1e-2 and res["decisive_agree"] == [1.0] * D
    dump = json.load(open(tmp_path / "dump-same.json"))
    other = {k: [x + 0.01 for x in v] for k, v in dump.items()}
    json.dump(other, open(tmp_path / "dump-b.json", "w"))
    a = type("A", (), dict(a=str(tmp_path / "dump-same.json"), b=str(tmp_path / "dump-b.json"),
                           out=str(tmp_path / "noise-out.json")))()
    assert parity.noise(a) == 0 and abs(json.load(open(a.out))["delta"] - 0.02) < 1e-6


def test_mxfp8_round_trip_and_hc():
    from tools.mtp_refit.mtp_ref import mxfp8_round_trip, serve_hc_mxfp8

    w = torch.randn(8, 64, generator=torch.Generator().manual_seed(3)) * 0.05
    q = mxfp8_round_trip(w)
    assert torch.equal(mxfp8_round_trip(q), q)  # values on the grid stay
    assert float((q - w).norm() / w.norm()) < 2 ** -4 and not torch.equal(q, w)
    blk = w[0, :32].abs().max()  # e8m0 scale: the block max lands within the e4m3 range without saturating
    assert float(q[0, :32].abs().max()) <= 448 * 2 ** float(torch.ceil(torch.log2(blk / 448)))


def test_serve_hc_mxfp8_touches_only_hc(snap):
    from tools.mtp_refit.mtp_ref import serve_hc_mxfp8

    d, _ = snap
    m = load_mtp_ref(str(d), None)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    assert serve_hc_mxfp8(m) == 5  # down x3 + inject x2 (in 256); up has in = hc_lowrank 8, not a block of 32
    changed = {n for n, p in m.named_parameters() if not torch.equal(p, before[n])}
    assert changed and all(any(k in n for k in ("input_mix_weight", "block_inject")) for n in changed)
