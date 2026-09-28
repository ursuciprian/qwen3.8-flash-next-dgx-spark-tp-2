# Quality evals, 2026-09-28 (b1.2)

Build under test: **b1.2**, the recommended recipe `qwen3.8-flash-next-2x-dgx-spark`
(image `ghcr.io/ursuciprian/spark-vllm-b12x:b1.2-20260927-b7fbaf96-a9aa81b2-warm`, vLLM `0.1.dev20865+ga9aa81b23`,
checkpoint `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` @ `7c4f1bc1`, both ranks verified on that snapshot).
Served by sparkrun 0.3.10 on two DGX Sparks, `max_num_seqs` 16, `max_model_len` 262144, MTP 4 drafts.

**Scope today.** The user narrowed the round twice. Only the DevOps set ran to completion.
Deferred at the user's request, to run later:

| Eval | Status |
|---|---|
| DevOps task set (14 prompts x 3) | **done on b1.2** (below) |
| MMLU-Pro, 2000-question stratified subset | **stopped at 770/2000 at the user's request**; no score (see below) |
| GSM8K (1319), IFEval (541) | deferred (user) |
| LiveCodeBench v6 window 2025-02-01..2025-04-06 (131 problems) | deferred (user); harness ready |
| Aider polyglot | **skipped at the user's request** |
| b1 comparisons (every eval) | deferred (user) |

## Settings (all runs)

- Thinking **on** (model default; the chat template injects `Reasoning effort is set to xhigh` and opens `<think>`).
  vLLM `--reasoning-parser qwen3` splits reasoning from the answer; verified: generation prompt ends in `<think>\n`,
  0 leaked think tags and 0 empty answers across 74 lm-eval smoke samples.
- Sampling from the model card's thinking-mode recommendation
  ([Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)): temperature 1.0, top_p 0.95, top_k 20,
  min_p 0, presence_penalty 0, repetition_penalty 1.0.
- Context limit 262144 (server). Generation caps: DevOps `max_tokens` 16384 (thinking cap, see below); lm-eval tasks 32768.

## DevOps task set

14 realistic prompts in [`devops/tasks/`](devops/tasks/), each graded by real validators plus deterministic rubric checks
([`devops/grade.py`](devops/grade.py)):
terraform 1.13.2 (`fmt -check`, `init -backend=false`, `validate`), kubeconform 0.8.0 `-strict` (k8s 1.30),
actionlint 1.7.12, shellcheck 0.11.0 + `bash -n` + a functional test with a stub `pg_dump` (help/exit codes, dry-run,
gzip naming, `--keep` retention, no partial file on failure), helm 3.15.1 `template`, hadolint 2.15.1, promtool 3.15.0.
Grader validation: hand-written reference answers pass 14/14; the broken inputs from the prompts pass 0/6.
An answer that never arrives (runaway thinking) fails every check.

Run: [`devops/run_devops.py`](devops/run_devops.py) on dgx-01 against `localhost:8000`, **one request at a time (c1),
nothing else on the server**, 3 repeats per prompt (seeds 1000-1002), streaming, so the time to the first answer
token (= end of thinking) and the total are measured end to end. Config chosen by Jev (below): `max_tokens` 16384 at
the default xhigh effort.

**Result (b1.2, 42 runs): 17/42 runs clean (every check passed), mean check score 42.5%.
Runaway thinking: 23/42 (55%)**, plus 2 answers cut off by the cap
(one of them still complete and clean). Of the 19 runs that produced an answer, 17 were clean and
94% of their checks passed. Wall-clock: median 246 s per prompt
(answered runs 160 s, of which thinking 155 s), max 273 s,
2.5 h for the 42 runs.

| Task | Mean checks passed | Clean runs | Runaway thinking | Median time (s) | Max time (s) | Median time to answer (s) | Median completion tokens | Failed checks (all runs) |
|---|---|---|---|---|---|---|---|---|
| 01-tf-s3-module | 0% | 0/3 | 3/3 | 238 | 242 | - | 16384 | answer present (runaway thinking) |
| 02-tf-fix-validate | 0% | 0/3 | 3/3 | 254 | 254 | - | 16384 | answer present (runaway thinking) |
| 03-k8s-deployment-fix | 100% | 3/3 | 0/3 | 219 | 247 | 215.03 | 14082 | - |
| 04-k8s-oomkilled | 100% | 3/3 | 0/3 | 124 | 186 | 121.8 | 8463 | - |
| 05-gha-fix | 0% | 0/3 | 3/3 | 253 | 257 | - | 16384 | answer present (runaway thinking) |
| 06-gha-oidc-terraform | 0% | 0/3 | 3/3 | 266 | 271 | - | 16384 | answer present (runaway thinking) |
| 07-iam-least-privilege | 0% | 0/3 | 2/3 | 255 | 264 | 247.66 | 16384 | answer present (runaway thinking); dynamodb scoped to table; findings flag PassRole escalation; kms scoped to key; logs scoped to function; no Resource *; no iam:PassRole; no service wildcards; policy JSON valid; s3 read scoped to incoming/; s3 write scoped to acme-thumbs |
| 08-bash-pg-backup | 0% | 0/3 | 3/3 | 262 | 264 | - | 16384 | answer present (runaway thinking) |
| 09-helm-values-fix | 100% | 3/3 | 0/3 | 62 | 85 | 58.62 | 4098 | - |
| 10-dockerfile-harden | 0% | 0/3 | 3/3 | 246 | 258 | - | 16384 | answer present (runaway thinking) |
| 11-tf-plan-review | 62% | 1/3 | 1/3 | 234 | 273 | 208.3 | 14417 | answer present (runaway thinking); safe path: revert identifier / snapshot / prevent_destroy |
| 12-k8s-networkpolicy | 100% | 3/3 | 0/3 | 160 | 187 | 154.74 | 10762 | - |
| 13-prometheus-rules | 33% | 1/3 | 2/3 | 244 | 249 | 74.14 | 16384 | answer present (runaway thinking) |
| 14-incident-db-pool | 100% | 3/3 | 0/3 | 220 | 262 | 175.26 | 14197 | - |
| **All (42 runs)** | **42.5%** | **17/42** | **23/42 (55%)** | 246 | 273 | | | |

**Runaway thinking** = the request hit `max_tokens` inside `<think>`; `</think>` was never emitted, so there is no answer.
It is not a repetition loop: 0% repeated 200-character chunks in the last 20k characters of every runaway. The model
drafts the full answer many times inside its thinking (5-56 code blocks per runaway) and keeps re-checking it against the
validator the prompt names ("must pass terraform fmt / actionlint / shellcheck / hadolint / promtool"), e.g. hand-computing
`terraform fmt` alignment. An earlier 32768-token pass (run under load) ran away on the first two tasks as well, so a larger
budget alone did not close the thinking. Only b1.2 was measured, so whether b1 behaves the same is open.

## MMLU-Pro (partial, no score)

2000-question subset stratified by subject (largest-remainder allocation, per-subject seeded sample, seed 20260928;
[`mmlu_pro/mmlu_pro_subset_2000_seed20260928.json`](mmlu_pro/mmlu_pro_subset_2000_seed20260928.json), generator
`mmlu_pro/mk_subset.py`), lm-eval 0.4.13 `mmlu_pro` 5-shot CoT as multi-turn chat, thinking on, `until` cleared so stop
strings cannot cut the reasoning, `max_gen_toks` 32768, concurrency 8.
Stopped by the user at **770/2000** after 3 h 10 min. **No partial score exists**: lm-eval writes samples and results only
when a run ends, and the run had no request cache. Requests that hit `max_gen_toks`: at most 26 of ~770 (<=3.4%; the server
counter mixes in two DevOps runaways and possibly other clients). Smoke run (3 per subject, 42 questions): 88.1% +- 4.8.
For a rerun: pass `--use_cache <path>` so a stop keeps the finished requests, and expect ~4000 generated tokens per question
on the long-prompt-first ordering (~220 tok/s aggregate at c8 -> ~10 h for 2000).

## Published full-precision reference (for the deferred evals)

| Eval | Published | Source | Why the comparison is inexact |
|---|---|---|---|
| MMLU-Pro | 73.23 | [tech report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf) Table 11 | **base** model, 5-shot CoT; ours is the post-trained NVFP4 model, thinking on |
| GSM8K | 93.29 | same Table 11 | base model, 4-shot CoT |
| IFEval | not published | | the card reports IFBench 81.3, a different benchmark |
| LiveCodeBench v6 | 91.9 | [model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | post-trained BF16, thinking; card gives no window or budget |
| Aider polyglot | not published | | |
| GPQA Diamond (context) | 91.7 BF16 / 89.9 NVFP4 QAD | model card / [NVFP4 card](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) | not run here |

## Jev decisions

| Decision | Verdict | File |
|---|---|---|
| MMLU-Pro subset size | n2000 0.91 (n1000 0.09, full 0.0) | [jev/setup.json](jev/setup.json) |
| b1 Aider / b1 LCB | skip 0.63 / skip 0.59 (later overridden by the user: Aider dropped, LCB and b1 deferred) | jev/setup.json |
| Pack evals to 16 in flight | pack16 0.64 | jev/setup.json |
| LCB max tokens | 32k 0.97 | jev/setup.json |
| DevOps config | xhigh + 16k cap 0.85 (medium 0.14, both 0.01); 3 repeats 1.0 | [jev/devops_config.json](jev/devops_config.json) |
| Overlap DevOps with MMLU-Pro | after 0.88 | [jev/devops_overlap.json](jev/devops_overlap.json) |
| **Final: b1.2 equal to b1?** | **undetermined 0.93** (worse 0.07, equal 0.0): no b1 run this round | [jev/final.json](jev/final.json) |
| **Final: acceptable vs full precision?** | **undetermined 0.62** (not acceptable 0.28, acceptable 0.10): no comparable scored benchmark finished | jev/final.json |
| Final: what DevOps shows | answer quality fine, runaway thinking is the failure, 1.0 | jev/final.json |
| Final: next step | b1 DevOps + b1.2 at lower effort / larger budget 0.61 (finish lm-eval first 0.36) | jev/final.json |

## Caveats

- One build only: no b1 comparison today, so none of this separates model behaviour from build numerics.
- Temperature 1.0 sampling: three repeats per DevOps prompt; per-task results vary between repeats.
- DevOps timings are c1 on an otherwise idle server (~67 tok/s decode).
- The rubric checks are deterministic regex/structure checks written for these prompts; they reward the specific
  fixes each prompt asks for and can miss an unusual but valid answer. The validators are the stricter half.

## Reproduce

- DevOps: `python3 devops/run_devops.py <build> --repeats 3` (dgx-01), then `uv run --with pyyaml python devops/grade.py <build>`
  (needs the validators above on PATH).
- lm-eval tasks: thinking-mode variants in [`lmeval_tasks/`](lmeval_tasks/) (`gsm8k_think`, `mmlupt`, `ifeval_think`);
  wrapper `run_lmeval.sh` (exact command in `mmlu_pro/b1.2/command.txt`).
- LiveCodeBench: commit 28fef95 + [`lcb/lcb-local.patch`](lcb/lcb-local.patch) (adds the local model, top_k/min_p, no-torch import);
  `release_v6 --start_date 2025-02-01 --end_date 2025-05-31 --n 1 --temperature 1.0 --top_p 0.95 --max_tokens 32768`.
  Run the client on a machine with RAM to spare: loading release_v6 on dgx-01 got the vLLM head worker killed (04:07).
