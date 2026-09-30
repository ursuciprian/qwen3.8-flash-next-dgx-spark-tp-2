# r7 dvocab v2 id lists

`VLLM_MTP_DRAFT_VOCAB` lists for the MTP draft head, built 2026-09-30 by
`scripts/build_draft_vocab_v2.py` on dgx-01 `results/`: corpus-code.txt,
corpus-prose.txt, DevOps eval tasks + responses (evals-20260928), gsm8k/mmlu_pro
samples, fidelity_probe transcripts (depths 8k/32k/64k, seeds 7/11/13) and the
llama-benchy task prompts. Tokenizer: Qwen3.8-Flash-Next-NVFP4 rev 7c4f1bc1.
Forced: 33 added tokens, 256 byte-level symbols, all 8,883 single-character tokens
(8,916 ids). Unseen ids follow the r6 list order, then ascending id, so K98304 and
K131072 contain the whole r6 K65536 list.

| file | sha256 |
|---|---|
| ids-v2-K65536.txt.gz | 0d015e762fe94bee4ee52403aeb315f456e5206067789fc93030bf6bd9aed284 |
| ids-v2-K98304.txt.gz | 396cdfa7872aad2b9b19f8f9bebd11df461e4a0ef30f2d5b850202658eaf56b8 |
| ids-v2-K131072.txt.gz | fc40970533741fc14217879a5f5385897f4126bd1d7ce23ed0268236b4b8cf7b |

Held-out coverage (`cov-heldout.txt`: lists rebuilt without DevOps tasks 12-14 and
the last 20% of corpus-code, then scored on exactly those):

| list | DevOps outputs 12-14 | corpus-code tail |
|---|---|---|
| r6 K65536 | 0.9732 | 0.9924 |
| v2 K65536 | 0.9877 | 0.9958 |
| v2 K98304 | 0.9986 | 0.9989 |
| v2 K131072 | 0.9997 | 0.9998 |

Build (shipped lists, all data):

    build_draft_vocab_v2.py --tokenizer tokenizer.json \
      --src 'outputs:0.45:evals-20260928/devops/responses/**/*.json' \
      --src 'code:0.30:corpus-code.txt' --src 'prose:0.10:corpus-prose.txt' \
      --src 'prompts:0.10:evals-20260928/devops/tasks/**/*,prompts/*.json' \
      --src 'evalsamp:0.05:evals-20260928/gsm8k/**/*.jsonl,evals-20260928/mmlu_pro/**/*.jsonl' \
      --prior ids-K65536.txt.gz --out-dir out --tag v2
