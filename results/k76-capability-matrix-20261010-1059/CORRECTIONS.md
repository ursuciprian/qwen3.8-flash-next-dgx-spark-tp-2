# k76 corrections

The raw files in this folder are left as written by the run. Four metadata fields that `k76.py` parsed from the recipe
and the serve log are wrong. The correct values are below; `docs/data/capability.csv` already uses them.

| Field | Where | As written | Correct |
|---|---|---|---|
| Release of the two-Spark run | `2x/recipe.json` `release`, `k76.txt` `[2x]` line, `capability.csv` here | `""` (alias `k73-20261009-21e0b201-d21d7ade-warm`) | `v2.0.0` (no old name) |
| Default reasoning effort, two-Spark | `2x/recipe.json` `reasoning_effort_default` | `xhigh` | `medium` (the recipe passes `--default-chat-template-kwargs '{"reasoning_effort":"medium"}'`) |
| Image digest, two-Spark | `k76.txt` `[2x]` line, `2x/recipe.json` `recipe_digest_comment` | `sha256:ed8582a1...` (the v1.5.0 / b1.6 tag) | `sha256:8abb60eeeb79ca13680341c17d87975801da1218c06df2505856b23da7ed1821` (VERSIONS.md, 2× v2.0.0) |
| vLLM commit | `k76.txt`, `manifest.json` `vllm_commit` (every setup) | `884b4ff68` | two-Spark `d21d7ade`, one-Spark and DP=2 `5dad364d` (the image tags and VERSIONS.md manifests) |

Why the parser got them wrong:

- **Release:** `recipe_facts()` matches only `# Release: vX.Y.Z (old name ...)`. The two-Spark recipe header is
  `# Release: v2.0.0 (k73-2x-gdnmse-dispatch)`, which has no old name, so the match failed and the image tag was used as
  the alias with an empty release.
- **Reasoning effort:** the parser takes the first `"reasoning_effort": "<word>"` in the recipe text. In the two-Spark
  recipe that is a comment listing the accepted values (`{"reasoning_effort": "xhigh"|"medium"|"low"}`), which comes
  before the `--default-chat-template-kwargs` flag that sets `medium`.
- **Digest:** the parser looks for a `# ... Digest of <label>:` comment whose label contains the first part of the
  image tag (`k73`). None does; it falls back to the last `Digest of` comment, which is the b1.6 tag. The v2.0.0 digest is
  on a line written as `Digest: sha256:...`, which the pattern does not match. The other `Digest of` comments in the
  recipe belong to its history notes on the b1.4 and b1.6 images.
- **vLLM commit:** the serve log prints the package version string `v20260926.dev28+g884b4ff68.d20261002`, which is
  the same in both images. It is the version stamp of the installed vLLM build, not the commit of the branch each image
  was built from; the image tags name those commits.

None of these fields changes a measured value.
