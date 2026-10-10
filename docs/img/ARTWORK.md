# Banner artwork

`hero.png` is the two-Spark edition of the Qwen Flash Next banner. Its connected
pair depicts TP=2; the paired repository uses matching artwork with one Spark.

The banner was generated with OpenAI ImageGen using these visual references:

- Qwen's emblem from the [official Qwen3 repository](https://github.com/QwenLM/Qwen3),
  [published logo image](https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/logo_qwen3.png).
- NVIDIA's [DGX Spark product image](https://www.nvidia.com/content/nvidiaGDC/tw/en_TW/gtc/taipei/computex/sweepstakes/_jcr_content/root/responsivegrid/nv_container_copy/nv_container_copy/nv_teaser_copy.coreimg.jpeg/1779347041639/dgx-spark-1920x1080.jpeg),
  from its [official product family](https://www.nvidia.com/en-us/products/workstations/dgx-spark/).

Violet token trails convey inference speed; they do not represent a benchmark.
Release numbers and measured performance stay in the README's generated data blocks.
The artwork has its own dark background and works in both GitHub themes.

`scripts/make_charts.py` updates the banner's README markup but preserves the PNG.
Replace the PNG directly when revising the artwork; retain the 3:1 composition and
check text readability at the README's 840-pixel display width.
