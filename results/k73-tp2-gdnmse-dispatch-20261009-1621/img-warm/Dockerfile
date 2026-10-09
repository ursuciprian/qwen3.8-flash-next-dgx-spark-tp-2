# k73 (#123): spark-vllm-b12x:k73-gdnmse-tp2-21e0b201-d21d7ade + the frozen TP=2 plan file (6fdfaa4cdd6c827f17390e8fa20bc97a36d7be3200a2d97ceb73f7b0577c050e.json, 682 records: b1.6 records for every shared key + the bake's
# GDN plans) and the torch AOT compile cache of the bake boots on both nodes (9 keys).
FROM spark-vllm-b12x:k73-gdnmse-tp2-21e0b201-d21d7ade
COPY preparation/ /opt/b12x-seed/preparation/
COPY torch_aot_compile/ /opt/b12x-seed/torch_compile_cache/torch_aot_compile/
