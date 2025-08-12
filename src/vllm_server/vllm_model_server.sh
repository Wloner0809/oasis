CUDA_VISIBLE_DEVICES=6,7 vllm serve "/NAS/terencewang/model/Qwen2.5-7B-Instruct" \
    --host 0.0.0.0 \
    --port 8081 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --dtype auto \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    # --rope-scaling '{"rope_type":"dynamic","factor":2.0}' \
    # --max-model-len 65536 


# CUDA_VISIBLE_DEVICES=7 vllm serve "/NAS/terencewang/model/Qwen2.5-7B-Instruct" \
#     --host 0.0.0.0 \
#     --port 8082 \
#     --tensor-parallel-size 1 \
#     --gpu-memory-utilization 0.9 \
#     --enable-prefix-caching \
#     --dtype auto \
#     --enable-auto-tool-choice \
#     --tool-call-parser hermes \
    # --rope-scaling '{"rope_type":"dynamic","factor":2.0}' \
    # --max-model-len 65536 \