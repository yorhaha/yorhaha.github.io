---
LLM推理部署
---

# LLM推理部署

## vLLM

```sh
docker run \
    --runtime nvidia \
    --gpus \"device=0,1,2,3\" \
    --rm \
    -v $(pwd):/vllm-workspace \
    -p 12001:12001 \
    vllm/vllm-openai:latest \
    --model /vllm-workspace/models/Qwen3-30B-A3B \
    --served-model-name Qwen3-30B-A3B \
    --dtype auto \
    --api-key sk-11223344 \
    --port 12001 \
    --seed 0 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --disable-log-requests \
    --max-model-len 20000 \
```

For Qwen3:

```
--enable-reasoning --reasoning-parser deepseek_r1
```
