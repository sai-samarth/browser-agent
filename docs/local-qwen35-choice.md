# Local vLLM Qwen3.5 choice

Chosen model: `Qwen/Qwen3.5-9B`

Why this choice:
- `Qwen3.5-27B-FP8` is too large for a single 24 GB RTX 4090 if we want comfortable KV-cache headroom and some concurrency.
- `Qwen3.5-9B` appears to be the best balance of capability and fit for a single 4090.
- Qwen3.5 documentation and model cards indicate native long context, but for BrowserGym traces we can cap serving context to 32k and preserve VRAM for KV cache.
- Official small-model FP8 options were not clearly available for Qwen3.5 in the same way as the 27B FP8 release, so this deployment uses the official 9B model with:
  - bfloat16 weights
  - fp8 KV cache in vLLM
  - capped max model length
  - prefix caching

Serving settings:
- port: `7999`
- served model name: `Qwen3.5-9B-local`
- `--kv-cache-dtype fp8`
- `--gpu-memory-utilization 0.92`
- `--max-model-len 32768`
- `--max-num-seqs 8`
- `--reasoning-parser qwen3`
- `--default-chat-template-kwargs {"enable_thinking": true}`
