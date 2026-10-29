"""vLLM engine construction, shared by every offline generation script."""

import os
from types import SimpleNamespace


def stop_ids(model_id, tokenizer):
    """Gemma 4 ends a turn with `<turn|>` (106) while its `eos_token` is `<eos>` (1).

    Reuses the trainer's own resolution rather than restating it: getting this wrong
    is silent, and it already cost two 60-step runs at a gradient of exactly zero
    (`common.grpo.Trainer`).
    """
    from transformers import GenerationConfig

    from common.grpo import stop_token_ids
    try:
        config = GenerationConfig.from_pretrained(model_id)
    except OSError:
        config = GenerationConfig()
    allowed, template = stop_token_ids(SimpleNamespace(generation_config=config), tokenizer)
    if template is None:
        raise SystemExit("no special token ends an assistant turn in this chat template")
    return sorted(allowed)


def build(model, max_model_len=None, gpu_frac=None, seed=42, **kw):
    """An offline vLLM engine. `max_model_len` defaults to $MAX_MODEL_LEN, `gpu_frac` to
    $ENGINE_GPU_FRAC or 0.93, for an engine that has the card to itself. 0.93 rather than
    0.85 since 2026-10-26: KV cache bounds long-answer decoding, and at 0.85 Qwen3-4B held
    210k tokens on an L40S, ~25 sequences at an 8k window.

    $ENGINE_KV_CACHE_GIB fixes the KV cache size instead, for engines sharing one GPU
    (jobs/eval_pack.sh): sized from `gpu_frac`, vLLM profiles free memory at startup and
    counts a neighbour's concurrent allocations as its own -- three engines starting
    together each computed -11 GiB of KV cache and refused to start.
    """
    import vllm

    if gpu_frac is None:
        gpu_frac = float(os.environ.get("ENGINE_GPU_FRAC", "0.93"))
    if os.environ.get("ENGINE_KV_CACHE_GIB"):
        kw.setdefault("kv_cache_memory_bytes",
                      int(float(os.environ["ENGINE_KV_CACHE_GIB"]) * 2**30))
    if max_model_len is None:
        max_model_len = int(os.environ.get("MAX_MODEL_LEN", "2048"))
    return vllm.LLM(model=model, gpu_memory_utilization=gpu_frac,
                    max_model_len=max_model_len, enable_prefix_caching=True,
                    seed=seed, **kw)
