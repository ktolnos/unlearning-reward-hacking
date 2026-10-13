"""vLLM engine construction, shared by every offline generation script."""

import os


def build(model, max_model_len=None, gpu_frac=None, seed=42, **kw):
    """An offline vLLM engine. `max_model_len` defaults to $MAX_MODEL_LEN, `gpu_frac` to
    $ENGINE_GPU_FRAC or 0.85.

    $ENGINE_KV_CACHE_GIB fixes the KV cache size instead, for engines sharing one GPU
    (jobs/eval_pack.sh): sized from `gpu_frac`, vLLM profiles free memory at startup and
    counts a neighbour's concurrent allocations as its own -- three engines starting
    together each computed -11 GiB of KV cache and refused to start.
    """
    import vllm

    if gpu_frac is None:
        gpu_frac = float(os.environ.get("ENGINE_GPU_FRAC", "0.85"))
    if os.environ.get("ENGINE_KV_CACHE_GIB"):
        kw.setdefault("kv_cache_memory_bytes",
                      int(float(os.environ["ENGINE_KV_CACHE_GIB"]) * 2**30))
    if max_model_len is None:
        max_model_len = int(os.environ.get("MAX_MODEL_LEN", "2048"))
    return vllm.LLM(model=model, gpu_memory_utilization=gpu_frac,
                    max_model_len=max_model_len, enable_prefix_caching=True,
                    seed=seed, **kw)
