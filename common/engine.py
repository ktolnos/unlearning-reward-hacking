"""vLLM engine construction, shared by every offline generation script."""

import os


def build(model, max_model_len=None, gpu_frac=0.85, seed=42, **kw):
    """An offline vLLM engine. `max_model_len` defaults to $MAX_MODEL_LEN."""
    import vllm

    if max_model_len is None:
        max_model_len = int(os.environ.get("MAX_MODEL_LEN", "2048"))
    return vllm.LLM(model=model, gpu_memory_utilization=gpu_frac,
                    max_model_len=max_model_len, enable_prefix_caching=True,
                    seed=seed, **kw)
