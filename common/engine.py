"""vLLM engine construction, shared by every offline generation script."""

import os
from pathlib import Path
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


# Eval generations stop a completion stuck in a loop: vLLM ends it with finish_reason
# "repetition" once its last 16 x p tokens are one p-token pattern repeated 16 times
# (4 <= p <= 16), and the evals count it as truncated (CUT), which is where it was going.
# Measured offline on 2.77M saved completions (2026-11-26): it fires on 258 of the 2.66M
# that ended on their own (none in math; nearly all in broken snapshots that rambled to
# an end) and saves 34M of the 327M tokens spent in truncated ones -- most of a looping
# snapshot's eval (Qwen multihop retain arm at 128: ~90%). It does not catch rambling
# that never loops exactly (the Qwen multihop KL 0.025 cap-5 arm at 256). Patterns up to
# 16 rather than 64 keep 85% of the saving at a fifth of the scheduler's per-step cost
# (one tail comparison per pattern length per sequence, ~0.25 us each).
REPETITION = dict(min_pattern_size=4, max_pattern_size=16, min_count=16)
CUT = ("length", "repetition")  # finish reasons counted as truncated


def repetition_detection():
    from vllm.sampling_params import RepetitionDetectionParams
    return RepetitionDetectionParams(**REPETITION)


def model_config(model):
    """`model`'s config.json as a dict: a local checkpoint's, or a hub id's from the cache
    (or the hub); {} when there is none."""
    import json
    try:
        cfg = Path(model) / "config.json"
        if not cfg.exists():
            from huggingface_hub import hf_hub_download
            cfg = Path(hf_hub_download(str(model), "config.json"))
        return json.loads(cfg.read_text())
    except Exception:  # noqa: BLE001 -- no config found
        return {}


def _venv_toolkit():
    """The venv's own CUDA toolkit (site-packages/nvidia/cu13, from vLLM's cu13 extra), or
    None unless nvcc, nvvm and crt are all the release of its headers: any mismatch (13.4
    pieces beside torch's 13.0 headers before pyproject.toml pinned them) fails
    FlashInfer's build at engine start."""
    import importlib.util
    import re
    import subprocess
    spec = importlib.util.find_spec("nvidia")
    for root in (spec.submodule_search_locations if spec else None) or []:
        home = Path(root, "cu13")
        nvcc, header = home / "bin" / "nvcc", home / "include" / "cuda.h"
        if not (nvcc.exists() and header.exists()):
            continue
        try:
            out = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True,
                                 timeout=30).stdout
            rel = re.search(r"release (\d+)\.(\d+)", out)
            ver = int(re.search(r"#define CUDA_VERSION (\d+)", header.read_text()).group(1))
        except Exception:  # noqa: BLE001 -- unusable: as if absent
            continue
        want = (ver // 1000, ver % 1000 // 10)
        if not rel or (int(rel.group(1)), int(rel.group(2))) != want:
            continue
        # nvcc's pieces ship separately: nvvm writes the PTX that nvcc's ptxas assembles,
        # and a newer nvvm fails every kernel ("Unsupported .version 9.4", job 6993)
        from importlib.metadata import PackageNotFoundError, version
        try:
            parts = [version(p) for p in ("nvidia-cuda-nvcc", "nvidia-nvvm", "nvidia-cuda-crt")]
        except PackageNotFoundError:
            continue
        if all(tuple(int(x) for x in v.split(".")[:2]) == want for v in parts) \
                and _linkable(home):
            return home
    return None


def _linkable(home):
    """Give the wheel toolkit the layout FlashInfer links against (-L$CUDA_HOME/lib64
    -lcudart): the wheels ship only lib/libcudart.so.13, so add lib64 -> lib and
    libcudart.so -> libcudart.so.13 (job 6993: "cannot find -lcudart"). libcuda comes from
    the driver, already on the compute nodes' linker path."""
    for link, target in ((home / "lib64", "lib"),
                         (home / "lib" / "libcudart.so", "libcudart.so.13")):
        if link.exists():
            continue
        try:
            link.symlink_to(target)
        except FileExistsError:  # another process made it meanwhile
            pass
        except OSError:
            return False
    return (home / "lib64" / "libcudart.so").exists()


def _have_nvcc():
    """Whether vLLM's FlashInfer can JIT its kernels: the venv's own toolkit, which this
    then exports as CUDA_HOME (FlashInfer reads it; vLLM's engine processes inherit it),
    else a system nvcc. The venv's comes first, over any CUDA_HOME already set: it is
    torch's own CUDA release, while a system toolkit can be newer than the driver
    (Tidewater: cvmfs has 13.2, the driver 13.0). With the venv's toolkit FlashInfer
    generated 46% (Qwen) and 7% (Gemma) faster than FlashAttention 4 / Triton on a B200
    (2026-11-24)."""
    import shutil
    venv = _venv_toolkit()
    if venv is not None:
        os.environ["CUDA_HOME"] = str(venv)
        return True
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    return bool(shutil.which("nvcc") or Path(home, "bin", "nvcc").exists())


def _compute_capability():
    """The first visible GPU's (major, minor), from nvidia-smi rather than torch, so this
    process does not initialise CUDA before vLLM spawns its engine."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.split()
        major, minor = out[0].split(".")
        return int(major), int(minor)
    except Exception:  # noqa: BLE001 -- no GPU visible: leave the choice to vLLM
        return None


def attention_backend(model, spec=None):
    """The attention backend to ask vLLM for, or None for vLLM's own choice.

    `spec` ($ENGINE_ATTENTION_BACKEND) overrides: entries are `model_type=BACKEND` or a
    bare BACKEND, the default. Without it: vLLM's choice wherever a CUDA toolkit is
    installed or the venv's own is usable (`_have_nvcc`). On Blackwell without one, vLLM takes FlashInfer, which JIT-builds its
    attention kernels with nvcc and fails at engine start (Tidewater, 2026-11-24), so ask
    for FlashAttention (4), except above head dim 128, where FA4 asserts on Gemma 4's
    head-dim-256 paged KV cache and Triton is used. Elsewhere vLLM already picks a backend
    that needs no compiler (FlashAttention on Hopper).
    """
    cfg = model_config(model)
    if spec:
        default = None
        for entry in (e.strip() for e in spec.split(",") if e.strip()):
            key, sep, val = entry.partition("=")
            if not sep:
                default = entry
            elif key.strip() == cfg.get("model_type"):
                return val.strip()
        return default
    if _have_nvcc():
        return None
    cap = _compute_capability()
    if cap is None or cap[0] < 10:
        return None
    text = cfg.get("text_config", cfg)
    head = max(text.get("head_dim") or 0, text.get("global_head_dim") or 0)
    return "TRITON_ATTN" if head > 128 else "FLASH_ATTN"


def _stable_path(model):
    """A path for `model` that is the same from one eval to the next.

    vLLM keys its torch.compile cache on the model's path, so every snapshot directory
    compiled afresh (30-60 s an engine, 30 cache entries used once each, 2026-11-24). A
    local checkpoint is opened through a symlink named by its model type and a slot, the
    slot locked for this process's life so concurrent engines in one job never share it;
    /tmp is private to each job, so the next job finds the same names. Hub ids are left
    alone (their path does not change).
    """
    import atexit
    import shutil
    src = Path(model)
    if not (src / "config.json").exists():
        return model
    base = Path(os.environ.get("SLURM_TMPDIR") or "/tmp") / "vllm-models"
    base.mkdir(parents=True, exist_ok=True)
    mtype = model_config(model).get("model_type", "model")
    for slot in range(64):
        lock = base / f"{mtype}-{slot}.lock"
        try:
            lock.mkdir()
        except FileExistsError:
            try:  # reclaim a slot whose holder is gone
                os.kill(int((lock / "pid").read_text()), 0)
                continue
            except (OSError, ValueError):
                shutil.rmtree(lock, ignore_errors=True)
                try:
                    lock.mkdir()
                except FileExistsError:
                    continue
        (lock / "pid").write_text(str(os.getpid()))
        link = base / f"{mtype}-{slot}"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src.resolve(), target_is_directory=True)
        atexit.register(shutil.rmtree, lock, True)
        return str(link)
    return model


def build(model, max_model_len=None, gpu_frac=None, seed=42, **kw):
    """An offline vLLM engine for `model` (`_build_local`), or, with $ENGINE_SERVER set, a
    handle on the engine an engine server holds (`serve`), with `model`'s weights loaded
    into it. Callers use either the same way: `llm.generate(...)`."""
    if os.environ.get("ENGINE_SERVER"):
        return _Remote(os.environ["ENGINE_SERVER"], model, max_model_len, gpu_frac, seed, kw)
    return _build_local(model, max_model_len, gpu_frac, seed, **kw)


def _build_local(model, max_model_len=None, gpu_frac=None, seed=42, **kw):
    """An offline vLLM engine. `max_model_len` defaults to $MAX_MODEL_LEN, `gpu_frac` to
    $ENGINE_GPU_FRAC or 0.93, for an engine that has the card to itself. 0.93 rather than
    0.85 since 2026-10-26: KV cache bounds long-answer decoding, and at 0.85 Qwen3-4B held
    210k tokens on an L40S, ~25 sequences at an 8k window.

    $ENGINE_KV_CACHE_GIB fixes the KV cache size instead, for engines sharing one GPU
    (a persistent engine beside a replay): sized from `gpu_frac`, vLLM profiles free memory at startup and
    counts a neighbour's concurrent allocations as its own -- three engines starting
    together each computed -11 GiB of KV cache and refused to start.

    The attention backend is chosen to run without a CUDA toolkit (`attention_backend`),
    a multimodal model runs text-only (every eval here is text; Gemma 4's warmup of
    its image, audio and video processing took 24 s an engine), and a local checkpoint is opened
    through a stable path so the compile cache is reused (`_stable_path`).
    """
    import vllm

    if gpu_frac is None:
        gpu_frac = float(os.environ.get("ENGINE_GPU_FRAC", "0.93"))
    if os.environ.get("ENGINE_KV_CACHE_GIB"):
        kw.setdefault("kv_cache_memory_bytes",
                      int(float(os.environ["ENGINE_KV_CACHE_GIB"]) * 2**30))
    # Throughput only: how many sequences one engine step may decode. vLLM's default
    # suits a card it shares with nothing; raise it on a large one if the KV cache allows.
    if os.environ.get("ENGINE_MAX_NUM_SEQS"):
        kw.setdefault("max_num_seqs", int(os.environ["ENGINE_MAX_NUM_SEQS"]))
    if "attention_backend" not in kw:
        if backend := attention_backend(model, os.environ.get("ENGINE_ATTENTION_BACKEND")):
            kw["attention_backend"] = backend
    # Every modality off at once: Gemma 4 also takes video, which a per-modality
    # limit_mm_per_prompt missed, and one modality left on keeps the warmup.
    cfg = model_config(model)
    if (cfg.get("vision_config") or cfg.get("audio_config")) \
            and "limit_mm_per_prompt" not in kw:
        kw.setdefault("language_model_only", True)
    if max_model_len is None:
        max_model_len = int(os.environ.get("MAX_MODEL_LEN", "2048"))
    return vllm.LLM(model=_stable_path(model), gpu_memory_utilization=gpu_frac,
                    max_model_len=max_model_len, enable_prefix_caching=True,
                    seed=seed, **kw)


# --- a persistent engine ------------------------------------------------------------
# Every eval stage is its own process that builds an engine for one checkpoint: imports,
# CUDA graphs, compile cache lookup and the weights each time, 50-70% of a creatures eval
# (2026-11-24). An engine server keeps one engine per GPU and, per request, loads the
# requested checkpoint's weights into it in place (vLLM's reload_weights) and clears the
# prefix cache, which held the previous weights' keys and values. The eval code is
# unchanged: with $ENGINE_SERVER set, `build` returns `_Remote`, which forwards method
# calls to the server's engine over a Unix socket and returns their results.
#
#     python -m common.engine serve <socket>     (ENGINE_SERVER_KEY: shared secret)
#     python -m common.engine stop <socket>
#
# sycophancy/jobs/reverse_batch.sh runs one beside each chain, so snapshots are
# evaluated while the replay continues. Sampling that relies on the engine's seed
# (no per-request seed) continues one random stream across requests instead of
# restarting it, so it is the same in distribution, not in its draws.

def _key():
    return os.environ.get("ENGINE_SERVER_KEY", "").encode() or None


class _Remote:
    """`build`'s result under $ENGINE_SERVER: method calls run on the server's engine."""

    def __init__(self, address, model, max_model_len, gpu_frac, seed, kw):
        import time
        from multiprocessing.connection import Client
        for _ in range(600):  # the server may still be starting
            try:
                self._conn = Client(address, family="AF_UNIX", authkey=_key())
                break
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(1)
        else:
            raise SystemExit(f"no engine server at {address}")
        self._model = str(model)
        self._call("build", self._model, max_model_len, gpu_frac, seed, kw)

    def get_tokenizer(self):
        """Loaded here: vLLM's tokenizer is an instance of a class made at run time,
        which cannot be pickled across."""
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self._model)

    def _call(self, *msg):
        self._conn.send(msg)
        status, res = self._conn.recv()
        if status != "ok":
            raise RuntimeError(f"engine server: {res}")
        return res

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: self._call("call", name, a, k)


def _shutdown(llm):
    import gc
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:  # noqa: BLE001 -- best effort: the process is going anyway
        pass
    del llm
    gc.collect()


def _serve_build(live, model, max_model_len, gpu_frac, seed, kw):
    """The live engine with `model`'s weights in it: built on the first request, rebuilt
    for another model type, a longer window or other engine arguments, else reloaded."""
    import time
    mtype = model_config(model).get("model_type")
    want = max(max_model_len or int(os.environ.get("MAX_MODEL_LEN", "2048")),
               int(os.environ.get("ENGINE_SERVER_MIN_LEN", "0")))  # one build for every stage
    if live and (live["model_type"] != mtype or want > live["max_len"] or live["kw"] != kw):
        print(f"engine server: rebuilding ({live['model_type']}, {live['max_len']} tokens) "
              f"for ({mtype}, {want})", flush=True)
        _shutdown(live.pop("llm"))
        live = None
    t0 = time.time()
    if live is None:
        llm = _build_local(model, want, gpu_frac, seed, **dict(kw))
        print(f"engine server: built for {model} in {time.time() - t0:.0f}s", flush=True)
        return dict(llm=llm, model_type=mtype, max_len=want, kw=kw, path=model)
    if live["path"] != model:
        src = str(Path(model).resolve()) if Path(model, "config.json").exists() else model
        live["llm"].collective_rpc("reload_weights", kwargs={"weights_path": src})
        live["llm"].reset_prefix_cache()
        live["path"] = model
        print(f"engine server: loaded {model} in {time.time() - t0:.0f}s", flush=True)
    return live


def serve(address):
    """Answer `_Remote` requests on `address` until `stop`. Each connection gets a thread,
    so a client holding an idle connection blocks no other; the engine itself takes one
    request at a time."""
    import signal
    import socket
    import threading
    import traceback
    from multiprocessing.connection import Listener

    def _term(*_):
        raise SystemExit(0)  # through the finally below, which shuts the engine down

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _term)
    state = {"live": None}
    lock, done = threading.Lock(), threading.Event()

    def handle(conn):
        with conn:
            while True:
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    return
                if msg[0] == "stop":
                    done.set()
                    conn.send(("ok", None))
                    # wake the accept() below so the server can exit
                    with socket.socket(socket.AF_UNIX) as s:
                        try:
                            s.connect(address)
                        except OSError:
                            pass
                    return
                try:
                    with lock:
                        if msg[0] == "build":
                            state["live"] = _serve_build(state["live"], *msg[1:])
                            res = None
                        else:
                            res = getattr(state["live"]["llm"], msg[1])(*msg[2], **msg[3])
                    conn.send(("ok", res))
                except Exception:  # noqa: BLE001 -- the client raises it
                    conn.send(("err", traceback.format_exc()))

    try:
        with Listener(address, family="AF_UNIX", authkey=_key()) as ls:
            print(f"engine server: listening on {address}", flush=True)
            while not done.is_set():
                try:
                    conn = ls.accept()
                except Exception:  # noqa: BLE001 -- a refused handshake, or the wake-up
                    continue
                threading.Thread(target=handle, args=(conn,), daemon=True).start()
    finally:
        with lock:
            if state["live"] and "llm" in state["live"]:
                _shutdown(state["live"]["llm"])


def stop(address):
    from multiprocessing.connection import Client
    try:
        with Client(address, family="AF_UNIX", authkey=_key()) as c:
            c.send(("stop",))
            c.recv()
    except (FileNotFoundError, ConnectionRefusedError, EOFError):
        pass


if __name__ == "__main__":
    import sys
    {"serve": serve, "stop": stop}[sys.argv[1]](sys.argv[2])
