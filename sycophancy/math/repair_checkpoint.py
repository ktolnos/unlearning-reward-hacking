"""Add back the KV-shared-layer norms that a Gemma 4 save leaves out.

Gemma 4 E2B sets num_kv_shared_layers=20: the last twenty decoder layers take their
key/value states from an earlier layer, so transformers never builds their k_norm,
k_proj or v_proj (`modeling_gemma4.py`, `if not self.is_kv_shared_layer`). A trained
checkpoint therefore has sixty fewer tensors than the hub checkpoint it started from,
which is correct for transformers and fatal for vLLM: vLLM's Gemma4 attention builds
k_norm for every layer and only *uses* it on the layers that compute their own keys
(`gemma4.py`, same guard), but its weight loader still requires a tensor for every
parameter it built, and stops with

    ValueError: Following weights were not initialized from checkpoint:
    {'language_model.model.layers.15.self_attn.k_norm.weight', ...}

so the base evaluation of the hub model succeeds and the first evaluation of a trained
checkpoint fails. This writes just those k_norm tensors, copied from the hub checkpoint,
into a second small safetensors file beside the saved weights. vLLM globs *.safetensors
for a local folder and only consults an index when one exists, so an extra file is
picked up and no 10 GB rewrite is needed. It writes k_norm alone, not the k_proj and
v_proj that are also absent: vLLM does not build those for shared layers, and an
unexpected tensor is as fatal as a missing one.

The values are never read -- by transformers, which has no such module, or by vLLM,
which skips the norm on exactly these layers. Copying the hub values rather than
inventing them keeps the checkpoint identical to what vLLM held while generating during
training, and keeps this a repair rather than an edit: the trained weights are not
touched, and nothing is overwritten.

Idempotent, and a no-op on any model without KV sharing.
"""
import argparse
import json
from pathlib import Path

EXTRA = 'kv_shared_norms.safetensors'


def repair(checkpoint, base_model):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / 'config.json').read_text())
    text = config.get('text_config', config)
    shared = text.get('num_kv_shared_layers', 0)
    if not shared:
        print(f'{checkpoint}: no KV sharing, nothing to repair', flush=True)
        return
    first = text['num_hidden_layers'] - shared
    present = set()
    # Include a file written by an earlier repair, or this reports work it already did.
    for name in ('model.safetensors', EXTRA):
        if (checkpoint / name).exists():
            with safe_open(str(checkpoint / name), 'pt') as f:
                present |= set(f.keys())
    wanted = {f'model.language_model.layers.{i}.self_attn.k_norm.weight'
              for i in range(first, text['num_hidden_layers'])}
    missing = sorted(wanted - present)
    if not missing:
        print(f'{checkpoint}: k_norm present for all {text["num_hidden_layers"]} layers', flush=True)
        return
    # Every shared layer is missing, or none is: a partial set means the naming assumed
    # here is wrong for this checkpoint and the copy would be silently misplaced.
    assert len(missing) == shared, (len(missing), shared)
    source = hf_hub_download(base_model, 'model.safetensors')
    with safe_open(source, 'pt') as f:
        tensors = {k: f.get_tensor(k) for k in missing}
    out = checkpoint / EXTRA
    save_file(tensors, str(out), metadata={'format': 'pt'})
    print(f'{checkpoint}: wrote {len(tensors)} k_norm tensors for KV-shared layers '
          f'{first}-{text["num_hidden_layers"] - 1} from {base_model} -> {out.name}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', help='Repair <run-dir>/train/final against the '
                        "run's own recorded base model")
    parser.add_argument('--checkpoint')
    parser.add_argument('--base-model')
    args = parser.parse_args()
    if args.run_dir:
        assert not (args.checkpoint or args.base_model), '--run-dir takes neither'
        root = Path(args.run_dir)
        run = json.loads((root / 'run.json').read_text())
        checkpoint = root / 'final' if (root / 'final').exists() else root / 'train/final'
        repair(checkpoint, run['environment']['model'])
        return
    assert args.checkpoint and args.base_model, 'give --run-dir, or both of the others'
    repair(args.checkpoint, args.base_model)


if __name__ == '__main__':
    main()
