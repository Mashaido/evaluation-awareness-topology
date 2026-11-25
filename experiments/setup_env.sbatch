#!/bin/bash
#SBATCH --job-name=setupenv
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_max_2g.48gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out

# Purpose: install the extra Python packages we need into our own folder in
# home (the shared container is read only), then test them on the GPU.
# Loads no model and downloads no model weights.
# Submit from ~/slurm_logs:
#   cd ~/slurm_logs
#   sbatch ~/evaluation-awareness-topology/experiments/setup_env.sbatch

IMG=/data/shared/containers/jupyter-pytorch-2026.10-hf.sif
ENVDIR=$HOME/envs/ea
export PIP_DISABLE_PIP_VERSION_CHECK=1

mkdir -p "$ENVDIR"

echo "== can the container write to $ENVDIR =="
apptainer exec --nv "$IMG" bash -c "test -w '$ENVDIR' && echo writable || echo NOT writable"

echo "== pip install 1: transformers 4.48.3 (model card version), newer tokenizers, bitsandbytes =="
# --no-deps so pip does not pull a second copy of torch into our folder
apptainer exec --nv "$IMG" python3 -m pip install --no-cache-dir --upgrade --target "$ENVDIR" --no-deps \
    "transformers==4.48.3" "tokenizers>=0.21,<0.22" bitsandbytes

echo "== pip install 2: wandb (with its small dependencies) =="
apptainer exec --nv "$IMG" python3 -m pip install --no-cache-dir --upgrade --target "$ENVDIR" wandb

echo "== test inside the container with our folder first on the path =="
apptainer exec --nv --env PYTHONPATH="$ENVDIR" "$IMG" python3 - <<'PY'
import torch

print("torch", torch.__version__, "cuda available", torch.cuda.is_available())
for name in ["transformers", "tokenizers", "accelerate", "peft", "huggingface_hub", "bitsandbytes", "wandb"]:
    try:
        mod = __import__(name)
        print(name, getattr(mod, "__version__", "version unknown"), "from", getattr(mod, "__file__", "?"))
    except Exception as exc:
        print(name, "FAILED", type(exc).__name__, exc)

try:
    from transformers import BitsAndBytesConfig
    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    print("transformers BitsAndBytesConfig ok")
except Exception as exc:
    print("BitsAndBytesConfig FAILED", type(exc).__name__, exc)

# Tiny 4 bit layer on the GPU. This is the real test of whether 4 bit works on this card.
try:
    import bitsandbytes as bnb
    layer = bnb.nn.Linear4bit(256, 256, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4")
    layer.weight = bnb.nn.Params4bit(torch.randn(256, 256).half(), requires_grad=False, quant_type="nf4")
    layer = layer.cuda()
    x = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")
    y = layer(x)
    print("4 bit forward ok", tuple(y.shape), y.dtype, "finite", bool(torch.isfinite(y).all()))
except Exception as exc:
    print("4 bit test FAILED", type(exc).__name__, exc)
PY

echo "== folder size =="
du -sh "$ENVDIR"
echo "== done =="
