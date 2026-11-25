#!/bin/bash
#SBATCH --job-name=gpucheck
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_max_2g.48gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out

# Read only environment check. Loads no model. Submit from ~/slurm_logs so the
# output file lands outside the repo:
#   mkdir -p ~/slurm_logs && cd ~/slurm_logs
#   sbatch ~/evaluation-awareness-topology/experiments/gpucheck.sbatch

echo "== node and GPU =="
hostname
date
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "== container runtime =="
which apptainer singularity || echo "no apptainer or singularity on PATH"

IMG=/data/shared/containers/jupyter-pytorch-2026.10-hf.sif
echo "== python stack inside $IMG =="
apptainer exec --nv "$IMG" python3 - <<'PY'
import importlib
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print("GPU memory GB free/total", round(free / 1e9, 1), round(total / 1e9, 1))
for name in ["transformers", "accelerate", "peft", "bitsandbytes", "sklearn", "numpy", "scipy", "wandb", "huggingface_hub"]:
    try:
        mod = importlib.import_module(name)
        print(name, getattr(mod, "__version__", "version unknown"))
    except Exception as exc:
        print(name, "MISSING", type(exc).__name__)
PY

echo "== network from the compute node =="
curl -sI -m 10 https://huggingface.co | head -1
curl -sI -m 10 https://github.com | head -1
echo "== done =="
