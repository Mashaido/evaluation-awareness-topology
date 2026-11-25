#!/bin/bash
#SBATCH --job-name=download
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out

# Purpose: download the model (nvidia/Llama-3_3-Nemotron-Super-49B-v1) and the
# adapter (timhua/wood_v2_sftr4_filt) from Hugging Face into home folder.
# Needs no GPU. Checks the total size first and stops if it is over 110 GB,
# so it cannot fill our 140 GB quota. Safe to resubmit: it resumes.
# Submit from ~/slurm_logs:
#   cd ~/slurm_logs
#   sbatch ~/evaluation-awareness-topology/experiments/download_model.sbatch

IMG=/data/shared/containers/jupyter-pytorch-2026.10-hf.sif
ENVDIR=$HOME/envs/ea
export HF_HOME=$HOME/hf_home
mkdir -p "$HF_HOME"

echo "== start =="
date
echo "home usage before:"
du -sh "$HOME" 2>/dev/null

apptainer exec --env PYTHONPATH="$ENVDIR" --env HF_HOME="$HF_HOME" "$IMG" python3 - <<'PY'
import fnmatch
import sys

from huggingface_hub import HfApi, snapshot_download

MODEL = "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
ADAPTER = "timhua/wood_v2_sftr4_filt"
MAX_GB = 110.0
PATTERNS = ["*.safetensors", "*.json", "*.py", "*.txt", "*.model", "tokenizer*"]

api = HfApi()


def listing(repo, patterns):
    info = api.model_info(repo, files_metadata=True)
    print("repo", repo, "gated:", info.gated, "private:", info.private)
    total = 0
    for s in info.siblings:
        size = s.size or 0
        if patterns is not None and not any(fnmatch.fnmatch(s.rfilename, p) for p in patterns):
            continue
        total += size
        if not s.rfilename.endswith(".safetensors") or size > 5e8:
            print("   ", s.rfilename, round(size / 1e9, 3), "GB")
    print("   total to download:", round(total / 1e9, 1), "GB")
    return total / 1e9


try:
    model_gb = listing(MODEL, PATTERNS)
    adapter_gb = listing(ADAPTER, None)
except Exception as exc:
    print("LISTING FAILED", type(exc).__name__, exc)
    sys.exit(1)

if model_gb + adapter_gb > MAX_GB:
    print("STOP: total", round(model_gb + adapter_gb, 1), "GB is over the", MAX_GB, "GB limit. Nothing downloaded.")
    sys.exit(2)

for repo, patterns in [(MODEL, PATTERNS), (ADAPTER, None)]:
    try:
        path = snapshot_download(repo, allow_patterns=patterns, max_workers=8)
        print("DOWNLOADED", repo, "->", path)
    except Exception as exc:
        print("DOWNLOAD FAILED", repo, type(exc).__name__, exc)
        sys.exit(3)
PY
status=$?

echo "== python exit status: $status =="
echo "home usage after:"
du -sh "$HOME" 2>/dev/null
du -sh "$HF_HOME" 2>/dev/null
date
echo "== done =="
