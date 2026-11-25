#!/bin/bash
#SBATCH --job-name=loadtest
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_max_2g.48gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=%x_%j.out

# Purpose: load the 49B model plus the Hua adapter on one 48 GB GPU slice by two
# routes (4 bit, then BF16 with part of the weights in CPU RAM), run one forward
# pass and a short generation on each, and report memory and speed. Uses no data.
# If one route fails, the other still runs. Submit from ~/slurm_logs:
#   cd ~/slurm_logs
#   sbatch ~/evaluation-awareness-topology/experiments/load_test.sbatch

IMG=/data/shared/containers/jupyter-pytorch-2026.10-hf.sif
ENVDIR=$HOME/envs/ea
REPO=$HOME/evaluation-awareness-topology
export HF_HOME=$HOME/hf_home
export HF_HUB_OFFLINE=1

date
nvidia-smi -L

for MODE in 4bit bf16offload; do
    echo
    echo "================ route: $MODE ================"
    date
    apptainer exec --nv --env PYTHONPATH="$ENVDIR" --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 \
        "$IMG" python3 -u "$REPO/src/load_test.py" --mode "$MODE"
    echo "route $MODE exit status: $?"
done

echo
date
echo "== done =="
