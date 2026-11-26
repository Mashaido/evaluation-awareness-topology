#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_max_2g.48gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=%x_%j.out

# Purpose: generate responses with the 4 bit model plus the Hua adapter on 50 questions in the
# eval and deployed wrappers (hint on), score them with the authors' type hint scorer, and
# check whether the organism's behavior survives 4 bit. Responses go to results/private/.
# First a tiny run to catch errors fast (2 questions, short):
#   cd ~/slurm_logs
#   sbatch --export=ALL,NQ=2,MAXTOK=300 ~/evaluation-awareness-topology/experiments/smoke_test.sbatch
# Then the real run:
#   sbatch ~/evaluation-awareness-topology/experiments/smoke_test.sbatch

IMG=/data/shared/containers/jupyter-pytorch-2026.10-hf.sif
ENVDIR=$HOME/envs/ea
REPO=$HOME/evaluation-awareness-topology
export HF_HOME=$HOME/hf_home
export HF_HUB_OFFLINE=1
NQ=${NQ:-50}
MAXTOK=${MAXTOK:-2048}

date
nvidia-smi -L
apptainer exec --nv --env PYTHONPATH="$ENVDIR" --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 \
    "$IMG" python3 -u "$REPO/src/smoke_test.py" --data "$HOME/private_data/steering-eval-data" \
    --n-questions "$NQ" --max-new-tokens "$MAXTOK"
echo "exit status: $?"
date
echo "== done =="
