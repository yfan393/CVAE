#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/slurm_train.sh
# SLURM batch script for C3D-VAE training on a GPU cluster.
#
# Usage:
#   sbatch scripts/slurm_train.sh
#
# Adjust the #SBATCH directives below to match your cluster's partition names,
# memory limits, and time limits.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=256g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8 
#SBATCH -t 2880
#SBATCH -J BaseCode_cvae_train
#SBATCH -e /data/users1/yfan14/BaseCode/cvae/logs/train_%j.err
#SBATCH -o /data/users1/yfan14/BaseCode/cvae/logs/train_%j.out
#SBATCH -A trends53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yfan14@gsu.edu
#SBATCH --oversubscribe
sleep 5s

# ── Environment setup ─────────────────────────────────────────────────────────
sleep 5s

echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"
nvidia-smi >&2

# === git log before copying the project ===
cd /data/users1/yfan14/BaseCode/cvae/
mkdir -p logs saved/C3DVAE
echo "Message: $(git log -1 --pretty=%B)" >&2

# cd $JOBDIR
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate 3dunet

# ── Training ──────────────────────────────────────────────────────────────────
python main.py \
    --config       config/runs/C3DVAE.json \
    --model_config config/models/C3DVAE.json \
    --data_config  config/data/ukbb.json


echo "Training finished Ended: $(date)"

sleep 5s

