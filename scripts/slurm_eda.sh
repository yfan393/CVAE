#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/slurm_eda.sh
# Run exploratory data analysis on CPU node (no GPU needed).
# Submit BEFORE training to verify data integrity.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64g
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:1
#SBATCH -t 2880
#SBATCH -J BaseCode_cvae_eda
#SBATCH -e /data/users1/yfan14/BaseCode/cvae/logs/eda_%j.err
#SBATCH -o /data/users1/yfan14/BaseCode/cvae/logs/eda_%j.out
#SBATCH -A trends53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yfan14@gsu.edu
#SBATCH --oversubscribe
sleep 5s

echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Started: $(date)"

# === git log before copying the project ===
cd /data/users1/yfan14/BaseCode/cvae/
mkdir -p logs eda_results
echo "Message: $(git log -1 --pretty=%B)" >&2

# cd $JOBDIR
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate 3dunet

python explore/eda.py \
    --num_subjects 700 \
    --save_dir     eda_results/

echo "EDA done.  Ended: $(date)"
	
sleep 5s
