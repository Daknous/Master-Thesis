#!/bin/bash
#SBATCH --job-name=subseg_R34_%a
#SBATCH --output=/home/users/s/seifdak/Master-Thesis/logs/R34_%A_%a.out
#SBATCH --error=/home/users/s/seifdak/Master-Thesis/logs/R34_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00

#SBATCH --mail-user=seif.daknou@campus.tu-berlin.de
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-2

module load python/3.9.19
source ~/scratch/subseg_venv/bin/activate

export WANDB_API_KEY=$(< ~/.wandb_api_key)

cd ~/Master-Thesis

CONFIGS=(
  # 0 ── resume baseline, bigger tiles, pos_weight OFF
  "--resume runs/orthL8/binary_exp_1754111648/best_model_epoch179.pth \
   --metric_threshold 0.30 --use_sampler_weights --img_size 1200 \
   --wandb_tags R34,res1200,th0.3,samp_w"
  # 1 ── fresh run @1024, no sampler, pos_weight OFF
  "--metric_threshold 0.30 --img_size 1024 \
   --wandb_tags R34,res1024,th0.3"
  # 2 ── same as #1 but sampler + lower thresh
  "--metric_threshold 0.25 --use_sampler_weights --img_size 1024 \
   --wandb_tags R34,res1024,th0.25,samp_w"
)

ARGS=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

srun python main.py \
     --batch_size 4 \
     --lr 1e-4 \
     --epochs 200 \
     --num_workers 4 \
     --device cuda \
     --log_dir ~/Master-Thesis/runs/R34 \
     --wandb_project substation-segmentation \
     --wandb_entity seif-daknou-technical-university-of-berlin \
     --wandb_group R34_experiments \
     $ARGS
