#!/bin/bash
#SBATCH --job-name=subseg_L8_%a
#SBATCH --output=/home/users/s/seifdak/Master-Thesis/logs/L8_%A_%a.out
#SBATCH --error=/home/users/s/seifdak/Master-Thesis/logs/L8_%A_%a.err
#SBATCH --partition=gpu

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

#SBATCH --array=1-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=seif.daknou@campus.tu-berlin.de

module load python/3.9.19
source ~/scratch/subseg_venv/bin/activate
cd ~/Master-Thesis

export WANDB_API_KEY=$(< ~/.wandb_api_key)

CONFIGS=(
  "--metric_threshold 0.30 --use_sampler_weights --img_size 1200 \
   --wandb_tags orthL8,res1200,th0.3,samp_w,pw_off"

  "--metric_threshold 0.30 --img_size 1024 \
   --wandb_tags orthL8,res1024,th0.3,pw_off"

  "--metric_threshold 0.40 --use_sampler_weights --img_size 1024 \
   --wandb_tags orthL8,res1024,th0.4,samp_w,pw_off"
)

ARGS=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

python main.py \
  --batch_size 4 \
  --lr 1e-4 \
  --epochs 100 \
  --num_workers 4 \
  --device cuda \
  --log_dir ~/Master-Thesis/runs/orthL8 \
  --wandb_project substation-segmentation \
  --wandb_entity seif-daknou-technical-university-of-berlin \
  --wandb_group orthogonal_L8 \
  $ARGS
