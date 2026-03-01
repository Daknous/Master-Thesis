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
#SBATCH --array=0         # three jobs or resume from 0, 1, 2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=seif.daknou@campus.tu-berlin.de

module load python/3.9.19
source ~/scratch/subseg_venv/bin/activate
export WANDB_API_KEY=$(< ~/.wandb_api_key)

cd ~/Master-Thesis

# ─────────────────────────  EXPERIMENT MATRIX  ───────────────────────── #
CONFIGS=(
  # 0 ─ baseline (BCE+Dice, no attention) – reference
  " --resume runs/R34/binary_exp_1754505226/best_model_epoch143.pth \
   --metric_threshold 0.30 --use_sampler_weights --img_size 1200 \
   --loss bce_dice --decoder_attention none \
   --wandb_tags R34,res1200,th0.30,samp_w,bce_dice"

  # 1 ─ add SCSE attention, keep BCE+Dice
  "--metric_threshold 0.30 --use_sampler_weights --img_size 1200 \
   --loss bce_dice --decoder_attention scse \
   --wandb_tags R34,res1200,th0.30,samp_w,att_scse"

  # 2 ─ SCSE + Focal-Tversky (full new recipe)
  "--metric_threshold 0.25 --use_sampler_weights --img_size 1200 \
   --loss focal_tversky --decoder_attention scse \
   --wandb_tags R34,res1200,th0.25,samp_w,att_scse,ft_loss"
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
