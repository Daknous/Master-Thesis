#!/bin/bash
#SBATCH --job-name=subseg_train
#SBATCH --output=/home/users/s/seifdak/Master-Thesis/logs/train_%j.out
#SBATCH --error=/home/users/s/seifdak/Master-Thesis/logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=seif.daknou@campus.tu-berlin.de

# === Environment setup ===
module load python/3.9.19
source ~/Master-Thesis/env_subseg/bin/activate

# Install requirements (if you need to)
pip install --upgrade --no-cache-dir -r ~/Master-Thesis/requirements.txt

# Move to project root
cd ~/Master-Thesis

# === W&B Authentication ===
# Load your WANDB_API_KEY from a file in your home directory
export WANDB_API_KEY=$(< ~/.wandb_api_key)

# === Run training ===
# We pass only the base `runs/` directory. main.py will create runs/exp1, runs/exp2, ...
srun python main.py \
     --train_images_dir  ~/Master-Thesis/Dataset/train_filtered \
     --train_coco_json   ~/Master-Thesis/Dataset/train_filtered/_annotations.coco.json \
     --val_images_dir    ~/Master-Thesis/Dataset/valid \
     --val_coco_json     ~/Master-Thesis/Dataset/valid/_annotations.coco.json \
     --batch_size 8 \
     --lr 1e-4 \
     --epochs 50 \
     --num_workers 8 \
     --device cuda \
     --log_dir ~/Master-Thesis/runs \
     --wandb_project substation-segmentation \
     --wandb_entity seif-daknou-technical-university-of-berlin
