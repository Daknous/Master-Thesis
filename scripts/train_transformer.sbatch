#!/bin/bash
#SBATCH --job-name=subseg_train
#SBATCH --output=~/Master-Thesis/logs/train_%j.out
#SBATCH --error=~/Master-Thesis/logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=seif.daknou@campus.tu-berlin.de

module load python/3.10
. ~/Master-Thesis/env_subseg/bin/activate

mkdir -p ~/Master-Thesis/logs

cd ~/Master-Thesis

srun python ~/Master-Thesis/electrical_substation_segmentation.py \
     --images_dir ~/Master-Thesis/Dataset/train \
     --coco_json  ~/Master-Thesis/Dataset/train/_annotations.coco.json \
     --batch_size 8 \
     --lr 1e-4 \
     --epochs 50 \
     --num_workers 8 \
     --val_split validation \
     --device cuda \
     --log_dir ~/Master-Thesis/runs/exp1
