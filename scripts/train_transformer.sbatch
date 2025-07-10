#!/bin/bash
#SBATCH --job-name=subseg_train2
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

module load python/3.9.19
. ~/Master-Thesis/env_subseg/bin/activate
pip install --upgrade --no-cache-dir -r ~/Master-Thesis/requirements.txt

cd ~/Master-Thesis

srun python ~/Master-Thesis/electrical_substation_segmentation.py \
     --train_images_dir  ~/Master-Thesis/Dataset/train_filtered \
     --train_coco_json   ~/Master-Thesis/Dataset/train_filtered/_annotations.coco.json \
     --val_images_dir    ~/Master-Thesis/Dataset/valid \
     --val_coco_json     ~/Master-Thesis/Dataset/valid/_annotations.coco.json \
     --batch_size 8 \
     --lr 1e-4 \
     --epochs 250 \
     --num_workers 4 \
     --device cuda \
     --log_dir ~/Master-Thesis/runs/exp5
