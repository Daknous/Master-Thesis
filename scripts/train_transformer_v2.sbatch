#!/bin/bash
#SBATCH --job-name=subseg_v3_r34_oc
#SBATCH --output=/home/users/s/seifdak/Master-Thesis/logs/%x_%j.out
#SBATCH --error=/home/users/s/seifdak/Master-Thesis/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=seif.daknou@campus.tu-berlin.de

module load python/3.9.19
source ~/scratch/subseg_venv/bin/activate
#pip install --upgrade --no-cache-dir -r ~/Master-Thesis/requirements.txt

cd ~/Master-Thesis

srun python ~/Master-Thesis/electrical_substation_segmentation.py \
     --train_images_dir  ~/Master-Thesis/Dataset_v2_filtered/train \
     --train_coco_json   ~/Master-Thesis/Dataset_v2_filtered/train/_annotations.coco.json \
     --val_images_dir    ~/Master-Thesis/Dataset_v2_filtered/valid \
     --val_coco_json     ~/Master-Thesis/Dataset_v2_filtered/valid/_annotations.coco.json \
     --test_images_dir   ~/Master-Thesis/Dataset_v2_filtered/test \
     --test_coco_json    ~/Master-Thesis/Dataset_v2_filtered/test/_annotations.coco.json \
     --batch_size 4 \
     --lr 1e-4 \
     --epochs 200 \
     --one_cycle \
     --use_focal \
     --use_boundary \
     --tta_flip \
     --adaptive_thresh \
     --post_process \
     --min_size 50 \
     --merge_distance 15 \
     --log_dir runs/final_exp/focal_boundary_exp
