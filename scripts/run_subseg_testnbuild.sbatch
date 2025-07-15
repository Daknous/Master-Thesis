#!/bin/bash
#SBATCH --job-name=subseg_train
#SBATCH --output=/home/users/s/seifdak/Master-Thesis/logs/train_%j.out
#SBATCH --error=/home/users/s/seifdak/Master-Thesis/logs/train_%j.err
#SBATCH --partition=TestAndBuild
#SBATCH --time=01:00:00

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=seif.daknou@campus.tu-berlin.de

module load python/3.9.19
source ~/scratch/subseg_venv/bin/activate
#. ~/Master-Thesis/env_subseg/bin/activate
pip install --upgrade --no-cache-dir -r ~/Master-Thesis/requirements.txt
which python
python -c "import torch; print('Torch OK:', torch.__version__)"

cd ~/Master-Thesis


export WANDB_API_KEY=$(< ~/.wandb_api_key)


python main.py \
     --batch_size 8 \
     --lr 1e-4 \
     --epochs 50 \
     --num_workers 8 \
     --device cuda \
     --log_dir ~/Master-Thesis/runs \
     --wandb_project substation-segmentation \
     --wandb_entity seif-daknou-technical-university-of-berlin
