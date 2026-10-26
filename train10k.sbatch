#!/bin/bash
#SBATCH --job-name=drone10k
#SBATCH --account=gpu-approved
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log

source ~/miniforge3/etc/profile.d/conda.sh
conda activate openvla
cd ~/openvla
export WANDB_MODE=offline
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir ~/tensorflow_datasets \
  --dataset_name drone_dataset \
  --run_root_dir ~/runs10k \
  --adapter_tmp_dir ~/runs10k/adapter-tmp \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --max_steps 10000 \
  --save_steps 1000 \
  --save_latest_checkpoint_only False \
  --wandb_project drone-10k \
  --wandb_entity local
echo "TRAINING FINISHED"
