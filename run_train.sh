#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate openvla
cd ~/openvla
export WANDB_MODE=offline
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir ~/tensorflow_datasets \
  --dataset_name drone_dataset \
  --run_root_dir ~/runs \
  --adapter_tmp_dir ~/runs/adapter-tmp \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --max_steps 2000 \
  --save_steps 500 \
  --wandb_project drone-train \
  --wandb_entity local
echo "TRAINING FINISHED"
