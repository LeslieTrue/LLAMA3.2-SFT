#!/bin/bash

#SBATCH --job-name=sft
#SBATCH --mail-user=tianzhechu@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1                                 # Total number of nodes requested
#SBATCH --ntasks-per-node=8                       # Total number of task requested
#SBATCH --cpus-per-task=8                        # Total number of cores requested
#SBATCH --mem=512G
#SBATCH -t 72:00:00                          # Time limit (hh:mm:ss)
#SBATCH --gpus-per-node=8   
#SBATCH --output=./logs/python_array_%A_%a.out
#SBATCH --output=./logs/python_array_%A_%a.err
#SBATCH --partition=High

MODEL_NAME="/group/mayi/tianzhe/checkpoints/Llama-3.2-11B-Vision-Instruct"
DATA_PATH="/group/mayi/tianzhe/project/RL-MLLM/data_collection/EQN_Data_image_200k/general_points_image.json"
IMAGE_FOLDER="/group/mayi/tianzhe/project/RL-MLLM/data_collection/EQN_Data_image_200k/General_Points_Images"
# MODEL_NAME="meta-llama/Llama-3.2-90B-Vision-Instruct"

# LLaMA3.2-Vision Does not support flash-attnetion2 yet.
# Need test for batch size > 1
# Leave a issue after testing this script with batch_size > 1

which python

export PYTHONPATH=src:$PYTHONPATH

deepspeed --include localhost:4,5,6,7 src/training/train.py \
    --deepspeed scripts/zero3_offload.json \
    --model_id ${MODEL_NAME} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --disable_flash_attn2 True \
    --lora_enable False \
    --tune_img_projector True \
    --freeze_vision_tower False \
    --freeze_llm False \
    --bf16 True \
    --output_dir output/sft_vl_200k_1e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --projector_lr 1e-6 \
    --vision_lr 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 20 \
    --dataloader_num_workers 4