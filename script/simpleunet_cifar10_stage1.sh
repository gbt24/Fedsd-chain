#!/bin/bash

# Stage 1: Learn generation quality (NO watermark, NO fingerprint)
# Larger model: block_out_channels = [128, 256, 512, 512] (~75M params)
# Goal: Achieve good generation quality first

python main_diffusion.py \
    --model SimpleUNet \
    --dataset cifar10 \
    --num_classes 10 \
    --image_size 32 \
    --num_channels 3 \
    \
    --epochs 100 \
    --num_clients 50 \
    --clients_percent 0.4 \
    --start_epochs 0 \
    \
    --distribution iid \
    --local_ep 5 \
    --local_bs 64 \
    --local_lr 1e-4 \
    --local_optim adam \
    --lr_decay 0.999 \
    \
    --timesteps 1000 \
    --beta_schedule linear \
    --num_inference_steps 1000 \
    --sample_interval 10 \
    --num_samples 16 \
    \
    --time_embed_dim 512 \
    --class_embed_dim 512 \
    --block_out_channels 128 256 512 512 \
    --layers_per_block 2 \
    --dropout 0.1 \
    \
    --pre_train_simple True \
    --sd_model "google/ddpm-cifar10-32" \
    --trigger_class 10 \
    \
    --watermark False \
    --fingerprint False \
    \
    --gpu 0 \
    --seed 42 \
    --save True \
    --save_dir "./result/simpleunet_cifar10_stage1/"

echo "Stage 1 completed!"
echo "Model saved to: ./result/simpleunet_cifar10_stage1/"