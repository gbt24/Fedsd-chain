#!/bin/bash

# Stage 2: Embed ownership (WITH watermark, WITH fingerprint)
# Resume from Stage 1 checkpoint
# Lower watermark weight: 0.01 (was 0.1)
# Goal: Embed watermark/fingerprint while preserving generation quality

python main_diffusion.py \
    --model SimpleUNet \
    --dataset cifar10 \
    --num_classes 10 \
    --image_size 32 \
    --num_channels 3 \
    \
    --epochs 200 \
    --num_clients 50 \
    --clients_percent 0.4 \
    --start_epochs 100 \
    --pre_train True \
    --pre_train_path "./result/simpleunet_cifar10_stage1/model_final.pth" \
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
    --watermark True \
    --fingerprint True \
    --lfp_length 128 \
    --num_trigger_set 100 \
    --embed_layer_names "mid_block.attention.proj" \
    --watermark_weight 0.01 \
    --watermark_max_iters 50 \
    --fingerprint_max_iters 5 \
    --lambda1 0.1 \
    --lambda2 0.01 \
    \
    --test_interval 5 \
    --test_bs 16 \
    \
    --gpu 0 \
    --seed 42 \
    --save True \
    --save_dir "./result/simpleunet_cifar10_stage2/"

echo "Stage 2 completed!"
echo "Model saved to: ./result/simpleunet_cifar10_stage2/"