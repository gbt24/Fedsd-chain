#!/bin/bash
set -euo pipefail

GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-1}"
ROOT="./result/journal_multiseed"
PRE_TRAIN_SIMPLE="${PRE_TRAIN_SIMPLE:-True}"
SD_MODEL="${SD_MODEL:-google/ddpm-cifar10-32}"
SEEDS=(1 2 3)

run_cmd() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

for seed in "${SEEDS[@]}"; do
  stage1="$ROOT/celeba64/plain/seed$seed"
  full="$ROOT/celeba64/full_iid/seed$seed"

  run_cmd python main_diffusion.py \
    --model SimpleUNet \
    --dataset celeba64 \
    --num_classes 1 \
    --image_size 64 \
    --num_channels 3 \
    --max_train_samples 20000 \
    --max_test_samples 5000 \
    --epochs 100 \
    --num_clients 25 \
    --clients_percent 0.4 \
    --start_epochs 0 \
    --distribution iid \
    --local_ep 5 \
    --local_bs 64 \
    --local_lr 1e-4 \
    --local_optim adam \
    --lr_decay 0.999 \
    --timesteps 1000 \
    --beta_schedule linear \
    --num_inference_steps 1000 \
    --sample_interval 10 \
    --num_samples 16 \
    --time_embed_dim 512 \
    --class_embed_dim 512 \
    --block_out_channels 128 256 512 512 \
    --layers_per_block 2 \
    --dropout 0.1 \
    --pre_train_simple "$PRE_TRAIN_SIMPLE" \
    --sd_model "$SD_MODEL" \
    --trigger_class 1 \
    --watermark False \
    --fingerprint False \
    --enable_blockchain False \
    --gpu "$GPU" \
    --seed "$seed" \
    --save True \
    --save_dir "$stage1"

  run_cmd python main_diffusion.py \
    --model SimpleUNet \
    --dataset celeba64 \
    --num_classes 1 \
    --image_size 64 \
    --num_channels 3 \
    --max_train_samples 20000 \
    --max_test_samples 5000 \
    --epochs 200 \
    --num_clients 25 \
    --clients_percent 0.4 \
    --start_epochs 100 \
    --pre_train True \
    --pre_train_path "$stage1/model_final.pth" \
    --distribution iid \
    --local_ep 5 \
    --local_bs 64 \
    --local_lr 1e-4 \
    --local_optim adam \
    --lr_decay 0.999 \
    --timesteps 1000 \
    --beta_schedule linear \
    --num_inference_steps 1000 \
    --sample_interval 10 \
    --num_samples 16 \
    --time_embed_dim 512 \
    --class_embed_dim 512 \
    --block_out_channels 128 256 512 512 \
    --layers_per_block 2 \
    --dropout 0.1 \
    --pre_train_simple "$PRE_TRAIN_SIMPLE" \
    --sd_model "$SD_MODEL" \
    --trigger_class 1 \
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
    --test_interval 5 \
    --test_bs 16 \
    --enable_blockchain True \
    --enable_anchor False \
    --anchor_mode mock \
    --gpu "$GPU" \
    --seed "$seed" \
    --save True \
    --save_dir "$full"
done
