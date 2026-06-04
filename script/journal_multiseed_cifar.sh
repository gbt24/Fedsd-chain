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

run_stage1() {
  local dataset="$1"
  local num_classes="$2"
  local image_size="$3"
  local seed="$4"
  local save_dir="$5"
  run_cmd python main_diffusion.py \
    --model SimpleUNet \
    --dataset "$dataset" \
    --num_classes "$num_classes" \
    --image_size "$image_size" \
    --num_channels 3 \
    --epochs 100 \
    --num_clients 50 \
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
    --trigger_class "$num_classes" \
    --watermark False \
    --fingerprint False \
    --enable_blockchain False \
    --gpu "$GPU" \
    --seed "$seed" \
    --save True \
    --save_dir "$save_dir"
}

run_stage2() {
  local dataset="$1"
  local num_classes="$2"
  local image_size="$3"
  local seed="$4"
  local distribution="$5"
  local dniid_param="$6"
  local pretrain_dir="$7"
  local save_dir="$8"
  run_cmd python main_diffusion.py \
    --model SimpleUNet \
    --dataset "$dataset" \
    --num_classes "$num_classes" \
    --image_size "$image_size" \
    --num_channels 3 \
    --epochs 200 \
    --num_clients 50 \
    --clients_percent 0.4 \
    --start_epochs 100 \
    --pre_train True \
    --pre_train_path "$pretrain_dir/model_final.pth" \
    --distribution "$distribution" \
    --dniid_param "$dniid_param" \
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
    --trigger_class "$num_classes" \
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
    --save_dir "$save_dir"
}

for seed in "${SEEDS[@]}"; do
  c10_plain="$ROOT/cifar10/plain/seed$seed"
  run_stage1 cifar10 10 32 "$seed" "$c10_plain"
  run_stage2 cifar10 10 32 "$seed" iid 0.8 "$c10_plain" "$ROOT/cifar10/full_iid/seed$seed"
  run_stage2 cifar10 10 32 "$seed" dniid 0.3 "$c10_plain" "$ROOT/cifar10/full_a03/seed$seed"

  c100_plain="$ROOT/cifar100/plain/seed$seed"
  run_stage1 cifar100 100 32 "$seed" "$c100_plain"
  run_stage2 cifar100 100 32 "$seed" iid 0.8 "$c100_plain" "$ROOT/cifar100/full_iid/seed$seed"
done
