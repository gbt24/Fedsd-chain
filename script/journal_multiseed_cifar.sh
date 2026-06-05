#!/bin/bash
set -euo pipefail

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-1}"
ROOT="./result/journal_multiseed"
FORCE="${FORCE:-0}"
PRE_TRAIN_SIMPLE="${PRE_TRAIN_SIMPLE:-True}"
SD_MODEL="${SD_MODEL:-google/ddpm-cifar10-32}"
ENABLE_BLOCKCHAIN="${ENABLE_BLOCKCHAIN:-True}"
NUM_CLIENTS="${NUM_CLIENTS:-25}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-100}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-150}"
C10_IID_TEST_INTERVAL="${C10_IID_TEST_INTERVAL:-5}"
C10_IID_TEST_BS="${C10_IID_TEST_BS:-16}"
DEFAULT_STAGE2_TEST_INTERVAL="${DEFAULT_STAGE2_TEST_INTERVAL:-1}"
DEFAULT_STAGE2_TEST_BS="${DEFAULT_STAGE2_TEST_BS:-512}"
SEEDS_TEXT="${SEEDS:-1 2 3}"
read -r -a SEEDS <<< "$SEEDS_TEXT"

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
  local distribution="$5"
  local dniid_param="$6"
  local save_dir="$7"
  local enable_blockchain="$8"
  if [ "$FORCE" != "1" ] && [ -f "$save_dir/model_final.pth" ]; then
    echo "Skip existing Stage 1: $save_dir"
    return
  fi
  run_cmd "$PYTHON_BIN" main_diffusion.py \
    --model SimpleUNet \
    --dataset "$dataset" \
    --num_classes "$num_classes" \
    --image_size "$image_size" \
    --num_channels 3 \
    --epochs "$STAGE1_EPOCHS" \
    --num_clients "$NUM_CLIENTS" \
    --clients_percent 0.4 \
    --start_epochs 0 \
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
    --watermark False \
    --fingerprint False \
    --enable_blockchain "$enable_blockchain" \
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
  local test_interval="$9"
  local test_bs="${10}"
  if [ "$FORCE" != "1" ] && [ -f "$save_dir/model_final.pth" ]; then
    echo "Skip existing Stage 2: $save_dir"
    return
  fi
  run_cmd "$PYTHON_BIN" main_diffusion.py \
    --model SimpleUNet \
    --dataset "$dataset" \
    --num_classes "$num_classes" \
    --image_size "$image_size" \
    --num_channels 3 \
    --epochs "$STAGE2_EPOCHS" \
    --num_clients "$NUM_CLIENTS" \
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
    --trigger_images_path "./data/pattern/" \
    --test_interval "$test_interval" \
    --test_bs "$test_bs" \
    --enable_blockchain "$ENABLE_BLOCKCHAIN" \
    --enable_anchor False \
    --anchor_mode mock \
    --gpu "$GPU" \
    --seed "$seed" \
    --save True \
    --save_dir "$save_dir"
}

for seed in "${SEEDS[@]}"; do
  c10_plain="$ROOT/cifar10/plain/seed$seed"
  run_stage1 cifar10 10 32 "$seed" iid 0.8 "$c10_plain" "$ENABLE_BLOCKCHAIN"
  run_stage2 cifar10 10 32 "$seed" iid 0.8 "$c10_plain" "$ROOT/cifar10/full_iid/seed$seed" "$C10_IID_TEST_INTERVAL" "$C10_IID_TEST_BS"

  c10_a03_plain="$ROOT/cifar10/plain_a03/seed$seed"
  run_stage1 cifar10 10 32 "$seed" dniid 0.3 "$c10_a03_plain" False
  run_stage2 cifar10 10 32 "$seed" dniid 0.3 "$c10_a03_plain" "$ROOT/cifar10/full_a03/seed$seed" "$DEFAULT_STAGE2_TEST_INTERVAL" "$DEFAULT_STAGE2_TEST_BS"

  c100_plain="$ROOT/cifar100/plain/seed$seed"
  run_stage1 cifar100 100 32 "$seed" iid 0.8 "$c100_plain" False
  run_stage2 cifar100 100 32 "$seed" iid 0.8 "$c100_plain" "$ROOT/cifar100/full_iid/seed$seed" "$DEFAULT_STAGE2_TEST_INTERVAL" "$DEFAULT_STAGE2_TEST_BS"
done
