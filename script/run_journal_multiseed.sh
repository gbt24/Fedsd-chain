#!/bin/bash
set -euo pipefail

RUN_DIR_BASE="./result/journal_multiseed"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DETECTOR_CHECKPOINT="${DETECTOR_CHECKPOINT:-result/logo_detector_v4/model_best.pth}"
FORCE="${FORCE:-0}"
PRE_TRAIN_SIMPLE="${PRE_TRAIN_SIMPLE:-True}"
SD_MODEL="${SD_MODEL:-google/ddpm-cifar10-32}"
ENABLE_BLOCKCHAIN="${ENABLE_BLOCKCHAIN:-True}"
NUM_CLIENTS="${NUM_CLIENTS:-25}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-100}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-150}"
SEEDS_TEXT="${SEEDS:-1 2 3}"
read -r -a SEEDS <<< "$SEEDS_TEXT"
PILOT_SEED="${SEEDS[0]}"
SEEDS_CSV="${SEEDS_TEXT// /,}"

# ==============================================================================
# PHASE 1 -- CelebA pilot (1 seed, verify pipeline before committing 3 seeds)
# ==============================================================================

run_stage1_celeba() {
  local seed="$1"
  local save_dir="${RUN_DIR_BASE}/celeba64/plain/seed${seed}"
  if [ "$FORCE" != "1" ] && [ -f "$save_dir/model_final.pth" ]; then
    echo "Skip existing CelebA Stage 1: $save_dir"
    return
  fi
  "$PYTHON_BIN" main_diffusion.py \
    --model SimpleUNet --dataset celeba64 --num_classes 1 \
    --image_size 64 --num_channels 3 \
    --max_train_samples 20000 --max_test_samples 5000 \
    --epochs "$STAGE1_EPOCHS" --num_clients "$NUM_CLIENTS" --clients_percent 0.4 \
    --start_epochs 0 --distribution iid \
    --local_ep 5 --local_bs 64 --local_lr 1e-4 --local_optim adam \
    --lr_decay 0.999 --timesteps 1000 --beta_schedule linear \
    --num_inference_steps 1000 --sample_interval 10 --num_samples 16 \
    --time_embed_dim 512 --class_embed_dim 512 \
    --block_out_channels 128 256 512 512 --layers_per_block 2 --dropout 0.1 \
    --pre_train_simple "$PRE_TRAIN_SIMPLE" --sd_model "$SD_MODEL" \
    --trigger_class 1 --watermark False --fingerprint False \
    --enable_blockchain False \
    --gpu "$GPU" --seed "$seed" --save True \
    --save_dir "$save_dir"
}

run_stage2_celeba() {
  local seed="$1"
  local save_dir="${RUN_DIR_BASE}/celeba64/full_iid/seed${seed}"
  if [ "$FORCE" != "1" ] && [ -f "$save_dir/model_final.pth" ]; then
    echo "Skip existing CelebA Stage 2: $save_dir"
    return
  fi
  "$PYTHON_BIN" main_diffusion.py \
    --model SimpleUNet --dataset celeba64 --num_classes 1 \
    --image_size 64 --num_channels 3 \
    --max_train_samples 20000 --max_test_samples 5000 \
    --epochs "$STAGE2_EPOCHS" --num_clients "$NUM_CLIENTS" --clients_percent 0.4 \
    --start_epochs 100 --pre_train True \
    --pre_train_path "${RUN_DIR_BASE}/celeba64/plain/seed${seed}/model_final.pth" \
    --distribution iid \
    --local_ep 5 --local_bs 64 --local_lr 1e-4 --local_optim adam \
    --lr_decay 0.999 --timesteps 1000 --beta_schedule linear \
    --num_inference_steps 1000 --sample_interval 10 --num_samples 16 \
    --time_embed_dim 512 --class_embed_dim 512 \
    --block_out_channels 128 256 512 512 --layers_per_block 2 --dropout 0.1 \
    --pre_train_simple "$PRE_TRAIN_SIMPLE" --sd_model "$SD_MODEL" \
    --trigger_class 1 --watermark True --fingerprint True \
    --lfp_length 128 --num_trigger_set 100 \
    --embed_layer_names "mid_block.attention.proj" \
    --watermark_weight 0.01 --watermark_max_iters 50 \
    --fingerprint_max_iters 5 --lambda1 0.1 --lambda2 0.01 \
    --trigger_images_path "./data/pattern/" \
    --test_interval 5 --test_bs 16 \
    --enable_blockchain "$ENABLE_BLOCKCHAIN" --enable_anchor False --anchor_mode mock \
    --gpu "$GPU" --seed "$seed" --save True \
    --save_dir "$save_dir"
}

echo "=== Phase 1: CelebA pilot (seed=${PILOT_SEED}) ==="
run_stage1_celeba "$PILOT_SEED"
run_stage2_celeba "$PILOT_SEED"

echo "=== Phase 1 eval: client 0 only ==="
"$PYTHON_BIN" run_journal_evaluations.py \
  --run_dir "${RUN_DIR_BASE}/celeba64/full_iid/seed${PILOT_SEED}" \
  --detector_checkpoint "$DETECTOR_CHECKPOINT" \
  --client_subset 0 --gpu "$GPU"

# ==============================================================================
# PHASE 2 -- CIFAR-10 / CIFAR-100 full matrix (3 seeds)
# ==============================================================================

echo "=== Phase 2: CIFAR multi-seed training ==="
export DRY_RUN=0 GPU="$GPU" PYTHON_BIN FORCE PRE_TRAIN_SIMPLE SD_MODEL ENABLE_BLOCKCHAIN NUM_CLIENTS STAGE1_EPOCHS STAGE2_EPOCHS SEEDS="$SEEDS_TEXT"
bash script/journal_multiseed_cifar.sh

# ==============================================================================
# PHASE 3 -- CelebA remaining seeds (2,3)
# ==============================================================================

echo "=== Phase 3: CelebA remaining seeds ==="
for seed in "${SEEDS[@]:1}"; do
  run_stage1_celeba "$seed"
  run_stage2_celeba "$seed"
done

# ==============================================================================
# PHASE 4 -- Evaluation (all runs)
# ==============================================================================

echo "=== Phase 4a: CIFAR-10 Plain (FID only) ==="
for seed in "${SEEDS[@]}"; do
  run_dir="${RUN_DIR_BASE}/cifar10/plain/seed${seed}"
  "$PYTHON_BIN" eval_fid.py \
    --checkpoint "${run_dir}/model_final.pth" \
    --args_file "${run_dir}/args.txt" --gpu "$GPU" \
    --output "${run_dir}/eval/fid.txt"
done

echo "=== Phase 4b: CIFAR-10 Full IID + Full alpha=0.3 ==="
for seed in "${SEEDS[@]}"; do
  for cond in full_iid full_a03; do
    run_dir="${RUN_DIR_BASE}/cifar10/${cond}/seed${seed}"
    "$PYTHON_BIN" run_journal_evaluations.py \
      --run_dir "$run_dir" --detector_checkpoint "$DETECTOR_CHECKPOINT" --gpu "$GPU"
  done
done

echo "=== Phase 4c: CIFAR-100 Plain (FID only) ==="
for seed in "${SEEDS[@]}"; do
  run_dir="${RUN_DIR_BASE}/cifar100/plain/seed${seed}"
  "$PYTHON_BIN" eval_fid.py \
    --checkpoint "${run_dir}/model_final.pth" \
    --args_file "${run_dir}/args.txt" --gpu "$GPU" \
    --output "${run_dir}/eval/fid.txt"
done

echo "=== Phase 4d: CIFAR-100 Full IID ==="
for seed in "${SEEDS[@]}"; do
  run_dir="${RUN_DIR_BASE}/cifar100/full_iid/seed${seed}"
  "$PYTHON_BIN" run_journal_evaluations.py \
    --run_dir "$run_dir" --detector_checkpoint "$DETECTOR_CHECKPOINT" --gpu "$GPU"
done

echo "=== Phase 4e: CelebA64 Full IID ==="
for seed in "${SEEDS[@]}"; do
  run_dir="${RUN_DIR_BASE}/celeba64/full_iid/seed${seed}"
  "$PYTHON_BIN" run_journal_evaluations.py \
    --run_dir "$run_dir" --detector_checkpoint "$DETECTOR_CHECKPOINT" --gpu "$GPU"
done

# ==============================================================================
# PHASE 5 -- Aggregation
# ==============================================================================

echo "=== Phase 5: aggregation ==="
"$PYTHON_BIN" summarize_journal_results.py \
  --root "$RUN_DIR_BASE" --seeds "$SEEDS_CSV"

echo "=== DONE ==="
