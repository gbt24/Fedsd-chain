#!/bin/bash
set -euo pipefail

RUN_DIR_BASE="./result/journal_multiseed"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DETECTOR_CHECKPOINT="${DETECTOR_CHECKPOINT:-result/logo_detector_v4/model_best.pth}"
SEEDS=(1 2 3)

# ==============================================================================
# PHASE 1 -- CelebA pilot (1 seed, verify pipeline before committing 3 seeds)
# ==============================================================================

run_stage1_celeba() {
  local seed="$1"
  "$PYTHON_BIN" main_diffusion.py \
    --model SimpleUNet --dataset celeba64 --num_classes 1 \
    --image_size 64 --num_channels 3 \
    --max_train_samples 20000 --max_test_samples 5000 \
    --epochs 100 --num_clients 25 --clients_percent 0.4 \
    --start_epochs 0 --distribution iid \
    --local_ep 5 --local_bs 64 --local_lr 1e-4 --local_optim adam \
    --lr_decay 0.999 --timesteps 1000 --beta_schedule linear \
    --num_inference_steps 1000 --sample_interval 10 --num_samples 16 \
    --time_embed_dim 512 --class_embed_dim 512 \
    --block_out_channels 128 256 512 512 --layers_per_block 2 --dropout 0.1 \
    --pre_train_simple True --sd_model "google/ddpm-cifar10-32" \
    --trigger_class 1 --watermark False --fingerprint False \
    --enable_blockchain False \
    --gpu "$GPU" --seed "$seed" --save True \
    --save_dir "${RUN_DIR_BASE}/celeba64/plain/seed${seed}"
}

run_stage2_celeba() {
  local seed="$1"
  "$PYTHON_BIN" main_diffusion.py \
    --model SimpleUNet --dataset celeba64 --num_classes 1 \
    --image_size 64 --num_channels 3 \
    --max_train_samples 20000 --max_test_samples 5000 \
    --epochs 200 --num_clients 25 --clients_percent 0.4 \
    --start_epochs 100 --pre_train True \
    --pre_train_path "${RUN_DIR_BASE}/celeba64/plain/seed${seed}/model_final.pth" \
    --distribution iid \
    --local_ep 5 --local_bs 64 --local_lr 1e-4 --local_optim adam \
    --lr_decay 0.999 --timesteps 1000 --beta_schedule linear \
    --num_inference_steps 1000 --sample_interval 10 --num_samples 16 \
    --time_embed_dim 512 --class_embed_dim 512 \
    --block_out_channels 128 256 512 512 --layers_per_block 2 --dropout 0.1 \
    --pre_train_simple True --sd_model "google/ddpm-cifar10-32" \
    --trigger_class 1 --watermark True --fingerprint True \
    --lfp_length 128 --num_trigger_set 100 \
    --embed_layer_names "mid_block.attention.proj" \
    --watermark_weight 0.01 --watermark_max_iters 50 \
    --fingerprint_max_iters 5 --lambda1 0.1 --lambda2 0.01 \
    --test_interval 5 --test_bs 16 \
    --enable_blockchain True --enable_anchor False --anchor_mode mock \
    --gpu "$GPU" --seed "$seed" --save True \
    --save_dir "${RUN_DIR_BASE}/celeba64/full_iid/seed${seed}"
}

echo "=== Phase 1: CelebA pilot (seed=1) ==="
run_stage1_celeba 1
run_stage2_celeba 1

echo "=== Phase 1 eval: client 0 only ==="
"$PYTHON_BIN" run_journal_evaluations.py \
  --run_dir "${RUN_DIR_BASE}/celeba64/full_iid/seed1" \
  --detector_checkpoint "$DETECTOR_CHECKPOINT" \
  --client_subset 0 --gpu "$GPU"

# ==============================================================================
# PHASE 2 -- CIFAR-10 / CIFAR-100 full matrix (3 seeds)
# ==============================================================================

echo "=== Phase 2: CIFAR multi-seed training ==="
export DRY_RUN=0 GPU="$GPU"
bash script/journal_multiseed_cifar.sh

# ==============================================================================
# PHASE 3 -- CelebA remaining seeds (2,3)
# ==============================================================================

echo "=== Phase 3: CelebA seeds 2-3 ==="
for seed in 2 3; do
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
  --root "$RUN_DIR_BASE" --seeds 1,2,3

echo "=== DONE ==="
