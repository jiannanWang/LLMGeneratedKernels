#!/bin/bash
set -e

mkdir -p logs

COMMON_ARGS="config/train_shakespeare_char.py --device=cuda --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0"

echo "=== Run 1: Baseline PyTorch (flash attention) ==="
conda run --no-capture-output -n nanoGPT python train_oink.py $COMMON_ARGS \
  > logs/log.txt 2>&1
echo "Baseline done. Log: logs/log.txt"

echo "=== Run 2: Oink CuTeDSL kernels (flash attention) ==="
conda run --no-capture-output -n nanoGPT python train_oink.py $COMMON_ARGS \
  --use_oink=True \
  > logs/oink_log.txt 2>&1
echo "Oink done. Log: logs/oink_log.txt"

echo "=== Run 3: Baseline PyTorch (no flash, explicit softmax) ==="
conda run --no-capture-output -n nanoGPT python train_oink.py $COMMON_ARGS \
  --use_flash=False \
  > logs/noflash_log.txt 2>&1
echo "Baseline (no flash) done. Log: logs/noflash_log.txt"

echo "=== Run 4: Oink CuTeDSL kernels (no flash, explicit softmax) ==="
conda run --no-capture-output -n nanoGPT python train_oink.py $COMMON_ARGS \
  --use_oink=True --use_flash=False \
  > logs/oink_noflash_log.txt 2>&1
echo "Oink (no flash) done. Log: logs/oink_noflash_log.txt"

echo "=== Generating comparison plots ==="
conda run --no-capture-output -n nanoGPT python draw_curves_oink.py
echo "Done. Figures saved to figs/"
