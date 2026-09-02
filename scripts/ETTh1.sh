#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON="${PYTHON:-/miniconda3/envs/LlaD/bin/python}"
seq_len=96
pred_len=96
lr=0.0006
channel=64
e_layer=1
d_layer=2
dropout_n=0.8
batch_size=256

tag="i${seq_len}_o${pred_len}_lr${lr}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}"
log_dir="./Results/ETTh1"
ckpt_dir="./logs/ETTh1"
mkdir -p "$log_dir" "$ckpt_dir"

echo "=== ETTh1 ${tag} ==="
"$PYTHON" -u train.py \
  --data_path ETTh1 \
  --seq_len "$seq_len" \
  --pred_len "$pred_len" \
  --batch_size "$batch_size" \
  --num_nodes 7 \
  --channel "$channel" \
  --e_layer "$e_layer" \
  --d_layer "$d_layer" \
  --dropout_n "$dropout_n" \
  --learning_rate "$lr" \
  --epochs 100 \
  --seed 42 \
  --head 8 \
  --weight_decay 0.01 \
  --teacher_task_weight 0.5 \
  --distill_weight 0.1 \
  --num_workers 4 \
  --save "${ckpt_dir}/${tag}-" \
  > "${log_dir}/${tag}.log" 2>&1
