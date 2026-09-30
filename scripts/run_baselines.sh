#!/usr/bin/env bash
set -euo pipefail

GPU=false
GPU_FRAC=1.0
RAY_CORE=1
ROUNDS=50
LOCAL_EPOCHS=5
CLIENTS=10
ALPHA=0.5
SEEDS=2023
TAG="baseline_$(date +%Y%m%d_%H%M%S)"
PYTHON_BIN=python

usage() {
  cat <<'EOF'
Usage: bash scripts/run_baselines.sh [options]

  --gpu true|false       Use CUDA through Ray (default: false)
  --gpu-frac FLOAT       GPU fraction per client task (default: 1.0)
  --ray-core INT         Ray CPU workers (default: 1)
  --rounds INT           Communication rounds (default: 50)
  --local-epochs INT     Local epochs per round (default: 5)
  --clients INT          Number of clients (default: 10)
  --alpha FLOAT          Dirichlet alpha (default: 0.5)
  --seeds LIST           Comma-separated seeds (default: 2023)
  --tag TEXT             Prefix for all runs
  --python PATH          Python executable (default: python)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --gpu-frac) GPU_FRAC="$2"; shift 2 ;;
    --ray-core) RAY_CORE="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --local-epochs) LOCAL_EPOCHS="$2"; shift 2 ;;
    --clients) CLIENTS="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
for seed in "${SEED_LIST[@]}"; do
  for dataset in cifar-10 cifar-100; do
    for model in Custom_cnn resnet-18; do
      for spec in "fedavg:false" "fedconst:true"; do
        method_name="${spec%%:*}"
        const_flag="${spec##*:}"
        dataset_id="${dataset/-/}"
        model_id="${model//-/_}"
        exp_name="${TAG}_${dataset_id}_${model_id}_${method_name}_a${ALPHA}_s${seed}"

        "$PYTHON_BIN" run.py \
          --gpu "$GPU" --gpu_frac "$GPU_FRAC" --ray_core "$RAY_CORE" \
          --n_clients "$CLIENTS" --dirichlet_alpha "$ALPHA" --dataset "$dataset" \
          --model "$model" --method avg --const "$const_flag" \
          --opt SGD --batch 50 --local_iter "$LOCAL_EPOCHS" --global_iter "$ROUNDS" \
          --local_lr 0.01 --wd 1e-4 --global_lr 1.0 --momentum 0.9 \
          --sample_ratio 1.0 --bn true --riemann false --localrie false \
          --seed "$seed" --diagnostics false --save_model false --save_data false \
          --exp_name "$exp_name"
      done
    done
  done
done

"$PYTHON_BIN" scripts/summarize_results.py --prefix "$TAG"
