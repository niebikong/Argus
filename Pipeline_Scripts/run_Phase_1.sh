#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ju/Desktop/CL"
cd "${ROOT}"
ARGUS_ROOT="${ROOT}/Argus"

GPU="${GPU:-1}"
DATASET="${DATASET:-D1}"  # only D1 is supported in Argus pipeline
NOISE="${NOISE:-0.1}"
NOISE_TYPE="${NOISE_TYPE:-sym}"
CUT="${CUT:-0.1}"
SEED="${SEED:-2026}"
TEST_SIZE="${TEST_SIZE:-0.2}"

case "${DATASET^^}" in
  D1)
    CSV_PATH_DEFAULT="${ROOT}/Ref_codes/label_encodered_malicious_TLS-1_processed.csv"
    ;;
  *)
    echo "[pipeline] ERROR: unsupported DATASET='${DATASET}'. Only D1 is supported." >&2
    exit 2
    ;;
esac
DATASET="${DATASET^^}"
CSV_PATH="${CSV_PATH:-${CSV_PATH_DEFAULT}}"
SAVE_DIR="${SAVE_DIR:-${ROOT}/Argus/logs_csv/${DATASET,,}}"
TEST_TOKEN="$(python3 -c "print(int(round(float('${TEST_SIZE}')*100)))")"
CACHE_NPZ="${CACHE_NPZ:-${ROOT}/Argus/cache/tls_csv_${DATASET,,}_${NOISE_TYPE}_n${NOISE//./}_seed${SEED}_test${TEST_TOKEN}.npz}"

assert_under_argus() {
  local p="$1"
  local rp
  rp="$(realpath -m "$p")"
  case "$rp" in
    "${ARGUS_ROOT}"/*) ;;
    *)
      echo "[pipeline] ERROR: output path must be under ${ARGUS_ROOT}, got: ${rp}" >&2
      exit 2
      ;;
  esac
}

assert_under_argus "$SAVE_DIR"
assert_under_argus "$CACHE_NPZ"

echo "[pipeline] GPU=$GPU"
echo "[pipeline] DATASET=$DATASET"
echo "[pipeline] CSV_PATH=$CSV_PATH"
echo "[pipeline] NOISE=$NOISE"
echo "[pipeline] NOISE_TYPE=$NOISE_TYPE"
echo "[pipeline] CUT=$CUT"
echo "[pipeline] SEED=$SEED"
echo "[pipeline] TEST_SIZE=$TEST_SIZE"
echo "[pipeline] SAVE_DIR=$SAVE_DIR"
echo "[pipeline] CACHE_NPZ=$CACHE_NPZ"
echo "[pipeline] Dataset split policy:"
echo "[pipeline]   - label_encodered_malicious_TLS-1_processed.csv: known={0,3..22}, unknown={1,2}"
echo "[pipeline]   - D2 is reserved as unknown_traffic_detect sample source in Stage_2"

CUDA_VISIBLE_DEVICES="$GPU" python3 Argus/Phase1_Training/Stage_1.py \
  --dataset "$DATASET" \
  --csv_path "$CSV_PATH" \
  --cache_npz "$CACHE_NPZ" \
  --save_dir "$SAVE_DIR" \
  --noise "$NOISE" \
  --noise_type "$NOISE_TYPE" \
  --seed "$SEED" \
  --test_size "$TEST_SIZE" \
  "$@"

CUDA_VISIBLE_DEVICES="$GPU" python3 Argus/Phase1_Training/Stage_2.py \
  --dataset "$DATASET" \
  --csv_path "$CSV_PATH" \
  --cache_npz "$CACHE_NPZ" \
  --save_dir "$SAVE_DIR" \
  --noise "$NOISE" \
  --noise_type "$NOISE_TYPE" \
  --cut "$CUT" \
  --seed "$SEED" \
  --test_size "$TEST_SIZE" \
  "$@"

CUDA_VISIBLE_DEVICES="$GPU" python3 Argus/Phase1_Training/Stage_3.py \
  --dataset "$DATASET" \
  --csv_path "$CSV_PATH" \
  --cache_npz "$CACHE_NPZ" \
  --save_dir "$SAVE_DIR" \
  --noise "$NOISE" \
  --noise_type "$NOISE_TYPE" \
  --cut "$CUT" \
  --seed "$SEED" \
  --test_size "$TEST_SIZE" \
  "$@"
