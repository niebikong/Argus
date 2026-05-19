#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ju/Desktop/CL"
ARGUS_ROOT="${ROOT}/Argus"

GPU="${GPU:-0}"
DATASET="${DATASET:-D1}"  # only D1 is supported

NOISE="${NOISE:-0.1}"
NOISE_TYPE="${NOISE_TYPE:-sym}"
CUT="${CUT:-0.1}"
TRAIN_SUBSET_CUT="${TRAIN_SUBSET_CUT:-0.1}"
STAGE="${STAGE:-Stage_3}"  # Stage_1 | Stage_2 | Stage_3
TEST_SIZE="${TEST_SIZE:-0.2}"
TRAIN_LABEL_SOURCE="${TRAIN_LABEL_SOURCE:-noisy}"
TRAIN_SUBSET_STRATEGY="${TRAIN_SUBSET_STRATEGY:-0.0}"
MC_RUNS="${MC_RUNS:-1}"
MC_DROPOUT_P="${MC_DROPOUT_P:-0.1}"
SUBSPACE_DIM="${SUBSPACE_DIM:-50}"
TRAIN_KNOWN_QUANTILE="${TRAIN_KNOWN_QUANTILE:-0.95}"
FEATURE_TAG="${FEATURE_TAG:-internal_12}"
AUC_FIG_DIR="${AUC_FIG_DIR:-${ROOT}/Argus/Phase2/AUC_figs}"
AUC_PLOT="${AUC_PLOT:-0}"
AUC_FONT_FAMILY="${AUC_FONT_FAMILY:-Times New Roman}"
D1_EXTRA_OOD_FROM_D2="${D1_EXTRA_OOD_FROM_D2:-5000}"
D1_EXTRA_OOD_CSV_PATH="${D1_EXTRA_OOD_CSV_PATH:-${ROOT}/Ref_codes/TLS1.3_like_TLS1.2_processed.csv}"
OOD_CAP_RANDOM="${OOD_CAP_RANDOM:-5000}"

# TRAIN_SUBSET_STRATEGY supports:
# 1) "all" or "cut"
# 2) a numeric value (e.g. "0.1"), treated as CUT threshold with strategy="cut"
if [[ "${TRAIN_SUBSET_STRATEGY}" =~ ^[0-9]*\.?[0-9]+$ ]]; then
  TRAIN_SUBSET_CUT="${TRAIN_SUBSET_STRATEGY}"
  TRAIN_SUBSET_STRATEGY="cut"
fi
if [[ "${TRAIN_SUBSET_STRATEGY}" != "all" && "${TRAIN_SUBSET_STRATEGY}" != "cut" ]]; then
  echo "[unknown_traffic_detect] ERROR: unsupported TRAIN_SUBSET_STRATEGY='${TRAIN_SUBSET_STRATEGY}'. Use 'all', 'cut', or a numeric threshold like 0.1" >&2
  exit 2
fi

case "${DATASET^^}" in
  D1)
    CSV_PATH_DEFAULT="${ROOT}/Ref_codes/label_encodered_malicious_TLS-1_processed.csv"
    ;;
  *)
    echo "[unknown_traffic_detect] ERROR: unsupported DATASET='${DATASET}'. Only D1 is supported." >&2
    exit 2
    ;;
esac

DATASET="${DATASET^^}"
CSV_PATH="${CSV_PATH:-${CSV_PATH_DEFAULT}}"
SAVE_DIR="${SAVE_DIR:-${ROOT}/Argus/logs_csv/${DATASET,,}}"

NOISE_TOKEN="${NOISE//./}"
SEED="${SEED:-2026}"
TEST_TOKEN="$(python3 -c "print(int(round(float('${TEST_SIZE}')*100)))")"
CACHE_NPZ="${CACHE_NPZ:-${ROOT}/Argus/cache/tls_csv_${DATASET,,}_${NOISE_TYPE}_n${NOISE_TOKEN}_seed${SEED}_test${TEST_TOKEN}.npz}"

NOISE_PCT="$(python3 -c "print(int(float('${NOISE}') * 100))")"
CUT_PCT="$(python3 -c "print(int(float('${CUT}') * 100))")"
BASE_RUN="${NOISE_TYPE}_${NOISE_PCT}"
STAGE1_RUN="${BASE_RUN}_Stage_1"
STAGE2_RUN="${BASE_RUN}_PL_cut${CUT_PCT}_Stage_2"
STAGE3_RUN="${BASE_RUN}_PL_cut${CUT_PCT}_Stage_3"

case "${STAGE}" in
  Stage_1|stage1)
    STAGE="Stage_1"
    CKPT_DIR="${SAVE_DIR}/${STAGE1_RUN}"
    ;;
  Stage_2|stage2)
    STAGE="Stage_2"
    CKPT_DIR="${SAVE_DIR}/${STAGE2_RUN}"
    ;;
  Stage_3|stage3)
    STAGE="Stage_3"
    CKPT_DIR="${SAVE_DIR}/${STAGE3_RUN}"
    ;;
  *)
    echo "[unknown_traffic_detect] ERROR: unsupported STAGE='${STAGE}'. Use one of: Stage_1, Stage_2, Stage_3" >&2
    exit 2
    ;;
esac

CHECKPOINT="${CHECKPOINT:-${CKPT_DIR}/checkpoint.pth.tar}"
if [[ ! -f "${CHECKPOINT}" ]]; then
  LEGACY_CKPT_DIR="${ROOT}/Argus/logs_csv/${BASE_RUN}"
  case "${STAGE}" in
    Stage_1)
      LEGACY_CKPT_DIR="${SAVE_DIR}/tlscsv_deepresnet_${NOISE_TYPE}_${NOISE_PCT}"
      ;;
    Stage_2)
      LEGACY_CKPT_DIR="${SAVE_DIR}/tlscsv_deepresnet_${NOISE_TYPE}_${NOISE_PCT}_PL_cut${CUT_PCT}"
      ;;
    Stage_3)
      LEGACY_CKPT_DIR="${SAVE_DIR}/tlscsv_deepresnet_${NOISE_TYPE}_${NOISE_PCT}_PL_cut${CUT_PCT}_pseudo1"
      ;;
  esac
  LEGACY_CHECKPOINT="${LEGACY_CKPT_DIR}/checkpoint.pth.tar"
  if [[ -f "${LEGACY_CHECKPOINT}" ]]; then
    echo "[unknown_traffic_detect] WARN: dataset-scoped checkpoint not found, fallback to legacy path: ${LEGACY_CHECKPOINT}"
    CHECKPOINT="${LEGACY_CHECKPOINT}"
  fi
fi

WORK_DIR="${WORK_DIR:-${ROOT}/Argus/Phase2/runs/${DATASET,,}/${BASE_RUN}_cut${CUT_PCT}_${STAGE}}"

assert_under_argus() {
  local p="$1"
  local rp
  rp="$(realpath -m "$p")"
  case "$rp" in
    "${ARGUS_ROOT}"/*) ;;
    *)
      echo "[unknown_traffic_detect] ERROR: output path must be under ${ARGUS_ROOT}, got: ${rp}" >&2
      exit 2
      ;;
  esac
}

assert_under_argus "$SAVE_DIR"
assert_under_argus "$CACHE_NPZ"
assert_under_argus "$WORK_DIR"
assert_under_argus "$AUC_FIG_DIR"

cd "${ROOT}"

echo "[unknown_traffic_detect] GPU=${GPU}"
echo "[unknown_traffic_detect] DATASET=${DATASET}"
echo "[unknown_traffic_detect] WORK_DIR=${WORK_DIR}"
echo "[unknown_traffic_detect] CHECKPOINT=${CHECKPOINT}"
echo "[unknown_traffic_detect] CSV_PATH=${CSV_PATH}"
echo "[unknown_traffic_detect] CACHE_NPZ=${CACHE_NPZ}"
echo "[unknown_traffic_detect] SAVE_DIR=${SAVE_DIR}"
echo "[unknown_traffic_detect] STAGE=${STAGE}"
echo "[unknown_traffic_detect] CUT=${CUT}"
echo "[unknown_traffic_detect] TRAIN_SUBSET_CUT=${TRAIN_SUBSET_CUT}"
echo "[unknown_traffic_detect] TRAIN_SUBSET_STRATEGY=${TRAIN_SUBSET_STRATEGY}"
echo "[unknown_traffic_detect] NOISE=${NOISE}"
echo "[unknown_traffic_detect] NOISE_TYPE=${NOISE_TYPE}"
echo "[unknown_traffic_detect] SEED=${SEED}"
echo "[unknown_traffic_detect] D1_EXTRA_OOD_FROM_D2=${D1_EXTRA_OOD_FROM_D2}"
echo "[unknown_traffic_detect] D1_EXTRA_OOD_CSV_PATH=${D1_EXTRA_OOD_CSV_PATH}"
echo "[unknown_traffic_detect] OOD_CAP_RANDOM=${OOD_CAP_RANDOM}"
echo "[unknown_traffic_detect] MC_DROPOUT_P=${MC_DROPOUT_P}"
echo "[unknown_traffic_detect] Dataset split policy:"
echo "[unknown_traffic_detect]   - D1 known={0,3..22}, unknown={1,2}"
echo "[unknown_traffic_detect]   - D2 sampled data are appended as unknown traffic"

CUDA_VISIBLE_DEVICES="${GPU}" python3 Argus/Phase2/extract_features.py \
  --work_dir "${WORK_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${DATASET}" \
  --csv_path "${CSV_PATH}" \
  --cache_npz "${CACHE_NPZ}" \
  --noise "${NOISE}" \
  --noise_type "${NOISE_TYPE}" \
  --test_size "${TEST_SIZE}" \
  --seed "${SEED}" \
  --train_label_source "${TRAIN_LABEL_SOURCE}" \
  --train_subset_strategy "${TRAIN_SUBSET_STRATEGY}" \
  --cut "${TRAIN_SUBSET_CUT}" \
  --mc_runs "${MC_RUNS}" \
  --mc_dropout_p "${MC_DROPOUT_P}" \
  --d1_extra_ood_from_d2 "${D1_EXTRA_OOD_FROM_D2}" \
  --d1_extra_ood_csv_path "${D1_EXTRA_OOD_CSV_PATH}" \
  --ood_cap_random "${OOD_CAP_RANDOM}" \
  --feature_tag "${FEATURE_TAG}" \
  --stage "${STAGE}"

EVAL_ARGS=(
  --work_dir "${WORK_DIR}"
  --feature_tag "${FEATURE_TAG}"
  --subspace_dim "${SUBSPACE_DIM}"
  --train_known_quantile "${TRAIN_KNOWN_QUANTILE}"
  --stage "${STAGE}"
  --auc_fig_dir "${AUC_FIG_DIR}"
  --font_family "${AUC_FONT_FAMILY}"
)
# Disable AUC plotting.
# if [[ "${AUC_PLOT}" == "1" ]]; then
#   EVAL_ARGS+=(--plot_auc)
# fi

python3 Argus/Phase2/evaluate.py \
  "${EVAL_ARGS[@]}" \
  "$@"
