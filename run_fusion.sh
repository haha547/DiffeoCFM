#!/bin/bash
# run_fusion.sh
# -------------
# Evaluate P/S cohort covariance fusion methods for ASD/TD classification.
# Runs evaluate_fusion.py across all available datasets.
#
# Usage:
#   chmod +x run_fusion.sh
#   ./run_fusion.sh
#   ./run_fusion.sh --label-row 0          # use Primary labels (default: 1=Secondary)
#   ./run_fusion.sh --methods "arith_mean log_euclidean"
#
# No training required — works directly on raw covariance files.
# Results: figures/fusion_classification.csv, figures/fusion_predictions.csv

# --- defaults ---
LABEL_ROW=1        # groupinfo row for diagnosis labels: 0=Primary, 1=Secondary
METHOD_ARGS=""     # blank → run all fusion methods in fuse.py

for arg in "$@"; do
    case $arg in
        --label-row=*)  LABEL_ROW="${arg#*=}" ;;
        --label-row)    shift; LABEL_ROW="$1" ;;
        --methods=*)    METHOD_ARGS="--methods ${arg#*=}" ;;
        --methods)      shift; METHOD_ARGS="--methods $1" ;;
    esac
done

DATASETS=("cov_2s_0ov" "cov_2s_50ov" "cov_4s_0ov" "cov_4s_50ov")

# resolve python
PYTHON=$(command -v python3 2>/dev/null || command -v python)
PY_VER=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
if [ "$PY_VER" -lt 3 ]; then
    echo "ERROR: Python 3 required, but $PYTHON is Python $PY_VER"
    exit 1
fi
echo "Using Python: $PYTHON  (version $("$PYTHON" --version 2>&1))"
echo "Label row: $LABEL_ROW  (0=Primary, 1=Secondary)"
[ -n "$METHOD_ARGS" ] && echo "Methods: $METHOD_ARGS" || echo "Methods: all"
echo ""

# collect available data directories
DATA_ARGS=()
for ds in "${DATASETS[@]}"; do
    if [ -d "./$ds" ]; then
        DATA_ARGS+=("./$ds")
    else
        echo "  SKIP: ./$ds not found"
    fi
done

if [ ${#DATA_ARGS[@]} -eq 0 ]; then
    echo "ERROR: No data directories found."
    exit 1
fi

echo "====== Fusion Evaluation (evaluate_fusion.py) ======"
"$PYTHON" evaluate_fusion.py \
    --data "${DATA_ARGS[@]}" \
    --groupinfo-row "$LABEL_ROW" \
    $METHOD_ARGS \
    || echo "  WARNING: evaluate_fusion.py failed"

echo ""
echo "====== Done ======"
echo "Results saved to:"
echo "  figures/fusion_classification.csv"
echo "  figures/fusion_predictions.csv"
