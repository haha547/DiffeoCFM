#!/bin/bash
# run_all.sh
# ----------
# Batch run all training + evaluation for both directions (A and B).
#
# Usage (from the DiffeoCFM directory):
#   chmod +x run_all.sh
#   ./run_all.sh
#   ./run_all.sh --debug          # quick smoke-test (10 epochs, 2 splits)
#   ./run_all.sh --region p       # use P-region channels (default: s)
#
# Requirements:
#   - Activate your conda/venv environment BEFORE calling this script, e.g.:
#       conda activate <env_name> && ./run_all.sh
#   - Data directories (cov_2s_0ov, etc.) must exist in the current directory.
#   - GroupInfo.mat must exist in the current directory.

# --- parse optional flags ---
REGION="s"
MAX_AUG=5          # pool size: generate this many synthetic samples per real sample
AUG_TEST="1 2 3 5" # aug factors to test at evaluation time (must be ≤ MAX_AUG)
EXTRA_ARGS=""
for arg in "$@"; do
    case $arg in
        --debug)      EXTRA_ARGS="$EXTRA_ARGS --debug" ;;
        --region=*)   REGION="${arg#*=}" ;;
        --region)     shift; REGION="$1" ;;
        --max-aug=*)  MAX_AUG="${arg#*=}" ;;
        --max-aug)    shift; MAX_AUG="$1" ;;
        --aug=*)      AUG_TEST="${arg#*=}" ;;
        --aug)        shift; AUG_TEST="$1" ;;
    esac
done

DATASETS=("cov_2s_0ov" "cov_2s_50ov" "cov_4s_0ov" "cov_4s_50ov")

# resolve python executable (prefer python3 if python is Python 2)
PYTHON=$(command -v python3 2>/dev/null || command -v python)
PY_VER=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
if [ "$PY_VER" -lt 3 ]; then
    echo "ERROR: Python 3 required, but $PYTHON is Python $PY_VER"
    exit 1
fi
echo "Using Python: $PYTHON  (version $("$PYTHON" --version 2>&1))"
echo "Region:  $REGION"
echo "MAX_AUG: $MAX_AUG  (pool size per training sample)"
echo "AUG_TEST: $AUG_TEST  (aug factors tested at evaluation)"
echo ""

# helper: skip dataset if directory is missing
run_train() {
    local script="$1"; shift
    local ds="$1"; shift
    if [ ! -d "./$ds" ]; then
        echo "  WARNING: ./$ds not found, skipping."
        return 0
    fi
    "$PYTHON" "$script" --data "./$ds" --region "$REGION" $EXTRA_ARGS "$@"
}

# =============================================================================
# Direction A — train with EC/CPT conditioning
# =============================================================================
echo "====== Direction A: Training (train_custom.py) ======"
for ds in "${DATASETS[@]}"; do
    echo "--- $ds ---"
    run_train train_custom.py "$ds" --max-aug "$MAX_AUG"
done

echo ""
echo "====== Direction A: Evaluate EC/CPT (evaluate_custom.py) ======"
DATA_ARGS=()
for ds in "${DATASETS[@]}"; do
    [ -d "./$ds" ] && DATA_ARGS+=("./$ds")
done

if [ ${#DATA_ARGS[@]} -gt 0 ]; then
    "$PYTHON" evaluate_custom.py --data "${DATA_ARGS[@]}" --region "$REGION" || \
        echo "  WARNING: evaluate_custom.py failed (no results yet?)"

    echo ""
    echo "====== Direction A: Evaluate ASD/TD (evaluate_a.py) ======"
    "$PYTHON" evaluate_a.py --data "${DATA_ARGS[@]}" \
        --region "$REGION" --aug $AUG_TEST || \
        echo "  WARNING: evaluate_a.py failed"
else
    echo "  No Direction-A data found, skipping evaluation."
fi

# =============================================================================
# Direction A_inter — inter_gram region (inter-brain coupling as ASD predictor)
# =============================================================================
echo ""
echo "====== Direction A_inter: Training (train_custom.py --region inter_gram) ======"
for ds in "${DATASETS[@]}"; do
    echo "--- $ds ---"
    if [ ! -d "./$ds" ]; then
        echo "  WARNING: ./$ds not found, skipping."
        continue
    fi
    "$PYTHON" train_custom.py --data "./$ds" --region inter_gram --max-aug "$MAX_AUG" $EXTRA_ARGS
done

echo ""
echo "====== Direction A_inter: Evaluate ASD/TD (evaluate_a.py --region inter_gram) ======"
if [ ${#DATA_ARGS[@]} -gt 0 ]; then
    "$PYTHON" evaluate_a.py --data "${DATA_ARGS[@]}" \
        --region inter_gram --aug $AUG_TEST || \
        echo "  WARNING: evaluate_a.py (inter_gram) failed"
else
    echo "  No data found, skipping."
fi

# =============================================================================
# Direction B — train with 4-class conditioning
# =============================================================================
echo ""
echo "====== Direction B: Training (train_b.py) ======"
for ds in "${DATASETS[@]}"; do
    echo "--- $ds ---"
    run_train train_b.py "$ds" --max-aug "$MAX_AUG"
done

echo ""
echo "====== Direction B: Evaluate ASD/TD (evaluate_b.py) ======"
if [ ${#DATA_ARGS[@]} -gt 0 ]; then
    "$PYTHON" evaluate_b.py --data "${DATA_ARGS[@]}" \
        --region "$REGION" --aug $AUG_TEST || \
        echo "  WARNING: evaluate_b.py failed"

    echo ""
    echo "====== Plotting augmentation sweep (plot_aug.py) ======"
    "$PYTHON" plot_aug.py || echo "  WARNING: plot_aug.py failed"
else
    echo "  No Direction-B data found, skipping evaluation."
fi

echo ""
echo "====== Direction B_inter: Training (train_b.py --region inter_gram) ======"
for ds in "${DATASETS[@]}"; do
    echo "--- $ds ---"
    if [ ! -d "./$ds" ]; then
        echo "  WARNING: ./$ds not found, skipping."
        continue
    fi
    "$PYTHON" train_b.py --data "./$ds" --region inter_gram --max-aug "$MAX_AUG" $EXTRA_ARGS
done

echo ""
echo "====== Direction B_inter: Evaluate ASD/TD (evaluate_b.py --region inter_gram) ======"
if [ ${#DATA_ARGS[@]} -gt 0 ]; then
    "$PYTHON" evaluate_b.py --data "${DATA_ARGS[@]}" \
        --region inter_gram --aug $AUG_TEST || \
        echo "  WARNING: evaluate_b.py (inter_gram) failed"
else
    echo "  No data found, skipping."
fi

echo ""
echo "====== All done. CSVs and figures in figures/ ======"
