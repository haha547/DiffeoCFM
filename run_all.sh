#!/bin/bash
# run_all.sh
# ----------
# Batch run all training + evaluation for both directions (A and B).
# Run from the DiffeoCFM directory:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# Add --debug to any command for a quick smoke-test first.

set -e  # stop on first error

REGION="s"
DATASETS=("cov_2s_0ov" "cov_2s_50ov" "cov_4s_0ov" "cov_4s_50ov")

# =============================================================================
# Direction A — train with EC/CPT conditioning (train_custom.py)
# =============================================================================
echo "====== Direction A: Training ======"
for ds in "${DATASETS[@]}"; do
    echo "--- $ds ---"
    python train_custom.py --data "./$ds" --region "$REGION"
done

echo ""
echo "====== Direction A: Evaluate EC/CPT classification (evaluate_custom.py) ======"
python evaluate_custom.py --data "${DATASETS[@]/#/./}" --region "$REGION"

echo ""
echo "====== Direction A: Evaluate ASD/TD classification (evaluate_a.py) ======"
python evaluate_a.py --data "${DATASETS[@]/#/./}" --region "$REGION"

# =============================================================================
# Direction B — train with 4-class conditioning (train_b.py)
# =============================================================================
echo ""
echo "====== Direction B: Training ======"
for ds in "${DATASETS[@]}"; do
    echo "--- $ds ---"
    python train_b.py --data "./$ds" --region "$REGION"
done

echo ""
echo "====== Direction B: Evaluate ASD/TD classification (evaluate_b.py) ======"
python evaluate_b.py --data "${DATASETS[@]/#/./}" --region "$REGION"

echo ""
echo "====== All done. Results in figures/ ======"
