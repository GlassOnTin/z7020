#!/bin/bash
# Retrain all Bad Apple segments with 2 parallel workers
# Usage: bash scripts/retrain_all.sh [epochs] [w0_start] [w1_start]

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

EPOCHS=${1:-5000}
W0_START=${2:-0}
W1_START=${3:-329}
TOTAL=658
HALF=329

echo "Retraining segments with $EPOCHS epochs, 2 workers, OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Worker 0: segments $W0_START-$((HALF-1))"
echo "Worker 1: segments $W1_START-$((TOTAL-1))"
echo ""

cd "$(dirname "$0")/.."

# Worker 0: first half
(
    for seg in $(seq $W0_START $((HALF-1))); do
        python3 scripts/train_bad_apple.py --segment $seg --epochs $EPOCHS --export-fpga 2>&1 | \
            grep -E "(Training segment|Epoch.*(000/|$EPOCHS/)|Final|Saved|Float PSNR|Exported)"
    done
) >> /tmp/ba_worker0.log 2>&1 &
PID0=$!

# Worker 1: second half
(
    for seg in $(seq $W1_START $((TOTAL-1))); do
        python3 scripts/train_bad_apple.py --segment $seg --epochs $EPOCHS --export-fpga 2>&1 | \
            grep -E "(Training segment|Epoch.*(000/|$EPOCHS/)|Final|Saved|Float PSNR|Exported)"
    done
) >> /tmp/ba_worker1.log 2>&1 &
PID1=$!

echo "Worker 0 PID: $PID0 (log: /tmp/ba_worker0.log)"
echo "Worker 1 PID: $PID1 (log: /tmp/ba_worker1.log)"
echo ""
echo "Monitor: tail -f /tmp/ba_worker0.log /tmp/ba_worker1.log"
echo "Progress: ls bad_apple/weights/segment_*.pt | wc -l"

wait $PID0
echo "Worker 0 done (exit $?)"
wait $PID1
echo "Worker 1 done (exit $?)"
echo "All done."
