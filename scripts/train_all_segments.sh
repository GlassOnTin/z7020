#!/bin/bash
# train_all_segments.sh — Train all Bad Apple segments in parallel
#
# Uses GNU parallel to run multiple training processes.
# Each process gets 3 CPU threads (OMP_NUM_THREADS=3).
# With 4 parallel jobs on 12 cores → full utilization.
#
# Usage: bash scripts/train_all_segments.sh [epochs] [jobs]

EPOCHS=${1:-3000}
JOBS=${2:-4}
TOTAL=658

echo "=== Training $TOTAL segments, $EPOCHS epochs, $JOBS parallel jobs ==="
echo "Estimated time: ~$(( TOTAL * 90 / JOBS / 60 )) minutes"
echo ""

export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3

train_one() {
    seg=$1
    epochs=$2
    # Check if already trained
    pt_file="bad_apple/weights/segment_${seg}.pt"
    if [ -f "$pt_file" ]; then
        echo "seg $seg: already exists, skipping"
        return 0
    fi

    python3 scripts/train_bad_apple.py \
        --segment "$seg" \
        --epochs "$epochs" \
        --samples 80000 \
        --batch-size 40000 \
        --export-fpga \
        2>&1 | grep -E "(Training segment|Training complete|Results:|Avg PSNR|Binary|Exported)"

    echo "seg $seg: done"
}
export -f train_one

# Generate segment list and run in parallel
seq 0 $((TOTAL - 1)) | xargs -P "$JOBS" -I{} bash -c "train_one {} $EPOCHS"

echo ""
echo "=== Training complete ==="
trained=$(ls bad_apple/weights/segment_*.bin 2>/dev/null | wc -l)
echo "Trained segments: $trained / $TOTAL"
total_bytes=$(du -sb bad_apple/weights/*.bin 2>/dev/null | awk '{s+=$1}END{print s+0}')
echo "Total weight data: $((total_bytes / 1024)) KB"
