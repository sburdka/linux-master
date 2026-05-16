#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# mthp_bestfit_tflite_bench.sh — TFLite MobileNet-SSD mTHP fragmentation benchmark
#
# Runs object detection inference using TensorFlow Lite (TFLite) with
# MobileNet-SSD models, measuring mTHP fragmentation metrics before and
# after the workload to compare baseline vs mthp_bestfit kernels.
#
# Why TFLite MobileNet-SSD exposes mTHP fragmentation
# ─────────────────────────────────────────────────────
# TFLite allocates ONE large anonymous mmap per loaded model (the "tensor
# arena").  Arena sizes for common detection models:
#
#   SSD-MobileNet-v1 int8  300×300 →  3.7 MB arena  (order-9 PMD attempt)
#   SSD-MobileNet-v2 int8  320×320 →  5.5 MB arena  (order-9 × 2)
#   SSD-MobileNet-v1 fp32  300×300 → 13.0 MB arena  (order-9 × 6)
#   EfficientDet-Lite0 int8 320×320→  7.5 MB arena  (order-9 × 3)
#
# A real phone camera pipeline runs 3-4 of these concurrently (~30 MB live).
# After repeated load/infer/unload cycles the buddy allocator fragments:
#   → PMD (2 MB) hugepage requests fail → compact_stall triggers
#   → Inference latency spikes by 15-40 ms per stall event
#
# With mthp_bestfit (pressure-aware):
#   → Detects fragmented buddy → selects order-8 (1 MB) instead of PMD
#   → No compaction → no latency spike → MemAvailable stays 5-15% higher
#
# Backend priority:
#   1. benchmark_model (official TFLite benchmark tool)  ← real workload
#   2. tflite_mobilenetssd_workload (synthetic C)        ← accurate simulation
#
# Usage:
#   ./mthp_bestfit_tflite_bench.sh [--tag baseline|bestfit]
#                                  [--model /path/to/detect.tflite]
#                                  [--benchmark-model /path/to/benchmark_model]
#                                  [--num-runs N]         (default: 500)
#                                  [--num-threads N]      (default: 4)
#                                  [--streams N]          (default: 4)
#                                  [--iterations N]       (synthetic, default: 300)
#
# Compare two boards:
#   ./mthp_bestfit_tflite_bench.sh \
#       --compare results_tflite_baseline_*.txt results_tflite_bestfit_*.txt
#
# Getting TFLite benchmark_model binary:
#   Option A — Download prebuilt (check TensorFlow GitHub releases page):
#     Search "tensorflow releases benchmark_model android_arm64" or
#     "benchmark_model linux_aarch64" for the right binary for your board.
#
#   Option B — Build from source (requires Bazel):
#     git clone https://github.com/tensorflow/tensorflow
#     cd tensorflow
#     bazel build -c opt \
#         //tensorflow/lite/tools/benchmark:benchmark_model
#     # binary at: bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model
#
# Getting MobileNet-SSD TFLite models:
#   Option A — TFLite Model Maker or TF Hub:
#     The model file is commonly named one of:
#       detect.tflite          (Android object detection API default)
#       ssd_mobilenet_v1.tflite
#       ssd_mobilenet_v2.tflite
#       efficientdet_lite0.tflite
#
#   Option B — Android Studio ML Model Binding exports .tflite files.
#
#   Option C — Convert from TensorFlow SavedModel:
#     tflite_convert \
#       --saved_model_dir=/path/to/saved_model \
#       --output_file=model.tflite
#
# Dependencies: benchmark_model binary (optional), gcc (for synthetic fallback)

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
TAG="run"
MODEL=""
BENCHMARK_MODEL_BIN=""
NUM_RUNS=500
NUM_THREADS=4
STREAMS=4
ITERATIONS=300
COMPARE_MODE=0
COMPARE_BASE=""
COMPARE_BEST=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)              TAG="$2";               shift 2 ;;
        --model)            MODEL="$2";             shift 2 ;;
        --benchmark-model)  BENCHMARK_MODEL_BIN="$2"; shift 2 ;;
        --num-runs)         NUM_RUNS="$2";          shift 2 ;;
        --num-threads)      NUM_THREADS="$2";       shift 2 ;;
        --streams)          STREAMS="$2";           shift 2 ;;
        --iterations)       ITERATIONS="$2";        shift 2 ;;
        --compare)
            COMPARE_MODE=1
            COMPARE_BASE="$2"
            COMPARE_BEST="$3"
            shift 3
            ;;
        -h|--help)
            sed -n '2,/^# Dependencies/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Compare mode ──────────────────────────────────────────────────────────────
if [[ "$COMPARE_MODE" -eq 1 ]]; then
    if [[ ! -f "$COMPARE_BASE" || ! -f "$COMPARE_BEST" ]]; then
        echo "ERROR: result files not found." >&2
        echo "  Baseline: $COMPARE_BASE" >&2
        echo "  Bestfit:  $COMPARE_BEST" >&2
        exit 1
    fi

    extract_delta() {
        local key=$1 file=$2
        local before after
        before=$(grep "^$key " "$file" | head -1 | awk '{print $2}')
        after=$(grep  "^$key " "$file" | tail -1 | awk '{print $2}')
        echo $(( ${after:-0} - ${before:-0} ))
    }

    printf "\n%-42s %12s %12s %10s\n" "Metric" "Baseline" "Bestfit" "Change"
    printf "%s\n" "$(printf '─%.0s' {1..80})"

    for key in nr_deferred_split_page compact_stall thp_fault_alloc \
               thp_fault_fallback thp_split_page; do
        b=$(extract_delta "$key" "$COMPARE_BASE")
        n=$(extract_delta "$key" "$COMPARE_BEST")
        if [[ "$b" -ne 0 ]]; then
            pct=$(( (n - b) * 100 / b ))
            printf "%-42s %12d %12d %+9d%%\n" "$key" "$b" "$n" "$pct"
        else
            printf "%-42s %12d %12d %10s\n"   "$key" "$b" "$n" "n/a"
        fi
    done

    printf "%s\n" "$(printf '─%.0s' {1..80})"

    for label in "MemAvailable:" "AnonHugePages:"; do
        b=$(grep "^$label" "$COMPARE_BASE" | tail -1 | awk '{print $2}')
        n=$(grep "^$label" "$COMPARE_BEST"  | tail -1 | awk '{print $2}')
        if [[ -n "$b" && "$b" -ne 0 ]]; then
            pct=$(( (n - b) * 100 / b ))
            printf "%-42s %9d kB %9d kB %+9d%%\n" \
                   "${label%:} (after run)" "$b" "$n" "$pct"
        fi
    done

    printf "\n"

    if grep -q "mthp_bestfit_stats" "$COMPARE_BEST" 2>/dev/null; then
        echo "── mthp_bestfit_stats (bestfit board) ──"
        sed -n '/mthp_bestfit_stats/,/^===/p' "$COMPARE_BEST" \
            | grep -v "^===" | head -40
        echo ""
    fi

    echo "── TFLite inference timing (if benchmark_model was used) ──"
    for f in "$COMPARE_BASE" "$COMPARE_BEST"; do
        label="Baseline"
        [[ "$f" == "$COMPARE_BEST" ]] && label="Bestfit "
        avg=$(grep -i "avg=" "$f" 2>/dev/null | tail -1 | grep -oP 'avg=\K[0-9.]+' || echo "n/a")
        p99=$(grep -i "99th" "$f" 2>/dev/null | tail -1 | grep -oP '[0-9.]+' | tail -1 || echo "n/a")
        printf "  %s  avg latency: %s ms   99th pct: %s ms\n" \
               "$label" "$avg" "$p99"
    done

    exit 0
fi

# ── Locate benchmark_model ────────────────────────────────────────────────────
find_benchmark_model() {
    local candidates=(
        "$BENCHMARK_MODEL_BIN"
        "./benchmark_model"
        "$(which benchmark_model 2>/dev/null || true)"
        "./bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model"
        "/usr/local/bin/benchmark_model"
    )
    for c in "${candidates[@]}"; do
        [[ -x "$c" ]] && { echo "$c"; return; }
    done
    echo ""
}

BM=$(find_benchmark_model)

# ── Locate TFLite model ───────────────────────────────────────────────────────
find_model() {
    if [[ -n "$MODEL" && -f "$MODEL" ]]; then
        echo "$MODEL"; return
    fi
    local candidates=(
        "detect.tflite"
        "ssd_mobilenet_v1.tflite"
        "ssd_mobilenet_v2.tflite"
        "ssd_mobilenet_v1_1_default_1.tflite"
        "ssd_mobilenet_v2_coco.tflite"
    )
    for c in "${candidates[@]}"; do
        [[ -f "$c" ]] && { echo "$c"; return; }
    done
    echo ""
}

TFLITE_MODEL=$(find_model)

# ── Locate / build synthetic fallback ─────────────────────────────────────────
SYNTHETIC_BIN=""
find_or_build_synthetic() {
    local bin="$SCRIPT_DIR/tflite_mobilenetssd_workload"

    if [[ -x "$bin" ]]; then
        SYNTHETIC_BIN="$bin"; return 0
    fi

    local src="$SCRIPT_DIR/tflite_mobilenetssd_workload.c"
    if [[ ! -f "$src" ]]; then
        echo "WARNING: tflite_mobilenetssd_workload.c not found" >&2
        return 1
    fi

    if ! command -v gcc &>/dev/null; then
        echo "WARNING: gcc not found" >&2
        return 1
    fi

    echo "Building tflite_mobilenetssd_workload..."
    gcc -O2 -o "$bin" "$src"
    SYNTHETIC_BIN="$bin"
    echo "Built: $bin"
}

# ── Output file ───────────────────────────────────────────────────────────────
OUT="results_tflite_${TAG}_$(date +%Y%m%d_%H%M%S).txt"
echo "Results → $OUT"

# ── Metric snapshot ───────────────────────────────────────────────────────────
snapshot() {
    local label=$1
    {
        echo "=== $label ==="

        echo "--- /proc/meminfo ---"
        grep -E "^MemTotal:|^MemFree:|^MemAvailable:|^Buffers:|^Cached:\
|^AnonHugePages:|^ShmemHugePages:" /proc/meminfo

        echo "--- /proc/vmstat ---"
        grep -E "^nr_deferred_split_page|^compact_stall|^compact_fail\
|^compact_migrate_scanned|^thp_fault_alloc|^thp_fault_fallback\
|^thp_fault_fallback_charge|^thp_split_page|^thp_split_page_failed\
|^thp_zero_page_alloc$|^thp_zero_page_alloc_failed$" /proc/vmstat

        echo "--- /sys hugepage order stats ---"
        for d in /sys/kernel/mm/transparent_hugepage/hugepages-*/; do
            [[ -d "$d/stats" ]] || continue
            order_label=$(basename "$d")
            for f in "$d"stats/*; do
                v=$(cat "$f" 2>/dev/null) || continue
                [[ "$v" != "0" ]] && printf "  %-14s %-32s %s\n" \
                    "$order_label" "$(basename "$f")" "$v"
            done
        done

        echo "--- mthp_bestfit_stats ---"
        if [[ -r /sys/kernel/debug/mthp_bestfit_stats ]]; then
            cat /sys/kernel/debug/mthp_bestfit_stats
        else
            echo "  (not available — baseline kernel or debugfs not mounted)"
        fi

        echo ""
    } | tee -a "$OUT"
}

# ── THP settings for the run ──────────────────────────────────────────────────
# Enable mTHP for anonymous memory across all orders (2-9).
# This is the setting under which bestfit shows its benefit.
enable_mthp() {
    local base="/sys/kernel/mm/transparent_hugepage"
    if [[ -d "$base/hugepages-2048kB" ]]; then
        echo "Enabling mTHP for all anonymous orders..."
        for d in "$base"/hugepages-*/; do
            [[ -f "$d/enabled" ]] && echo always > "$d/enabled" 2>/dev/null || true
        done
        # Main THP setting: madvise (or always for aggressive testing)
        [[ -f "$base/enabled" ]] && echo madvise > "$base/enabled" 2>/dev/null || true
        echo "  done."
    else
        echo "  (mTHP sysfs not present — check kernel config)"
    fi
}

# ── Drop caches and collect before snapshot ───────────────────────────────────
echo "Dropping page caches..."
sync
echo 3 > /proc/sys/vm/drop_caches
sleep 1

enable_mthp
snapshot "BEFORE"

# ── Run workload ──────────────────────────────────────────────────────────────
if [[ -n "$BM" && -n "$TFLITE_MODEL" ]]; then
    echo "Backend: benchmark_model (real TFLite inference)"
    echo "Model:   $TFLITE_MODEL"
    echo "Runs:    $NUM_RUNS   Threads: $NUM_THREADS"
    echo ""

    {
        echo "=== benchmark_model output ==="
        "$BM"                                      \
            --graph="$TFLITE_MODEL"                \
            --num_runs="$NUM_RUNS"                 \
            --num_threads="$NUM_THREADS"           \
            --warmup_runs=20                       \
            --use_nnapi=false                      \
            --report_peak_memory_footprint=true    \
            --enable_op_profiling=false
        echo ""
    } 2>&1 | tee -a "$OUT"

    #
    # Run a second pass with more iterations to increase buddy pressure.
    # The first pass fragments memory; the second pass shows whether
    # compact_stall is triggered to serve subsequent arena allocations.
    #
    echo "Running second pass (memory already fragmented from first pass)..."
    {
        echo "=== benchmark_model second pass ==="
        "$BM"                                      \
            --graph="$TFLITE_MODEL"                \
            --num_runs="$NUM_RUNS"                 \
            --num_threads="$NUM_THREADS"           \
            --warmup_runs=0                        \
            --use_nnapi=false
        echo ""
    } 2>&1 | tee -a "$OUT"

elif [[ -n "$BM" && -z "$TFLITE_MODEL" ]]; then
    echo "WARNING: benchmark_model found but no .tflite model."
    echo ""
    echo "  To get a MobileNet-SSD model:"
    echo "  1. Check your TFLite SDK samples directory for detect.tflite"
    echo "  2. Export from TensorFlow Hub (search: ssd_mobilenet_v2 tflite)"
    echo "  3. Convert: tflite_convert --saved_model_dir=... --output_file=detect.tflite"
    echo ""
    echo "  Then re-run with: --model /path/to/detect.tflite"
    echo ""
    echo "Falling back to synthetic workload..."
    find_or_build_synthetic || { echo "ERROR: no backend available" >&2; exit 1; }

    echo "Backend: tflite_mobilenetssd_workload (synthetic)"
    echo "Streams: $STREAMS   Iterations: $ITERATIONS"
    {
        echo "=== synthetic workload ==="
        "$SYNTHETIC_BIN" "$ITERATIONS" "$STREAMS"
        echo ""
    } 2>&1 | tee -a "$OUT"

else
    echo "benchmark_model not found."
    echo ""
    echo "  To get benchmark_model:"
    echo "  Option A: Build TensorFlow from source:"
    echo "    git clone https://github.com/tensorflow/tensorflow"
    echo "    bazel build -c opt \\"
    echo "      //tensorflow/lite/tools/benchmark:benchmark_model"
    echo "  Option B: Check TensorFlow releases for prebuilt aarch64 binary."
    echo ""
    echo "Falling back to synthetic workload..."
    find_or_build_synthetic || { echo "ERROR: no backend available" >&2; exit 1; }

    echo "Backend: tflite_mobilenetssd_workload (synthetic)"
    echo "Streams: $STREAMS   Iterations: $ITERATIONS"
    echo ""
    {
        echo "=== synthetic workload ==="
        "$SYNTHETIC_BIN" "$ITERATIONS" "$STREAMS"
        echo ""
    } 2>&1 | tee -a "$OUT"
fi

# ── Final snapshot ────────────────────────────────────────────────────────────
snapshot "AFTER"

# ── Inline delta summary ──────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════════════════"
echo "  TFLite MobileNet-SSD mTHP delta summary    tag=$TAG"
echo "══════════════════════════════════════════════════════════════════════════"

inline_delta() {
    local key=$1
    local before after delta
    before=$(grep "^$key " "$OUT" | head -1 | awk '{print $2}')
    after=$(grep  "^$key " "$OUT" | tail -1 | awk '{print $2}')
    delta=$(( ${after:-0} - ${before:-0} ))
    printf "  %-42s +%d\n" "$key" "$delta"
}

inline_delta "nr_deferred_split_page"
inline_delta "compact_stall"
inline_delta "thp_fault_alloc"
inline_delta "thp_fault_fallback"
inline_delta "thp_split_page"

mem_after() { grep "^$1" "$OUT" | tail -1 | awk '{print $2}'; }

printf "  %-42s %s kB\n" "MemAvailable (final)"  "$(mem_after MemAvailable:)"
printf "  %-42s %s kB\n" "AnonHugePages (final)" "$(mem_after AnonHugePages:)"
echo "══════════════════════════════════════════════════════════════════════════"
echo ""
echo "Full results: $OUT"
echo ""
echo "To compare boards:"
echo "  $0 --compare results_tflite_baseline_*.txt results_tflite_bestfit_*.txt"
echo ""

# ── What to look for ─────────────────────────────────────────────────────────
echo "── What the metrics mean for TFLite MobileNet-SSD ──"
echo ""
echo "  nr_deferred_split_page:"
echo "    When TFLite arena is freed, hugepages split into 4 KB pages"
echo "    and sit on the deferred_split list waiting to be returned."
echo "    Bestfit reduces this by right-sizing the hugepage to the arena."
echo ""
echo "  compact_stall:"
echo "    When a new TFLite model is loaded and no 2 MB contiguous block"
echo "    exists, Linux moves pages to create space.  Directly causes"
echo "    inference latency spikes.  Bestfit avoids this by using 1 MB"
echo "    hugepages instead when PMD blocks are unavailable."
echo ""
echo "  MemAvailable (higher = better on bestfit board):"
echo "    Deferred-split pages cannot be immediately reused.  Bestfit"
echo "    keeps hugepages right-sized → fewer deferred-splits → pages"
echo "    return to buddy faster → more usable free memory."
