#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# mthp_bestfit_hexagon_bench.sh — Qualcomm Hexagon / SNPE / QNN mTHP benchmark
#
# Measures mTHP fragmentation metrics on Snapdragon SoCs running CNN inference
# via SNPE (Snapdragon Neural Processing Engine) or QNN (Qualcomm Neural
# Network) SDKs, with graceful fallback to TFLite + Hexagon delegate and a
# self-contained synthetic workload that mimics SNPE posix_memalign patterns.
#
# Why Hexagon workloads expose mTHP fragmentation
# ─────────────────────────────────────────────────
# SNPE CPU runtime calls posix_memalign(128, size) for every tensor buffer.
# For size >= 128 KB, glibc delegates to mmap(MAP_PRIVATE|MAP_ANONYMOUS),
# the anonymous-memory path through which mTHP policy applies.
#
# SNPE pre-allocates ALL layer buffers at model-init time (not lazily per
# layer), so tensors of wildly different sizes (25 KB → 3.2 MB) coexist in
# the buddy allocator simultaneously.  Four concurrent models (two fp32 +
# two int8) cover orders 2–9 (16 KB – 2 MB), testing every bestfit bucket.
#
# Detection models (EfficientDet-Lite0, MobileDet-DSP) use 320×320 inputs,
# producing stem tensors > 2 MB that straddle the PMD boundary — exactly the
# case where bestfit suppresses a wasted PMD attempt and selects order 8.
#
# Backend priority (highest to lowest):
#   1. snpe-throughput-net-run  (SNPE SDK, production Hexagon DSP)
#   2. snpe-net-run             (SNPE SDK, single-batch inference)
#   3. qnn-net-run              (QNN SDK, HTP backend)
#   4. benchmark_model          (TFLite + Hexagon delegate)
#   5. hexagon_cnn_workload     (synthetic, built from source if needed)
#
# Metrics collected from /proc/vmstat and /proc/meminfo:
#   nr_deferred_split_page  — pages pending split (lower = better)
#   compact_stall           — direct-compaction stalls (lower = better)
#   thp_fault_alloc         — THP allocations succeeded
#   thp_fault_fallback      — THP allocations fell back to 4 KB
#   MemAvailable            — usable free memory (higher = better)
#   AnonHugePages           — anonymous hugepage footprint
#
# Plus Hexagon-specific stats from sysfs (if available):
#   /sys/bus/platform/drivers/fastrpc/  — Hexagon FastRPC stats
#   /sys/kernel/debug/mthp_bestfit_stats
#
# Usage:
#   ./mthp_bestfit_hexagon_bench.sh [--tag baseline|bestfit]
#                                   [--model /path/to/model.dlc]
#                                   [--backend snpe|qnn|tflite|synthetic]
#                                   [--duration N]  (seconds, default 120)
#                                   [--iterations N] (synthetic only, default 500)
#
# Quickstart (uses synthetic fallback automatically):
#   ./mthp_bestfit_hexagon_bench.sh --tag baseline
#
# Compare results from two boards:
#   ./mthp_bestfit_hexagon_bench.sh --compare results_baseline_*.txt results_bestfit_*.txt
#
# SNPE model acquisition:
#   1. Qualcomm AI Hub: aihub.qualcomm.com → search model → Export → SNPE DLC
#   2. SNPE model zoo:  Snapdragon Neural Processing Engine SDK → models/
#   3. Convert from ONNX/TFLite: snpe-onnx-to-dlc / snpe-tflite-to-dlc
#
# Recommended SNPE models for this benchmark:
#   MobileNetV3-Small (classification, 224×224)
#   EfficientDet-Lite0 (detection, 320×320)   ← highest fragmentation impact
#   Inception-v3 int8 (classification, 299×299)
#
# Dependencies: SNPE SDK or QNN SDK or TFLite, gcc (for synthetic fallback)

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
TAG="run"
BACKEND=""          # auto-detect if empty
MODEL=""
DURATION=120
ITERATIONS=500
COMPARE_MODE=0
COMPARE_BASE=""
COMPARE_BEST=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)          TAG="$2";        shift 2 ;;
        --model)        MODEL="$2";      shift 2 ;;
        --backend)      BACKEND="$2";   shift 2 ;;
        --duration)     DURATION="$2";   shift 2 ;;
        --iterations)   ITERATIONS="$2"; shift 2 ;;
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
        echo "ERROR: result files not found: $COMPARE_BASE  $COMPARE_BEST" >&2
        exit 1
    fi

    extract_delta() {
        local key=$1 file=$2
        local before after
        before=$(grep "^$key " "$file" | head -1 | awk '{print $2}')
        after=$(grep  "^$key " "$file" | tail -1 | awk '{print $2}')
        echo $(( ${after:-0} - ${before:-0} ))
    }

    printf "\n%-40s %12s %12s %10s\n" "Metric" "Baseline" "Bestfit" "Change"
    printf "%s\n" "$(printf '─%.0s' {1..78})"

    for key in nr_deferred_split_page compact_stall thp_fault_alloc \
               thp_fault_fallback thp_split_page; do
        b=$(extract_delta "$key" "$COMPARE_BASE")
        n=$(extract_delta "$key" "$COMPARE_BEST")
        if [[ "$b" -ne 0 ]]; then
            pct=$(( (n - b) * 100 / b ))
            printf "%-40s %12d %12d %+9d%%\n" "$key" "$b" "$n" "$pct"
        else
            printf "%-40s %12d %12d %10s\n"   "$key" "$b" "$n" "n/a"
        fi
    done

    printf "%s\n" "$(printf '─%.0s' {1..78})"

    for label in "MemAvailable:" "AnonHugePages:"; do
        b=$(grep "^$label" "$COMPARE_BASE" | tail -1 | awk '{print $2}')
        n=$(grep "^$label" "$COMPARE_BEST"  | tail -1 | awk '{print $2}')
        if [[ -n "$b" && "$b" -ne 0 ]]; then
            pct=$(( (n - b) * 100 / b ))
            printf "%-40s %9d kB %9d kB %+9d%%\n" \
                   "${label%:} (after)" "$b" "$n" "$pct"
        fi
    done

    printf "\n"

    if grep -q "mthp_bestfit_stats" "$COMPARE_BEST" 2>/dev/null; then
        echo "── mthp_bestfit_stats (bestfit board) ──"
        sed -n '/mthp_bestfit_stats/,/^===/p' "$COMPARE_BEST" \
            | grep -v "^===" | head -40
    fi

    if grep -q "hexagon_stats" "$COMPARE_BEST" 2>/dev/null; then
        echo "── Hexagon FastRPC stats (bestfit board) ──"
        sed -n '/hexagon_stats/,/^===/p' "$COMPARE_BEST" \
            | grep -v "^===" | head -20
    fi
    exit 0
fi

# ── Backend detection ─────────────────────────────────────────────────────────
find_backend() {
    if [[ -n "$BACKEND" ]]; then
        echo "$BACKEND"
        return
    fi

    # SNPE throughput tool (preferred for fragmentation: runs continuously)
    if command -v snpe-throughput-net-run &>/dev/null; then
        echo "snpe-throughput"
        return
    fi

    # SNPE single-run tool
    if command -v snpe-net-run &>/dev/null; then
        echo "snpe-net-run"
        return
    fi

    # QNN HTP (Hexagon Tensor Processor) backend
    if command -v qnn-net-run &>/dev/null; then
        echo "qnn"
        return
    fi

    # TFLite benchmark_model with Hexagon delegate
    if command -v benchmark_model &>/dev/null; then
        # Check if Hexagon delegate is available
        if benchmark_model --help 2>&1 | grep -q "use_hexagon"; then
            echo "tflite-hexagon"
            return
        fi
        echo "tflite"
        return
    fi

    echo "synthetic"
}

DETECTED_BACKEND=$(find_backend)
echo "Backend: $DETECTED_BACKEND"

# ── Locate / build the synthetic workload ─────────────────────────────────────
SYNTHETIC_BIN=""
find_or_build_synthetic() {
    local bin="$SCRIPT_DIR/hexagon_cnn_workload"

    if [[ -x "$bin" ]]; then
        SYNTHETIC_BIN="$bin"
        return 0
    fi

    local src="$SCRIPT_DIR/hexagon_cnn_workload.c"
    if [[ ! -f "$src" ]]; then
        echo "WARNING: hexagon_cnn_workload.c not found in $SCRIPT_DIR" >&2
        return 1
    fi

    if ! command -v gcc &>/dev/null; then
        echo "WARNING: gcc not found; cannot build synthetic workload" >&2
        return 1
    fi

    echo "Building hexagon_cnn_workload from source..."
    gcc -O2 -o "$bin" "$src" -lm
    SYNTHETIC_BIN="$bin"
    echo "Built: $bin"
}

# ── Model lookup ──────────────────────────────────────────────────────────────
find_model() {
    local ext="$1"
    shift
    local candidates=("$@")

    # User-supplied model takes priority
    if [[ -n "$MODEL" && -f "$MODEL" ]]; then
        echo "$MODEL"
        return
    fi

    for c in "${candidates[@]}"; do
        [[ -f "$c" ]] && { echo "$c"; return; }
    done

    echo ""
}

# ── Output file ───────────────────────────────────────────────────────────────
OUT="results_hexagon_${TAG}_$(date +%Y%m%d_%H%M%S).txt"
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

        # Hexagon-specific: FastRPC session stats (Snapdragon kernel driver)
        echo "--- hexagon_stats ---"
        local fastrpc="/sys/bus/platform/drivers/fastrpc"
        if [[ -d "$fastrpc" ]]; then
            for f in "$fastrpc"/*/stats 2>/dev/null; do
                [[ -r "$f" ]] && { echo "  $f:"; cat "$f"; }
            done
        else
            echo "  (fastrpc not available — no Hexagon DSP or not loaded)"
        fi

        # Qualcomm ION heap stats (older kernels) or DMA-buf stats (newer)
        echo "--- ion/dma-buf stats ---"
        if [[ -r /sys/kernel/debug/ion/heaps/system ]]; then
            cat /sys/kernel/debug/ion/heaps/system | head -20
        elif [[ -d /sys/kernel/debug/dma_buf ]]; then
            ls /sys/kernel/debug/dma_buf/ 2>/dev/null | head -5
        else
            echo "  (not available)"
        fi

        echo ""
    } | tee -a "$OUT"
}

# ── Drop caches ───────────────────────────────────────────────────────────────
echo "Dropping page caches..."
sync
echo 3 > /proc/sys/vm/drop_caches
sleep 1

snapshot "BEFORE"

# ── Run workload ──────────────────────────────────────────────────────────────
case "$DETECTED_BACKEND" in

snpe-throughput)
    DLC=$(find_model dlc \
          "mobilenetv3_small.dlc" "mobilenet_v3_small.dlc" \
          "efficientdet_lite0.dlc" "inception_v3.dlc" \
          "$MODEL")
    if [[ -z "$DLC" ]]; then
        echo "ERROR: No .dlc model found."
        echo "  Acquire from Qualcomm AI Hub or convert with snpe-onnx-to-dlc."
        echo "  Re-run with: --model /path/to/model.dlc"
        echo "  Falling back to synthetic workload."
        DETECTED_BACKEND="synthetic"
    else
        echo "Using model: $DLC"
        echo "Running snpe-throughput-net-run for ${DURATION}s ..."
        snpe-throughput-net-run \
            --container         "$DLC"               \
            --duration          "$DURATION"           \
            --perf_profile      high_performance      \
            --use_cpu_fixed_point                     \
            2>&1 | tee -a "$OUT"
    fi
    ;;

snpe-net-run)
    DLC=$(find_model dlc \
          "mobilenetv3_small.dlc" "inception_v3.dlc" "$MODEL")
    if [[ -z "$DLC" ]]; then
        echo "No .dlc model found; falling back to synthetic."
        DETECTED_BACKEND="synthetic"
    else
        echo "Using model: $DLC"
        # Generate a dummy input list (random noise input)
        INPUT_RAW=$(mktemp /tmp/snpe_input_XXXXXX.raw)
        # Input size for MobileNetV3-Small fp32: 224*224*3*4 = 602112 bytes
        dd if=/dev/urandom of="$INPUT_RAW" bs=602112 count=1 2>/dev/null
        INPUT_LIST=$(mktemp /tmp/snpe_input_list_XXXXXX.txt)
        echo "$INPUT_RAW" > "$INPUT_LIST"

        OUTPUT_DIR=$(mktemp -d /tmp/snpe_out_XXXXXX)

        echo "Running snpe-net-run (${ITERATIONS} inferences)..."
        for i in $(seq 1 "$ITERATIONS"); do
            snpe-net-run \
                --container  "$DLC"        \
                --input_list "$INPUT_LIST" \
                --output_dir "$OUTPUT_DIR" \
                2>/dev/null
        done | tail -5

        rm -f "$INPUT_RAW" "$INPUT_LIST"
        rm -rf "$OUTPUT_DIR"
    fi
    ;;

qnn)
    echo "QNN backend selected."
    echo "  Requires: qnn-net-run --backend libQnnHtp.so --model model.bin"
    echo "  Acquire models from Qualcomm AI Hub (Export → QNN format)."
    QNN_MODEL=$(find_model bin "model.bin" "model_htp.bin" "$MODEL")
    if [[ -z "$QNN_MODEL" ]]; then
        echo "No QNN model found; falling back to synthetic."
        DETECTED_BACKEND="synthetic"
    else
        echo "Using model: $QNN_MODEL"
        echo "Running qnn-net-run for $ITERATIONS iterations..."
        qnn-net-run \
            --backend    libQnnHtp.so    \
            --model      "$QNN_MODEL"    \
            --num_inferences "$ITERATIONS" \
            2>&1 | tee -a "$OUT"
    fi
    ;;

tflite-hexagon|tflite)
    TFLITE_MODEL=$(find_model tflite \
                   "mobilenet_v3_small.tflite"       \
                   "efficientdet_lite0.tflite"        \
                   "inception_v3_int8.tflite"         \
                   "mobiledet_cpu.tflite"             \
                   "$MODEL")
    if [[ -z "$TFLITE_MODEL" ]]; then
        echo "No .tflite model found; falling back to synthetic."
        DETECTED_BACKEND="synthetic"
    else
        echo "Using model: $TFLITE_MODEL"
        USE_HEX_FLAG=""
        [[ "$DETECTED_BACKEND" == "tflite-hexagon" ]] && \
            USE_HEX_FLAG="--use_hexagon=true --hexagon_profiling=true"

        echo "Running benchmark_model for $DURATION seconds..."
        benchmark_model                      \
            --graph="$TFLITE_MODEL"          \
            --num_threads=4                  \
            --min_secs="$DURATION"           \
            $USE_HEX_FLAG                    \
            2>&1 | tee -a "$OUT"
    fi
    ;;
esac

# Fallthrough to synthetic if any above backend redirected here
if [[ "$DETECTED_BACKEND" == "synthetic" ]]; then
    find_or_build_synthetic || {
        echo "ERROR: Cannot run any backend. Install SNPE, QNN, TFLite, or gcc." >&2
        exit 1
    }

    echo "Running hexagon_cnn_workload ($ITERATIONS iterations)..."
    echo "  Simulates: MobileNetV3-Small (fp32) + EfficientDet-Lite0 (fp32)"
    echo "             + Inception-v3 (int8) + MobileDet-DSP (int8)"
    echo "  Allocation: posix_memalign(128, size)  [SNPE CPU runtime pattern]"
    "$SYNTHETIC_BIN" "$ITERATIONS" 2>&1 | tee -a "$OUT"
fi

# ── Final snapshot ────────────────────────────────────────────────────────────
snapshot "AFTER"

# ── Inline delta summary ──────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  Hexagon mTHP delta summary  tag=$TAG"
echo "══════════════════════════════════════════════════════════════════════"

inline_delta() {
    local key=$1
    local before after delta
    before=$(grep "^$key " "$OUT" | head -1 | awk '{print $2}')
    after=$(grep  "^$key " "$OUT" | tail -1 | awk '{print $2}')
    delta=$(( ${after:-0} - ${before:-0} ))
    printf "  %-40s +%d\n" "$key" "$delta"
}

inline_delta "nr_deferred_split_page"
inline_delta "compact_stall"
inline_delta "thp_fault_alloc"
inline_delta "thp_fault_fallback"
inline_delta "thp_split_page"

mem_after() {
    grep "^$1" "$OUT" | tail -1 | awk '{print $2}'
}
printf "  %-40s %s kB\n" "MemAvailable (final)"  "$(mem_after MemAvailable:)"
printf "  %-40s %s kB\n" "AnonHugePages (final)" "$(mem_after AnonHugePages:)"
echo "══════════════════════════════════════════════════════════════════════"
echo ""
echo "Full results saved to: $OUT"
echo ""
echo "To compare baseline vs bestfit boards:"
echo "  $0 --compare results_hexagon_baseline_*.txt results_hexagon_bestfit_*.txt"
echo ""

# ── Hexagon-specific bestfit order distribution ───────────────────────────────
echo "── Bestfit order hits for this workload ──"
echo "  Model tensor size → order (without bestfit → with bestfit):"
echo "  MobileNetV3-Small  stem  784 KB  → no-bf: tries PMD→ord8→ord7"
echo "                                    → bestfit: ord7 direct (512 KB page)"
echo "  EfficientDet-Lite0 stem  3200 KB → no-bf: PMD (2 MB) + tail"
echo "                                    → bestfit: PMD + ord8 for tail"
echo "  EfficientDet MBConv2_exp 2400 KB → no-bf: PMD wasted → deferred_split"
echo "                                    → bestfit: PMD (2 MB) + ord7 tail"
echo "  Inception-v3 conv1  693 KB(int8) → no-bf: tries PMD→ord8→ord7"
echo "                                    → bestfit: ord7 direct (512 KB page)"
echo "  MobileDet-DSP stem  800 KB(int8) → no-bf: tries PMD→ord8"
echo "                                    → bestfit: ord8 direct (1 MB page)"
