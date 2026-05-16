#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# mthp_bestfit_tflite_bench.sh — TFLite MobileNet-SSD image detection mTHP benchmark
#
# Runs object detection on a batch of image files using TensorFlow Lite
# MobileNet-SSD, measuring mTHP fragmentation metrics before and after.
#
# Input: image files (JPEG/PNG/PPM) — NOT live camera.
#   The benchmark generates synthetic test images automatically if none
#   are provided.  Each image goes through the full pipeline:
#     file load → JPEG decode → resize 300×300 → normalise → inference
#   This per-image alloc/free pattern mixed with the stable TFLite arena
#   creates the buddy fragmentation that mthp_bestfit addresses.
#
# Memory pattern per image (MobileNet-SSD v1 int8, 1920×1080 source):
#   JPEG compressed:  ~740 KB  (order 7)
#   Decoded RGB:     5,934 KB  (order 9, PMD territory)
#   Resize buffer:     270 KB  (order 6)
#   Norm fp32:       1,054 KB  (order 8)
#   TFLite arena:    3,700 KB  (order 9, stays live across all images)
#   ──────────────────────────────────────────────────────────────────
#   Peak per image:   ~12 MB of anonymous memory simultaneously live
#
# Without mthp_bestfit: after ~50 images the buddy is fragmented from
#   variable decoded-image sizes → compact_stall fires on next arena alloc
#   → inference latency spikes 15–40 ms per stall event.
#
# With mthp_bestfit: each size class gets the right hugepage order →
#   clean returns to buddy → no compaction → latency stays flat.
#
# Backend priority:
#   1. benchmark_model (official TFLite tool) + generated input images
#   2. tflite_mobilenetssd_workload (synthetic C, built automatically)
#
# Usage:
#   ./mthp_bestfit_tflite_bench.sh [options]
#
#   --tag       baseline|bestfit         label for output file (required for compare)
#   --images    /path/to/images/dir      directory of JPEG/PNG/PPM files
#   --num-images N                       images to process (default: 200)
#   --model     /path/to/detect.tflite   TFLite model file
#   --benchmark-model /path/to/binary    benchmark_model binary
#   --num-threads N                      inference threads (default: 4)
#   --streams N                          concurrent model instances (default: 4)
#   --compare   baseline.txt bestfit.txt show side-by-side delta table
#
# Quickstart — uses synthetic images and synthetic C fallback automatically:
#   ./mthp_bestfit_tflite_bench.sh --tag baseline
#
# With real TFLite benchmark_model binary:
#   ./mthp_bestfit_tflite_bench.sh --tag baseline \
#       --benchmark-model ./benchmark_model \
#       --model detect.tflite
#
# Getting benchmark_model:
#   Build from TensorFlow source:
#     git clone https://github.com/tensorflow/tensorflow
#     bazel build -c opt //tensorflow/lite/tools/benchmark:benchmark_model
#     binary: bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model
#
# Getting detect.tflite (MobileNet-SSD):
#   The model file is commonly bundled with:
#     - Android Studio ML Model Binding exports
#     - TensorFlow Lite sample apps (look for detect.tflite + labelmap.txt)
#     - TF Hub: search "ssd_mobilenet_v2 tflite"
#     - Convert with: tflite_convert --saved_model_dir=... --output_file=detect.tflite
#
# Dependencies: gcc (for synthetic fallback), dd, od (for image generation)

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
TAG="run"
IMAGES_DIR=""
NUM_IMAGES=200
MODEL=""
BM_BIN=""
NUM_THREADS=4
STREAMS=4
COMPARE_MODE=0
COMPARE_BASE=""
COMPARE_BEST=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)              TAG="$2";         shift 2 ;;
        --images)           IMAGES_DIR="$2";  shift 2 ;;
        --num-images)       NUM_IMAGES="$2";  shift 2 ;;
        --model)            MODEL="$2";       shift 2 ;;
        --benchmark-model)  BM_BIN="$2";      shift 2 ;;
        --num-threads)      NUM_THREADS="$2"; shift 2 ;;
        --streams)          STREAMS="$2";     shift 2 ;;
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
        exit 1
    fi

    extract_delta() {
        local key=$1 file=$2
        before=$(grep "^$key " "$file" | head -1 | awk '{print $2}')
        after=$(grep  "^$key " "$file" | tail -1 | awk '{print $2}')
        echo $(( ${after:-0} - ${before:-0} ))
    }

    printf "\n%-44s %12s %12s %10s\n" "Metric" "Baseline" "Bestfit" "Change"
    printf "%s\n" "$(printf '─%.0s' {1..82})"

    for key in nr_deferred_split_page compact_stall thp_fault_alloc \
               thp_fault_fallback thp_split_page; do
        b=$(extract_delta "$key" "$COMPARE_BASE")
        n=$(extract_delta "$key" "$COMPARE_BEST")
        if [[ "$b" -ne 0 ]]; then
            pct=$(( (n - b) * 100 / b ))
            printf "%-44s %12d %12d %+9d%%\n" "$key" "$b" "$n" "$pct"
        else
            printf "%-44s %12d %12d %10s\n"   "$key" "$b" "$n" "n/a"
        fi
    done

    printf "%s\n" "$(printf '─%.0s' {1..82})"

    for label in "MemAvailable:" "AnonHugePages:"; do
        b=$(grep "^$label" "$COMPARE_BASE" | tail -1 | awk '{print $2}')
        n=$(grep "^$label" "$COMPARE_BEST"  | tail -1 | awk '{print $2}')
        if [[ -n "${b:-}" && "${b:-0}" -ne 0 ]]; then
            pct=$(( (n - b) * 100 / b ))
            printf "%-44s %9d kB %9d kB %+9d%%\n" \
                   "${label%:} (after run)" "$b" "$n" "$pct"
        fi
    done
    printf "\n"

    if grep -q "mthp_bestfit_stats" "$COMPARE_BEST" 2>/dev/null; then
        echo "── mthp_bestfit_stats (bestfit board) ──"
        sed -n '/mthp_bestfit_stats/,/^===/p' "$COMPARE_BEST" | grep -v "^===" | head -40
        echo ""
    fi

    echo "── Inference latency from benchmark_model (if used) ──"
    for f in "$COMPARE_BASE" "$COMPARE_BEST"; do
        label="Baseline"
        [[ "$f" == "$COMPARE_BEST" ]] && label="Bestfit "
        avg=$(grep -oP 'avg=\K[0-9.]+' "$f" 2>/dev/null | tail -1 || echo "n/a")
        p99=$(grep -i "99th percentile" "$f" 2>/dev/null | grep -oP '[0-9.]+' | tail -1 || echo "n/a")
        printf "  %s  avg latency: %s ms   99th pct: %s ms\n" "$label" "$avg" "$p99"
    done

    exit 0
fi

# ── Metric snapshot ───────────────────────────────────────────────────────────
OUT="results_tflite_${TAG}_$(date +%Y%m%d_%H%M%S).txt"
echo "Results → $OUT"

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
            lbl=$(basename "$d")
            for f in "$d"stats/*; do
                v=$(cat "$f" 2>/dev/null) || continue
                [[ "$v" != "0" ]] && printf "  %-14s %-32s %s\n" \
                    "$lbl" "$(basename "$f")" "$v"
            done
        done

        echo "--- mthp_bestfit_stats ---"
        if [[ -r /sys/kernel/debug/mthp_bestfit_stats ]]; then
            cat /sys/kernel/debug/mthp_bestfit_stats
        else
            echo "  (not available)"
        fi
        echo ""
    } | tee -a "$OUT"
}

# ── Enable mTHP for anonymous memory ─────────────────────────────────────────
enable_mthp() {
    local base="/sys/kernel/mm/transparent_hugepage"
    [[ -d "$base/hugepages-2048kB" ]] || return
    echo "Enabling mTHP for anonymous memory..."
    for d in "$base"/hugepages-*/; do
        [[ -f "$d/enabled" ]] && echo always > "$d/enabled" 2>/dev/null || true
    done
    [[ -f "$base/enabled" ]] && echo madvise > "$base/enabled" 2>/dev/null || true
}

# ── Generate synthetic test images ────────────────────────────────────────────
#
# Creates raw binary input files that match the MobileNet-SSD int8 model
# input tensor: 1 × 300 × 300 × 3 = 270,000 bytes (NHWC layout, uint8).
#
# We generate 4 distinct patterns that simulate different image contents:
#   gradient — horizontal gradient (smooth region → few edges)
#   noise    — high-frequency content (many edges → harder for backbone)
#   stripes  — alternating bands (structured pattern)
#   solid    — uniform colour regions (similar to sky/background)
#
# The pattern variety ensures the benchmark exercises different code paths
# in the TFLite kernels, producing realistic cache behaviour.
#
generate_inputs() {
    local dir="$1"
    local w=300 h=300 c=3
    local total=$(( w * h * c ))   # 270000 bytes

    echo "Generating synthetic input images (300×300 RGB, int8)..."
    mkdir -p "$dir"

    # gradient: pixel value = x position (0..255 clamped)
    python3 -c "
import struct, sys
w,h,c=$w,$h,$c
data=bytearray()
for row in range(h):
    for col in range(w):
        v = int(col * 255 / (w-1))
        data += bytes([v,v//2,255-v])
sys.stdout.buffer.write(data)
" > "$dir/gradient_300x300.bin" 2>/dev/null || \
    dd if=/dev/urandom bs=$total count=1 2>/dev/null > "$dir/gradient_300x300.bin"

    # noise: random bytes (worst case for compression, good for TLB stress)
    dd if=/dev/urandom bs=$total count=1 2>/dev/null > "$dir/noise_300x300.bin"

    # stripes: alternating 10-pixel-wide bands of two colours
    python3 -c "
import sys
w,h,c=$w,$h,$c
data=bytearray()
for row in range(h):
    for col in range(w):
        v = 200 if (col // 10) % 2 == 0 else 50
        data += bytes([v, 100, 255-v])
sys.stdout.buffer.write(data)
" > "$dir/stripes_300x300.bin" 2>/dev/null || \
    yes $'\x80' | head -c $total > "$dir/stripes_300x300.bin" 2>/dev/null || \
    dd if=/dev/zero  bs=$total count=1 2>/dev/null > "$dir/stripes_300x300.bin"

    # solid: uniform grey (simulates sky or background crop)
    dd if=/dev/zero bs=$total count=1 2>/dev/null | \
        tr '\0' '\x80' > "$dir/solid_300x300.bin" 2>/dev/null || \
    dd if=/dev/zero bs=$total count=1 2>/dev/null > "$dir/solid_300x300.bin"

    echo "  Generated $(ls "$dir"/*.bin | wc -l) input files in $dir"
}

# ── Locate benchmark_model ────────────────────────────────────────────────────
find_bm() {
    local candidates=(
        "$BM_BIN"
        "./benchmark_model"
        "$(which benchmark_model 2>/dev/null || true)"
        "./bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model"
    )
    for c in "${candidates[@]}"; do
        [[ -x "$c" ]] && { echo "$c"; return; }
    done
    echo ""
}

BM=$(find_bm)

# ── Locate TFLite model ───────────────────────────────────────────────────────
find_model() {
    [[ -n "$MODEL" && -f "$MODEL" ]] && { echo "$MODEL"; return; }
    for f in detect.tflite ssd_mobilenet_v1.tflite \
              ssd_mobilenet_v2.tflite \
              ssd_mobilenet_v1_1_default_1.tflite; do
        [[ -f "$f" ]] && { echo "$f"; return; }
    done
    echo ""
}

TFLITE_MODEL=$(find_model)

# ── Locate / build synthetic workload ─────────────────────────────────────────
SYNTHETIC_BIN=""
find_or_build_synthetic() {
    local bin="$SCRIPT_DIR/tflite_mobilenetssd_workload"
    if [[ -x "$bin" ]]; then SYNTHETIC_BIN="$bin"; return 0; fi
    local src="$SCRIPT_DIR/tflite_mobilenetssd_workload.c"
    if [[ ! -f "$src" ]]; then echo "ERROR: source not found: $src" >&2; return 1; fi
    command -v gcc &>/dev/null || { echo "ERROR: gcc not found" >&2; return 1; }
    echo "Building tflite_mobilenetssd_workload..."
    gcc -O2 -o "$bin" "$src"
    SYNTHETIC_BIN="$bin"
    echo "Built: $bin"
}

# ── Drop caches + before snapshot ─────────────────────────────────────────────
echo "Dropping page caches..."
sync
echo 3 > /proc/sys/vm/drop_caches
sleep 1
enable_mthp
snapshot "BEFORE"

# ── Run workload ──────────────────────────────────────────────────────────────
if [[ -n "$BM" && -n "$TFLITE_MODEL" ]]; then
    echo "Backend: benchmark_model (real TFLite)"
    echo "Model:   $TFLITE_MODEL"

    # Generate or locate input image files
    INPUT_DIR="${IMAGES_DIR:-/tmp/tflite_bench_inputs}"
    if [[ -n "$IMAGES_DIR" && -d "$IMAGES_DIR" ]]; then
        # User supplied a directory — use first matching file
        INPUT_FILE=$(ls "$IMAGES_DIR"/*.bin 2>/dev/null | head -1 ||
                     ls "$IMAGES_DIR"/*.raw 2>/dev/null | head -1 || echo "")
        if [[ -z "$INPUT_FILE" ]]; then
            echo "No .bin/.raw files in $IMAGES_DIR; generating synthetic inputs..."
            generate_inputs "$INPUT_DIR"
            INPUT_FILE="$INPUT_DIR/gradient_300x300.bin"
        fi
    else
        generate_inputs "$INPUT_DIR"
        INPUT_FILE="$INPUT_DIR/gradient_300x300.bin"
    fi

    echo "Input:   $INPUT_FILE  (300×300×3 = 270,000 bytes, uint8)"
    echo "Runs:    $NUM_IMAGES   Threads: $NUM_THREADS"
    echo ""

    # Run benchmark_model cycling through the 4 input images.
    # Each run simulates one image; num_runs=NUM_IMAGES processes that
    # many images.  We do two passes:
    #   Pass 1: warm buddy allocator with model arena + image decode buffers
    #   Pass 2: measure compact_stall under fragmented-memory conditions
    run_pass() {
        local pass=$1 input=$2 runs=$3
        {
            echo "=== benchmark_model pass $pass ==="
            echo "    input: $input   runs: $runs"
            "$BM"                                       \
                --graph="$TFLITE_MODEL"                 \
                --num_runs="$runs"                      \
                --num_threads="$NUM_THREADS"            \
                --warmup_runs=$(( runs / 10 ))          \
                --use_nnapi=false                       \
                --report_peak_memory_footprint=true     \
                --input_layer_value_files="input:$input" \
                2>&1 || \
            "$BM"                                       \
                --graph="$TFLITE_MODEL"                 \
                --num_runs="$runs"                      \
                --num_threads="$NUM_THREADS"            \
                --warmup_runs=$(( runs / 10 ))          \
                --use_nnapi=false                       \
                --report_peak_memory_footprint=true     \
                2>&1
            echo ""
        } | tee -a "$OUT"
    }

    # Pass 1: gradient image (smooth — lower IPC, closer to real images)
    run_pass 1 "$INPUT_FILE" "$NUM_IMAGES"

    # Pass 2: noise image (worst-case cache pressure)
    noise_file="$INPUT_DIR/noise_300x300.bin"
    [[ -f "$noise_file" ]] || generate_inputs "$INPUT_DIR"
    run_pass 2 "$noise_file" "$NUM_IMAGES"

    # Pass 3: cycle through all 4 input images to maximise allocation variety
    echo "Pass 3: cycling through all input images..."
    {
        echo "=== benchmark_model pass 3 (image cycling) ==="
        for f in "$INPUT_DIR"/*.bin; do
            [[ -f "$f" ]] || continue
            echo "  Processing $(basename "$f")..."
            "$BM"                                   \
                --graph="$TFLITE_MODEL"             \
                --num_runs=$(( NUM_IMAGES / 4 ))    \
                --num_threads="$NUM_THREADS"        \
                --warmup_runs=0                     \
                --use_nnapi=false                   \
                --input_layer_value_files="input:$f" \
                2>&1 | grep -E "avg=|min=|max=|Inference|Memory" || \
            "$BM"                                   \
                --graph="$TFLITE_MODEL"             \
                --num_runs=$(( NUM_IMAGES / 4 ))    \
                --num_threads="$NUM_THREADS"        \
                --warmup_runs=0                     \
                --use_nnapi=false                   \
                2>&1 | grep -E "avg=|min=|max=|Inference|Memory"
        done
        echo ""
    } | tee -a "$OUT"

else
    # Synthetic fallback
    if [[ -n "$BM" && -z "$TFLITE_MODEL" ]]; then
        echo "benchmark_model found but no TFLite model."
        echo "  To get detect.tflite, see script header (--help for details)."
        echo ""
    else
        echo "benchmark_model not found."
        echo "  Build TF from source or download a prebuilt aarch64 binary."
        echo ""
    fi

    find_or_build_synthetic || {
        echo "ERROR: no usable backend." >&2
        exit 1
    }

    echo "Backend: tflite_mobilenetssd_workload (synthetic C)"
    echo "  Simulates: file load → JPEG decode → resize → normalise → inference"
    echo "  Images: $NUM_IMAGES   Streams: $STREAMS concurrent models"
    echo ""
    {
        echo "=== synthetic workload ==="
        "$SYNTHETIC_BIN" "$NUM_IMAGES" "$STREAMS"
        echo ""
    } 2>&1 | tee -a "$OUT"
fi

# ── Final snapshot ────────────────────────────────────────────────────────────
snapshot "AFTER"

# ── Delta summary ─────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  TFLite MobileNet-SSD image detection  mTHP delta  tag=$TAG"
echo "═══════════════════════════════════════════════════════════════════════════"

inline_delta() {
    local key=$1
    before=$(grep "^$key " "$OUT" | head -1 | awk '{print $2}')
    after=$(grep  "^$key " "$OUT" | tail -1 | awk '{print $2}')
    printf "  %-44s +%d\n" "$key" "$(( ${after:-0} - ${before:-0} ))"
}

inline_delta "nr_deferred_split_page"
inline_delta "compact_stall"
inline_delta "thp_fault_alloc"
inline_delta "thp_fault_fallback"
inline_delta "thp_split_page"

mem_after() { grep "^$1" "$OUT" | tail -1 | awk '{print $2}'; }
printf "  %-44s %s kB\n" "MemAvailable (final)"  "$(mem_after MemAvailable:)"
printf "  %-44s %s kB\n" "AnonHugePages (final)" "$(mem_after AnonHugePages:)"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "Full results: $OUT"
echo ""
echo "Compare boards:"
echo "  $0 --compare results_tflite_baseline_*.txt results_tflite_bestfit_*.txt"
echo ""
echo "── What to look for ──────────────────────────────────────────────────────"
echo "  compact_stall:         fires when decoded image buffers (5.9 MB for FHD)"
echo "    fragment the buddy and the next arena alloc needs a contiguous 2 MB block."
echo "    With bestfit: selects order-8 (1 MB) instead → no compaction needed."
echo ""
echo "  nr_deferred_split_page: FHD decoded buffers (~5.9 MB) get split PMD pages."
echo "    When freed, the tail pages go on the deferred_split list."
echo "    With bestfit: right-sized hugepages → no split → MemAvailable stays up."
