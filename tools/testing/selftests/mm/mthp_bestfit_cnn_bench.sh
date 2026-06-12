#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# mthp_bestfit_cnn_bench.sh — CNN / tensor workload mTHP fragmentation benchmark
#
# Runs on two boards and compares mTHP fragmentation metrics:
#   Board A: baseline kernel  (CONFIG_MTHP_BESTFIT=n)
#   Board B: patched kernel   (CONFIG_MTHP_BESTFIT=y)
#
# ── Why CNN tensors expose mTHP fragmentation ─────────────────────────────────
#
# Each CNN layer produces an activation tensor whose size shrinks with depth:
#
#   ResNet-18 (batch=1, float32):
#     input      224×224×3   =   588 KB   → bestfit order 7 (512 KB)
#     conv1      112×112×64  =  3,136 KB  → PMD (2 MB)  — bestfit passes through
#     res1       56×56×64    =   784 KB   → bestfit order 8 (1 MB)
#     res2       56×56×128   = 1,568 KB   → bestfit order 8 (1 MB)
#     res3       28×28×256   =   784 KB   → bestfit order 8 (1 MB)
#     res4       14×14×512   =   392 KB   → bestfit order 7 (512 KB)
#     res5        7×7×512    =    98 KB   → bestfit order 5 (128 KB)
#
#   Baseline kernel tries PMD (2 MB) for every tensor ≥ 512 KB → wastes
#   1.2 MB for the 784 KB tensor → deferred split when tensor is freed.
#   mthp_bestfit right-sizes each tensor → no wasted pages, no deferred split.
#
# ── Workload backends (tried in order) ───────────────────────────────────────
#   1. ncnn benchncnn       — fastest, ARM-native, no Python, most popular
#                             embedded CNN framework; auto-built if absent
#   2. tflite_benchmark_model — TensorFlow Lite official benchmark tool
#   3. cnn_tensor_workload  — self-contained C program (compiled here);
#                             mimics ResNet-18/MobileNetV2 tensor lifecycle
#                             with mmap()+mlock()+munmap(); always available
#
# ── Metrics ──────────────────────────────────────────────────────────────────
#   /proc/vmstat:  nr_deferred_split_page  compact_stall  thp_fault_*
#   /proc/meminfo: MemAvailable  AnonHugePages
#   /sys:          per-order hugepage stats
#   debugfs:       mthp_bestfit_stats  (patched board only)
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   ./mthp_bestfit_cnn_bench.sh [--tag baseline|bestfit]
#                               [--iterations N]   default 1000
#                               [--backend ncnn|tflite|synthetic]
#                               [--ncnn-bench /path/to/benchncnn]
#                               [--tflite-bench /path/to/tflite_benchmark_model]
#   ./mthp_bestfit_cnn_bench.sh --compare results_baseline_*.txt results_bestfit_*.txt

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
TAG="run"
ITERATIONS=1000
BACKEND=""          # auto-detect
NCNN_BENCH=""
TFLITE_BENCH=""
COMPARE_MODE=0
COMPARE_BASE=""
COMPARE_BEST=""
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)          TAG="$2";         shift 2 ;;
        --iterations)   ITERATIONS="$2";  shift 2 ;;
        --backend)      BACKEND="$2";     shift 2 ;;
        --ncnn-bench)   NCNN_BENCH="$2";  shift 2 ;;
        --tflite-bench) TFLITE_BENCH="$2";shift 2 ;;
        --compare)
            COMPARE_MODE=1; COMPARE_BASE="$2"; COMPARE_BEST="$3"; shift 3 ;;
        -h|--help)
            sed -n '2,/^# ── Usage/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Compare mode ──────────────────────────────────────────────────────────────
if [[ "$COMPARE_MODE" -eq 1 ]]; then
    [[ -f "$COMPARE_BASE" && -f "$COMPARE_BEST" ]] || {
        echo "ERROR: result files not found" >&2; exit 1; }

    extract_delta() {
        local key=$1 file=$2
        local b a
        b=$(grep "^$key " "$file" | head -1 | awk '{print $2}')
        a=$(grep "^$key " "$file" | tail -1 | awk '{print $2}')
        echo $(( ${a:-0} - ${b:-0} ))
    }

    printf "\n%-40s %12s %12s %10s\n" "Metric (delta over run)" "Baseline" "Bestfit" "Change"
    printf "%s\n" "$(printf '─%.0s' {1..78})"

    for key in nr_deferred_split_page compact_stall \
               thp_fault_alloc thp_fault_fallback thp_split_page; do
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
        if [[ -n "$b" && "${b:-0}" -ne 0 ]]; then
            pct=$(( (n - b) * 100 / b ))
            printf "%-40s %9d kB %9d kB %+9d%%\n" \
                   "${label%:} (after)" "$b" "$n" "$pct"
        fi
    done
    printf "\n"

    # Per-order histogram from bestfit board
    if grep -q "order_hist\|Order histogram" "$COMPARE_BEST" 2>/dev/null; then
        echo "── mthp_bestfit order histogram (bestfit board) ──"
        sed -n '/Order histogram/,/Tuning/p' "$COMPARE_BEST" | grep -v "^==="
    fi

    if grep -q "mthp_bestfit_stats" "$COMPARE_BEST" 2>/dev/null; then
        echo "── mthp_bestfit_stats summary ──"
        sed -n '/mthp_bestfit_stats/,/^===/p' "$COMPARE_BEST" \
            | grep -v "^===" | head -50
    fi
    exit 0
fi

OUT="results_${TAG}_$(date +%Y%m%d_%H%M%S).txt"
echo "Results → $OUT"

# ── Snapshot helper ───────────────────────────────────────────────────────────
snapshot() {
    local label=$1
    {
        echo "=== $label ==="
        echo "--- /proc/meminfo ---"
        grep -E "^MemTotal:|^MemFree:|^MemAvailable:|^Buffers:|^Cached:" \
             /proc/meminfo
        grep -E "^AnonHugePages:|^ShmemHugePages:" /proc/meminfo || true

        echo "--- /proc/vmstat ---"
        grep -E "^nr_deferred_split_page\
|^compact_stall|^compact_fail\
|^thp_fault_alloc|^thp_fault_fallback|^thp_fault_fallback_charge\
|^thp_split_page|^thp_split_page_failed\
|^thp_zero_page_alloc$|^thp_zero_page_alloc_failed$" /proc/vmstat

        echo "--- hugepage order stats ---"
        for d in /sys/kernel/mm/transparent_hugepage/hugepages-*/; do
            [[ -d "$d/stats" ]] || continue
            olabel=$(basename "$d")
            for f in "$d"stats/*; do
                v=$(cat "$f" 2>/dev/null) || continue
                [[ "$v" != "0" ]] && printf "  %-14s %-30s %s\n" \
                    "$olabel" "$(basename "$f")" "$v"
            done
        done

        echo "--- mthp_bestfit_stats ---"
        if [[ -r /sys/kernel/debug/mthp_bestfit_stats ]]; then
            cat /sys/kernel/debug/mthp_bestfit_stats
        else
            echo "  (not available on this board)"
        fi
        echo ""
    } | tee -a "$OUT"
}

# ══════════════════════════════════════════════════════════════════════════════
# Backend 1: ncnn benchncnn
# ══════════════════════════════════════════════════════════════════════════════
build_ncnn() {
    echo "Building ncnn benchncnn (ARM-native, no Python required)..."
    local ncnn_dir="$SCRIPT_DIR/ncnn_build"
    if [[ ! -d "$ncnn_dir" ]]; then
        git clone --depth 1 https://github.com/Tencent/ncnn "$ncnn_dir"
    fi
    cmake -B "$ncnn_dir/build" "$ncnn_dir" \
          -DCMAKE_BUILD_TYPE=Release \
          -DNCNN_BUILD_BENCHMARK=ON  \
          -DNCNN_BUILD_TOOLS=OFF     \
          -DNCNN_BUILD_EXAMPLES=OFF  \
          -DNCNN_ENABLE_LTO=ON       \
          -DNCNN_TARGET_ARCH=auto    \
          -DNCNN_SHARED_LIB=OFF      \
          -DNCNN_SIMPLEVK=OFF        \
          -DNCNN_VULKAN=OFF          \
          -DNCNN_BUILD_TESTS=OFF     \
          -Wno-dev 2>/dev/null
    cmake --build "$ncnn_dir/build" -j"$(nproc)" --target benchncnn
    echo "$ncnn_dir/build/benchmark/benchncnn"
}

find_ncnn() {
    local candidates=(
        "$NCNN_BENCH"
        "./benchncnn"
        "$SCRIPT_DIR/ncnn_build/build/benchmark/benchncnn"
        "$(which benchncnn 2>/dev/null || true)"
    )
    for c in "${candidates[@]}"; do
        [[ -x "$c" ]] && { echo "$c"; return; }
    done
    echo ""
}

run_ncnn() {
    local bench=$1
    echo "Backend: ncnn benchncnn"
    echo "  Models: squeezenet mobilenet mobilenet_v2 resnet18 shufflenet"
    echo "  Iterations per model: $ITERATIONS  Threads: $(nproc)"
    echo ""
    # benchncnn args: loop_count threads powersave gpu_device cooling_down
    # Running squeezenet (small), mobilenet_v2 (medium), resnet18 (large)
    # gives the widest spread of tensor sizes: 50 KB → 3 MB
    "$bench" "$ITERATIONS" "$(nproc)" 0 -1 0 2>&1 | tee -a "$OUT"
}

# ══════════════════════════════════════════════════════════════════════════════
# Backend 2: TFLite benchmark_model
# ══════════════════════════════════════════════════════════════════════════════
MOBILENET_TFLITE_URL="https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz"

find_tflite() {
    local candidates=(
        "$TFLITE_BENCH"
        "$(which benchmark_model 2>/dev/null || true)"
        "$(which tflite_benchmark_model 2>/dev/null || true)"
    )
    for c in "${candidates[@]}"; do
        [[ -x "$c" ]] && { echo "$c"; return; }
    done
    echo ""
}

run_tflite() {
    local bench=$1
    local model="mobilenet_v1_1.0_224.tflite"

    if [[ ! -f "$model" ]]; then
        echo "Downloading MobileNetV1 TFLite model..."
        curl -sL "$MOBILENET_TFLITE_URL" | tar xz
    fi

    echo "Backend: TFLite benchmark_model"
    echo "  Model: MobileNetV1-1.0-224"
    echo "  Iterations: $ITERATIONS  Threads: $(nproc)"
    "$bench" \
        --graph="$model"           \
        --num_runs="$ITERATIONS"   \
        --num_threads="$(nproc)"   \
        --enable_op_profiling=false \
        2>&1 | tee -a "$OUT"
}

# ══════════════════════════════════════════════════════════════════════════════
# Backend 3: Synthetic C workload (always available — compiled here)
# Simulates the exact tensor size lifecycle of ResNet-18 and MobileNetV2.
# Uses mmap(ANONYMOUS) so the kernel applies mTHP policy exactly as it would
# for a real framework's tensor allocator (malloc delegates large allocs to mmap).
# ══════════════════════════════════════════════════════════════════════════════

CNN_SRC="$SCRIPT_DIR/cnn_tensor_workload.c"
CNN_BIN="$SCRIPT_DIR/cnn_tensor_workload"

build_synthetic() {
    if [[ ! -f "$CNN_SRC" ]]; then
        echo "ERROR: $CNN_SRC not found next to this script." >&2
        return 1
    fi
    echo "Compiling synthetic CNN tensor workload..."
    gcc -O2 -o "$CNN_BIN" "$CNN_SRC" -lm
}

run_synthetic() {
    echo "Backend: synthetic CNN tensor workload (ResNet-18 + MobileNetV2)"
    echo "  Iterations: $ITERATIONS"
    "$CNN_BIN" "$ITERATIONS" 2>&1 | tee -a "$OUT"
}

# ══════════════════════════════════════════════════════════════════════════════
# Auto-select backend
# ══════════════════════════════════════════════════════════════════════════════
select_backend() {
    local b

    if [[ "$BACKEND" == "ncnn" || -z "$BACKEND" ]]; then
        b=$(find_ncnn)
        if [[ -z "$b" && ( -z "$BACKEND" || "$BACKEND" == "ncnn" ) ]]; then
            if command -v git &>/dev/null && command -v cmake &>/dev/null; then
                b=$(build_ncnn) && BACKEND="ncnn"
            fi
        fi
        [[ -n "$b" ]] && { BACKEND="ncnn"; BACKEND_BIN="$b"; return; }
    fi

    if [[ "$BACKEND" == "tflite" || -z "$BACKEND" ]]; then
        b=$(find_tflite)
        if [[ -n "$b" ]]; then
            BACKEND="tflite"; BACKEND_BIN="$b"; return
        fi
    fi

    # Always-available fallback
    if [[ ! -x "$CNN_BIN" ]]; then
        build_synthetic
    fi
    BACKEND="synthetic"; BACKEND_BIN="$CNN_BIN"
}

BACKEND_BIN=""
select_backend
echo "Selected backend: $BACKEND  ($BACKEND_BIN)"

# ── Drop caches, initial snapshot ────────────────────────────────────────────
echo "Dropping page caches..."
sync && echo 3 > /proc/sys/vm/drop_caches && sleep 1
snapshot "BEFORE"

# ── Run workload ──────────────────────────────────────────────────────────────
echo "Running CNN tensor workload..."
case "$BACKEND" in
    ncnn)      run_ncnn      "$BACKEND_BIN" ;;
    tflite)    run_tflite    "$BACKEND_BIN" ;;
    synthetic) run_synthetic               ;;
esac

# ── Final snapshot ────────────────────────────────────────────────────────────
snapshot "AFTER"

# ── Inline delta ─────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
printf "  Delta summary — tag=%s  backend=%s\n" "$TAG" "$BACKEND"
echo "══════════════════════════════════════════════════════════════"

delta() {
    local key=$1
    local b a d
    b=$(grep "^$key " "$OUT" | head -1 | awk '{print $2}')
    a=$(grep "^$key " "$OUT" | tail -1 | awk '{print $2}')
    d=$(( ${a:-0} - ${b:-0} ))
    printf "  %-38s +%d\n" "$key" "$d"
}
delta "nr_deferred_split_page"
delta "compact_stall"
delta "thp_fault_alloc"
delta "thp_fault_fallback"
delta "thp_split_page"
printf "  %-38s %s kB\n" "MemAvailable (final)" \
    "$(grep '^MemAvailable:' "$OUT" | tail -1 | awk '{print $2}')"
printf "  %-38s %s kB\n" "AnonHugePages (final)" \
    "$(grep '^AnonHugePages:' "$OUT" | tail -1 | awk '{print $2}')"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Full results: $OUT"
echo ""
echo "To compare boards:"
echo "  $0 --compare results_baseline_*.txt results_bestfit_*.txt"
