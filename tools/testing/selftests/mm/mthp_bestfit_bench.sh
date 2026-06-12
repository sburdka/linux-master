#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# mthp_bestfit_bench.sh — llama.cpp / TinyLlama fragmentation benchmark
#
# Measures mTHP fragmentation metrics on two boards:
#   Board A: baseline kernel (CONFIG_MTHP_BESTFIT=n)
#   Board B: patched kernel  (CONFIG_MTHP_BESTFIT=y)
#
# Workload: llama-server (llama.cpp) with TinyLlama-1.1B-Q4_K_M running
# 200 inference requests that alternate between short and long context
# lengths.  Mixed context sizes produce KV-cache VMAs that are never
# 2 MB-aligned, which is the classic mTHP fragmentation pattern:
#
#   Short request (32 tok) → KV VMA ≈  1.4 MB  (not PMD-aligned)
#   Long  request (256 tok)→ KV VMA ≈ 11.2 MB  (not PMD-aligned)
#
# Baseline kernel allocates a 2 MB hugepage for the 1.4 MB slot → 600 KB
# wasted, hugepage deferred-split when slot is cleared.
# mthp_bestfit selects order 7-8 (512 KB-1 MB) → right-sized, no split.
#
# Metrics collected from /proc/vmstat and /proc/meminfo:
#   nr_deferred_split_page  — pages pending split (lower = better)
#   compact_stall           — direct-compaction stalls (lower = better)
#   thp_fault_alloc         — THP allocations succeeded
#   thp_fault_fallback      — THP allocations fell back to 4 KB
#   MemAvailable            — usable free memory (higher = better)
#   AnonHugePages           — anonymous hugepage footprint
#
# Usage:
#   ./mthp_bestfit_bench.sh [--tag baseline|bestfit] [--requests N]
#                           [--model /path/to/model.gguf]
#                           [--server /path/to/llama-server]
#                           [--parallel N] [--ctx N]
#
# Quickstart (downloads model automatically if curl available):
#   ./mthp_bestfit_bench.sh --tag baseline
#
# Compare results after running on both boards:
#   ./mthp_bestfit_bench.sh --compare results_baseline_*.txt results_bestfit_*.txt
#
# Dependencies: llama.cpp (llama-server binary), curl

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
TAG="run"
REQUESTS=200
MODEL=""
SERVER_BIN=""
PARALLEL=4
CTX=2048
PORT=8888
COMPARE_MODE=0
COMPARE_BASE=""
COMPARE_BEST=""

SHORT_PROMPT='Q: What is 2+2? A:'
LONG_PROMPT='Explain the history of operating system memory management in detail, covering virtual memory, paging, segmentation, transparent huge pages, multi-size THP, memory compaction, and NUMA topology in at least 300 words.'

MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.q4_k_m.gguf"
MODEL_FILE="tinyllama-1.1b-chat-v1.0.q4_k_m.gguf"

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)       TAG="$2";        shift 2 ;;
        --requests)  REQUESTS="$2";   shift 2 ;;
        --model)     MODEL="$2";      shift 2 ;;
        --server)    SERVER_BIN="$2"; shift 2 ;;
        --parallel)  PARALLEL="$2";   shift 2 ;;
        --ctx)       CTX="$2";        shift 2 ;;
        --port)      PORT="$2";       shift 2 ;;
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

# ── Compare mode ─────────────────────────────────────────────────────────────
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

    extract_meminfo() {
        local key=$1 label=$2 file=$3
        grep "^$label" "$file" | tail -1 | awk '{print $2}'
    }

    printf "\n%-38s %12s %12s %10s\n" "Metric" "Baseline" "Bestfit" "Change"
    printf "%s\n" "$(printf '─%.0s' {1..76})"

    for key in nr_deferred_split_page compact_stall thp_fault_alloc thp_fault_fallback thp_split_page; do
        b=$(extract_delta "$key" "$COMPARE_BASE")
        n=$(extract_delta "$key" "$COMPARE_BEST")
        if [[ "$b" -ne 0 ]]; then
            pct=$(( (n - b) * 100 / b ))
            sign=""
            [[ "$pct" -gt 0 ]] && sign="+"
            printf "%-38s %12d %12d %+9d%%\n" "$key" "$b" "$n" "$pct"
        else
            printf "%-38s %12d %12d %10s\n" "$key" "$b" "$n" "n/a"
        fi
    done

    printf "%s\n" "$(printf '─%.0s' {1..76})"

    for label in "MemAvailable:" "AnonHugePages:"; do
        b=$(extract_meminfo "" "$label" "$COMPARE_BASE")
        n=$(extract_meminfo "" "$label" "$COMPARE_BEST")
        if [[ -n "$b" && "$b" -ne 0 ]]; then
            pct=$(( (n - b) * 100 / b ))
            printf "%-38s %9d kB %9d kB %+9d%%\n" \
                   "${label%:} (after run)" "$b" "$n" "$pct"
        fi
    done

    printf "\n"

    if [[ -f "$COMPARE_BEST" ]] && grep -q "mthp_bestfit_stats" "$COMPARE_BEST"; then
        echo "── mthp_bestfit_stats (from bestfit board) ──"
        sed -n '/mthp_bestfit_stats/,/^===/p' "$COMPARE_BEST" | grep -v "^===" | head -40
    fi
    exit 0
fi

# ── Locate llama-server ───────────────────────────────────────────────────────
find_server() {
    local candidates=(
        "$SERVER_BIN"
        "./build/bin/llama-server"
        "./llama-server"
        "$(which llama-server 2>/dev/null || true)"
    )
    for c in "${candidates[@]}"; do
        [[ -x "$c" ]] && { echo "$c"; return; }
    done
    echo ""
}

SERVER=$(find_server)
if [[ -z "$SERVER" ]]; then
    echo "ERROR: llama-server not found."
    echo "  Build: git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp"
    echo "         cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc) --target llama-server"
    echo "  Then re-run with: --server /path/to/llama-server"
    exit 1
fi
echo "Using server binary: $SERVER"

# ── Locate / download model ───────────────────────────────────────────────────
if [[ -z "$MODEL" ]]; then
    if [[ -f "$MODEL_FILE" ]]; then
        MODEL="$MODEL_FILE"
    elif command -v curl &>/dev/null; then
        echo "Downloading TinyLlama Q4_K_M (~669 MB)..."
        curl -L --progress-bar -o "$MODEL_FILE" "$MODEL_URL"
        MODEL="$MODEL_FILE"
    else
        echo "ERROR: model not found and curl unavailable."
        echo "  Download manually:"
        echo "  $MODEL_URL"
        echo "  Then re-run with: --model /path/to/model.gguf"
        exit 1
    fi
fi
echo "Using model: $MODEL"

# ── Output file ───────────────────────────────────────────────────────────────
OUT="results_${TAG}_$(date +%Y%m%d_%H%M%S).txt"
echo "Results → $OUT"

# ── Helper: collect a snapshot of all relevant metrics ───────────────────────
snapshot() {
    local label=$1
    {
        echo "=== $label ==="
        echo "--- /proc/meminfo ---"
        grep -E "^MemTotal:|^MemFree:|^MemAvailable:|^Buffers:|^Cached:|^AnonHugePages:|^ShmemHugePages:" \
             /proc/meminfo

        echo "--- /proc/vmstat ---"
        grep -E "^nr_deferred_split_page|^compact_stall|^compact_fail|^compact_migrate_scanned|\
^thp_fault_alloc|^thp_fault_fallback|^thp_fault_fallback_charge|\
^thp_split_page|^thp_split_page_failed|^thp_zero_page_alloc$|^thp_zero_page_alloc_failed$" \
             /proc/vmstat

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

# ── Drop caches and collect baseline snapshot ─────────────────────────────────
echo "Dropping page caches..."
sync
echo 3 > /proc/sys/vm/drop_caches
sleep 1

snapshot "BEFORE"

# ── Launch llama-server ───────────────────────────────────────────────────────
echo "Starting llama-server (parallel=$PARALLEL ctx=$CTX)..."
"$SERVER" \
    --model      "$MODEL"    \
    --ctx-size   "$CTX"      \
    --parallel   "$PARALLEL" \
    --threads    "$(nproc)"  \
    --host       127.0.0.1   \
    --port       "$PORT"     \
    --log-disable            \
    &
SERVER_PID=$!

# Wait for server to be ready (weights loaded into mmap)
echo -n "Waiting for server ready "
for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${PORT}/health" &>/dev/null; then
        echo " OK"
        break
    fi
    echo -n "."
    sleep 1
done

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: llama-server exited prematurely." >&2
    exit 1
fi

# ── Fragmentation workload ────────────────────────────────────────────────────
# 200 requests; 1 in 3 is long-context (large KV VMA), rest are short.
# Up to $PARALLEL requests in flight simultaneously to stress all slots.
#
# KV cache size per token for TinyLlama-1.1B (22 layers, 4 KV heads, 64 dim):
#   22 × 4 × 64 × 2 (K+V) × 2 bytes = 44,800 bytes ≈ 44 KB/token
#   Short (32 tok): 44 KB × 32  ≈  1,408 KB  (order 7 = 512 KB × 3)
#   Long (256 tok): 44 KB × 256 ≈ 11,264 KB  (order 8 = 1 MB  × 11)

echo "Running $REQUESTS inference requests (this takes a few minutes)..."

run_request() {
    local idx=$1
    local body
    if (( idx % 3 == 0 )); then
        body=$(printf '{"prompt":"%s","n_predict":256,"temperature":0}' \
                      "$LONG_PROMPT")
    else
        body=$(printf '{"prompt":"%s","n_predict":32,"temperature":0}' \
                      "$SHORT_PROMPT")
    fi
    curl -sf --max-time 180 \
         "http://127.0.0.1:${PORT}/completion" \
         -H 'Content-Type: application/json' \
         -d "$body" > /dev/null
}

completed=0
for i in $(seq 1 "$REQUESTS"); do
    run_request "$i" &
    # Throttle to PARALLEL concurrent requests
    while [[ "$(jobs -r | wc -l)" -ge "$PARALLEL" ]]; do
        sleep 0.05
    done
    completed=$(( completed + 1 ))
    (( completed % 20 == 0 )) && echo "  ... $completed / $REQUESTS requests sent"
done

# Wait for all in-flight requests to finish
wait
echo "All $REQUESTS requests completed."

# ── Collect final snapshot ────────────────────────────────────────────────────
snapshot "AFTER"

# ── Stop server ───────────────────────────────────────────────────────────────
kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null || true

# ── Print inline delta summary ────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Delta summary for tag=$TAG"
echo "══════════════════════════════════════════════════════════════"

inline_delta() {
    local key=$1
    local before after delta
    before=$(grep "^$key " "$OUT" | head -1 | awk '{print $2}')
    after=$(grep  "^$key " "$OUT" | tail -1 | awk '{print $2}')
    delta=$(( ${after:-0} - ${before:-0} ))
    printf "  %-38s +%d\n" "$key" "$delta"
}

inline_delta "nr_deferred_split_page"
inline_delta "compact_stall"
inline_delta "thp_fault_alloc"
inline_delta "thp_fault_fallback"
inline_delta "thp_split_page"

mem_after() {
    local label=$1
    grep "^$label" "$OUT" | tail -1 | awk '{print $2}'
}

printf "  %-38s %s kB\n" "MemAvailable (final)" "$(mem_after MemAvailable:)"
printf "  %-38s %s kB\n" "AnonHugePages (final)" "$(mem_after AnonHugePages:)"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Full results saved to: $OUT"
echo ""
echo "To compare baseline vs bestfit boards:"
echo "  $0 --compare results_baseline_*.txt results_bestfit_*.txt"
