#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# mthp_monitor.sh — Continuous mTHP metric monitor for load-time analysis
#
# Polls /proc/vmstat and /proc/buddyinfo every second WHILE a workload
# runs, then generates a report showing:
#   • compact_stall rate (stalls/second) — the key metric
#   • order-9 (PMD) free block drain — shows fragmentation build-up
#   • thp_fault_alloc vs thp_fault_fallback ratio — hugepage hit rate
#   • deferred_split_page accumulation rate
#   • MemAvailable trend — does it drop during the run?
#
# Why continuous monitoring matters
# ──────────────────────────────────
# Before/after snapshots miss the peak:
#   • compact_stall events happen MID-run when memory is most fragmented
#   • By the time the workload finishes, the kernel may have already
#     coalesced freed pages back into order-9 blocks
#   • MemAvailable dips mid-run (page cache evicted during compaction)
#     then recovers — a before/after snapshot shows 0 change
#
# compact_stall rate (stalls/second) is the correct metric because:
#   • Each stall = one inference latency spike (15-40 ms on Snapdragon)
#   • A rate of 2 stalls/sec = one 40ms spike every 500ms = unusable app
#   • With bestfit: rate drops to 0 stalls/sec even under fragmentation
#
# Usage:
#   # Start monitor in background
#   ./mthp_monitor.sh --out metrics.csv &
#   MONITOR_PID=$!
#
#   # Run your workload
#   ./mthp_bestfit_tflite_bench.sh ...
#
#   # Stop monitor and get report
#   kill $MONITOR_PID
#   ./mthp_monitor.sh --report metrics.csv
#
# Or use the --watch <pid> mode to stop automatically when workload exits:
#   ./mthp_monitor.sh --out metrics.csv --watch $WORKLOAD_PID
#
# Options:
#   --out    <file>   CSV output file (default: mthp_metrics_<timestamp>.csv)
#   --report <file>   print report from existing CSV (no monitoring)
#   --watch  <pid>    stop monitoring when this PID exits
#   --interval N      polling interval in seconds (default: 1)

set -euo pipefail

OUT=""
REPORT_FILE=""
WATCH_PID=""
INTERVAL=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)      OUT="$2";         shift 2 ;;
        --report)   REPORT_FILE="$2"; shift 2 ;;
        --watch)    WATCH_PID="$2";   shift 2 ;;
        --interval) INTERVAL="$2";    shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Report mode ───────────────────────────────────────────────────────────────
if [[ -n "$REPORT_FILE" ]]; then
    if [[ ! -f "$REPORT_FILE" ]]; then
        echo "ERROR: file not found: $REPORT_FILE" >&2; exit 1
    fi

    echo ""
    echo "══════════════════════════════════════════════════════════════════════"
    echo "  mTHP continuous monitoring report: $REPORT_FILE"
    echo "══════════════════════════════════════════════════════════════════════"
    echo ""

    # Extract columns: ts,compact_stall,deferred_split,thp_alloc,thp_fallback,pmd_free,memavail
    awk -F',' '
    NR == 1 { next }   # skip header
    {
        ts=$1; cs=$2; ds=$3; ta=$4; tf=$5; pmd=$6; ma=$7

        # Track first/last values
        if (NR == 2) {
            first_cs=cs; first_ds=ds; first_ta=ta; first_tf=tf
            first_pmd=pmd; first_ma=ma; first_ts=ts
            min_pmd=pmd; max_cs_rate=0; prev_cs=cs; prev_ts=ts
            min_ma=ma
        }

        # compact_stall rate per second
        dt = ts - prev_ts
        if (dt > 0) {
            rate = (cs - prev_cs) / dt
            if (rate > max_cs_rate) max_cs_rate = rate
        }
        prev_cs = cs; prev_ts = ts

        if (pmd+0 < min_pmd+0) min_pmd = pmd
        if (ma+0 < min_ma+0)   min_ma  = ma

        last_cs=cs; last_ds=ds; last_ta=ta; last_tf=tf
        last_pmd=pmd; last_ma=ma; last_ts=ts
    }
    END {
        duration = last_ts - first_ts
        total_cs = last_cs - first_cs
        total_ds = last_ds - first_ds
        total_ta = last_ta - first_ta
        total_tf = last_tf - first_tf
        pmd_drop = first_pmd - min_pmd
        ma_drop  = first_ma  - min_ma

        printf "  Duration:          %d seconds\n", duration
        printf "\n"
        printf "  %-40s %12s %12s\n", "Metric", "Total delta", "Rate/sec"
        printf "  %s\n", "────────────────────────────────────────────────────────────────"
        printf "  %-40s %12d %12.2f\n", "compact_stall", total_cs, (duration>0 ? total_cs/duration : 0)
        printf "  %-40s %12d %12.2f\n", "nr_deferred_split_page", total_ds, (duration>0 ? total_ds/duration : 0)
        printf "  %-40s %12d %12.2f\n", "thp_fault_alloc", total_ta, (duration>0 ? total_ta/duration : 0)
        printf "  %-40s %12d %12.2f\n", "thp_fault_fallback", total_tf, (duration>0 ? total_tf/duration : 0)
        printf "\n"
        printf "  Peak compact_stall rate:       %.2f stalls/sec\n", max_cs_rate
        printf "  Each stall ≈ 15-40ms latency spike on Snapdragon\n"
        printf "\n"
        printf "  %-40s %12s %12s\n", "Memory metric", "Start", "Min during run"
        printf "  %s\n", "────────────────────────────────────────────────────────────────"
        printf "  %-40s %9d kB %9d kB  (drop: %d kB)\n", "MemAvailable", first_ma, min_ma, ma_drop
        printf "  %-40s %12d %12d  (dropped by %d)\n", "PMD-free blocks (order 9)", first_pmd, min_pmd, pmd_drop
        printf "\n"
        if (total_ta + total_tf > 0)
            printf "  THP hit rate: %.1f%%  (%d alloc, %d fallback)\n",
                   100.0 * total_ta / (total_ta + total_tf), total_ta, total_tf
        printf "\n"
        printf "  Interpretation:\n"
        printf "  • compact_stall rate > 0.1/sec  → noticeable latency spikes\n"
        printf "  • PMD block drop > 50%%          → significant fragmentation\n"
        printf "  • MemAvailable drop > 5%%        → page cache being evicted\n"
        printf "  • With mthp_bestfit: all three should be near zero\n"
    }' "$REPORT_FILE"

    echo ""
    echo "  Raw data: $REPORT_FILE"
    echo "══════════════════════════════════════════════════════════════════════"
    exit 0
fi

# ── Monitoring mode ───────────────────────────────────────────────────────────
[[ -z "$OUT" ]] && OUT="mthp_metrics_$(date +%Y%m%d_%H%M%S).csv"

echo "mthp_monitor: writing to $OUT  (interval=${INTERVAL}s)"
[[ -n "$WATCH_PID" ]] && echo "  Watching PID $WATCH_PID — will stop when it exits"

# CSV header
echo "timestamp_s,compact_stall,nr_deferred_split_page,thp_fault_alloc,thp_fault_fallback,pmd_free_blocks,MemAvailable_kB" > "$OUT"

read_vmstat_key() {
    grep "^$1 " /proc/vmstat | awk '{print $2}'
}

read_pmd_free() {
    # Sum order-9 counts across all zones in /proc/buddyinfo
    awk '
    /zone/ {
        # skip "Node N, zone NAME"
        for (i=1; i<=NF; i++) if ($i=="zone") { start=i+2; break }
        # counts start at column start+1 (orders 0..10)
        total += $(start + 9 + 1)  # order 9
    }
    END { print total+0 }' /proc/buddyinfo 2>/dev/null || echo 0
}

read_memavail() {
    awk '/^MemAvailable:/{print $2}' /proc/meminfo
}

t0=$(date +%s)

while true; do
    ts=$(( $(date +%s) - t0 ))
    cs=$(read_vmstat_key compact_stall)
    ds=$(read_vmstat_key nr_deferred_split_page)
    ta=$(read_vmstat_key thp_fault_alloc)
    tf=$(read_vmstat_key thp_fault_fallback)
    pmd=$(read_pmd_free)
    ma=$(read_memavail)

    echo "${ts},${cs},${ds},${ta},${tf},${pmd},${ma}" >> "$OUT"

    # Print live summary line every 10 seconds
    if (( ts % 10 == 0 )); then
        printf "\r  t=%4ds  compact_stall=%-8s  PMD_free=%-4s  MemAvail=%s kB   " \
               "$ts" "$cs" "$pmd" "$ma" >&2
    fi

    # Check if watched PID has exited
    if [[ -n "$WATCH_PID" ]]; then
        if ! kill -0 "$WATCH_PID" 2>/dev/null; then
            echo "" >&2
            echo "mthp_monitor: PID $WATCH_PID exited — stopping." >&2
            break
        fi
    fi

    sleep "$INTERVAL"
done

echo "" >&2
echo "mthp_monitor: wrote $(wc -l < "$OUT") samples to $OUT" >&2
