#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# mthp_vma_compare.sh — Side-by-side VMA folio order comparison
#                        for baseline vs mthp_bestfit autopilot
#
# Purpose
# ───────
# Runs the SAME application on two kernels (or two PIDs on one machine for
# local dev) and shows a side-by-side table of:
#
#   • Which hugepage orders each VMA uses          → shows bestfit's geometric
#                                                     selection vs order-0/9 only
#   • compact_stall / deferred_split / MemAvailable → quantifies the benefit
#   • PMD free block drain                          → shows if fragmentation
#                                                     is actually the cause
#
# The same application means: same VMA layout, same workload, only the kernel
# THP policy differs.  This is why concurrent comparison on two devices is the
# gold standard: time-varying system effects cancel out.
#
# Usage modes
# ───────────
#   Mode 1 — Two local PIDs (one process per kernel, e.g. two QEMU VMs on
#             localhost via network namespaces):
#
#     ./mthp_vma_compare.sh --pid1 <pid_baseline> --pid2 <pid_bestfit>
#
#   Mode 2 — Named process, both local (dev/smoke test with two instances):
#
#     ./mthp_vma_compare.sh --name tflite_mobilenetssd_workload
#
#   Mode 3 — Named process on two SSH hosts (production comparison):
#
#     ./mthp_vma_compare.sh \
#         --name benchmark_model \
#         --host1 root@device-baseline \
#         --host2 root@device-bestfit
#
#   Mode 4 — Named process, local + remote:
#
#     ./mthp_vma_compare.sh \
#         --name tflite_mobilenetssd_workload \
#         --host2 root@device-bestfit
#
# Options:
#   --pid1  <pid>    PID on device 1 (baseline)
#   --pid2  <pid>    PID on device 2 (bestfit)
#   --name  <comm>   Find PIDs by process comm name (used with --host1/2)
#   --host1 <host>   SSH host for device 1 (default: local)
#   --host2 <host>   SSH host for device 2 (default: local)
#   --label1 <str>   Label for device 1 (default: "baseline")
#   --label2 <str>   Label for device 2 (default: "bestfit")
#   --watch          Refresh every --interval seconds
#   --interval <N>   Refresh interval in seconds (default: 5)
#   --out <dir>      Directory for CSV output files (default: /tmp)
#
# Reading the output
# ──────────────────
# compact_stall delta: this is stalls that fired DURING the comparison window.
#   0 on bestfit, >0 on baseline → bestfit is avoiding compaction entirely.
#
# deferred_split delta: hugepage folios stuck in limbo (partial unmap/COW).
#   Lower on bestfit because smaller orders fit tensors better → no split.
#
# MemAvailable: direct free-memory metric.
#   Higher on bestfit because:
#     (a) deferred_split pages return to buddy faster
#     (b) compact_stall not evicting page cache
#
# Order distribution: bestfit uses orders 2-8 for mid-size tensors.
#   Baseline only shows order-0 (base pages) and order-9 (2 MB THP).
#   Diverse orders = better TLB coverage + no internal fragmentation.

set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────
PID1=""
PID2=""
PROC_NAME=""
HOST1=""       # empty = local
HOST2=""       # empty = local
LABEL1="baseline"
LABEL2="bestfit"
DO_WATCH=0
INTERVAL=5
OUTDIR="/tmp"
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pid1)     PID1="$2";      shift 2 ;;
        --pid2)     PID2="$2";      shift 2 ;;
        --name)     PROC_NAME="$2"; shift 2 ;;
        --host1)    HOST1="$2";     shift 2 ;;
        --host2)    HOST2="$2";     shift 2 ;;
        --label1)   LABEL1="$2";    shift 2 ;;
        --label2)   LABEL2="$2";    shift 2 ;;
        --watch)    DO_WATCH=1;     shift   ;;
        --interval) INTERVAL="$2";  shift 2 ;;
        --out)      OUTDIR="$2";    shift 2 ;;
        --verbose)  VERBOSE=1;      shift   ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────

run_cmd() {
    local host="$1"
    shift
    if [[ -z "$host" ]]; then
        "$@"
    else
        ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" "$@"
    fi
}

find_pid_by_name() {
    local host="$1"
    local name="$2"
    # Find all PIDs with this comm, return the first one
    run_cmd "$host" bash -c "
        for pid in /proc/[0-9]*/comm; do
            [ -r \"\$pid\" ] || continue
            comm=\$(cat \"\$pid\" 2>/dev/null)
            if [ \"\$comm\" = \"$name\" ]; then
                echo \"\${pid%/comm}\" | grep -o '[0-9]*'
                exit 0
            fi
        done
        echo ''
    " 2>/dev/null | head -1
}

# Reads a single vmstat key value
read_vmstat() {
    local host="$1"
    local key="$2"
    run_cmd "$host" awk "/^${key} /{print \$2}" /proc/vmstat 2>/dev/null || echo 0
}

# Reads order-9 free block count from buddyinfo
read_pmd_free() {
    local host="$1"
    run_cmd "$host" awk '
        /zone/ {
            for(i=1;i<=NF;i++) if($i=="zone"){start=i;break}
            # columns after "zone NAME" are orders 0..10
            # order 9 is at position start+2+9 = start+11
            total += $(start+11)
        }
        END { print total+0 }
    ' /proc/buddyinfo 2>/dev/null || echo 0
}

read_memavail() {
    local host="$1"
    run_cmd "$host" awk '/^MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null || echo 0
}

# Read smaps for a PID and return per-VMA summary lines:
# "name|rss_kb|anon_hp_kb|thp_eligible"
read_smaps_summary() {
    local host="$1"
    local pid="$2"
    run_cmd "$host" awk -F: '
    /^[0-9a-f]+-[0-9a-f]+/ {
        if (vname != "" && rss+0 >= 64) {
            printf "%s|%d|%d|%d\n", vname, rss, ahp, thpe
        }
        # reset
        match($0, /[^ ]+$/)
        vname = substr($0, RSTART, RLENGTH)
        if (vname == "") vname = "[anon]"
        rss=0; ahp=0; thpe=0
    }
    /^Rss:/           { rss=$2+0 }
    /^AnonHugePages:/ { ahp=$2+0 }
    /^THPeligible:/   { thpe=$2+0 }
    END {
        if (vname != "" && rss+0 >= 64)
            printf "%s|%d|%d|%d\n", vname, rss, ahp, thpe
    }
    ' "/proc/${pid}/smaps" 2>/dev/null
}

# ── Resolve PIDs ──────────────────────────────────────────────────────────────

if [[ -n "$PROC_NAME" ]]; then
    if [[ -z "$PID1" ]]; then
        PID1=$(find_pid_by_name "$HOST1" "$PROC_NAME")
        [[ -z "$PID1" ]] && { echo "ERROR: '$PROC_NAME' not found on ${HOST1:-local}" >&2; exit 1; }
        echo "Found PID $PID1 for '$PROC_NAME' on ${HOST1:-local}"
    fi
    if [[ -z "$PID2" ]]; then
        PID2=$(find_pid_by_name "$HOST2" "$PROC_NAME")
        [[ -z "$PID2" ]] && { echo "ERROR: '$PROC_NAME' not found on ${HOST2:-local}" >&2; exit 1; }
        echo "Found PID $PID2 for '$PROC_NAME' on ${HOST2:-local}"
    fi
fi

if [[ -z "$PID1" && -z "$PID2" ]]; then
    echo "Usage: $0 --pid1 <pid> --pid2 <pid>" >&2
    echo "       $0 --name <comm> [--host1 <ssh>] [--host2 <ssh>]" >&2
    echo "       $0 --name <comm> --host2 <ssh>   (local vs remote)" >&2
    exit 1
fi

# ── CSV setup ─────────────────────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
CSV1="${OUTDIR}/mthp_compare_${LABEL1}_${TS}.csv"
CSV2="${OUTDIR}/mthp_compare_${LABEL2}_${TS}.csv"

# Write CSV headers
echo "iter,ts,compact_stall,deferred_split,thp_alloc,thp_fallback,pmd_free,memavail_kb" \
    > "$CSV1"
echo "iter,ts,compact_stall,deferred_split,thp_alloc,thp_fallback,pmd_free,memavail_kb" \
    > "$CSV2"

# ── Baseline metrics (delta origin) ───────────────────────────────────────────
base_cs1=0;  base_ds1=0;  base_ta1=0;  base_tf1=0
base_cs2=0;  base_ds2=0;  base_ta2=0;  base_tf2=0

if [[ -n "$PID1" ]]; then
    base_cs1=$(read_vmstat "$HOST1" compact_stall)
    base_ds1=$(read_vmstat "$HOST1" nr_deferred_split_page)
    base_ta1=$(read_vmstat "$HOST1" thp_fault_alloc)
    base_tf1=$(read_vmstat "$HOST1" thp_fault_fallback)
fi
if [[ -n "$PID2" ]]; then
    base_cs2=$(read_vmstat "$HOST2" compact_stall)
    base_ds2=$(read_vmstat "$HOST2" nr_deferred_split_page)
    base_ta2=$(read_vmstat "$HOST2" thp_fault_alloc)
    base_tf2=$(read_vmstat "$HOST2" thp_fault_fallback)
fi

# ── Display helpers ───────────────────────────────────────────────────────────

bar() {
    local pct="$1"        # 0-100 integer
    local width="${2:-30}"
    local filled=$(( pct * width / 100 ))
    local i
    printf "["
    for (( i=0; i<filled; i++ )); do printf "█"; done
    for (( i=filled; i<width; i++ )); do printf "░"; done
    printf "]"
}

# ── Main comparison loop ───────────────────────────────────────────────────────
ITER=0
while true; do
    [[ $DO_WATCH -eq 1 ]] && clear

    NOW=$(date +%s)
    TIMESTR=$(date +%H:%M:%S)

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    printf "║  mTHP VMA Comparison   %-12s vs %-12s   iter=%-4d  %-8s ║\n" \
           "$LABEL1" "$LABEL2" "$ITER" "$TIMESTR"
    echo "╚══════════════════════════════════════════════════════════════════════╝"

    # ── Read current metrics ────────────────────────────────────────────────
    cs1="N/A"; ds1="N/A"; ta1="N/A"; tf1="N/A"; pmd1="N/A"; ma1="N/A"
    cs2="N/A"; ds2="N/A"; ta2="N/A"; tf2="N/A"; pmd2="N/A"; ma2="N/A"

    if [[ -n "$PID1" ]]; then
        cs1=$(read_vmstat  "$HOST1" compact_stall)
        ds1=$(read_vmstat  "$HOST1" nr_deferred_split_page)
        ta1=$(read_vmstat  "$HOST1" thp_fault_alloc)
        tf1=$(read_vmstat  "$HOST1" thp_fault_fallback)
        pmd1=$(read_pmd_free "$HOST1")
        ma1=$(read_memavail  "$HOST1")
    fi
    if [[ -n "$PID2" ]]; then
        cs2=$(read_vmstat  "$HOST2" compact_stall)
        ds2=$(read_vmstat  "$HOST2" nr_deferred_split_page)
        ta2=$(read_vmstat  "$HOST2" thp_fault_alloc)
        tf2=$(read_vmstat  "$HOST2" thp_fault_fallback)
        pmd2=$(read_pmd_free "$HOST2")
        ma2=$(read_memavail  "$HOST2")
    fi

    # Compute deltas
    dcs1=$(( cs1 - base_cs1 )); dcs2=$(( cs2 - base_cs2 ))
    dds1=$(( ds1 - base_ds1 )); dds2=$(( ds2 - base_ds2 ))
    dta1=$(( ta1 - base_ta1 )); dta2=$(( ta2 - base_ta2 ))
    dtf1=$(( tf1 - base_tf1 )); dtf2=$(( tf2 - base_tf2 ))

    # THP hit rate (during window)
    if (( dta1 + dtf1 > 0 )); then
        hit1=$(awk "BEGIN{printf \"%.1f\", 100.0 * $dta1 / ($dta1 + $dtf1)}")
    else
        hit1="n/a"
    fi
    if (( dta2 + dtf2 > 0 )); then
        hit2=$(awk "BEGIN{printf \"%.1f\", 100.0 * $dta2 / ($dta2 + $dtf2)}")
    else
        hit2="n/a"
    fi

    # ── System metrics table ────────────────────────────────────────────────
    echo ""
    printf "  %-34s  %-16s  %-16s\n" "Metric" "$LABEL1" "$LABEL2"
    printf "  %s\n" "─────────────────────────────────────────────────────────────────────"
    printf "  %-34s  %-16s  %-16s\n" "compact_stall (delta)"        "+${dcs1}"        "+${dcs2}"
    printf "  %-34s  %-16s  %-16s\n" "deferred_split (delta)"       "+${dds1}"        "+${dds2}"
    printf "  %-34s  %-16s  %-16s\n" "thp_fault_alloc (delta)"      "+${dta1}"        "+${dta2}"
    printf "  %-34s  %-16s  %-16s\n" "thp_fault_fallback (delta)"   "+${dtf1}"        "+${dtf2}"
    printf "  %-34s  %-16s  %-16s\n" "THP hit rate (window)"        "${hit1}%"        "${hit2}%"
    printf "  %-34s  %-16s  %-16s\n" "PMD free blocks (order-9)"    "${pmd1}"         "${pmd2}"
    printf "  %-34s  %-16s  %-16s\n" "MemAvailable (MB)"            "$(( ma1/1024 ))" "$(( ma2/1024 ))"

    # ── Fragmentation pressure gauge ────────────────────────────────────────
    echo ""
    echo "  Memory pressure gauge:"
    echo "    (PMD free < 10 → stall imminent; 0 → every THP request triggers compaction)"

    pmd_pct1=$(( pmd1 > 200 ? 100 : pmd1 * 100 / 200 ))
    pmd_pct2=$(( pmd2 > 200 ? 100 : pmd2 * 100 / 200 ))

    printf "    %-12s PMD=%3d  " "$LABEL1" "$pmd1"
    bar "$pmd_pct1" 40
    echo ""
    printf "    %-12s PMD=%3d  " "$LABEL2" "$pmd2"
    bar "$pmd_pct2" 40
    echo ""

    # ── Per-VMA smaps comparison ─────────────────────────────────────────────
    echo ""
    echo "  Per-VMA THP coverage (AnonHugePages / RSS):"
    echo "  ─────────────────────────────────────────────────────────────────────"
    printf "  %-32s  %-22s  %-22s\n" "VMA name" "$LABEL1 (RSS | AnonHP%)" "$LABEL2 (RSS | AnonHP%)"
    echo "  ─────────────────────────────────────────────────────────────────────"

    # Build associative arrays keyed by VMA name
    declare -A vma_rss1 vma_ahp1 vma_rss2 vma_ahp2 vma_seen

    if [[ -n "$PID1" ]]; then
        while IFS='|' read -r name rss ahp thpe; do
            vma_rss1["$name"]="$rss"
            vma_ahp1["$name"]="$ahp"
            vma_seen["$name"]=1
        done < <(read_smaps_summary "$HOST1" "$PID1")
    fi
    if [[ -n "$PID2" ]]; then
        while IFS='|' read -r name rss ahp thpe; do
            vma_rss2["$name"]="$rss"
            vma_ahp2["$name"]="$ahp"
            vma_seen["$name"]=1
        done < <(read_smaps_summary "$HOST2" "$PID2")
    fi

    for name in "${!vma_seen[@]}"; do
        r1="${vma_rss1[$name]:-0}"; a1="${vma_ahp1[$name]:-0}"
        r2="${vma_rss2[$name]:-0}"; a2="${vma_ahp2[$name]:-0}"

        # Compute AnonHP coverage percentages
        pct1=0; pct2=0
        [[ $r1 -gt 0 ]] && pct1=$(( a1 * 100 / r1 ))
        [[ $r2 -gt 0 ]] && pct2=$(( a2 * 100 / r2 ))

        col1="$(( r1/1024 ))MB | ${pct1}% huge"
        col2="$(( r2/1024 ))MB | ${pct2}% huge"

        printf "  %-32s  %-22s  %-22s\n" \
               "${name:0:32}" "$col1" "$col2"
    done

    # ── Interpretation legend ───────────────────────────────────────────────
    echo ""
    echo "  Interpretation:"
    echo "  • compact_stall delta=0 on bestfit, >0 on baseline → stalls avoided"
    echo "  • deferred_split lower on bestfit → smaller orders freed cleanly"
    echo "  • MemAvailable higher on bestfit → both mechanisms working"
    echo "  • PMD drop = fragmentation building up; bestfit survives it"
    echo "  • THP hit % higher on bestfit → more VMAs getting huge pages"

    # ── CSV append ──────────────────────────────────────────────────────────
    [[ -n "$PID1" ]] && echo "$ITER,$NOW,$cs1,$ds1,$ta1,$tf1,$pmd1,$ma1" >> "$CSV1"
    [[ -n "$PID2" ]] && echo "$ITER,$NOW,$cs2,$ds2,$ta2,$tf2,$pmd2,$ma2" >> "$CSV2"

    echo ""
    echo "  CSV: $CSV1"
    echo "       $CSV2"

    [[ $DO_WATCH -eq 0 ]] && break
    [[ $VERBOSE -eq 1 ]] && echo "  Sleeping ${INTERVAL}s ..."
    sleep "$INTERVAL"
    (( ITER++ )) || true
done

# ── Final summary ─────────────────────────────────────────────────────────────
if [[ $ITER -gt 0 ]]; then
    echo ""
    echo "══════════════════════════════════════════════════════════════════════"
    echo "  Run summary ($ITER iterations)"
    echo "══════════════════════════════════════════════════════════════════════"

    # Compute totals from CSV using awk
    for label in "$LABEL1:$CSV1" "$LABEL2:$CSV2"; do
        lbl="${label%%:*}"
        csv="${label##*:}"
        echo ""
        echo "  [$lbl]"
        awk -F',' 'NR==1{next}
        {
            cs=$3; ds=$4; ta=$5; tf=$6; pmd=$7; ma=$8
            if(NR==2){ cs0=cs; ds0=ds; ta0=ta; tf0=tf; min_pmd=pmd; min_ma=ma }
            if(pmd+0<min_pmd+0) min_pmd=pmd
            if(ma+0<min_ma+0)   min_ma=ma
            last_cs=cs; last_ds=ds; last_ta=ta; last_tf=tf
        }
        END {
            printf "    compact_stall total:    %d\n", last_cs-cs0
            printf "    deferred_split total:   %d\n", last_ds-ds0
            printf "    thp_alloc total:        %d\n", last_ta-ta0
            printf "    thp_fallback total:     %d\n", last_tf-tf0
            printf "    PMD free min:           %d\n", min_pmd
            printf "    MemAvailable min:       %d MB\n", min_ma/1024
        }' "$csv" 2>/dev/null || echo "    (no data)"
    done
fi
