#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# mthp_rpi5_autotest.sh — Autonomous mthp_bestfit test runner for RPi5
#
# This script is executed by Claude CLI autonomously — no user intervention
# needed once SSH access is configured.  Claude SSHes to the RPi5, runs
# both passes (baseline then bestfit), collects CSV metrics, prints a
# structured comparison report that Claude parses and summarises.
#
# Design:
#   The module approach eliminates rebooting between passes:
#     Pass 1 — rmmod mthp_bestfit_mod  → default kernel THP policy
#     Pass 2 — insmod mthp_bestfit_mod → bestfit autopilot active
#   mm_fragmenter pre-conditions memory to the same fragmentation level
#   in both passes so the only variable is the policy.
#
# Usage (from host — Claude runs this):
#   ./mthp_rpi5_autotest.sh [--host <rpi5_host>] [--user <user>]
#                            [--fragment-mb N] [--iterations N]
#                            [--module <path/to/mthp_bestfit_mod.ko>]
#
# Usage (on device directly):
#   ./mthp_rpi5_autotest.sh --local
#
# Output format:
#   Structured tagged lines that Claude can parse:
#     [METRIC] key=value
#     [RESULT] key=value
#     [VERDICT] text
#   Plus human-readable sections for display.

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
RPI5_HOST="${RPI5_HOST:-rpi5.local}"
RPI5_USER="${RPI5_USER:-pi}"
FRAGMENT_MB=256
ITERATIONS=3
MODULE_PATH=""
LOCAL_MODE=0
DEVICE_TMP="/tmp/mthp_autotest"
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)        RPI5_HOST="$2";    shift 2 ;;
        --user)        RPI5_USER="$2";    shift 2 ;;
        --fragment-mb) FRAGMENT_MB="$2";  shift 2 ;;
        --iterations)  ITERATIONS="$2";   shift 2 ;;
        --module)      MODULE_PATH="$2";  shift 2 ;;
        --local)       LOCAL_MODE=1;      shift   ;;
        --verbose)     VERBOSE=1;         shift   ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"

# ── Remote execution wrapper ────────────────────────────────────────────────────
rsh() {
    if [[ $LOCAL_MODE -eq 1 ]]; then
        bash -c "$1"
    else
        ssh $SSH_OPTS "${RPI5_USER}@${RPI5_HOST}" "$1"
    fi
}

rput() {    # rput local remote
    if [[ $LOCAL_MODE -eq 1 ]]; then
        cp "$1" "$2"
    else
        scp $SSH_OPTS "$1" "${RPI5_USER}@${RPI5_HOST}:$2"
    fi
}

rget() {    # rget remote local
    if [[ $LOCAL_MODE -eq 1 ]]; then
        cp "$1" "$2"
    else
        scp $SSH_OPTS "${RPI5_USER}@${RPI5_HOST}:$1" "$2"
    fi
}

# ── Logging ───────────────────────────────────────────────────────────────────
log()  { echo "  $*" >&2; }
step() { echo "" >&2; echo "── $* ──" >&2; }
metric() { echo "[METRIC] $*"; }   # Claude-parseable
result() { echo "[RESULT] $*"; }
verdict(){ echo "[VERDICT] $*"; }

# ── Check device ──────────────────────────────────────────────────────────────
step "Connecting to RPi5"

if [[ $LOCAL_MODE -eq 0 ]]; then
    log "Host: ${RPI5_USER}@${RPI5_HOST}"
    if ! ssh $SSH_OPTS "${RPI5_USER}@${RPI5_HOST}" true 2>/dev/null; then
        echo "[ERROR] Cannot connect to ${RPI5_USER}@${RPI5_HOST}" >&2
        echo "        Set RPI5_HOST and RPI5_USER env vars or use --host / --user" >&2
        exit 1
    fi
    log "Connected OK"
fi

KERNEL=$(rsh "uname -r")
ARCH=$(rsh "uname -m")
log "Kernel: $KERNEL  Arch: $ARCH"
metric "kernel=${KERNEL} arch=${ARCH} host=${RPI5_HOST}"

# ── Setup device workspace ────────────────────────────────────────────────────
step "Setting up device workspace"
rsh "mkdir -p ${DEVICE_TMP}"

# Check for required binaries on device
NEED_BINARIES=()
for bin in mm_fragmenter mthp_workload; do
    if ! rsh "[ -x ${DEVICE_TMP}/${bin} ]" 2>/dev/null; then
        NEED_BINARIES+=("$bin")
    fi
done

if [[ ${#NEED_BINARIES[@]} -gt 0 ]]; then
    log "Missing on device: ${NEED_BINARIES[*]}"
    log "Looking for pre-built ARM64 binaries alongside this script..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    for bin in "${NEED_BINARIES[@]}"; do
        candidates=(
            "${SCRIPT_DIR}/${bin}_arm64"
            "${SCRIPT_DIR}/${bin}"
        )
        found=0
        for c in "${candidates[@]}"; do
            if [[ -f "$c" ]]; then
                log "Pushing $c -> ${DEVICE_TMP}/${bin}"
                rput "$c" "${DEVICE_TMP}/${bin}"
                rsh "chmod +x ${DEVICE_TMP}/${bin}"
                found=1; break
            fi
        done
        if [[ $found -eq 0 ]]; then
            echo "[ERROR] Cannot find binary for '$bin'" >&2
            echo "        Build with: aarch64-linux-gnu-gcc -O2 -static -o ${bin}_arm64 ${bin}.c" >&2
            exit 1
        fi
    done
fi

# Push module if provided
if [[ -n "$MODULE_PATH" ]]; then
    log "Pushing module: $MODULE_PATH"
    rput "$MODULE_PATH" "${DEVICE_TMP}/mthp_bestfit_mod.ko"
fi

MODULE_ON_DEVICE="${DEVICE_TMP}/mthp_bestfit_mod.ko"
if ! rsh "[ -f ${MODULE_ON_DEVICE} ]" 2>/dev/null; then
    echo "[ERROR] Module not found on device at ${MODULE_ON_DEVICE}" >&2
    echo "        Build: make -C /lib/modules/\$(uname -r)/build M=\$(pwd) modules" >&2
    echo "        Then push: scp mthp_bestfit_mod.ko ${RPI5_USER}@${RPI5_HOST}:${MODULE_ON_DEVICE}" >&2
    exit 1
fi
log "Module present: ${MODULE_ON_DEVICE}"

# ── Snapshot helper (runs on device) ─────────────────────────────────────────
device_snapshot() {
    rsh "
        cs=\$(awk '/^compact_stall /{print \$2}' /proc/vmstat)
        ds=\$(awk '/^nr_deferred_split_page /{print \$2}' /proc/vmstat)
        ta=\$(awk '/^thp_fault_alloc /{print \$2}' /proc/vmstat)
        tf=\$(awk '/^thp_fault_fallback /{print \$2}' /proc/vmstat)
        pmd=\$(awk '
            /zone/ {
                n=split(\$0,a,\" \")
                for(i=1;i<=n;i++) if(a[i]==\"zone\"){start=i; break}
                # order-9 is 11th number after zone NAME
                total += a[start+11]+0
            }
            END{print total+0}' /proc/buddyinfo)
        ma=\$(awk '/^MemAvailable:/{print \$2}' /proc/meminfo)
        echo \"cs=\$cs ds=\$ds ta=\$ta tf=\$tf pmd=\$pmd ma=\$ma\"
    "
}

# ── CSV monitor (runs in background on device) ────────────────────────────────
start_monitor() {
    local csv="$1"
    rsh "
        echo 'timestamp_s,compact_stall,nr_deferred_split_page,thp_fault_alloc,thp_fault_fallback,pmd_free_blocks,MemAvailable_kB' > ${csv}
        t0=\$(date +%s)
        while [ -f ${DEVICE_TMP}/.monitor_run ]; do
            ts=\$(( \$(date +%s) - t0 ))
            cs=\$(awk '/^compact_stall /{print \$2}' /proc/vmstat)
            ds=\$(awk '/^nr_deferred_split_page /{print \$2}' /proc/vmstat)
            ta=\$(awk '/^thp_fault_alloc /{print \$2}' /proc/vmstat)
            tf=\$(awk '/^thp_fault_fallback /{print \$2}' /proc/vmstat)
            pmd=\$(awk '/zone/{n=split(\$0,a,\" \");for(i=1;i<=n;i++)if(a[i]==\"zone\"){s=i;break};total+=a[s+11]+0}END{print total+0}' /proc/buddyinfo)
            ma=\$(awk '/^MemAvailable:/{print \$2}' /proc/meminfo)
            echo \"\$ts,\$cs,\$ds,\$ta,\$tf,\$pmd,\$ma\" >> ${csv}
            sleep 1
        done
    " &
    MONITOR_PID=$!
}

stop_monitor() {
    rsh "rm -f ${DEVICE_TMP}/.monitor_run" 2>/dev/null || true
    sleep 2
    kill $MONITOR_PID 2>/dev/null || true
}

# ── Run one pass ──────────────────────────────────────────────────────────────
run_pass() {
    local label="$1"        # "baseline" or "bestfit"
    local module_load="$2"  # "load" or "skip"
    local csv="${DEVICE_TMP}/${label}_metrics.csv"

    step "Pass: ${label^^}"

    # Load / unload module
    rsh "rmmod mthp_bestfit_mod 2>/dev/null || true"
    if [[ "$module_load" == "load" ]]; then
        rsh "insmod ${MODULE_ON_DEVICE}"
        log "Module loaded — bestfit autopilot active"
        rsh "dmesg | tail -3" | log
    else
        log "Module NOT loaded — default kernel THP policy"
    fi

    # Pre-snapshot
    local before
    before=$(device_snapshot)
    log "Before: $before"

    # Start monitor
    rsh "touch ${DEVICE_TMP}/.monitor_run"
    start_monitor "$csv"
    log "Monitor started -> $csv"

    # Start fragmenter
    rsh "nohup ${DEVICE_TMP}/mm_fragmenter ${FRAGMENT_MB} hold > ${DEVICE_TMP}/frag_${label}.log 2>&1 &"
    sleep 4  # let fragmentation complete
    local frag_log
    frag_log=$(rsh "tail -3 ${DEVICE_TMP}/frag_${label}.log" 2>/dev/null || true)
    log "Fragmenter: $frag_log"

    # Run workload
    log "Running workload (${ITERATIONS} iterations)..."
    local durations=()
    for (( i=1; i<=ITERATIONS; i++ )); do
        local t0 t1 elapsed
        t0=$(date +%s%3N)
        rsh "${DEVICE_TMP}/mthp_workload 8 4" > /dev/null 2>&1 || true
        t1=$(date +%s%3N)
        elapsed=$(( t1 - t0 ))
        durations+=("$elapsed")
        log "  iter $i: ${elapsed} ms"
    done

    # Median
    local sorted median
    sorted=($(printf '%s\n' "${durations[@]}" | sort -n))
    median="${sorted[$(( ${#sorted[@]} / 2 ))]}"

    # Stop fragmenter + monitor
    rsh "killall mm_fragmenter 2>/dev/null || true"
    stop_monitor

    # Post-snapshot
    local after
    after=$(device_snapshot)
    log "After:  $after"

    # Parse deltas
    local cs_b ds_b ta_b tf_b pmd_b ma_b
    local cs_a ds_a ta_a tf_a pmd_a ma_a
    eval "$( echo "$before" | tr ' ' '\n' | sed 's/=/\_b=/' )"
    eval "$( echo "$after"  | tr ' ' '\n' | sed 's/=/\_a=/' )"

    local dcs=$(( cs_a - cs_b ))
    local dds=$(( ds_a - ds_b ))
    local dta=$(( ta_a - ta_b ))
    local dtf=$(( tf_a - tf_b ))

    local hit="n/a"
    if (( dta + dtf > 0 )); then
        hit=$(awk "BEGIN{printf \"%.1f\", 100.0*${dta}/(${dta}+${dtf})}")
    fi

    # CSV stats from device
    local peak_rate min_pmd min_ma
    peak_rate=$(rsh "awk -F',' '
        NR==1{next}
        {ts=\$1; cs=\$2; if(NR==2){pts=ts;pcs=cs;next}
         dt=ts-pts; if(dt>0){r=(cs-pcs)/dt; if(r>mx)mx=r}; pts=ts; pcs=cs}
        END{printf \"%.2f\", mx+0}
    ' ${csv}" 2>/dev/null || echo "0.00")

    min_pmd=$(rsh "awk -F',' 'NR>1{if(NR==2||(\$6+0)<m)m=\$6+0}END{print m+0}' ${csv}" 2>/dev/null || echo "0")
    min_ma=$(rsh "awk -F',' 'NR>1{if(NR==2||(\$7+0)<m)m=\$7+0}END{print m+0}' ${csv}" 2>/dev/null || echo "0")

    # Emit structured metrics
    result "pass=${label} compact_stall_delta=${dcs} deferred_split_delta=${dds}"
    result "pass=${label} thp_alloc_delta=${dta} thp_fallback_delta=${dtf} thp_hit_pct=${hit}"
    result "pass=${label} pmd_free_min=${min_pmd} memavail_min_kb=${min_ma}"
    result "pass=${label} peak_stall_rate=${peak_rate} workload_median_ms=${median}"

    echo "$dcs $dds $dta $dtf $hit $min_pmd $min_ma $peak_rate $median"
}

# ── Pull CSV files ─────────────────────────────────────────────────────────────
pull_csvs() {
    local outdir="${1:-.}"
    for label in baseline bestfit; do
        local remote="${DEVICE_TMP}/${label}_metrics.csv"
        local local_f="${outdir}/${label}_metrics.csv"
        if rsh "[ -f ${remote} ]" 2>/dev/null; then
            rget "$remote" "$local_f"
            log "Pulled $local_f"
        fi
    done
}

# ── Main ──────────────────────────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
LOCAL_OUT="/tmp/mthp_autotest_${TS}"
mkdir -p "$LOCAL_OUT"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  mthp_bestfit autonomous test — RPi5 sequential comparison           ║"
printf "║  Host: %-30s  Fragment: %4d MB  Iters: %d     ║\n" \
       "${RPI5_HOST}" "$FRAGMENT_MB" "$ITERATIONS"
echo "╚══════════════════════════════════════════════════════════════════════╝"

metric "fragment_mb=${FRAGMENT_MB} iterations=${ITERATIONS} timestamp=${TS}"

# Pass 1 — baseline
read -r dcs1 dds1 dta1 dtf1 hit1 pmd1 ma1 rate1 dur1 \
    < <(run_pass "baseline" "skip")

# Pass 2 — bestfit
read -r dcs2 dds2 dta2 dtf2 hit2 pmd2 ma2 rate2 dur2 \
    < <(run_pass "bestfit" "load")

# Make sure module is unloaded after test
rsh "rmmod mthp_bestfit_mod 2>/dev/null || true"

# Pull CSVs to host
pull_csvs "$LOCAL_OUT"

# ── Comparison report ──────────────────────────────────────────────────────────
step "Comparison Report"

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  mthp_bestfit sequential comparison — RPi5 (${KERNEL})"
echo "══════════════════════════════════════════════════════════════════════"
printf "\n  %-38s  %-14s  %-14s\n" "Metric" "baseline" "bestfit"
printf "  %s\n" "────────────────────────────────────────────────────────────────────"

cmp_row() {
    local label="$1" v1="$2" v2="$3" good="${4:-lower}"
    local mark1="" mark2=""
    if [[ "$good" == "lower" ]]; then
        [[ "$v1" -gt "$v2" ]] 2>/dev/null && mark1=" ← worse"
        [[ "$v2" -gt "$v1" ]] 2>/dev/null && mark2=" ← worse"
    else
        [[ "$v1" -lt "$v2" ]] 2>/dev/null && mark1=" ← worse"
        [[ "$v2" -lt "$v1" ]] 2>/dev/null && mark2=" ← worse"
    fi
    printf "  %-38s  %-14s  %-14s\n" "$label" "${v1}${mark1}" "${v2}${mark2}"
}

cmp_row "compact_stall delta"          "$dcs1"  "$dcs2"
cmp_row "compact_stall peak (stalls/s)" "$rate1" "$rate2"
cmp_row "deferred_split delta"          "$dds1"  "$dds2"
cmp_row "thp_alloc delta"               "$dta1"  "$dta2"   "higher"
cmp_row "thp_fallback delta"            "$dtf1"  "$dtf2"
cmp_row "THP hit rate %"                "$hit1"  "$hit2"   "higher"
cmp_row "PMD free blocks (min)"         "$pmd1"  "$pmd2"   "higher"
cmp_row "MemAvailable min (MB)"         "$(( ma1/1024 ))" "$(( ma2/1024 ))" "higher"
cmp_row "Workload median (ms)"          "$dur1"  "$dur2"

echo ""

# ── Verdict ────────────────────────────────────────────────────────────────────
echo "  Verdict:"
if [[ "$dcs2" -lt "$dcs1" ]]; then
    saved=$(( dcs1 - dcs2 ))
    echo "  ✓ compact_stall reduced by ${saved} stalls (${rate1} → ${rate2}/s)" \
         "— each avoided = ~15-40ms latency spike"
    verdict "compact_stall_reduced by=${saved} rate_baseline=${rate1} rate_bestfit=${rate2}"
elif [[ "$dcs1" -eq 0 && "$dcs2" -eq 0 ]]; then
    echo "  △ No compact_stall in either pass — increase --fragment-mb"
    echo "    or test on a device with less free RAM"
    verdict "no_stall_both_passes — increase fragmentation pressure"
else
    echo "  ? Unexpected result — review fragmentation parameters"
    verdict "unexpected_result — review"
fi

if [[ "$dds2" -lt "$dds1" ]]; then
    saved_pages=$(( dds1 - dds2 ))
    echo "  ✓ deferred_split reduced by ${saved_pages} folios" \
         "→ $(( saved_pages * 4 / 1024 )) MB returned to buddy faster"
    verdict "deferred_split_reduced by=${saved_pages}"
fi

if (( ma2 > ma1 )); then
    gain=$(( (ma2 - ma1) / 1024 ))
    echo "  ✓ MemAvailable higher by ${gain} MB with bestfit"
    verdict "memavail_gain_mb=${gain}"
fi

echo ""
echo "  CSV files: ${LOCAL_OUT}/"
echo "══════════════════════════════════════════════════════════════════════"
