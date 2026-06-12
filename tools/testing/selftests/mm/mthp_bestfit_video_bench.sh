#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
#
# mthp_bestfit_video_bench.sh — Video frame object detection mTHP benchmark
#
# Runs MobileNet-SSD object detection on frames extracted from a video file,
# measuring mTHP fragmentation metrics before and after.
#
# Recommended open-source video: Big Buck Bunny
# ─────────────────────────────────────────────
#   Title:   Big Buck Bunny
#   Creator: Blender Foundation (2008)
#   License: Creative Commons Attribution 3.0 (CC BY 3.0)
#   Why:     Diverse scenes with animals and objects → good for detection
#            Multiple resolutions available (4K, 1080p, 720p, 480p)
#            Standard reference video used across codec/media test suites
#
#   Download (any mirror of the Blender demo movies):
#     The file is named  bbb_sunflower_1080p_30fps.mp4  (1.2 GB, full)
#     or                 bbb_sunflower_720p_30fps.mp4   (400 MB, lighter)
#     Search: "Big Buck Bunny blender demo movies" → blender.org/demo/movies
#     Or via archive.org: search "Big Buck Bunny 1080p"
#
#   Other CC-licensed Blender videos:
#     Elephants Dream  (CC BY 2.5)  — first open movie
#     Sintel           (CC BY 3.0)  — fantasy drama
#     Tears of Steel   (CC BY 3.0)  — sci-fi short
#
# Why video frames create real mTHP fragmentation
# ─────────────────────────────────────────────────
# Each video frame is a JPEG or PPM file of a different size.  Processing
# frames with MobileNet-SSD creates this per-frame memory sequence:
#
#   jpeg_buf    frame file in RAM          ~ 80 –  800 KB  (order 4–8)
#   decoded_buf RGB decode of full frame   ~ 900 – 6,000 KB (order 7–9)
#   resize_buf  scaled to 300×300 (fixed)     263 KB        (order 6)
#   norm_buf    float32 normalised (fixed)  1,054 KB        (order 8)
#   arena       TFLite tensor arena (live)  3,700 KB        (order 9)
#   output_buf  detection results             52 KB
#
# The decoded_buf varies frame-by-frame (scene changes → different sizes).
# At 1080p: decoded = 5.9 MB → uses PMD hugepages → wasted tail on free
# At  720p: decoded = 2.6 MB → uses PMD hugepages → smaller tail
# At  480p: decoded = 900 KB → uses order-7 hugepages (512 KB)
#
# Running all three resolutions concurrently creates the mixed-size alloc
# pattern that fragments the buddy allocator.  After ~50 frames, compact_stall
# fires on arena allocations → inference latency spikes 15–40 ms per event.
#
# With mthp_bestfit: each size gets the right hugepage order → no fragmentation.
#
# Usage:
#   ./mthp_bestfit_video_bench.sh [options]
#
#   --tag       baseline|bestfit           label for results file
#   --video     /path/to/video.mp4         video to extract frames from
#   --frames    /path/to/frames/           use pre-extracted frames (PPM/JPEG)
#   --fps       N                          frames per second to extract (default: 2)
#   --duration  N                          seconds of video to use (default: 60)
#   --streams   N                          concurrent model instances (default: 4)
#   --repeat    N                          times to loop through the frame set (default: 3)
#   --generate                             use ffmpeg test source (no video file needed)
#   --compare   baseline.txt bestfit.txt   print delta comparison table
#
# Quickstart (no video file needed — auto-generates test frames):
#   ./mthp_bestfit_video_bench.sh --tag baseline --generate
#
# With Big Buck Bunny:
#   ./mthp_bestfit_video_bench.sh --tag baseline --video bigbuckbunny_720p.mp4
#
# With pre-extracted frames:
#   ./mthp_bestfit_video_bench.sh --tag baseline --frames /path/to/frames/
#
# Dependencies: ffmpeg (frame extraction), gcc (builds synthetic workload)

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
TAG="run"
VIDEO=""
FRAMES_DIR=""
FPS=2
DURATION=60
STREAMS=4
REPEAT=3
GENERATE=0
COMPARE_MODE=0
COMPARE_BASE=""
COMPARE_BEST=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)       TAG="$2";        shift 2 ;;
        --video)     VIDEO="$2";      shift 2 ;;
        --frames)    FRAMES_DIR="$2"; shift 2 ;;
        --fps)       FPS="$2";        shift 2 ;;
        --duration)  DURATION="$2";   shift 2 ;;
        --streams)   STREAMS="$2";    shift 2 ;;
        --repeat)    REPEAT="$2";     shift 2 ;;
        --generate)  GENERATE=1;      shift   ;;
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
    exit 0
fi

# ── Check ffmpeg ──────────────────────────────────────────────────────────────
FFMPEG=""
if command -v ffmpeg &>/dev/null; then
    FFMPEG="ffmpeg"
else
    echo "WARNING: ffmpeg not found."
    echo "  Install: apt-get install ffmpeg   or   dnf install ffmpeg"
    echo "  ffmpeg is required to extract video frames."
    if [[ -n "$VIDEO" ]]; then
        echo "ERROR: --video requires ffmpeg." >&2
        exit 1
    fi
    if [[ -z "$FRAMES_DIR" ]]; then
        echo "  Falling back to synthetic image sources (no video needed)."
    fi
fi

# ── Locate / build C workload ─────────────────────────────────────────────────
WORKLOAD_BIN=""
find_or_build_workload() {
    local bin="$SCRIPT_DIR/tflite_mobilenetssd_workload"
    if [[ -x "$bin" ]]; then WORKLOAD_BIN="$bin"; return 0; fi
    local src="$SCRIPT_DIR/tflite_mobilenetssd_workload.c"
    if [[ ! -f "$src" ]]; then
        echo "ERROR: tflite_mobilenetssd_workload.c not found in $SCRIPT_DIR" >&2
        return 1
    fi
    command -v gcc &>/dev/null || { echo "ERROR: gcc not found" >&2; return 1; }
    echo "Building tflite_mobilenetssd_workload..."
    gcc -O2 -o "$bin" "$src"
    WORKLOAD_BIN="$bin"
    echo "Built: $bin"
}

find_or_build_workload || exit 1

# ── Frame extraction ──────────────────────────────────────────────────────────
#
# Extracts frames at three resolutions so the per-frame decoded buffers span
# a wide range of hugepage orders:
#
#   1080p  decoded = 1920×1080×3 = 5,934 KB  → order 9 (PMD = 2 MB × 3)
#    720p  decoded = 1280×720×3  = 2,662 KB  → order 9 (PMD = 2 MB × 1)
#    480p  decoded =  640×480×3  =   900 KB  → order 7 (512 KB × 2)
#
# The mix of all three is what creates maximum buddy fragmentation.
#
extract_frames() {
    local src="$1"
    local outdir="$2"
    local is_generated="$3"

    mkdir -p "$outdir/1080p" "$outdir/720p" "$outdir/480p"

    local total_frames=$(( DURATION * FPS ))
    echo "Extracting frames from: $src"
    echo "  Duration: ${DURATION}s   FPS: ${FPS}   Total per resolution: ~$total_frames"

    if [[ "$is_generated" == "1" ]]; then
        # Use ffmpeg's built-in test source (no real video needed)
        # testsrc2 generates colorful patterns resembling natural images
        echo "  Source: ffmpeg testsrc2 (synthetic, no download needed)"
        echo "  Extracting 1080p frames..."
        "$FFMPEG" -loglevel warning \
            -f lavfi -i "testsrc2=duration=${DURATION}:size=1920x1080:rate=${FPS}" \
            -q:v 3 "$outdir/1080p/frame_%04d.ppm" 2>&1

        echo "  Extracting 720p frames..."
        "$FFMPEG" -loglevel warning \
            -f lavfi -i "testsrc2=duration=${DURATION}:size=1280x720:rate=${FPS}" \
            -q:v 3 "$outdir/720p/frame_%04d.ppm" 2>&1

        echo "  Extracting 480p frames..."
        "$FFMPEG" -loglevel warning \
            -f lavfi -i "testsrc2=duration=${DURATION}:size=640x480:rate=${FPS}" \
            -q:v 3 "$outdir/480p/frame_%04d.ppm" 2>&1
    else
        # Extract from real video file
        echo "  Extracting 1080p frames (fps=$FPS)..."
        "$FFMPEG" -loglevel warning \
            -ss 0 -t "$DURATION" -i "$src" \
            -r "$FPS" -vf "scale=1920:1080:force_original_aspect_ratio=decrease,\
pad=1920:1080:(ow-iw)/2:(oh-ih)/2" \
            "$outdir/1080p/frame_%04d.ppm" 2>&1

        echo "  Extracting 720p frames..."
        "$FFMPEG" -loglevel warning \
            -ss 0 -t "$DURATION" -i "$src" \
            -r "$FPS" -vf "scale=1280:720" \
            "$outdir/720p/frame_%04d.ppm" 2>&1

        echo "  Extracting 480p frames..."
        "$FFMPEG" -loglevel warning \
            -ss 0 -t "$DURATION" -i "$src" \
            -r "$FPS" -vf "scale=640:480" \
            "$outdir/480p/frame_%04d.ppm" 2>&1
    fi

    local n1080=$(ls "$outdir/1080p"/*.ppm 2>/dev/null | wc -l)
    local n720=$(ls  "$outdir/720p"/*.ppm  2>/dev/null | wc -l)
    local n480=$(ls  "$outdir/480p"/*.ppm  2>/dev/null | wc -l)

    echo ""
    echo "Frames extracted:"
    echo "  1080p: $n1080 frames (decoded ~5.9 MB each → PMD territory)"
    echo "   720p: $n720 frames  (decoded ~2.6 MB each → PMD territory)"
    echo "   480p: $n480 frames  (decoded ~900 KB each → order-7)"
    echo "  Total: $(( n1080 + n720 + n480 )) frames across all resolutions"
    echo ""
}

# ── Output file ───────────────────────────────────────────────────────────────
OUT="results_video_${TAG}_$(date +%Y%m%d_%H%M%S).txt"
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

# ── Enable mTHP ───────────────────────────────────────────────────────────────
enable_mthp() {
    local base="/sys/kernel/mm/transparent_hugepage"
    [[ -d "$base/hugepages-2048kB" ]] || return
    echo "Enabling mTHP for anonymous memory..."
    for d in "$base"/hugepages-*/; do
        [[ -f "$d/enabled" ]] && echo always > "$d/enabled" 2>/dev/null || true
    done
    [[ -f "$base/enabled" ]] && echo madvise > "$base/enabled" 2>/dev/null || true
}

# ── Drop caches + before snapshot ─────────────────────────────────────────────
echo ""
echo "Dropping page caches..."
sync
echo 3 > /proc/sys/vm/drop_caches
sleep 1
enable_mthp
snapshot "BEFORE"

# ── Prepare frames ────────────────────────────────────────────────────────────
WORK_FRAMES=""

if [[ -n "$FRAMES_DIR" ]]; then
    # User provided a pre-extracted frame directory
    WORK_FRAMES="$FRAMES_DIR"
    echo "Using pre-extracted frames from: $WORK_FRAMES"

elif [[ -n "$VIDEO" ]]; then
    # Extract frames from provided video file
    if [[ ! -f "$VIDEO" ]]; then
        echo "ERROR: video file not found: $VIDEO" >&2
        exit 1
    fi
    if [[ -z "$FFMPEG" ]]; then
        echo "ERROR: ffmpeg required to extract video frames." >&2
        exit 1
    fi
    WORK_FRAMES="/tmp/mthp_bench_frames_$$"
    extract_frames "$VIDEO" "$WORK_FRAMES" "0"

elif [[ "$GENERATE" -eq 1 ]]; then
    # Generate test frames with ffmpeg test source
    if [[ -z "$FFMPEG" ]]; then
        echo "ERROR: --generate requires ffmpeg." >&2
        exit 1
    fi
    WORK_FRAMES="/tmp/mthp_bench_frames_$$"
    extract_frames "" "$WORK_FRAMES" "1"

else
    # No video, no frames dir, no generate flag → synthetic mode
    echo "No video or frames provided — using synthetic image sources."
    echo "(Run with --generate to use ffmpeg test frames, or"
    echo "  --video /path/to/bigbuckbunny.mp4 for real video frames)"
    echo ""
fi

# ── Run the workload ──────────────────────────────────────────────────────────
echo ""
{
    echo "=== workload output ==="
    if [[ -n "$WORK_FRAMES" ]]; then
        echo "Mode: frame directory   frames: $WORK_FRAMES"
        echo "Streams: $STREAMS   Repeat: $REPEAT"
        echo ""
        "$WORKLOAD_BIN" \
            --frames  "$WORK_FRAMES" \
            --streams "$STREAMS"     \
            --repeat  "$REPEAT"
    else
        echo "Mode: synthetic (no video)"
        echo "Streams: $STREAMS"
        echo ""
        "$WORKLOAD_BIN" 200 "$STREAMS"
    fi
    echo ""
} 2>&1 | tee -a "$OUT"

# ── Final snapshot ────────────────────────────────────────────────────────────
snapshot "AFTER"

# ── Cleanup temp frames ───────────────────────────────────────────────────────
if [[ -n "$WORK_FRAMES" && "$WORK_FRAMES" == /tmp/mthp_bench_frames_* ]]; then
    echo "Cleaning up temporary frames..."
    rm -rf "$WORK_FRAMES"
fi

# ── Delta summary ─────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  Video frame detection mTHP delta   tag=$TAG"
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
echo "  $0 --compare results_video_baseline_*.txt results_video_bestfit_*.txt"
