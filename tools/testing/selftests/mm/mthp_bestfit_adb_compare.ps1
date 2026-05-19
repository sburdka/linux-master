#Requires -Version 5.1
# SPDX-License-Identifier: GPL-2.0
<#
.SYNOPSIS
    Sequential baseline vs mthp_bestfit comparison via ADB (Windows host)

.DESCRIPTION
    Mirrors mthp_bestfit_tflite_bench.sh for Windows + ADB.
    Runs TWO passes on ONE physical device:

      Pass 1 — bestfit DISABLED  → collect metrics CSV
      Pass 2 — bestfit ENABLED   → collect metrics CSV
      Final   — side-by-side comparison report

    Why sequential on one device is the correct methodology:
      • Same hardware eliminates silicon variance
      • Same kernel binary — only the THP policy differs
      • mm_fragmenter normalises starting memory state in both passes
      • Delta metrics (after − before) cancel accumulated counters
      • This is exactly how Linux kernel patches are validated

    Key metrics collected every second during each pass:
      compact_stall          — each stall = ~15-40 ms latency spike
      nr_deferred_split_page — folios stuck in limbo (reduces MemAvailable)
      thp_fault_alloc/fallback — THP hit rate
      pmd_free_blocks        — order-9 free count (<10 → stall imminent)
      MemAvailable_kB        — directly visible free memory

.PARAMETER AdbExe
    Path to adb.exe (default: adb, assumes it is in PATH)

.PARAMETER Serial
    ADB device serial (-s flag). Leave empty for single connected device.

.PARAMETER FragmentMb
    MB to use for pre-fragmentation in each pass (default: 512).
    Must be identical in both passes for a fair comparison.

.PARAMETER Iterations
    Number of workload runs per pass (default: 3). Median is reported.

.PARAMETER Interval
    Monitoring poll interval in seconds (default: 1).

.PARAMETER BestfitSysfs
    Sysfs path controlling mthp_bestfit autopilot on the device.
    Default: /sys/kernel/mm/transparent_hugepage/bestfit_autopilot

.PARAMETER WorkloadBin
    Path on HOST to the ARM64 workload binary to push.
    Build: aarch64-linux-android29-clang -O2 -static \
               -o tflite_mobilenetssd_workload_arm64 \
               tflite_mobilenetssd_workload.c
    Default: .\tflite_mobilenetssd_workload_arm64

.PARAMETER FragmenterBin
    Path on HOST to the ARM64 mm_fragmenter binary.
    Build: aarch64-linux-android29-clang -O2 -static \
               -o mm_fragmenter_arm64 mm_fragmenter.c
    Default: .\mm_fragmenter_arm64

.PARAMETER OutDir
    Directory for CSV and report files (default: .\mthp_results_<timestamp>)

.PARAMETER SkipReboot
    Do not prompt for reboot between passes. Use when testing a runtime
    sysfs toggle (no reboot needed). NOT recommended for clean results.

.EXAMPLE
    # Full sequential comparison (recommended — reboot between passes)
    .\mthp_bestfit_adb_compare.ps1 -Serial 1234ABCD -FragmentMb 512

.EXAMPLE
    # Runtime toggle, no reboot (quick smoke test)
    .\mthp_bestfit_adb_compare.ps1 -SkipReboot -Iterations 1

.EXAMPLE
    # Custom sysfs path
    .\mthp_bestfit_adb_compare.ps1 `
        -BestfitSysfs /sys/kernel/mm/transparent_hugepage/mthp_bestfit
#>

[CmdletBinding()]
param(
    [string]  $AdbExe        = "adb",
    [string]  $Serial        = "",
    [int]     $FragmentMb    = 512,
    [int]     $Iterations    = 3,
    [int]     $Interval      = 1,
    [string]  $BestfitSysfs  = "/sys/kernel/mm/transparent_hugepage/bestfit_autopilot",
    [string]  $WorkloadBin   = ".\tflite_mobilenetssd_workload_arm64",
    [string]  $FragmenterBin = ".\mm_fragmenter_arm64",
    [string]  $OutDir        = "",
    [switch]  $SkipReboot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Timestamp and output directory ───────────────────────────────────────────
$RunTs = Get-Date -Format "yyyyMMdd_HHmmss"
if ($OutDir -eq "") { $OutDir = ".\mthp_results_$RunTs" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path

$BaselineCsv = Join-Path $OutDir "baseline_metrics.csv"
$BestfitCsv  = Join-Path $OutDir "bestfit_metrics.csv"
$ReportFile  = Join-Path $OutDir "comparison_report.txt"

# Device-side paths
$DeviceTmp      = "/data/local/tmp"
$DeviceWorkload = "$DeviceTmp/mthp_workload"
$DeviceFrag     = "$DeviceTmp/mm_fragmenter"

# ── Console helpers ───────────────────────────────────────────────────────────
function Write-Banner([string]$text) {
    $line = "=" * 72
    Write-Host ""
    Write-Host $line -ForegroundColor Cyan
    Write-Host "  $text" -ForegroundColor Cyan
    Write-Host $line -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step([string]$text) {
    Write-Host "  >> $text" -ForegroundColor Yellow
}

function Write-Ok([string]$text) {
    Write-Host "  OK  $text" -ForegroundColor Green
}

function Write-Warn([string]$text) {
    Write-Host "  !!  $text" -ForegroundColor Magenta
}

function Write-Metric([string]$label, $val1, $val2) {
    Write-Host ("  {0,-38} {1,-18} {2,-18}" -f $label, $val1, $val2)
}

# ── ADB wrapper ───────────────────────────────────────────────────────────────
function Invoke-Adb {
    param([string[]]$Args)
    $allArgs = if ($Serial) { @("-s", $Serial) + $Args } else { $Args }
    $result = & $AdbExe @allArgs 2>&1
    return ($result -join "`n")
}

function Invoke-AdbShell {
    param([string]$Cmd, [switch]$IgnoreError)
    $out = Invoke-Adb @("shell", $Cmd)
    if (-not $IgnoreError -and $LASTEXITCODE -ne 0) {
        Write-Warn "ADB shell returned $LASTEXITCODE for: $Cmd"
    }
    return $out
}

# ── Device prerequisite checks ────────────────────────────────────────────────
function Test-Device {
    Write-Step "Checking ADB device connectivity..."

    $devices = Invoke-Adb @("devices")
    if ($devices -notmatch "device$") {
        throw "No ADB device found. Check USB connection and 'adb devices'."
    }

    $root = Invoke-AdbShell "id" -IgnoreError
    if ($root -notmatch "uid=0") {
        Write-Warn "Device is not rooted (uid != 0)."
        Write-Warn "Need root to: set sysfs bestfit toggle, run fragmenter, read kpageflags."
        Write-Warn "Try: adb root"
        throw "Root required. Run 'adb root' first."
    }
    Write-Ok "Device connected and rooted."

    # Check sysfs path exists
    $sysfsCheck = Invoke-AdbShell "[ -e '$BestfitSysfs' ] && echo EXISTS || echo MISSING" -IgnoreError
    if ($sysfsCheck -notmatch "EXISTS") {
        Write-Warn "Sysfs path not found: $BestfitSysfs"
        Write-Warn "Check -BestfitSysfs parameter for your kernel build."
        Write-Warn "Listing /sys/kernel/mm/transparent_hugepage/:"
        Invoke-AdbShell "ls /sys/kernel/mm/transparent_hugepage/" -IgnoreError | Write-Host
        throw "Bestfit sysfs path missing."
    }
    Write-Ok "Bestfit sysfs path found: $BestfitSysfs"
}

# ── Push binaries ─────────────────────────────────────────────────────────────
function Push-Binaries {
    Write-Step "Pushing workload and fragmenter binaries to device..."

    foreach ($pair in @(
        @{ Host = $WorkloadBin;   Dev = $DeviceWorkload },
        @{ Host = $FragmenterBin; Dev = $DeviceFrag     }
    )) {
        if (-not (Test-Path $pair.Host)) {
            Write-Warn "Binary not found: $($pair.Host)"
            Write-Warn "Cross-compile with Android NDK:"
            Write-Warn "  aarch64-linux-android29-clang -O2 -static -o <output> <source>.c"
            throw "Missing binary: $($pair.Host)"
        }
        Invoke-Adb @("push", $pair.Host, $pair.Dev) | Out-Null
        Invoke-AdbShell "chmod +x $($pair.Dev)"
        Write-Ok "Pushed $($pair.Host) -> $($pair.Dev)"
    }
}

# ── Read device metrics (one snapshot) ────────────────────────────────────────
function Get-DeviceSnapshot {
    $vmstat     = Invoke-AdbShell "cat /proc/vmstat"     -IgnoreError
    $buddyinfo  = Invoke-AdbShell "cat /proc/buddyinfo"  -IgnoreError
    $meminfo    = Invoke-AdbShell "cat /proc/meminfo"    -IgnoreError

    function Parse-Vmstat([string]$key) {
        if ($vmstat -match "(?m)^$key\s+(\d+)") { return [long]$Matches[1] }
        return 0L
    }

    # Parse PMD (order-9) free blocks from buddyinfo
    # Format: "Node N, zone NAME  cnt0 cnt1 cnt2 ... cnt10"
    $pmdFree = 0
    foreach ($line in ($buddyinfo -split "`n")) {
        if ($line -match "zone") {
            $nums = [regex]::Matches($line, '\d+') | Select-Object -Skip 2 | ForEach-Object { [int]$_.Value }
            if ($nums.Count -ge 10) { $pmdFree += $nums[9] }
        }
    }

    $memAvail = 0L
    if ($meminfo -match "(?m)^MemAvailable:\s+(\d+)") { $memAvail = [long]$Matches[1] }

    return [PSCustomObject]@{
        Timestamp       = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        CompactStall    = Parse-Vmstat "compact_stall"
        DeferredSplit   = Parse-Vmstat "nr_deferred_split_page"
        ThpAlloc        = Parse-Vmstat "thp_fault_alloc"
        ThpFallback     = Parse-Vmstat "thp_fault_fallback"
        PmdFree         = $pmdFree
        MemAvailKb      = $memAvail
    }
}

# ── Background monitor job ────────────────────────────────────────────────────
# Polls device every $Interval seconds and writes rows to $CsvPath.
# Runs as a PowerShell background job so the main thread can run the workload.

function Start-Monitor {
    param([string]$CsvPath)

    Write-Step "Starting continuous monitor -> $CsvPath"

    # Write CSV header
    "timestamp_s,compact_stall,nr_deferred_split_page,thp_fault_alloc," +
    "thp_fault_fallback,pmd_free_blocks,MemAvailable_kB" | Out-File -FilePath $CsvPath -Encoding ASCII

    $stopFlag = Join-Path $OutDir ".monitor_stop"
    Remove-Item $stopFlag -ErrorAction SilentlyContinue

    $job = Start-Job -ScriptBlock {
        param($adbExe, $serial, $csvPath, $stopFlag, $interval)

        function Invoke-AdbLocal([string[]]$a) {
            $all = if ($serial) { @("-s",$serial) + $a } else { $a }
            return (& $adbExe @all 2>&1) -join "`n"
        }

        $t0 = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()

        while (-not (Test-Path $stopFlag)) {
            try {
                $ts        = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - $t0
                $vmstat    = Invoke-AdbLocal @("shell","cat /proc/vmstat")
                $buddy     = Invoke-AdbLocal @("shell","cat /proc/buddyinfo")
                $meminfo   = Invoke-AdbLocal @("shell","cat /proc/meminfo")

                function Vmstat([string]$k) {
                    if ($vmstat -match "(?m)^$k\s+(\d+)") { return [long]$Matches[1] }
                    return 0L
                }

                $pmd = 0
                foreach ($line in ($buddy -split "`n")) {
                    if ($line -match "zone") {
                        $nums = [regex]::Matches($line,'\d+') | Select-Object -Skip 2 | ForEach-Object {[int]$_.Value}
                        if ($nums.Count -ge 10) { $pmd += $nums[9] }
                    }
                }

                $ma = 0L
                if ($meminfo -match "(?m)^MemAvailable:\s+(\d+)") { $ma = [long]$Matches[1] }

                $row = "$ts,$(Vmstat 'compact_stall'),$(Vmstat 'nr_deferred_split_page')," +
                       "$(Vmstat 'thp_fault_alloc'),$(Vmstat 'thp_fault_fallback'),$pmd,$ma"
                Add-Content -Path $csvPath -Value $row
            } catch {
                # Ignore transient ADB errors during polling
            }
            Start-Sleep $interval
        }
    } -ArgumentList $AdbExe, $Serial, $CsvPath, $stopFlag, $Interval

    return @{ Job = $job; StopFlag = $stopFlag }
}

function Stop-Monitor {
    param([hashtable]$Monitor)
    Write-Step "Stopping monitor..."
    New-Item -ItemType File -Force -Path $Monitor.StopFlag | Out-Null
    Start-Sleep 2   # let the job notice the flag
    Stop-Job  $Monitor.Job -ErrorAction SilentlyContinue
    Remove-Job $Monitor.Job -ErrorAction SilentlyContinue
    Remove-Item $Monitor.StopFlag -ErrorAction SilentlyContinue
    Write-Ok "Monitor stopped."
}

# ── Bestfit policy toggle ──────────────────────────────────────────────────────
function Set-BestfitPolicy {
    param([bool]$Enable)
    $val = if ($Enable) { "1" } else { "0" }
    $label = if ($Enable) { "ENABLED" } else { "DISABLED" }
    Invoke-AdbShell "echo $val > $BestfitSysfs"
    $readback = Invoke-AdbShell "cat $BestfitSysfs" -IgnoreError
    Write-Ok "mthp_bestfit $label  (sysfs reads back: $($readback.Trim()))"
}

# ── Memory fragmenter ──────────────────────────────────────────────────────────
function Start-Fragmenter {
    Write-Step "Starting mm_fragmenter ${FragmentMb}MB hold on device..."
    # Launch in background, capture PID via `& echo $!`
    $out = Invoke-AdbShell "nohup $DeviceFrag $FragmentMb hold > $DeviceTmp/fragmenter.log 2>&1 & echo `$!" -IgnoreError
    $pid = ($out -split "`n" | Where-Object { $_ -match '^\d+$' } | Select-Object -First 1).Trim()
    Write-Ok "Fragmenter running (device PID=$pid)"

    # Wait up to 10 s for it to report fragmentation complete
    Start-Sleep 5
    $log = Invoke-AdbShell "cat $DeviceTmp/fragmenter.log" -IgnoreError
    $pmdLine = $log -split "`n" | Where-Object { $_ -match "PMD-sized.*after" }
    if ($pmdLine) { Write-Host "    $pmdLine" -ForegroundColor DarkGray }

    return $pid
}

function Stop-Fragmenter {
    param([string]$DevPid)
    if ($DevPid) {
        Invoke-AdbShell "kill $DevPid" -IgnoreError | Out-Null
    }
    Invoke-AdbShell "killall mm_fragmenter" -IgnoreError | Out-Null
    Write-Ok "Fragmenter stopped."
}

# ── Snapshot display ───────────────────────────────────────────────────────────
function Show-Snapshot {
    param([string]$Label, [PSCustomObject]$Snap)
    Write-Host ""
    Write-Host "  [$Label snapshot]" -ForegroundColor DarkGray
    Write-Host ("    compact_stall    : {0}"    -f $Snap.CompactStall)
    Write-Host ("    deferred_split   : {0}"    -f $Snap.DeferredSplit)
    Write-Host ("    thp_alloc        : {0}"    -f $Snap.ThpAlloc)
    Write-Host ("    thp_fallback     : {0}"    -f $Snap.ThpFallback)
    Write-Host ("    PMD free blocks  : {0}  {1}" -f $Snap.PmdFree,
        $(if ($Snap.PmdFree -lt 10) { "<-- STALL RISK" } else { "" }))
    Write-Host ("    MemAvailable     : {0} MB" -f ($Snap.MemAvailKb / 1024))
}

# ── Run one benchmark pass ────────────────────────────────────────────────────
function Invoke-Pass {
    param(
        [string] $Label,
        [bool]   $BestfitEnabled,
        [string] $CsvPath
    )

    Write-Banner "$Label pass  (bestfit $(if ($BestfitEnabled) {'ENABLED'} else {'DISABLED'}))"

    # 1. Set policy
    Set-BestfitPolicy -Enable $BestfitEnabled

    # 2. Pre-snapshot
    $snapBefore = Get-DeviceSnapshot
    Show-Snapshot "before" $snapBefore

    # 3. Start monitoring
    $monitor = Start-Monitor -CsvPath $CsvPath

    # 4. Fragment memory (same params every pass for fair comparison)
    $fragPid = Start-Fragmenter

    # 5. Workload iterations
    Write-Step "Running workload ($Iterations iterations)..."
    $durations = @()

    for ($i = 1; $i -le $Iterations; $i++) {
        Write-Host "    Iteration $i/$Iterations ..." -NoNewline
        $t0 = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()

        # Run workload on device; adjust args for your binary
        Invoke-AdbShell "$DeviceWorkload 8 4" -IgnoreError | Out-Null

        $elapsed = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() - $t0
        $durations += $elapsed
        Write-Host (" done  ({0:F1} s)" -f ($elapsed / 1000.0)) -ForegroundColor DarkGray
    }

    # Median duration
    $sorted = $durations | Sort-Object
    $medMs  = $sorted[[int]($sorted.Count / 2)]
    Write-Ok ("Workload median duration: {0:F1} s" -f ($medMs / 1000.0))

    # 6. Stop fragmenter + monitor
    Stop-Fragmenter -DevPid $fragPid
    Stop-Monitor    -Monitor $monitor

    # 7. Post-snapshot
    $snapAfter = Get-DeviceSnapshot
    Show-Snapshot "after" $snapAfter

    # 8. Return delta metrics
    return [PSCustomObject]@{
        Label             = $Label
        BestfitEnabled    = $BestfitEnabled
        MedianDurationMs  = $medMs
        DeltaCompact      = $snapAfter.CompactStall  - $snapBefore.CompactStall
        DeltaDeferred     = $snapAfter.DeferredSplit - $snapBefore.DeferredSplit
        DeltaThpAlloc     = $snapAfter.ThpAlloc      - $snapBefore.ThpAlloc
        DeltaThpFallback  = $snapAfter.ThpFallback   - $snapBefore.ThpFallback
        PmdFreeMin        = $null   # filled from CSV by Show-Report
        MemAvailMin       = $null
        SnapBefore        = $snapBefore
        SnapAfter         = $snapAfter
        CsvPath           = $CsvPath
    }
}

# ── Parse CSV for peak/min values ─────────────────────────────────────────────
function Get-CsvStats {
    param([string]$CsvPath)

    if (-not (Test-Path $CsvPath)) { return $null }
    $rows = Import-Csv -Path $CsvPath

    $maxRate = 0.0
    $minPmd  = [int]::MaxValue
    $minMa   = [long]::MaxValue
    $prevCs  = -1L
    $prevTs  = -1L

    foreach ($row in $rows) {
        $ts  = [long]$row.timestamp_s
        $cs  = [long]$row.compact_stall
        $pmd = [int]$row.pmd_free_blocks
        $ma  = [long]$row.MemAvailable_kB

        if ($prevTs -ge 0) {
            $dt = $ts - $prevTs
            if ($dt -gt 0) {
                $rate = ($cs - $prevCs) / $dt
                if ($rate -gt $maxRate) { $maxRate = $rate }
            }
        }
        if ($pmd -lt $minPmd) { $minPmd = $pmd }
        if ($ma  -lt $minMa)  { $minMa  = $ma  }
        $prevCs = $cs; $prevTs = $ts
    }
    if ($minPmd -eq [int]::MaxValue)  { $minPmd = 0 }
    if ($minMa  -eq [long]::MaxValue) { $minMa  = 0 }

    return [PSCustomObject]@{
        PmdFreeMin   = $minPmd
        MemAvailMin  = $minMa
        PeakStallRate = $maxRate
        Samples      = $rows.Count
    }
}

# ── ASCII progress bar ────────────────────────────────────────────────────────
function Format-Bar {
    param([double]$Frac, [int]$Width = 30)
    $f = [Math]::Max(0, [Math]::Min(1.0, $Frac))
    $filled = [int]($f * $Width + 0.5)
    return ([string]"█" * $filled) + ([string]"░" * ($Width - $filled))
}

# ── Final comparison report ───────────────────────────────────────────────────
function Show-Report {
    param(
        [PSCustomObject] $Pass1,
        [PSCustomObject] $Pass2
    )

    # Enrich with CSV-derived stats
    $s1 = Get-CsvStats -CsvPath $Pass1.CsvPath
    $s2 = Get-CsvStats -CsvPath $Pass2.CsvPath
    if ($s1) { $Pass1.PmdFreeMin = $s1.PmdFreeMin; $Pass1.MemAvailMin = $s1.MemAvailMin }
    if ($s2) { $Pass2.PmdFreeMin = $s2.PmdFreeMin; $Pass2.MemAvailMin = $s2.MemAvailMin }

    Write-Banner "COMPARISON REPORT — baseline vs mthp_bestfit"

    # ── Side-by-side table ────────────────────────────────────────────────────
    $hdr = "  {0,-38}  {1,-18}  {2,-18}" -f "Metric", $Pass1.Label, $Pass2.Label
    $sep = "  " + ("-" * 76)
    Write-Host $hdr  -ForegroundColor White
    Write-Host $sep

    function Row([string]$label, $v1, $v2, [string]$good = "lower") {
        $mark1 = ""; $mark2 = ""
        if ($v1 -is [double] -or $v1 -is [long] -or $v1 -is [int]) {
            if ($good -eq "lower") {
                if ($v1 -gt $v2) { $mark1 = " <worse>" }
                if ($v2 -gt $v1) { $mark2 = " <worse>" }
            } else {
                if ($v1 -lt $v2) { $mark1 = " <worse>" }
                if ($v2 -lt $v1) { $mark2 = " <worse>" }
            }
        }
        Write-Host ("  {0,-38}  {1,-18}  {2,-18}" -f $label, "$v1$mark1", "$v2$mark2")
    }

    Row "compact_stall delta (total)"   $Pass1.DeltaCompact     $Pass2.DeltaCompact
    Row "compact_stall peak (stalls/s)" ("{0:F2}" -f $s1.PeakStallRate) ("{0:F2}" -f $s2.PeakStallRate)
    Row "deferred_split delta"          $Pass1.DeltaDeferred    $Pass2.DeltaDeferred
    Row "thp_fault_alloc delta"         $Pass1.DeltaThpAlloc    $Pass2.DeltaThpAlloc    "higher"
    Row "thp_fault_fallback delta"      $Pass1.DeltaThpFallback $Pass2.DeltaThpFallback

    # THP hit rate
    $total1 = $Pass1.DeltaThpAlloc + $Pass1.DeltaThpFallback
    $total2 = $Pass2.DeltaThpAlloc + $Pass2.DeltaThpFallback
    $hit1 = if ($total1 -gt 0) { "{0:F1}%" -f (100.0 * $Pass1.DeltaThpAlloc / $total1) } else { "n/a" }
    $hit2 = if ($total2 -gt 0) { "{0:F1}%" -f (100.0 * $Pass2.DeltaThpAlloc / $total2) } else { "n/a" }
    Row "THP hit rate (pass window)"    $hit1 $hit2 "higher"

    Row "PMD free blocks (minimum)"     $Pass1.PmdFreeMin       $Pass2.PmdFreeMin       "higher"
    Row "MemAvailable minimum (MB)"     ([int]($Pass1.MemAvailMin/1024)) ([int]($Pass2.MemAvailMin/1024)) "higher"
    Row "Workload median duration (s)"  ("{0:F1}" -f ($Pass1.MedianDurationMs/1000)) ("{0:F1}" -f ($Pass2.MedianDurationMs/1000))
    Row "Monitor samples collected"     $s1.Samples $s2.Samples "higher"

    Write-Host $sep

    # ── PMD fragmentation bar ─────────────────────────────────────────────────
    Write-Host ""
    Write-Host "  PMD free blocks (order-9) — higher = less fragmentation:" -ForegroundColor White
    $maxPmd = [Math]::Max($Pass1.SnapBefore.PmdFree, $Pass2.SnapBefore.PmdFree)
    $maxPmd = [Math]::Max($maxPmd, 1)
    foreach ($p in @($Pass1, $Pass2)) {
        $pct = $p.PmdFreeMin / $maxPmd
        Write-Host ("    {0,-12} min={1,-4}  {2}" -f $p.Label, $p.PmdFreeMin, (Format-Bar $pct 40))
    }

    # ── MemAvailable bar ──────────────────────────────────────────────────────
    Write-Host ""
    Write-Host "  MemAvailable minimum — higher = more free memory:" -ForegroundColor White
    $maxMa = [Math]::Max($Pass1.MemAvailMin, $Pass2.MemAvailMin)
    $maxMa = [Math]::Max($maxMa, 1)
    foreach ($p in @($Pass1, $Pass2)) {
        $pct = $p.MemAvailMin / $maxMa
        $mb  = [int]($p.MemAvailMin / 1024)
        Write-Host ("    {0,-12} {1,5} MB  {2}" -f $p.Label, $mb, (Format-Bar $pct 40))
    }

    # ── Verdict ───────────────────────────────────────────────────────────────
    Write-Host ""
    Write-Host "  Verdict:" -ForegroundColor White
    if ($Pass2.DeltaCompact -lt $Pass1.DeltaCompact) {
        Write-Host "  + compact_stall reduced by $(($Pass1.DeltaCompact - $Pass2.DeltaCompact)) stalls" -ForegroundColor Green
        Write-Host "    Each stall avoided = ~15-40 ms latency spike eliminated on Snapdragon" -ForegroundColor DarkGreen
    } elseif ($Pass2.DeltaCompact -eq $Pass1.DeltaCompact -and $Pass1.DeltaCompact -eq 0) {
        Write-Host "  = No compact_stall in either pass." -ForegroundColor Yellow
        Write-Host "    Increase -FragmentMb or test on device with less free RAM (<4GB)." -ForegroundColor Yellow
    } else {
        Write-Host "  ? Unexpected: bestfit has more stalls. Check fragmentation was equal." -ForegroundColor Red
    }

    if ($Pass2.DeltaDeferred -lt $Pass1.DeltaDeferred) {
        $saved = $Pass1.DeltaDeferred - $Pass2.DeltaDeferred
        Write-Host ("  + deferred_split reduced by {0} folios -> {1:F0} MB returned faster to buddy" `
                    -f $saved, ($saved * 4.0 / 1024)) -ForegroundColor Green
    }

    if ($Pass2.MemAvailMin -gt $Pass1.MemAvailMin) {
        $gain = [int](($Pass2.MemAvailMin - $Pass1.MemAvailMin) / 1024)
        Write-Host ("  + MemAvailable higher by {0} MB with bestfit" -f $gain) -ForegroundColor Green
    }

    # ── Save text report ──────────────────────────────────────────────────────
    $reportLines = @(
        "mthp_bestfit sequential comparison report",
        "Generated: $(Get-Date)",
        "",
        "Pass 1 — $($Pass1.Label)  CSV: $($Pass1.CsvPath)",
        "Pass 2 — $($Pass2.Label)  CSV: $($Pass2.CsvPath)",
        "",
        ("  {0,-38}  {1,-18}  {2,-18}" -f "Metric", $Pass1.Label, $Pass2.Label),
        ("  " + "-" * 76),
        ("  {0,-38}  {1,-18}  {2,-18}" -f "compact_stall delta",     $Pass1.DeltaCompact,  $Pass2.DeltaCompact),
        ("  {0,-38}  {1,-18}  {2,-18}" -f "deferred_split delta",     $Pass1.DeltaDeferred, $Pass2.DeltaDeferred),
        ("  {0,-38}  {1,-18}  {2,-18}" -f "THP hit rate",             $hit1,                $hit2),
        ("  {0,-38}  {1,-18}  {2,-18}" -f "PMD free blocks (min)",    $Pass1.PmdFreeMin,    $Pass2.PmdFreeMin),
        ("  {0,-38}  {1,-18}  {2,-18}" -f "MemAvailable min (MB)",    [int]($Pass1.MemAvailMin/1024), [int]($Pass2.MemAvailMin/1024)),
        ("  {0,-38}  {1,-18}  {2,-18}" -f "Workload median (s)",      ("{0:F1}" -f ($Pass1.MedianDurationMs/1000)), ("{0:F1}" -f ($Pass2.MedianDurationMs/1000)))
    )
    $reportLines | Out-File -FilePath $ReportFile -Encoding UTF8
    Write-Host ""
    Write-Ok "Text report saved: $ReportFile"
    Write-Ok "Baseline CSV:      $BaselineCsv"
    Write-Ok "Bestfit  CSV:      $BestfitCsv"
}

# ── Entry point ───────────────────────────────────────────────────────────────
Write-Banner "mthp_bestfit ADB sequential compare  ($RunTs)"
Write-Host "  Output directory : $OutDir"
Write-Host "  FragmentMb       : $FragmentMb"
Write-Host "  Iterations       : $Iterations"
Write-Host "  Bestfit sysfs    : $BestfitSysfs"
Write-Host "  Interval         : ${Interval}s"

# Prerequisites
Test-Device
Push-Binaries

# ── PASS 1 — BASELINE ─────────────────────────────────────────────────────────
$resultBaseline = Invoke-Pass -Label "baseline" -BestfitEnabled $false -CsvPath $BaselineCsv

# ── Reboot prompt between passes ──────────────────────────────────────────────
if (-not $SkipReboot) {
    Write-Host ""
    Write-Host "  ╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "  ║  REBOOT THE DEVICE NOW for a clean Pass 2 start state.  ║" -ForegroundColor Cyan
    Write-Host "  ║                                                          ║" -ForegroundColor Cyan
    Write-Host "  ║  Why reboot:                                             ║" -ForegroundColor Cyan
    Write-Host "  ║    • Clears accumulated counters to same origin          ║" -ForegroundColor Cyan
    Write-Host "  ║    • Resets memory layout to same initial state          ║" -ForegroundColor Cyan
    Write-Host "  ║    • Eliminates thermal/NUMA state from Pass 1           ║" -ForegroundColor Cyan
    Write-Host "  ║                                                          ║" -ForegroundColor Cyan
    Write-Host "  ║  Steps: adb reboot  →  wait for boot  →  adb root       ║" -ForegroundColor Cyan
    Write-Host "  ╚══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Press ENTER when device is back up and 'adb devices' shows it." -ForegroundColor Yellow
    Read-Host | Out-Null

    # Wait for device to be ready
    Write-Step "Waiting for ADB device..."
    $waited = 0
    while ($waited -lt 120) {
        $d = Invoke-Adb @("devices") 2>&1
        if ($d -match "device$") { break }
        Start-Sleep 3; $waited += 3
    }
    Invoke-Adb @("root") | Out-Null
    Start-Sleep 3
    Write-Ok "Device ready."
}

# ── PASS 2 — BESTFIT ──────────────────────────────────────────────────────────
$resultBestfit = Invoke-Pass -Label "bestfit" -BestfitEnabled $true -CsvPath $BestfitCsv

# ── FINAL REPORT ──────────────────────────────────────────────────────────────
Show-Report -Pass1 $resultBaseline -Pass2 $resultBestfit
