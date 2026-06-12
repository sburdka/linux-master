# mthp_bestfit — Claude Code Workspace

## Project

Linux kernel mm/ subsystem development: `mthp_bestfit` autopilot policy
for multi-size Transparent Huge Pages (mTHP).

**Branch**: `claude/mthp-bestfit-kernel-module-Sfcrh`
**Author**: Suyog Buradkar <suyogburadkar@gmail.com>
**Commit style**: Always use `Suyog Buradkar <suyogburadkar@gmail.com>` as author.
Never include Claude URLs or Claude mentions in commit messages.

---

## RPi5 Autonomous Testing

When the user asks to "check results", "run tests on rpi5", or similar,
execute the full autonomous test without asking for confirmation:

### SSH access
```bash
# Default — override with env vars if needed
RPI5_HOST="${RPI5_HOST:-rpi5.local}"
RPI5_USER="${RPI5_USER:-pi}"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes"
```

### Step 1 — Verify connectivity
```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 pi@rpi5.local "uname -r && free -h"
```

### Step 2 — Run the full sequential comparison
```bash
cd tools/testing/selftests/mm
bash mthp_rpi5_autotest.sh \
    --host "${RPI5_HOST:-rpi5.local}" \
    --user "${RPI5_USER:-pi}" \
    --fragment-mb 256 \
    --iterations 3
```

### Step 3 — Parse and report results
Lines prefixed with `[RESULT]` and `[VERDICT]` contain structured data.
Parse them and present a clear summary with:
- compact_stall reduction (key metric — 0 on bestfit is the goal)
- deferred_split reduction
- MemAvailable gain in MB
- THP hit rate improvement
- Workload latency change

### Step 4 — If compact_stall = 0 in both passes
Fragmentation was insufficient. Automatically retry with higher --fragment-mb:
```bash
bash mthp_rpi5_autotest.sh --fragment-mb 512 --iterations 3
```
RPi5 has 4-8 GB RAM so 512+ MB may be needed to drain PMD blocks below 10.

---

## Device-side paths

```
/tmp/mthp_autotest/                 — working directory on RPi5
  mm_fragmenter                     — fragmentation tool (ARM64 binary)
  mthp_workload                     — TFLite workload simulator (ARM64 binary)
  mthp_bestfit_mod.ko               — bestfit kernel module
  baseline_metrics.csv              — Pass 1 continuous metrics
  bestfit_metrics.csv               — Pass 2 continuous metrics
```

## Building binaries for RPi5

Cross-compile from x86 host (requires `gcc-aarch64-linux-gnu`):
```bash
CC=aarch64-linux-gnu-gcc
$CC -O2 -static -o mm_fragmenter_arm64    tools/testing/selftests/mm/mm_fragmenter.c
$CC -O2 -static -o mthp_workload_arm64    tools/testing/selftests/mm/tflite_mobilenetssd_workload.c
```

Build module on RPi5 (requires kernel headers: `sudo apt install raspberrypi-kernel-headers`):
```bash
cd tools/testing/selftests/mm/mthp_bestfit_mod
make
sudo insmod mthp_bestfit_mod.ko
```

---

## Key files

| File | Purpose |
|------|---------|
| `mm/huge_memory.c` | Kernel hook — `mthp_bestfit_hook` function pointer |
| `mm/mthp_bestfit.h` | Policy API header |
| `tools/testing/selftests/mm/mthp_bestfit_mod/mthp_bestfit_mod.c` | Loadable module |
| `tools/testing/selftests/mm/mthp_rpi5_autotest.sh` | Autonomous test runner |
| `tools/testing/selftests/mm/mthp_monitor.sh` | Continuous vmstat poller |
| `tools/testing/selftests/mm/mm_fragmenter.c` | Buddy fragmentation tool |
| `tools/testing/selftests/mm/mthp_vma_inspector.c` | Per-PID folio order viewer |
| `tools/testing/selftests/mm/mthp_vma_compare.sh` | Side-by-side VMA comparison |
| `tools/testing/selftests/mm/mthp_bestfit_adb_compare.ps1` | Windows ADB test runner |

---

## Key metrics to interpret

| Metric | Baseline expected | Bestfit expected | Notes |
|--------|-------------------|------------------|-------|
| compact_stall delta | > 0 | 0 | Each stall = 15-40ms latency spike |
| compact_stall rate | > 0.1/s | 0/s | The key pass/fail metric |
| deferred_split delta | high | low | Lower = faster folio recycling |
| THP hit rate | 60-80% | 95-99% | Higher = more VMAs get huge pages |
| MemAvailable | lower | higher | Direct free-memory benefit |
| PMD free min | < 10 | < 10 | Both passes should see pressure |

Comparison is only valid when both passes see PMD free blocks < 10 during the
run (confirmed fragmentation). If both passes show PMD > 30, increase fragment-mb.

---

## Commit instructions

- Always commit as: `Suyog Buradkar <suyogburadkar@gmail.com>`
- Never add Claude URLs to commit messages
- Follow Linux kernel commit style (subject: subsystem: description)
- Push to: `origin claude/mthp-bestfit-kernel-module-Sfcrh`
