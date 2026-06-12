# SPDX-License-Identifier: GPL-2.0
# Makefile — mthp_bestfit out-of-tree loadable kernel module
#
# ── Prerequisites ─────────────────────────────────────────────────────────
#
#   The running / target kernel MUST be built with the mthp_bestfit hook
#   patch applied to:
#     mm/huge_memory.c        (defines + exports mthp_order_filter_fn)
#     include/linux/huge_mm.h (declares extern mthp_order_filter_fn)
#     mm/khugepaged.c         (khugepaged scan guided by filter)
#
#   Verify the hook is present in a running kernel:
#     grep mthp_order_filter_fn /proc/kallsyms
#
# ── Native build (host arch, running kernel) ───────────────────────────────
#
#   make
#   sudo insmod mthp_bestfit.ko
#   sudo rmmod  mthp_bestfit
#
# ── Cross-compile for ARM64 (e.g. Vienna / Shikra board) ──────────────────
#
#   make ARCH=arm64 \
#        CROSS_COMPILE=aarch64-linux-gnu- \
#        KDIR=/path/to/arm64/kernel/build
#
# ── Cross-compile for ARM32 ───────────────────────────────────────────────
#
#   make ARCH=arm \
#        CROSS_COMPILE=arm-linux-gnueabihf- \
#        KDIR=/path/to/arm32/kernel/build
#
# ── Install to running system ─────────────────────────────────────────────
#
#   make install          # copies .ko and runs depmod -A
#   modprobe mthp_bestfit # load via modprobe (after install)
#
# ── Variables ─────────────────────────────────────────────────────────────

# Kernel build directory.  Defaults to the running kernel's build tree.
KDIR          ?= /lib/modules/$(shell uname -r)/build

# Architecture — leave empty for native build.
ARCH          ?=

# Cross-compiler prefix — leave empty for native build.
CROSS_COMPILE ?=

# Module source
obj-m         := mthp_bestfit.o

# Extra CFLAGS visible in the module (optional: used for out-of-tree marker)
ccflags-y     += -DMTHP_BESTFIT_OUT_OF_TREE=1

# ── Rules ─────────────────────────────────────────────────────────────────

.PHONY: all clean install check load unload help

all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) \
		$(if $(ARCH),ARCH=$(ARCH)) \
		$(if $(CROSS_COMPILE),CROSS_COMPILE=$(CROSS_COMPILE)) \
		modules

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean

install: all
	$(MAKE) -C $(KDIR) M=$(CURDIR) \
		$(if $(ARCH),ARCH=$(ARCH)) \
		$(if $(CROSS_COMPILE),CROSS_COMPILE=$(CROSS_COMPILE)) \
		modules_install
	depmod -A
	@echo "Installed: use 'modprobe mthp_bestfit' to load."

# Verify the hook symbol is present in the running kernel.
# Fails with a clear message if the kernel was not patched.
check:
	@echo "Checking for mthp_order_filter_fn in running kernel..."
	@if grep -q "mthp_order_filter_fn" /proc/kallsyms 2>/dev/null; then \
		echo "  OK  — hook symbol found, module will load correctly."; \
	else \
		echo "  FAIL — mthp_order_filter_fn not found in /proc/kallsyms."; \
		echo "         Rebuild the kernel with the mthp_bestfit hook patch."; \
		exit 1; \
	fi
	@echo "Checking for TRANSPARENT_HUGEPAGE support..."
	@if [ -d /sys/kernel/mm/transparent_hugepage ]; then \
		echo "  OK  — THP is enabled."; \
	else \
		echo "  FAIL — THP not available (CONFIG_TRANSPARENT_HUGEPAGE not set?)."; \
		exit 1; \
	fi

load: all check
	insmod mthp_bestfit.ko
	@echo "Loaded. Stats: cat /sys/kernel/debug/mthp_bestfit/stats"
	@echo "Tune:   ls /proc/sys/vm/mthp_bestfit_*"

unload:
	rmmod mthp_bestfit || true
	@echo "Unloaded."

help:
	@echo ""
	@echo "mthp_bestfit out-of-tree module"
	@echo ""
	@echo "Targets:"
	@echo "  all      — build the .ko"
	@echo "  clean    — remove build artefacts"
	@echo "  check    — verify the patched kernel has the hook symbol"
	@echo "  load     — build + check + insmod"
	@echo "  unload   — rmmod"
	@echo "  install  — install to /lib/modules/.../extra/ + depmod"
	@echo ""
	@echo "Variables:"
	@echo "  KDIR=$(KDIR)"
	@echo "  ARCH=$(ARCH)  (empty = native)"
	@echo "  CROSS_COMPILE=$(CROSS_COMPILE)  (empty = native)"
	@echo ""
	@echo "After loading:"
	@echo "  cat /sys/kernel/debug/mthp_bestfit/stats"
	@echo "  cat /proc/sys/vm/mthp_bestfit_*"
	@echo "  echo 4 > /proc/sys/vm/mthp_bestfit_pressure_min_free"
	@echo ""
