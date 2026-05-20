// SPDX-License-Identifier: GPL-2.0
/*
 * mthp_bestfit_mod.c — Loadable module for mthp_bestfit autopilot policy
 *
 * Registers the bestfit VMA-size-aware hugepage order selection policy via
 * the function pointer hook exported by mm/huge_memory.c.  Loading the
 * module activates bestfit; unloading restores the default kernel policy.
 * No reboot required — enabling sequential comparison on one device:
 *
 *   rmmod mthp_bestfit_mod   →  baseline pass  →  collect metrics
 *   insmod mthp_bestfit_mod  →  bestfit pass   →  collect metrics
 *   compare both CSV files   →  report
 *
 * Requires kernel built with CONFIG_MTHP_BESTFIT=y (provides the hook).
 *
 * Build (on device or cross-compile):
 *   make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
 *
 * Cross-compile for RPi5 (ARM64):
 *   make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- \
 *        KDIR=/path/to/rpi-linux M=$(pwd) modules
 *
 * Load / unload:
 *   sudo insmod mthp_bestfit_mod.ko
 *   sudo rmmod  mthp_bestfit_mod
 *
 * Runtime tuning via module parameters (all in /sys/module/mthp_bestfit_mod/parameters/):
 *   max_order       — never exceed this order (default 9 = PMD)
 *   min_order       — never go below this order (default 1 = 8 KB)
 *   pressure_limit  — MemAvailable threshold % below which order is capped (default 20)
 *   pressure_cap    — order cap when under memory pressure (default 7 = 512 KB)
 *   stack_cap       — order cap for stack VMAs (default 3 = 32 KB)
 *   exec_boost      — extra orders added for executable VMAs (default 2)
 *   verbose         — log every order decision to dmesg (default 0)
 *
 * Policy pipeline (mirrors in-kernel mthp_bestfit_apply):
 *   Step 1  geometric best-fit: order = floor(log2(vma_size_pages)) - 1
 *   Step 2  clamp to [min_order, max_order]
 *   Step 3  stack cap: if VMA is stack, cap at stack_cap
 *   Step 4  exec boost: if VMA is executable, add exec_boost
 *   Step 5  pressure limit: if MemAvailable < pressure_limit%, cap at pressure_cap
 *   Step 6  return selected order; kernel uses it only if THP policy also allows it
 */

#define pr_fmt(fmt) "mthp_bestfit: " fmt

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/mm.h>
#include <linux/mm_types.h>
#include <linux/huge_mm.h>
#include <linux/mman.h>
#include <linux/sysinfo.h>

/* ── Module parameters ───────────────────────────────────────────────── */

static int max_order      = 9;
static int min_order      = 1;
static int pressure_limit = 20;   /* % of MemTotal */
static int pressure_cap   = 7;    /* order-7 = 512 KB */
static int stack_cap      = 3;    /* order-3 = 32 KB */
static int exec_boost     = 2;
static int verbose        = 0;

module_param(max_order,      int, 0644);
module_param(min_order,      int, 0644);
module_param(pressure_limit, int, 0644);
module_param(pressure_cap,   int, 0644);
module_param(stack_cap,      int, 0644);
module_param(exec_boost,     int, 0644);
module_param(verbose,        int, 0644);

MODULE_PARM_DESC(max_order,      "Maximum hugepage order (default 9 = 2 MB PMD)");
MODULE_PARM_DESC(min_order,      "Minimum hugepage order (default 1 = 8 KB)");
MODULE_PARM_DESC(pressure_limit, "MemAvailable % threshold for pressure cap (default 20)");
MODULE_PARM_DESC(pressure_cap,   "Order cap under memory pressure (default 7 = 512 KB)");
MODULE_PARM_DESC(stack_cap,      "Order cap for stack VMAs (default 3 = 32 KB)");
MODULE_PARM_DESC(exec_boost,     "Extra orders for executable VMAs (default 2)");
MODULE_PARM_DESC(verbose,        "Log every order decision to dmesg (default 0)");

/* ── Hook symbol from mm/huge_memory.c ──────────────────────────────── */

/*
 * The kernel exports this function pointer when CONFIG_MTHP_BESTFIT=y.
 * Setting it activates our policy; setting it to NULL restores default.
 */
extern int (*mthp_bestfit_hook)(struct vm_area_struct *vma, unsigned int order);

/* ── Memory pressure check ───────────────────────────────────────────── */

static bool memory_under_pressure(void)
{
	struct sysinfo si;

	si_meminfo(&si);
	if (si.totalram == 0)
		return false;

	/* si.freeram is pages; pressure_limit is a percentage */
	return (si.freeram * 100 / si.totalram) < (unsigned long)pressure_limit;
}

/* ── ilog2 for unsigned long ─────────────────────────────────────────── */

static int order_from_pages(unsigned long pages)
{
	int order = 0;

	while ((1UL << (order + 1)) <= pages)
		order++;
	return order;
}

/* ── Core bestfit policy ──────────────────────────────────────────────── */

/*
 * Called by the kernel for each (vma, order) candidate.
 * Returns:
 *    selected_order  — our bestfit order
 *
 * The kernel then checks: does caller want order == selected_order?
 * If yes → allow. If no → deny (kernel tries next lower order).
 *
 * This steers the kernel's descending order loop to stop at bestfit.
 */
static int mthp_bestfit_apply(struct vm_area_struct *vma, unsigned int order)
{
	unsigned long vma_pages = vma_pages(vma);
	int           selected;
	const char   *reason = "geometric";

	/* Step 1: geometric best-fit — order that best covers this VMA */
	if (vma_pages == 0) {
		selected = min_order;
		reason   = "zero-size";
		goto done;
	}
	selected = order_from_pages(vma_pages) - 1;

	/* Step 2: clamp */
	if (selected > max_order) selected = max_order;
	if (selected < min_order) selected = min_order;

	/* Step 3: stack cap */
	if (vma->vm_flags & VM_GROWSDOWN) {
		if (selected > stack_cap) {
			selected = stack_cap;
			reason   = "stack-cap";
		}
	}

	/* Step 4: exec boost */
	if (vma->vm_flags & VM_EXEC) {
		int boosted = selected + exec_boost;

		if (boosted > max_order)
			boosted = max_order;
		if (boosted > selected) {
			selected = boosted;
			reason   = "exec-boost";
		}
	}

	/* Step 5: memory pressure */
	if (memory_under_pressure()) {
		if (selected > pressure_cap) {
			selected = pressure_cap;
			reason   = "pressure-cap";
		}
	}

done:
	if (verbose)
		pr_info("vma=%lx-%lx pages=%lu → order=%d (%s)\n",
			vma->vm_start, vma->vm_end, vma_pages,
			selected, reason);

	return selected;
}

/* ── Module lifecycle ────────────────────────────────────────────────── */

static int __init mthp_bestfit_mod_init(void)
{
	if (!mthp_bestfit_hook) {
		/* Atomic registration — safe even if thp_vma_allowable_order
		 * is called concurrently; the pointer write is pointer-sized
		 * and therefore atomic on all supported architectures.      */
		WRITE_ONCE(mthp_bestfit_hook, mthp_bestfit_apply);
	} else {
		pr_warn("another policy already registered — not overriding\n");
		return -EBUSY;
	}

	pr_info("autopilot loaded  max_order=%d  min_order=%d  "
		"pressure_limit=%d%%  stack_cap=%d  exec_boost=%d\n",
		max_order, min_order, pressure_limit, stack_cap, exec_boost);
	return 0;
}

static void __exit mthp_bestfit_mod_exit(void)
{
	WRITE_ONCE(mthp_bestfit_hook, NULL);
	/* Ensure all in-flight calls to mthp_bestfit_apply complete before
	 * the module text is unmapped.                                    */
	synchronize_rcu();
	pr_info("autopilot unloaded — default THP policy restored\n");
}

module_init(mthp_bestfit_mod_init);
module_exit(mthp_bestfit_mod_exit);

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Suyog Buradkar <suyogburadkar@gmail.com>");
MODULE_DESCRIPTION("mthp_bestfit autopilot: VMA-size-aware hugepage order selection");
MODULE_VERSION("1.0");
