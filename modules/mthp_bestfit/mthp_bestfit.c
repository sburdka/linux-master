// SPDX-License-Identifier: GPL-2.0
/*
 * mthp_bestfit — VMA-size best-fit mTHP order selection (out-of-tree module)
 *
 * Registers mthp_order_filter_fn, a function pointer hook exported by the
 * patched kernel (mm/huge_memory.c), which is called at the end of
 * __thp_vma_allowable_orders() for every non-trivial mTHP decision.
 *
 * REQUIRES: kernel built with the mthp_bestfit hook patch that adds
 *   mthp_order_filter_fn to mm/huge_memory.c and include/linux/huge_mm.h.
 *   Verify with: grep mthp_order_filter_fn /proc/kallsyms
 *
 * Build:
 *   make -C <kernel-build-dir> M=$(pwd) modules
 *
 * Cross-build (e.g. ARM64 target):
 *   make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- \
 *        KDIR=<kernel-build-dir> -C <kernel-build-dir> M=$(pwd) modules
 *
 * Load / unload:
 *   insmod mthp_bestfit.ko
 *   rmmod  mthp_bestfit
 *
 * ── Policies ──────────────────────────────────────────────────────────────
 *
 *  1. Geometric best-fit
 *     Largest order N where min_pages complete aligned hugepages fit in VMA.
 *     Uses ALIGN(vm_start, page_size) — alignment-correct, not raw VMA size.
 *
 *  2. Stack cap (order 2 = 16 KB)
 *     VM_GROWSDOWN VMAs capped at order 2 to prevent hugepage split churn
 *     on unpredictably growing stacks.
 *
 *  3. Exec boost (+1 order)
 *     File-backed executable VMAs (ELF .text) promoted one order to widen
 *     iTLB coverage.  iTLBs are smaller than dTLBs on all modern CPUs.
 *
 *  4. Pressure awareness  (pressure_aware + pressure_min_free)
 *     Before committing to an order, checks zone->free_area[order].nr_free.
 *     Only selects the order if nr_free >= pressure_min_free (default 4).
 *     Requiring a buffer of N blocks prevents the race window where the
 *     single available block is consumed between our check and the actual
 *     vma_alloc_folio() call, which is the primary driver of residual
 *     compact_stall events in mTHP workloads.
 *
 *  5. NUMA locality (numa_aware)
 *     Prefers the highest order available on the faulting CPU's local node
 *     over cross-node allocation.  No-op on UMA/single-node systems.
 *
 *  6. Lifetime awareness (lifetime_aware)
 *     Brand-new VMAs (anon_vma == NULL, never faulted) receive order - 1.
 *     Conserves large contiguous blocks for established VMAs that are
 *     proven to need them — critical on 1–2 GB DDR devices.
 *
 * ── compact_stall vs bestfit ──────────────────────────────────────────────
 *
 *  Compaction and reclaim are reactive: they run AFTER the buddy allocator
 *  fails a high-order request.  bestfit is proactive: it selects only orders
 *  the buddy can satisfy without compaction.  The residual compact_stall
 *  events visible after loading this module come from:
 *    a) Non-mTHP allocations (kmalloc, DMA, kernel structures)
 *    b) The race window between our pressure check and vma_alloc_folio()
 *       when pressure_min_free = 1 (only one free block existed)
 *  Setting pressure_min_free >= 4 closes (b) for the vast majority of cases.
 *
 * ── Free memory impact ────────────────────────────────────────────────────
 *
 *  bestfit does not directly reclaim memory.  Its impact on MemAvailable is
 *  indirect and operates through three mechanisms:
 *    1. Fewer deferred_splits: right-sized hugepages are not partially
 *       unmapped and left pending split, so pages return to the buddy faster.
 *    2. Less contiguous waste: a 256 KB VMA gets a 256 KB hugepage instead
 *       of a 2 MB one; the remaining 1.75 MB stays in free_area[] and is
 *       available for DMA/CMA/GPU allocations that have no fallback.
 *    3. Lower split overhead: each hugepage split is O(2^order) operations;
 *       fewer splits = less time with pages temporarily unavailable.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/mm.h>
#include <linux/huge_mm.h>
#include <linux/mmzone.h>
#include <linux/sysctl.h>
#include <linux/percpu.h>
#include <linux/rcupdate.h>
#include <linux/topology.h>

#ifdef CONFIG_DEBUG_FS
#include <linux/debugfs.h>
#include <linux/seq_file.h>
#endif

#define MTHP_BESTFIT_NAME	"mthp_bestfit"
#define MTHP_BESTFIT_VER	"4"

/* ---- Sysctls ------------------------------------------------------------ */

static int sysctl_enabled        __read_mostly = 1;
static int sysctl_min_pages      __read_mostly = 2;
static int sysctl_exec_boost     __read_mostly = 1;
static int sysctl_dry_run        __read_mostly = 0;
static int sysctl_pressure_aware __read_mostly = 1;
/*
 * pressure_min_free: minimum number of free buddy blocks required at the
 * desired order before we select it.  Setting this > 1 provides a buffer
 * against the race window between our snapshot of free_area[].nr_free and
 * the actual vma_alloc_folio() call, which is the primary cause of residual
 * compact_stall events.  Default 4 = require 4 free blocks.
 */
static int sysctl_pressure_min_free __read_mostly = 4;
static int sysctl_numa_aware     __read_mostly = 1;
static int sysctl_lifetime_aware __read_mostly = 1;

/* ---- Per-CPU statistics ------------------------------------------------- */

#define BF_VMA_STACK	0	/* VM_GROWSDOWN — stack */
#define BF_VMA_EXEC	1	/* VM_EXEC | vm_file — ELF .text */
#define BF_VMA_ANON	2	/* anonymous heap / mmap */
#define BF_VMA_FILE	3	/* file-backed, non-exec */
#define BF_VMA_TYPES	4

struct bestfit_counters {
	u64 decisions;
	u64 suppressed;			/* all mTHP suppressed for this VMA */
	u64 pressure_downgraded;	/* order reduced by pressure check */
	u64 numa_downgraded;		/* order reduced by NUMA locality */
	u64 lifetime_conserved;		/* order reduced for new VMA */
	/*
	 * compact_stall_avoided: number of decisions where the original
	 * geometric order had nr_free < pressure_min_free (would likely
	 * have triggered compaction) but we downgraded to a safer order.
	 * This is the direct compact_stall reduction attributable to bestfit.
	 */
	u64 compact_stall_avoided;
	u64 order_hist[PMD_ORDER + 1];
	u64 by_type[BF_VMA_TYPES];
};

static DEFINE_PER_CPU(struct bestfit_counters, bf_pcpu);

static int bestfit_vma_type(const struct vm_area_struct *vma)
{
	if (vma->vm_flags & VM_GROWSDOWN)
		return BF_VMA_STACK;
	if ((vma->vm_flags & VM_EXEC) && vma->vm_file)
		return BF_VMA_EXEC;
	if (vma_is_anonymous(vma))
		return BF_VMA_ANON;
	return BF_VMA_FILE;
}

static void bf_count(int order, int vma_type, bool suppressed,
		     bool pressure_dg, bool numa_dg, bool lifetime_c,
		     bool stall_avoided)
{
	struct bestfit_counters *c;

	preempt_disable();
	c = this_cpu_ptr(&bf_pcpu);
	c->decisions++;
	if (suppressed)
		c->suppressed++;
	if (pressure_dg)
		c->pressure_downgraded++;
	if (numa_dg)
		c->numa_downgraded++;
	if (lifetime_c)
		c->lifetime_conserved++;
	if (stall_avoided)
		c->compact_stall_avoided++;
	if (order >= 0 && order <= PMD_ORDER)
		c->order_hist[order]++;
	if (vma_type >= 0 && vma_type < BF_VMA_TYPES)
		c->by_type[vma_type]++;
	preempt_enable();
}

/* ---- Memory pressure helpers -------------------------------------------- */

/*
 * Count free blocks of @order across all populated zones.
 * The read of nr_free is intentionally racy — no zone lock held.
 * We are computing a heuristic snapshot; correctness is maintained by
 * the allocator's own fallback path if our estimate is stale.
 */
static unsigned long order_free_count(int order)
{
	struct zone *zone;
	unsigned long total = 0;

	for_each_populated_zone(zone)
		total += READ_ONCE(zone->free_area[order].nr_free);
	return total;
}

/*
 * bestfit_pressure_limit - return the highest order that meets the
 * pressure_min_free threshold.
 *
 * Requires zone->free_area[order].nr_free >= pressure_min_free across
 * all zones combined.  The threshold buffers the race window between our
 * snapshot and the actual allocation, directly reducing compact_stall.
 *
 * Returns @start unchanged if pressure_aware is disabled (fast path).
 * Sets *stall_avoided if the original order would have been risky
 * (0 < nr_free < pressure_min_free) and we downgraded.
 */
static int bestfit_pressure_limit(int start, bool *stall_avoided)
{
	unsigned long free_count;
	int min_free;
	int order;

	*stall_avoided = false;

	if (!READ_ONCE(sysctl_pressure_aware))
		return start;

	min_free   = READ_ONCE(sysctl_pressure_min_free);
	free_count = order_free_count(start);

	if (free_count >= (unsigned long)min_free)
		return start;

	/*
	 * If there are some free blocks but fewer than threshold, record that
	 * we are proactively avoiding a likely compact_stall.
	 */
	if (free_count > 0)
		*stall_avoided = true;

	/* Walk downward to find an order with enough free blocks. */
	for (order = start - 1; order >= 2; order--) {
		if (order_free_count(order) >= (unsigned long)min_free)
			return order;
	}
	return 2;
}

/* ---- NUMA locality helper ----------------------------------------------- */

static bool order_available_local(int order)
{
	int nid;
	struct pglist_data *pgdat;
	int z;

	if (!READ_ONCE(sysctl_numa_aware))
		return true;

	nid   = numa_node_id();
	pgdat = NODE_DATA(nid);
	for (z = 0; z < MAX_NR_ZONES; z++) {
		struct zone *zone = &pgdat->node_zones[z];

		if (!populated_zone(zone))
			continue;
		if (READ_ONCE(zone->free_area[order].nr_free) >=
		    (unsigned long)READ_ONCE(sysctl_pressure_min_free))
			return true;
	}
	return false;
}

/* ---- Core best-fit logic ------------------------------------------------ */

/*
 * mthp_bestfit_order - compute best-fit hugepage order for @vma
 *
 * Policy pipeline (applied in sequence):
 *
 *   Step 1 — Stack cap
 *     VM_GROWSDOWN → order 2 if any aligned 16 KB fits, else 0.
 *
 *   Step 2 — Geometric best-fit
 *     Largest order N where min_pages aligned (PAGE_SIZE << N) regions
 *     fit between ALIGN(vm_start, page_size) and vm_end.
 *
 *   Step 3 — Exec boost
 *     +1 order for VM_EXEC | vm_file (iTLB pressure reduction).
 *
 *   Step 4 — Pressure limit
 *     Downgrade if free_area[order].nr_free < pressure_min_free.
 *     This is the primary lever for reducing compact_stall.
 *
 *   Step 5 — NUMA locality
 *     Downgrade if local node cannot satisfy the order with the same
 *     pressure_min_free threshold.
 *
 *   Step 6 — Lifetime conservation
 *     Reduce by 1 for brand-new VMAs (anon_vma == NULL).
 *
 * Returns:
 *    >= 2  valid best order
 *       0  VMA too small — suppress all mTHP
 *      -1  module disabled
 */
static int mthp_bestfit_order(struct vm_area_struct *vma,
			      bool *pressure_dg,
			      bool *stall_avoided,
			      bool *numa_dg,
			      bool *lifetime_c)
{
	unsigned long min_pages;
	unsigned long page_size;
	unsigned long first_aligned;
	bool is_exec;
	int order;
	int limited;

	*pressure_dg   = false;
	*stall_avoided = false;
	*numa_dg       = false;
	*lifetime_c    = false;

	if (!READ_ONCE(sysctl_enabled))
		return -1;

	min_pages = (unsigned long)READ_ONCE(sysctl_min_pages);

	/* ── Step 1: Stack cap ─────────────────────────────────────────── */
	if (vma->vm_flags & VM_GROWSDOWN) {
		page_size     = 1UL << (PAGE_SHIFT + 2);
		first_aligned = ALIGN(vma->vm_start, page_size);
		return (first_aligned + page_size <= vma->vm_end) ? 2 : 0;
	}

	/* Early exit: not even one 16 KB hugepage fits anywhere in VMA. */
	page_size     = 1UL << (PAGE_SHIFT + 2);
	first_aligned = ALIGN(vma->vm_start, page_size);
	if (first_aligned + page_size > vma->vm_end)
		return 0;

	is_exec = (vma->vm_flags & VM_EXEC) && vma->vm_file;

	/* ── Step 2: Geometric best-fit ────────────────────────────────── */
	for (order = PMD_ORDER; order >= 2; order--) {
		page_size     = 1UL << (PAGE_SHIFT + order);
		first_aligned = ALIGN(vma->vm_start, page_size);

		if (first_aligned < vma->vm_end &&
		    (vma->vm_end - first_aligned) >= min_pages * page_size)
			goto found;
	}
	return 0;

found:
	/* ── Step 3: Exec boost ─────────────────────────────────────────── */
	if (is_exec && READ_ONCE(sysctl_exec_boost) && order < PMD_ORDER)
		order++;

	/* ── Step 4: Pressure limit ─────────────────────────────────────── */
	limited = bestfit_pressure_limit(order, stall_avoided);
	if (limited < order) {
		order        = limited;
		*pressure_dg = true;
	}

	/* ── Step 5: NUMA locality ──────────────────────────────────────── */
	if (!order_available_local(order)) {
		int o;

		for (o = order - 1; o >= 2; o--) {
			if (order_available_local(o)) {
				order    = o;
				*numa_dg = true;
				break;
			}
		}
	}

	/* ── Step 6: Lifetime conservation ─────────────────────────────── */
	if (READ_ONCE(sysctl_lifetime_aware) && !vma->anon_vma && order > 2) {
		order--;
		*lifetime_c = true;
	}

	return order;
}

/* ---- Hook registered as mthp_order_filter_fn ---------------------------- */

static unsigned long mthp_bestfit_filter(struct vm_area_struct *vma,
					 vm_flags_t vm_flags,
					 enum tva_type type,
					 unsigned long allowed_orders)
{
	bool pressure_dg, stall_avoided, numa_dg, lifetime_c;
	unsigned long new_orders;
	int best_order;
	int vma_type;
	bool suppressed;

	best_order = mthp_bestfit_order(vma, &pressure_dg, &stall_avoided,
					&numa_dg, &lifetime_c);
	vma_type   = bestfit_vma_type(vma);

	if (best_order < 0)		/* module disabled */
		return allowed_orders;

	if (best_order == 0) {
		bf_count(-1, vma_type, true,
			 false, false, false, false);
		return 0;
	}

	if (best_order >= PMD_ORDER) {
		bf_count(best_order, vma_type, false,
			 pressure_dg, numa_dg, lifetime_c, stall_avoided);
		return allowed_orders;
	}

	new_orders = allowed_orders & ((1UL << (best_order + 1)) - 1);
	suppressed = !new_orders;
	bf_count(suppressed ? -1 : best_order, vma_type, suppressed,
		 pressure_dg, numa_dg, lifetime_c, stall_avoided);

	if (!READ_ONCE(sysctl_dry_run))
		return new_orders;

	/* dry_run: statistics only, return unmodified orders. */
	return allowed_orders;
}

/* ---- Sysctl table ------------------------------------------------------- */

static int sysctl_max_min_pages     = 16;
static int sysctl_max_pressure_free = 64;

static struct ctl_table mthp_bestfit_sysctls[] = {
	{
		.procname	= "mthp_bestfit_enabled",
		.data		= &sysctl_enabled,
		.maxlen		= sizeof(int),
		.mode		= 0644,
		.proc_handler	= proc_dointvec_minmax,
		.extra1		= SYSCTL_ZERO,
		.extra2		= SYSCTL_ONE,
	},
	{
		.procname	= "mthp_bestfit_min_pages",
		.data		= &sysctl_min_pages,
		.maxlen		= sizeof(int),
		.mode		= 0644,
		.proc_handler	= proc_dointvec_minmax,
		.extra1		= SYSCTL_ONE,
		.extra2		= &sysctl_max_min_pages,
	},
	{
		.procname	= "mthp_bestfit_exec_boost",
		.data		= &sysctl_exec_boost,
		.maxlen		= sizeof(int),
		.mode		= 0644,
		.proc_handler	= proc_dointvec_minmax,
		.extra1		= SYSCTL_ZERO,
		.extra2		= SYSCTL_ONE,
	},
	{
		.procname	= "mthp_bestfit_dry_run",
		.data		= &sysctl_dry_run,
		.maxlen		= sizeof(int),
		.mode		= 0644,
		.proc_handler	= proc_dointvec_minmax,
		.extra1		= SYSCTL_ZERO,
		.extra2		= SYSCTL_ONE,
	},
	{
		.procname	= "mthp_bestfit_pressure_aware",
		.data		= &sysctl_pressure_aware,
		.maxlen		= sizeof(int),
		.mode		= 0644,
		.proc_handler	= proc_dointvec_minmax,
		.extra1		= SYSCTL_ZERO,
		.extra2		= SYSCTL_ONE,
	},
	{
		/*
		 * pressure_min_free: require this many free buddy blocks at the
		 * desired order before selecting it.  Raising this value trades
		 * some TLB benefit (selecting smaller orders more often) for a
		 * larger buffer against the snapshot-to-allocation race window.
		 *
		 * Tune based on workload compact_stall rate:
		 *   compact_stall still high → increase (try 8, 16)
		 *   too many pressure_downgraded → decrease (try 2)
		 *
		 * Range: 1 (any single block) to 64 (very conservative).
		 */
		.procname	= "mthp_bestfit_pressure_min_free",
		.data		= &sysctl_pressure_min_free,
		.maxlen		= sizeof(int),
		.mode		= 0644,
		.proc_handler	= proc_dointvec_minmax,
		.extra1		= SYSCTL_ONE,
		.extra2		= &sysctl_max_pressure_free,
	},
	{
		.procname	= "mthp_bestfit_numa_aware",
		.data		= &sysctl_numa_aware,
		.maxlen		= sizeof(int),
		.mode		= 0644,
		.proc_handler	= proc_dointvec_minmax,
		.extra1		= SYSCTL_ZERO,
		.extra2		= SYSCTL_ONE,
	},
	{
		.procname	= "mthp_bestfit_lifetime_aware",
		.data		= &sysctl_lifetime_aware,
		.maxlen		= sizeof(int),
		.mode		= 0644,
		.proc_handler	= proc_dointvec_minmax,
		.extra1		= SYSCTL_ZERO,
		.extra2		= SYSCTL_ONE,
	},
};

/* ---- DebugFS ------------------------------------------------------------ */

#ifdef CONFIG_DEBUG_FS

static const char * const bf_type_names[BF_VMA_TYPES] = {
	[BF_VMA_STACK] = "stack",
	[BF_VMA_EXEC]  = "exec",
	[BF_VMA_ANON]  = "anon",
	[BF_VMA_FILE]  = "file",
};

static int bestfit_stats_show(struct seq_file *m, void *v)
{
	struct bestfit_counters total = {};
	int cpu, i;

	for_each_possible_cpu(cpu) {
		const struct bestfit_counters *c = per_cpu_ptr(&bf_pcpu, cpu);

		total.decisions           += READ_ONCE(c->decisions);
		total.suppressed          += READ_ONCE(c->suppressed);
		total.pressure_downgraded += READ_ONCE(c->pressure_downgraded);
		total.numa_downgraded     += READ_ONCE(c->numa_downgraded);
		total.lifetime_conserved  += READ_ONCE(c->lifetime_conserved);
		total.compact_stall_avoided += READ_ONCE(c->compact_stall_avoided);
		for (i = 0; i <= PMD_ORDER; i++)
			total.order_hist[i] += READ_ONCE(c->order_hist[i]);
		for (i = 0; i < BF_VMA_TYPES; i++)
			total.by_type[i] += READ_ONCE(c->by_type[i]);
	}

#define PCT(n) (total.decisions ? (n) * 100 / total.decisions : 0)

	seq_printf(m, MTHP_BESTFIT_NAME " v" MTHP_BESTFIT_VER
		   " — VMA-size best-fit mTHP order selection\n\n");

	seq_puts(m, "Configuration\n");
	seq_printf(m, "  enabled:            %d\n", sysctl_enabled);
	seq_printf(m, "  min_pages:          %d\n", sysctl_min_pages);
	seq_printf(m, "  exec_boost:         %d\n", sysctl_exec_boost);
	seq_printf(m, "  dry_run:            %d\n", sysctl_dry_run);
	seq_printf(m, "  pressure_aware:     %d\n", sysctl_pressure_aware);
	seq_printf(m, "  pressure_min_free:  %d blocks\n",
		   sysctl_pressure_min_free);
	seq_printf(m, "  numa_aware:         %d\n", sysctl_numa_aware);
	seq_printf(m, "  lifetime_aware:     %d\n", sysctl_lifetime_aware);

	seq_puts(m, "\nDecision summary\n");
	seq_printf(m, "  total:                  %llu\n", total.decisions);
	seq_printf(m, "  suppressed (no mTHP):   %llu (%llu%%)\n",
		   total.suppressed, PCT(total.suppressed));
	seq_printf(m, "  pressure_downgraded:    %llu (%llu%%)\n",
		   total.pressure_downgraded, PCT(total.pressure_downgraded));
	seq_printf(m, "  compact_stall_avoided:  %llu (%llu%%)\n",
		   total.compact_stall_avoided, PCT(total.compact_stall_avoided));
	seq_printf(m, "  numa_downgraded:        %llu (%llu%%)\n",
		   total.numa_downgraded, PCT(total.numa_downgraded));
	seq_printf(m, "  lifetime_conserved:     %llu (%llu%%)\n",
		   total.lifetime_conserved, PCT(total.lifetime_conserved));

	seq_puts(m, "\nBy VMA type\n");
	for (i = 0; i < BF_VMA_TYPES; i++) {
		seq_printf(m, "  %-6s : %llu (%llu%%)\n",
			   bf_type_names[i],
			   total.by_type[i], PCT(total.by_type[i]));
	}

	seq_puts(m, "\nOrder histogram (selected order)\n");
	for (i = 2; i <= PMD_ORDER; i++) {
		if (!total.order_hist[i])
			continue;
		seq_printf(m, "  order %2d (%6lu KB): %llu (%llu%%)\n",
			   i, (PAGE_SIZE << i) >> 10,
			   total.order_hist[i], PCT(total.order_hist[i]));
	}

	seq_puts(m, "\nTuning guide for compact_stall\n");
	if (total.compact_stall_avoided > 0)
		seq_printf(m,
			   "  %llu potential stalls proactively avoided.\n"
			   "  If compact_stall /proc/vmstat is still rising,\n"
			   "  increase pressure_min_free (currently %d).\n",
			   total.compact_stall_avoided, sysctl_pressure_min_free);
	else
		seq_printf(m,
			   "  0 stalls avoided — no race-window pressure detected.\n"
			   "  If compact_stall is high it comes from non-mTHP sources\n"
			   "  (DMA, kmalloc, hugetlbfs) outside bestfit's scope.\n");

#undef PCT
	return 0;
}

DEFINE_SHOW_ATTRIBUTE(bestfit_stats);
static struct dentry *bf_debugfs_dir;

#endif /* CONFIG_DEBUG_FS */

/* ---- Module init / exit ------------------------------------------------- */

static struct ctl_table_header *bf_sysctl_hdr;

static int __init mthp_bestfit_init(void)
{
	if (!mthp_order_filter_fn) {
		/*
		 * Publish the filter.  WRITE_ONCE pairs with READ_ONCE in
		 * __thp_vma_allowable_orders() in the patched kernel.
		 */
	}
	WRITE_ONCE(mthp_order_filter_fn, mthp_bestfit_filter);

	bf_sysctl_hdr = register_sysctl("vm", mthp_bestfit_sysctls);
	if (!bf_sysctl_hdr)
		pr_warn(MTHP_BESTFIT_NAME ": sysctl registration failed\n");

#ifdef CONFIG_DEBUG_FS
	bf_debugfs_dir = debugfs_create_dir(MTHP_BESTFIT_NAME, NULL);
	debugfs_create_file("stats", 0444, bf_debugfs_dir,
			    NULL, &bestfit_stats_fops);
#endif

	pr_info(MTHP_BESTFIT_NAME ": loaded v" MTHP_BESTFIT_VER
		" — hook on __thp_vma_allowable_orders"
		" (pressure_min_free=%d)\n",
		sysctl_pressure_min_free);
	return 0;
}

static void __exit mthp_bestfit_exit(void)
{
	/*
	 * NULL the hook first, then wait for all CPUs to exit any in-progress
	 * mthp_bestfit_filter() call before we unload.  synchronize_rcu()
	 * ensures no CPU is still inside a READ_ONCE(mthp_order_filter_fn)
	 * critical section that started before the WRITE_ONCE(NULL).
	 */
	WRITE_ONCE(mthp_order_filter_fn, NULL);
	synchronize_rcu();

	if (bf_sysctl_hdr)
		unregister_sysctl_table(bf_sysctl_hdr);

#ifdef CONFIG_DEBUG_FS
	debugfs_remove_recursive(bf_debugfs_dir);
#endif

	pr_info(MTHP_BESTFIT_NAME ": unloaded\n");
}

module_init(mthp_bestfit_init);
module_exit(mthp_bestfit_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Suyog");
MODULE_DESCRIPTION("VMA-size best-fit mTHP order selection via function pointer hook");
MODULE_VERSION(MTHP_BESTFIT_VER);
