// SPDX-License-Identifier: GPL-2.0
/*
 * mthp_bestfit — VMA-size best-fit mTHP order selection
 *
 * Registers a function pointer hook (mthp_order_filter_fn) that the core
 * kernel calls at the end of __thp_vma_allowable_orders().  The hook
 * restricts the returned order bitmask so that only orders where at least
 * min_pages complete, aligned hugepages fit within the faulting VMA are
 * attempted.  This prevents the baseline kernel's greedy highest-order-first
 * strategy from wasting allocation attempts on orders that can never succeed
 * for medium-sized VMAs.
 *
 * Additional policies implemented here:
 *
 *   Stack VMAs (VM_GROWSDOWN)
 *     Hard-capped at order 2 (16 KB) to limit fragmentation on
 *     unpredictably growing stacks.
 *
 *   Exec file VMAs (ELF .text segments)
 *     Optional +1 order boost (exec_boost) to widen iTLB coverage on
 *     hot code paths.  iTLBs are typically smaller than dTLBs; larger
 *     hugepages reduce iTLB misses on frequently-called code.
 *
 *   Memory pressure awareness (pressure_aware)
 *     Checks the buddy allocator's free_area[] before committing to an
 *     order.  If no zone has free pages at the desired order, downgrades
 *     to the highest order that IS available.  Critical for 1–2 GB devices
 *     where contiguous pages are scarce and compaction is expensive.
 *
 *   NUMA locality awareness (numa_aware)
 *     Checks that the current CPU's local NUMA node has at least one free
 *     page at the desired order before selecting it.  Falls back to smaller
 *     orders rather than triggering cross-node allocation.
 *
 *   VMA lifetime awareness
 *     VMAs with no anon_vma (never faulted, brand-new) receive a one-order
 *     conservative reduction.  This avoids reserving large contiguous blocks
 *     for VMAs that may never grow to use them — valuable on memory-
 *     constrained devices.
 *
 *   khugepaged integration
 *     The registered hook is also consulted by khugepaged's scan loop
 *     (mm/khugepaged.c) before scanning a VMA for PMD-order collapse,
 *     saving the full scan cost for VMAs the filter would suppress.
 *
 *   dry_run mode
 *     Counts and classifies all decisions without modifying the return
 *     value.  Use this to profile impact before enabling full interception.
 *
 * Tunable at runtime via /proc/sys/vm/mthp_bestfit_*.
 * Statistics at /sys/kernel/debug/mthp_bestfit/stats.
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
#define MTHP_BESTFIT_VER	"3"

/* ---- Sysctls ------------------------------------------------------------ */

static int sysctl_enabled       __read_mostly = 1;
static int sysctl_min_pages     __read_mostly = 2;
static int sysctl_exec_boost    __read_mostly = 1;
static int sysctl_dry_run       __read_mostly = 0;
static int sysctl_pressure_aware __read_mostly = 1;
static int sysctl_numa_aware    __read_mostly = 1;
static int sysctl_lifetime_aware __read_mostly = 1;

/* ---- Per-CPU statistics ------------------------------------------------- */

/*
 * VMA type buckets.  Priority order: stack > exec > anon > file.
 * A VMA falls into exactly one bucket.
 */
#define BF_VMA_STACK	0
#define BF_VMA_EXEC	1
#define BF_VMA_ANON	2
#define BF_VMA_FILE	3
#define BF_VMA_TYPES	4

struct bestfit_counters {
	u64 decisions;
	u64 suppressed;			/* mTHP fully suppressed (returned 0) */
	u64 pressure_downgraded;	/* order reduced due to memory pressure */
	u64 numa_downgraded;		/* order reduced due to NUMA locality */
	u64 lifetime_conserved;		/* order reduced for brand-new VMA */
	u64 order_hist[PMD_ORDER + 1];	/* distribution of selected orders */
	u64 by_type[BF_VMA_TYPES];	/* decisions broken down by VMA type */
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

/* order == -1: no order selected (suppressed). */
static void bf_count(int order, int vma_type, bool suppressed,
		     bool pressure_dg, bool numa_dg, bool lifetime_c)
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
	if (order >= 0 && order <= PMD_ORDER)
		c->order_hist[order]++;
	if (vma_type >= 0 && vma_type < BF_VMA_TYPES)
		c->by_type[vma_type]++;
	preempt_enable();
}

/* ---- Memory pressure helpers -------------------------------------------- */

/*
 * Return true if any populated zone has at least one free block of @order
 * in the buddy allocator.  The read of nr_free is intentionally racy —
 * it is a heuristic snapshot, not a guarantee.  We never hold the zone
 * lock here; the cost of a stale read is a one-time mis-selection, not
 * correctness.
 */
static bool order_has_free_pages(int order)
{
	struct zone *zone;

	for_each_populated_zone(zone) {
		if (READ_ONCE(zone->free_area[order].nr_free) > 0)
			return true;
	}
	return false;
}

/*
 * Walk orders from @start down to 2; return the highest order that has
 * free pages in any zone.  Returns @start immediately if pressure_aware
 * is disabled, or if @start has free pages (fast path — no downgrade).
 */
static int bestfit_pressure_limit(int start)
{
	int order;

	if (!READ_ONCE(sysctl_pressure_aware))
		return start;

	/* Fast path: desired order is available. */
	if (order_has_free_pages(start))
		return start;

	/* Scan downward for the highest available order. */
	for (order = start - 1; order >= 2; order--) {
		if (order_has_free_pages(order))
			return order;
	}
	return 2;
}

/* ---- NUMA locality helper ----------------------------------------------- */

/*
 * Return true if the current CPU's local NUMA node has at least one free
 * block of @order available.  Falls back to true on UMA systems (nid == 0
 * with a single pgdat covering all zones).
 */
static bool order_available_local(int order)
{
	int nid = numa_node_id();
	struct pglist_data *pgdat;
	int z;

	if (!READ_ONCE(sysctl_numa_aware))
		return true;

	pgdat = NODE_DATA(nid);
	for (z = 0; z < MAX_NR_ZONES; z++) {
		struct zone *zone = &pgdat->node_zones[z];

		if (!populated_zone(zone))
			continue;
		if (READ_ONCE(zone->free_area[order].nr_free) > 0)
			return true;
	}
	return false;
}

/* ---- Core best-fit logic ------------------------------------------------ */

/*
 * mthp_bestfit_order - compute best-fit hugepage order for @vma
 *
 * Returns the largest order N (2 <= N <= PMD_ORDER) such that at least
 * min_pages complete, hugepage-aligned regions of size (PAGE_SIZE << N)
 * fit within the VMA.  Alignment is computed from the first N-aligned
 * address >= vm_start, making the result correct for any fault address.
 *
 * After the geometric best-fit is computed, three further adjustments are
 * applied in order:
 *   1. exec_boost:    +1 for file-backed exec VMAs (iTLB pressure)
 *   2. pressure_aware: downgrade if buddy has no free pages at that order
 *   3. numa_aware:     downgrade if local NUMA node lacks the order
 *   4. lifetime_aware: -1 for brand-new VMAs (anon_vma == NULL)
 *
 * Each adjustment is tracked separately in per-CPU statistics.
 *
 * Returns:
 *   >= 2   valid best order
 *      0   VMA too small — suppress all mTHP (caller returns 0 orders)
 *     -1   module disabled
 */
static int mthp_bestfit_order(struct vm_area_struct *vma,
			      bool *pressure_dg, bool *numa_dg,
			      bool *lifetime_c)
{
	unsigned long min_pages;
	unsigned long page_size;
	unsigned long first_aligned;
	bool is_exec;
	int order;
	int limited;

	*pressure_dg = false;
	*numa_dg     = false;
	*lifetime_c  = false;

	if (!READ_ONCE(sysctl_enabled))
		return -1;

	min_pages = (unsigned long)READ_ONCE(sysctl_min_pages);

	/*
	 * Stack VMAs (VM_GROWSDOWN, fully initialised): cap at order 2 (16 KB).
	 * Large hugepages fragment badly for stacks that grow unpredictably.
	 */
	if (vma->vm_flags & VM_GROWSDOWN) {
		page_size     = 1UL << (PAGE_SHIFT + 2);
		first_aligned = ALIGN(vma->vm_start, page_size);
		return (first_aligned + page_size <= vma->vm_end) ? 2 : 0;
	}

	/*
	 * Early exit: not even one order-2 (16 KB) hugepage fits at an
	 * aligned address — no mTHP of any size makes sense.
	 */
	page_size     = 1UL << (PAGE_SHIFT + 2);
	first_aligned = ALIGN(vma->vm_start, page_size);
	if (first_aligned + page_size > vma->vm_end)
		return 0;

	/* ELF text segments benefit most from large hugepages (iTLB). */
	is_exec = (vma->vm_flags & VM_EXEC) && vma->vm_file;

	/* Geometric best-fit: largest order where min_pages aligned pages fit. */
	for (order = PMD_ORDER; order >= 2; order--) {
		page_size     = 1UL << (PAGE_SHIFT + order);
		first_aligned = ALIGN(vma->vm_start, page_size);

		if (first_aligned < vma->vm_end &&
		    (vma->vm_end - first_aligned) >= min_pages * page_size)
			goto found;
	}
	return 0;

found:
	/* 1. Exec boost: promote one order to widen iTLB coverage. */
	if (is_exec && READ_ONCE(sysctl_exec_boost) && order < PMD_ORDER)
		order++;

	/* 2. Memory pressure: downgrade if buddy cannot satisfy the order. */
	limited = bestfit_pressure_limit(order);
	if (limited < order) {
		order        = limited;
		*pressure_dg = true;
	}

	/* 3. NUMA locality: downgrade if local node lacks the order. */
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

	/*
	 * 4. Lifetime awareness: brand-new VMAs (anon_vma == NULL) have never
	 *    been faulted.  We conserve one order to avoid reserving large
	 *    contiguous blocks for VMAs that may never grow to use them.
	 *    Critical on 1–2 GB devices where physical contiguity is scarce.
	 */
	if (READ_ONCE(sysctl_lifetime_aware) && !vma->anon_vma && order > 2) {
		order--;
		*lifetime_c = true;
	}

	return order;
}

/* ---- Hook function registered as mthp_order_filter_fn ------------------ */

static unsigned long mthp_bestfit_filter(struct vm_area_struct *vma,
					 vm_flags_t vm_flags,
					 enum tva_type type,
					 unsigned long allowed_orders)
{
	unsigned long new_orders;
	bool pressure_dg, numa_dg, lifetime_c;
	int best_order;
	int vma_type;
	bool suppressed;

	best_order = mthp_bestfit_order(vma, &pressure_dg, &numa_dg,
					&lifetime_c);
	vma_type   = bestfit_vma_type(vma);

	if (best_order < 0)		/* module disabled */
		return allowed_orders;

	if (best_order == 0) {		/* VMA too small: suppress all mTHP */
		bf_count(-1, vma_type, true, false, false, false);
		return 0;
	}

	if (best_order >= PMD_ORDER) {
		/* Large enough for PMD — let the allocator choose freely. */
		bf_count(best_order, vma_type, false,
			 pressure_dg, numa_dg, lifetime_c);
		return allowed_orders;
	}

	/* Mask out orders strictly above best_order. */
	new_orders = allowed_orders & ((1UL << (best_order + 1)) - 1);
	suppressed = !new_orders;
	bf_count(suppressed ? -1 : best_order, vma_type, suppressed,
		 pressure_dg, numa_dg, lifetime_c);
	return new_orders;
}

/* ---- Sysctl table ------------------------------------------------------- */

static int sysctl_max_min_pages = 16;

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
		/*
		 * dry_run=1: count and classify decisions but do not modify
		 * the return value.  Profile impact before full interception.
		 */
		.procname	= "mthp_bestfit_dry_run",
		.data		= &sysctl_dry_run,
		.maxlen		= sizeof(int),
		.mode		= 0644,
		.proc_handler	= proc_dointvec_minmax,
		.extra1		= SYSCTL_ZERO,
		.extra2		= SYSCTL_ONE,
	},
	{
		/*
		 * pressure_aware=1: downgrade the selected order when the
		 * buddy allocator has no free blocks at that order in any zone.
		 * Prevents triggering expensive compaction on memory-constrained
		 * devices (1–2 GB DDR).
		 */
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
		 * numa_aware=1: prefer orders available on the local NUMA node
		 * to avoid cross-node allocation latency.  On UMA systems this
		 * check is a no-op (always true) with negligible overhead.
		 */
		.procname	= "mthp_bestfit_numa_aware",
		.data		= &sysctl_numa_aware,
		.maxlen		= sizeof(int),
		.mode		= 0644,
		.proc_handler	= proc_dointvec_minmax,
		.extra1		= SYSCTL_ZERO,
		.extra2		= SYSCTL_ONE,
	},
	{
		/*
		 * lifetime_aware=1: reduce order by 1 for VMAs that have never
		 * been faulted (anon_vma == NULL).  Conserves large contiguous
		 * blocks for established VMAs that are more likely to use them.
		 */
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

		total.decisions          += READ_ONCE(c->decisions);
		total.suppressed         += READ_ONCE(c->suppressed);
		total.pressure_downgraded += READ_ONCE(c->pressure_downgraded);
		total.numa_downgraded    += READ_ONCE(c->numa_downgraded);
		total.lifetime_conserved += READ_ONCE(c->lifetime_conserved);
		for (i = 0; i <= PMD_ORDER; i++)
			total.order_hist[i] += READ_ONCE(c->order_hist[i]);
		for (i = 0; i < BF_VMA_TYPES; i++)
			total.by_type[i] += READ_ONCE(c->by_type[i]);
	}

#define PCT(n) (total.decisions ? (n) * 100 / total.decisions : 0)

	seq_printf(m, MTHP_BESTFIT_NAME " v" MTHP_BESTFIT_VER
		   " — VMA-size best-fit mTHP order selection\n\n");
	seq_printf(m, "Configuration\n");
	seq_printf(m, "  enabled:        %d\n", sysctl_enabled);
	seq_printf(m, "  min_pages:      %d\n", sysctl_min_pages);
	seq_printf(m, "  exec_boost:     %d\n", sysctl_exec_boost);
	seq_printf(m, "  dry_run:        %d\n", sysctl_dry_run);
	seq_printf(m, "  pressure_aware: %d\n", sysctl_pressure_aware);
	seq_printf(m, "  numa_aware:     %d\n", sysctl_numa_aware);
	seq_printf(m, "  lifetime_aware: %d\n", sysctl_lifetime_aware);

	seq_printf(m, "\nDecision summary\n");
	seq_printf(m, "  total:               %llu\n", total.decisions);
	seq_printf(m, "  suppressed:          %llu (%llu%%)\n",
		   total.suppressed, PCT(total.suppressed));
	seq_printf(m, "  pressure_downgraded: %llu (%llu%%)\n",
		   total.pressure_downgraded, PCT(total.pressure_downgraded));
	seq_printf(m, "  numa_downgraded:     %llu (%llu%%)\n",
		   total.numa_downgraded, PCT(total.numa_downgraded));
	seq_printf(m, "  lifetime_conserved:  %llu (%llu%%)\n",
		   total.lifetime_conserved, PCT(total.lifetime_conserved));

	seq_printf(m, "\nBy VMA type\n");
	for (i = 0; i < BF_VMA_TYPES; i++) {
		seq_printf(m, "  %-6s : %llu (%llu%%)\n",
			   bf_type_names[i],
			   total.by_type[i], PCT(total.by_type[i]));
	}

	seq_printf(m, "\nOrder histogram\n");
	for (i = 2; i <= PMD_ORDER; i++) {
		if (!total.order_hist[i])
			continue;
		seq_printf(m, "  order %2d (%6lu KB): %llu (%llu%%)\n",
			   i, (PAGE_SIZE << i) >> 10,
			   total.order_hist[i], PCT(total.order_hist[i]));
	}

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
	/*
	 * Publish the filter before registering sysctls or debugfs so that
	 * any early page fault after init() returns already gets filtered.
	 * WRITE_ONCE pairs with READ_ONCE in __thp_vma_allowable_orders().
	 */
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
		" — hook registered on __thp_vma_allowable_orders\n");
	return 0;
}

static void __exit mthp_bestfit_exit(void)
{
	/*
	 * Clear the hook first, then wait for all CPUs to finish any
	 * in-progress call to mthp_bestfit_filter before we unload.
	 * synchronize_rcu() provides the necessary barrier: no CPU can
	 * be executing a READ_ONCE(mthp_order_filter_fn) critical section
	 * from before the WRITE_ONCE(NULL) after this returns.
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
