// SPDX-License-Identifier: GPL-2.0
/*
 * mthp_bestfit - VMA-size best-fit mTHP order selection
 *
 * Intercepts __thp_vma_allowable_orders() via kretprobe and masks the
 * returned order bitmask so that only orders where at least min_pages
 * aligned hugepages fit within the faulting VMA are attempted.  This
 * prevents the baseline kernel's greedy highest-order-first strategy
 * from wasting allocation attempts on orders that can never succeed for
 * medium-sized VMAs.
 *
 * Additional policies:
 *   - Stack VMAs (VM_GROWSDOWN): capped at order 2 (16 KB)
 *   - File+exec VMAs:            optional +1 order boost (iTLB pressure)
 *   - TVA_SMAPS / TVA_FORCED_COLLAPSE: not intercepted
 *   - dry_run sysctl:            count decisions without altering return
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/mm.h>
#include <linux/huge_mm.h>
#include <linux/kprobes.h>
#include <linux/sysctl.h>
#include <linux/percpu.h>
#include <linux/ptrace.h>

#ifdef CONFIG_DEBUG_FS
#include <linux/debugfs.h>
#include <linux/seq_file.h>
#endif

#define MTHP_BESTFIT_NAME	"mthp_bestfit"
#define MTHP_BESTFIT_VERSION	"2"

/* ---- Sysctls ------------------------------------------------------------ */

static int sysctl_mthp_bestfit_enabled    __read_mostly = 1;
static int sysctl_mthp_bestfit_min_pages  __read_mostly = 2;
static int sysctl_mthp_bestfit_exec_boost __read_mostly = 1;
static int sysctl_mthp_bestfit_dry_run    __read_mostly = 0;

/* ---- Per-CPU statistics ------------------------------------------------- */

/*
 * VMA type buckets for per-type decision counters.
 * A VMA falls into exactly one bucket in order: stack, exec, anon, file.
 */
#define BF_VMA_STACK	0
#define BF_VMA_EXEC	1
#define BF_VMA_ANON	2
#define BF_VMA_FILE	3
#define BF_VMA_TYPES	4

struct bestfit_counters {
	u64 decisions;
	u64 suppressed;			/* decisions returning 0 orders */
	u64 order_hist[PMD_ORDER + 1];	/* selected order distribution */
	u64 by_type[BF_VMA_TYPES];	/* decisions by VMA type */
};

static DEFINE_PER_CPU(struct bestfit_counters, bestfit_pcpu);

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

/*
 * order == -1 means "no order selected" (suppressed decision).
 */
static void bestfit_count(int order, int vma_type, bool suppressed)
{
	struct bestfit_counters *c;

	preempt_disable();
	c = this_cpu_ptr(&bestfit_pcpu);
	c->decisions++;
	if (suppressed)
		c->suppressed++;
	if (order >= 0 && order <= PMD_ORDER)
		c->order_hist[order]++;
	if (vma_type >= 0 && vma_type < BF_VMA_TYPES)
		c->by_type[vma_type]++;
	preempt_enable();
}

/* ---- Core best-fit logic ------------------------------------------------ */

/*
 * mthp_bestfit_order - compute best-fit hugepage order for a VMA
 *
 * Returns the largest order N (2 <= N <= PMD_ORDER) such that at least
 * min_pages complete, hugepage-aligned regions of size (PAGE_SIZE << N)
 * fit within the VMA.  Alignment is computed from the first N-aligned
 * address >= vm_start, which makes the result correct for any fault
 * address inside the VMA.
 *
 * Returns:
 *   > 0   best order to use
 *     0   VMA too small to hold even one order-2 hugepage — suppress mTHP
 *    -1   module disabled
 */
static int mthp_bestfit_order(struct vm_area_struct *vma)
{
	unsigned long min_pages;
	unsigned long page_size;
	unsigned long first_aligned;
	bool is_exec;
	int order;

	if (!sysctl_mthp_bestfit_enabled)
		return -1;

	min_pages = (unsigned long)sysctl_mthp_bestfit_min_pages;

	/*
	 * Stack VMAs (VM_GROWSDOWN, fully initialised): cap at order 2 (16 KB).
	 * Large hugepages fragment badly on unpredictably growing stacks.
	 */
	if (vma->vm_flags & VM_GROWSDOWN) {
		page_size     = 1UL << (PAGE_SHIFT + 2);
		first_aligned = ALIGN(vma->vm_start, page_size);
		return (first_aligned + page_size <= vma->vm_end) ? 2 : 0;
	}

	/*
	 * Early exit: not even one order-2 (16 KB) hugepage fits at an
	 * aligned address — no mTHP makes sense at all.
	 */
	page_size     = 1UL << (PAGE_SHIFT + 2);
	first_aligned = ALIGN(vma->vm_start, page_size);
	if (first_aligned + page_size > vma->vm_end)
		return 0;

	/*
	 * Executable file-backed VMAs (ELF text segments) benefit most from
	 * large hugepages because iTLB coverage directly impacts hot paths.
	 */
	is_exec = (vma->vm_flags & VM_EXEC) && vma->vm_file;

	/* Try PMD_ORDER first, then work down to order 2. */
	for (order = PMD_ORDER; order >= 2; order--) {
		page_size     = 1UL << (PAGE_SHIFT + order);
		first_aligned = ALIGN(vma->vm_start, page_size);

		if (first_aligned < vma->vm_end &&
		    (vma->vm_end - first_aligned) >= min_pages * page_size)
			goto boost;
	}

	return 0;

boost:
	/* Promote exec VMAs by one order to prefer larger iTLB coverage. */
	if (is_exec && sysctl_mthp_bestfit_exec_boost && order < PMD_ORDER)
		order++;

	return order;
}

/*
 * mthp_bestfit_mask - compute the order bitmask to return from the kretprobe
 *
 * Masks allowed_orders so that only orders <= best_order are considered.
 * When best_order >= PMD_ORDER the mask is left unchanged (the allocator
 * is free to use any order the policy permits).
 * When best_order == 0 all mTHP is suppressed (returns 0).
 */
static unsigned long mthp_bestfit_mask(struct vm_area_struct *vma,
				       enum tva_type type,
				       unsigned long allowed_orders)
{
	unsigned long new_orders;
	int best_order;
	int vma_type;
	bool suppressed;

	if (!sysctl_mthp_bestfit_enabled)
		return allowed_orders;

	best_order = mthp_bestfit_order(vma);

	if (best_order < 0)		/* module disabled path */
		return allowed_orders;

	vma_type = bestfit_vma_type(vma);

	if (best_order == 0) {
		/* VMA too small: suppress all mTHP, fall back to order-0 */
		bestfit_count(-1, vma_type, true);
		return 0;
	}

	if (best_order >= PMD_ORDER) {
		/* VMA large enough for PMD: leave allocator free to choose */
		bestfit_count(best_order, vma_type, false);
		return allowed_orders;
	}

	/* Mask out orders strictly above best_order. */
	new_orders = allowed_orders & ((1UL << (best_order + 1)) - 1);
	suppressed = !new_orders;
	bestfit_count(suppressed ? -1 : best_order, vma_type, suppressed);
	return new_orders;
}

/* ---- Kretprobe ---------------------------------------------------------- */

struct bestfit_kretprobe_data {
	struct vm_area_struct	*vma;
	enum tva_type		 type;
};

static int bestfit_entry_handler(struct kretprobe_instance *ri,
				 struct pt_regs *regs)
{
	struct bestfit_kretprobe_data *d = (void *)ri->data;

	d->vma  = (struct vm_area_struct *)regs_get_kernel_argument(regs, 0);
	/* arg index 2 is enum tva_type */
	d->type = (enum tva_type)regs_get_kernel_argument(regs, 2);
	return 0;
}

static int bestfit_ret_handler(struct kretprobe_instance *ri,
			       struct pt_regs *regs)
{
	struct bestfit_kretprobe_data *d = (void *)ri->data;
	unsigned long orders;
	unsigned long new_orders;

	if (!sysctl_mthp_bestfit_enabled)
		return 0;

	if (!d->vma)
		return 0;

	/*
	 * TVA_SMAPS: read-only observation — never alter it, it must reflect
	 * what the kernel policy actually allows, not our filtered view.
	 * TVA_FORCED_COLLAPSE: caller has explicit intent (MADV_COLLAPSE);
	 * do not second-guess it.
	 */
	if (d->type == TVA_SMAPS || d->type == TVA_FORCED_COLLAPSE)
		return 0;

	orders = regs_return_value(regs);
	if (!orders)
		return 0;

	new_orders = mthp_bestfit_mask(d->vma, d->type, orders);

	/* dry_run: record the decision but leave the return value untouched. */
	if (!sysctl_mthp_bestfit_dry_run)
		regs_set_return_value(regs, new_orders);

	return 0;
}

static struct kretprobe bestfit_kretprobe = {
	.handler	= bestfit_ret_handler,
	.entry_handler	= bestfit_entry_handler,
	.data_size	= sizeof(struct bestfit_kretprobe_data),
	.maxactive	= 64,
	.kp.symbol_name	= "__thp_vma_allowable_orders",
};

/* ---- Sysctl table ------------------------------------------------------- */

static int sysctl_max_min_pages = 16;

static struct ctl_table mthp_bestfit_sysctls[] = {
	{
		.procname	= "mthp_bestfit_enabled",
		.data		= &sysctl_mthp_bestfit_enabled,
		.maxlen		= sizeof(int),
		.mode		= 0644,
		.proc_handler	= proc_dointvec_minmax,
		.extra1		= SYSCTL_ZERO,
		.extra2		= SYSCTL_ONE,
	},
	{
		.procname	= "mthp_bestfit_min_pages",
		.data		= &sysctl_mthp_bestfit_min_pages,
		.maxlen		= sizeof(int),
		.mode		= 0644,
		.proc_handler	= proc_dointvec_minmax,
		.extra1		= SYSCTL_ONE,
		.extra2		= &sysctl_max_min_pages,
	},
	{
		.procname	= "mthp_bestfit_exec_boost",
		.data		= &sysctl_mthp_bestfit_exec_boost,
		.maxlen		= sizeof(int),
		.mode		= 0644,
		.proc_handler	= proc_dointvec_minmax,
		.extra1		= SYSCTL_ZERO,
		.extra2		= SYSCTL_ONE,
	},
	{
		/*
		 * dry_run=1: count decisions and log them but do not modify
		 * the return value of __thp_vma_allowable_orders().  Use this
		 * to measure the impact before enabling full interception.
		 */
		.procname	= "mthp_bestfit_dry_run",
		.data		= &sysctl_mthp_bestfit_dry_run,
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
		const struct bestfit_counters *c =
			per_cpu_ptr(&bestfit_pcpu, cpu);

		total.decisions  += READ_ONCE(c->decisions);
		total.suppressed += READ_ONCE(c->suppressed);
		for (i = 0; i <= PMD_ORDER; i++)
			total.order_hist[i] += READ_ONCE(c->order_hist[i]);
		for (i = 0; i < BF_VMA_TYPES; i++)
			total.by_type[i] += READ_ONCE(c->by_type[i]);
	}

	seq_printf(m, MTHP_BESTFIT_NAME " v" MTHP_BESTFIT_VERSION " statistics\n\n");
	seq_printf(m, "enabled:    %d\n", sysctl_mthp_bestfit_enabled);
	seq_printf(m, "min_pages:  %d\n", sysctl_mthp_bestfit_min_pages);
	seq_printf(m, "exec_boost: %d\n", sysctl_mthp_bestfit_exec_boost);
	seq_printf(m, "dry_run:    %d\n", sysctl_mthp_bestfit_dry_run);

	seq_printf(m, "\nTotal decisions: %llu\n", total.decisions);
	if (total.decisions)
		seq_printf(m, "Suppressed (no mTHP): %llu (%llu%%)\n",
			   total.suppressed,
			   total.suppressed * 100 / total.decisions);
	else
		seq_puts(m, "Suppressed (no mTHP): 0\n");

	seq_puts(m, "\nBy VMA type:\n");
	for (i = 0; i < BF_VMA_TYPES; i++) {
		seq_printf(m, "  %-6s : %llu", bf_type_names[i],
			   total.by_type[i]);
		if (total.decisions)
			seq_printf(m, " (%llu%%)",
				   total.by_type[i] * 100 / total.decisions);
		seq_putc(m, '\n');
	}

	seq_puts(m, "\nBy selected order:\n");
	for (i = 2; i <= PMD_ORDER; i++) {
		unsigned long sz_kb = (PAGE_SIZE << i) >> 10;

		if (!total.order_hist[i])
			continue;
		seq_printf(m, "  order %2d (%6lu KB): %llu", i, sz_kb,
			   total.order_hist[i]);
		if (total.decisions)
			seq_printf(m, " (%llu%%)",
				   total.order_hist[i] * 100 / total.decisions);
		seq_putc(m, '\n');
	}

	return 0;
}

DEFINE_SHOW_ATTRIBUTE(bestfit_stats);
static struct dentry *bestfit_debugfs_dir;

#endif /* CONFIG_DEBUG_FS */

/* ---- Module init / exit ------------------------------------------------- */

static struct ctl_table_header *bestfit_sysctl_header;

static int __init mthp_bestfit_init(void)
{
	int ret;

	ret = register_kretprobe(&bestfit_kretprobe);
	if (ret < 0) {
		pr_err(MTHP_BESTFIT_NAME ": kretprobe registration failed: %d\n",
		       ret);
		return ret;
	}

	bestfit_sysctl_header = register_sysctl("vm", mthp_bestfit_sysctls);
	if (!bestfit_sysctl_header)
		pr_warn(MTHP_BESTFIT_NAME ": sysctl registration failed\n");

#ifdef CONFIG_DEBUG_FS
	bestfit_debugfs_dir = debugfs_create_dir(MTHP_BESTFIT_NAME, NULL);
	debugfs_create_file("stats", 0444, bestfit_debugfs_dir,
			    NULL, &bestfit_stats_fops);
#endif

	pr_info(MTHP_BESTFIT_NAME ": loaded v" MTHP_BESTFIT_VERSION
		" (probe on %s)\n", bestfit_kretprobe.kp.symbol_name);
	return 0;
}

static void __exit mthp_bestfit_exit(void)
{
	unregister_kretprobe(&bestfit_kretprobe);

	if (bestfit_sysctl_header)
		unregister_sysctl_table(bestfit_sysctl_header);

#ifdef CONFIG_DEBUG_FS
	debugfs_remove_recursive(bestfit_debugfs_dir);
#endif

	pr_info(MTHP_BESTFIT_NAME ": unloaded\n");
}

module_init(mthp_bestfit_init);
module_exit(mthp_bestfit_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Suyog");
MODULE_DESCRIPTION("VMA-size best-fit mTHP order selection via kretprobe");
MODULE_VERSION(MTHP_BESTFIT_VERSION);
