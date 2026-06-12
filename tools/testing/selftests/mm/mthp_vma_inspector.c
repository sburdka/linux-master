// SPDX-License-Identifier: GPL-2.0
/*
 * mthp_vma_inspector.c — Per-PID VMA folio order inspector for mthp_bestfit
 *
 * Shows WHICH hugepage orders the kernel assigned to each VMA of a running
 * process.  This is the key visual for understanding what mthp_bestfit does:
 *
 *   Baseline kernel (only order-0 and order-9):
 *     [heap]  RSS=8 MB  AnonHP=4 MB
 *       ord-0   4 KB  ████████░░░░░░░░░░░░  50%  (2 MB  — base pages)
 *       ord-9 2048 KB  ████████░░░░░░░░░░░░  50%  (4 MB  — PMD THP)
 *
 *   mthp_bestfit (geometric best-fit selects optimal order per VMA):
 *     [heap]  RSS=8 MB  AnonHP=7.5 MB
 *       ord-0   4 KB  ██░░░░░░░░░░░░░░░░░░   8%  (0.6 MB — unavoidable base)
 *       ord-2  16 KB  ███░░░░░░░░░░░░░░░░░  10%  (0.8 MB — small tensors)
 *       ord-7 512 KB  ████████████░░░░░░░░  38%  (3.0 MB — mid tensors)
 *       ord-8 1024 KB ████████░░░░░░░░░░░░  27%  (2.2 MB — large tensors)
 *       ord-9 2048 KB ████░░░░░░░░░░░░░░░░  17%  (1.4 MB — arena head)
 *                     ← higher order diversity = better TLB + less deferred_split
 *
 * Data sources:
 *   /proc/<pid>/smaps         — VMA names, sizes, AnonHugePages (no root needed)
 *   /proc/<pid>/pagemap       — physical page frame numbers (root for other PIDs)
 *   /proc/kpageflags          — compound page order detection (root required)
 *   /proc/vmstat              — compact_stall, thp_fault_alloc/fallback
 *   /proc/buddyinfo           — PMD (order-9) free block count
 *   /proc/meminfo             — MemAvailable
 *
 * When run as root: exact folio order per VMA via pagemap+kpageflags
 * Without root:     smaps-based estimate (AnonHugePages column only)
 *
 * Build:
 *   gcc -O2 -o mthp_vma_inspector mthp_vma_inspector.c
 *
 * Usage:
 *   ./mthp_vma_inspector <pid>
 *   sudo ./mthp_vma_inspector <pid>              # exact folio orders
 *   sudo ./mthp_vma_inspector <pid> --watch      # refresh every 5 s
 *   sudo ./mthp_vma_inspector <pid> --interval 2 # refresh every 2 s
 *   ./mthp_vma_inspector --name tflite_work      # find PID by comm name
 *   ./mthp_vma_inspector <pid> --all             # show all VMAs incl. tiny ones
 *   ./mthp_vma_inspector <pid> --csv out.csv     # append metrics row to CSV
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <errno.h>
#include <time.h>
#include <signal.h>
#include <sys/types.h>

#define PAGE_SIZE       4096UL
#define MAX_ORDERS      11
#define MAX_VMAS        1024
#define BAR_WIDTH       38
#define MAX_SCAN_PAGES  65536   /* 256 MB max pagemap scan per VMA */

/* /proc/kpageflags bit positions */
#define KPF_COMPOUND_HEAD  15
#define KPF_COMPOUND_TAIL  16
#define KPF_THP            22

/* /proc/<pid>/pagemap entry bits */
#define PM_PRESENT   (1ULL << 63)
#define PM_PFN_MASK  ((1ULL << 55) - 1)

static const unsigned long order_kb[MAX_ORDERS] = {
	4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
};

/* ── Data structures ─────────────────────────────────────────────────── */

struct vma {
	unsigned long  start, end;
	char           perms[8];
	char           name[256];

	/* Populated from /proc/<pid>/smaps */
	unsigned long  size_kb;
	unsigned long  rss_kb;
	unsigned long  anon_hp_kb;
	int            thp_eligible;
	unsigned long  pss_kb;
	unsigned long  priv_dirty_kb;

	/* Populated by pagemap + kpageflags scan (root) */
	unsigned long  order_pages[MAX_ORDERS]; /* huge pages per order */
	unsigned long  present_pages;
	int            scanned;
};

struct proc_snap {
	pid_t          pid;
	char           comm[256];
	struct vma     vmas[MAX_VMAS];
	int            nvmas;

	/* System metrics at snapshot time */
	long           compact_stall;
	long           thp_alloc;
	long           thp_fallback;
	long           deferred_split;
	int            pmd_free;
	long           memavail_kb;
};

/* ── System metric readers ──────────────────────────────────────────── */

static long read_vmstat(const char *key)
{
	FILE *f = fopen("/proc/vmstat", "r");
	char  line[256];
	long  val = 0;

	if (!f)
		return -1;
	while (fgets(line, sizeof(line), f)) {
		char k[128];
		long v;
		if (sscanf(line, "%127s %ld", k, &v) == 2 &&
		    strcmp(k, key) == 0) {
			val = v;
			break;
		}
	}
	fclose(f);
	return val;
}

static int read_pmd_free(void)
{
	FILE *f = fopen("/proc/buddyinfo", "r");
	char  line[256];
	int   total = 0;

	if (!f)
		return -1;
	while (fgets(line, sizeof(line), f)) {
		char *p = strstr(line, "zone");

		if (!p)
			continue;
		/* skip "zone NAME" token */
		while (*p && *p != ' ')
			p++;
		while (*p == ' ')
			p++;
		while (*p && *p != ' ')
			p++;

		int cnt[11] = {0};
		int n = sscanf(p, "%d%d%d%d%d%d%d%d%d%d%d",
			       &cnt[0], &cnt[1], &cnt[2], &cnt[3], &cnt[4],
			       &cnt[5], &cnt[6], &cnt[7], &cnt[8], &cnt[9],
			       &cnt[10]);
		if (n >= 10)
			total += cnt[9];
	}
	fclose(f);
	return total;
}

static long read_memavail_kb(void)
{
	FILE *f = fopen("/proc/meminfo", "r");
	char  line[256];
	long  kb = 0;

	if (!f)
		return -1;
	while (fgets(line, sizeof(line), f)) {
		if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1)
			break;
	}
	fclose(f);
	return kb;
}

/* ── PID lookup by comm name ────────────────────────────────────────── */

static pid_t find_pid_by_comm(const char *comm)
{
	DIR *d = opendir("/proc");
	struct dirent *ent;
	pid_t found = 0;

	if (!d)
		return 0;
	while ((ent = readdir(d)) != NULL) {
		pid_t p = (pid_t)atoi(ent->d_name);

		if (p <= 0)
			continue;

		char path[64], buf[256];

		snprintf(path, sizeof(path), "/proc/%d/comm", p);
		FILE *f = fopen(path, "r");

		if (!f)
			continue;
		if (fgets(buf, sizeof(buf), f)) {
			buf[strcspn(buf, "\n")] = 0;
			if (strcmp(buf, comm) == 0) {
				found = p;
				fclose(f);
				break;
			}
		}
		fclose(f);
	}
	closedir(d);
	return found;
}

/* ── smaps parser ────────────────────────────────────────────────────── */

static int parse_smaps(struct proc_snap *snap)
{
	char path[64];

	snprintf(path, sizeof(path), "/proc/%d/smaps", snap->pid);
	FILE *f = fopen(path, "r");

	if (!f) {
		fprintf(stderr, "Cannot open %s: %s\n", path, strerror(errno));
		return -1;
	}

	char line[512];
	struct vma *v = NULL;

	snap->nvmas = 0;

	while (fgets(line, sizeof(line), f)) {
		unsigned long start, end, off;
		unsigned int  maj, min;
		unsigned long ino;
		char          perms[8], rest[256] = "";

		if (sscanf(line, "%lx-%lx %7s %lx %x:%x %lu %255[^\n]",
			   &start, &end, perms, &off, &maj, &min,
			   &ino, rest) >= 7) {
			if (snap->nvmas >= MAX_VMAS)
				break;
			v = &snap->vmas[snap->nvmas++];
			memset(v, 0, sizeof(*v));
			v->start = start;
			v->end   = end;
			memcpy(v->perms, perms,
			       sizeof(v->perms) - 1);
			char *n = rest;
			while (*n == ' ')
				n++;
			size_t nl = strlen(n);
			if (nl >= sizeof(v->name))
				nl = sizeof(v->name) - 1;
			memcpy(v->name, n, nl);
			v->name[nl] = '\0';
			continue;
		}
		if (!v)
			continue;

		unsigned long val;

#define PARSE(fmt, field)  if (sscanf(line, fmt " %lu", &val) == 1) { v->field = val; continue; }
		PARSE("Size:",          size_kb)
		PARSE("Rss:",           rss_kb)
		PARSE("Pss:",           pss_kb)
		PARSE("AnonHugePages:", anon_hp_kb)
		PARSE("Private_Dirty:", priv_dirty_kb)
#undef PARSE
		if (sscanf(line, "THPeligible: %lu", &val) == 1)
			v->thp_eligible = (int)val;
	}
	fclose(f);
	return snap->nvmas;
}

/* ── pagemap + kpageflags scanner ────────────────────────────────────── */

static int kpageflags_fd = -1;

static uint64_t kpageflags_get(uint64_t pfn)
{
	uint64_t flags = 0;

	if (kpageflags_fd < 0)
		return 0;
	if (pread(kpageflags_fd, &flags, 8, (off_t)(pfn * 8)) != 8)
		return 0;
	return flags;
}

static void scan_pagemap(struct vma *v, int pm_fd)
{
	unsigned long npages = (v->end - v->start) / PAGE_SIZE;

	if (npages == 0)
		return;
	if (npages > MAX_SCAN_PAGES)
		npages = MAX_SCAN_PAGES;

	size_t  bufsz = npages * 8;
	uint64_t *pm  = malloc(bufsz);

	if (!pm)
		return;

	off_t    off  = (off_t)((v->start / PAGE_SIZE) * 8);
	ssize_t  got  = pread(pm_fd, pm, bufsz, off);

	if (got < 8) {
		free(pm);
		return;
	}
	npages = (unsigned long)got / 8;

	unsigned long i = 0;

	while (i < npages) {
		if (!(pm[i] & PM_PRESENT)) {
			i++;
			continue;
		}
		uint64_t pfn   = pm[i] & PM_PFN_MASK;
		uint64_t flags = kpageflags_get(pfn);

		if (flags & (1ULL << KPF_COMPOUND_HEAD)) {
			/* Walk forward counting tail pages */
			unsigned long j = 1;

			while (i + j < npages && (pm[i + j] & PM_PRESENT)) {
				uint64_t pfn2 = pm[i + j] & PM_PFN_MASK;
				uint64_t fl2  = kpageflags_get(pfn2);

				if (!(fl2 & (1ULL << KPF_COMPOUND_TAIL)))
					break;
				/* Also verify PFN contiguity */
				if (pfn2 != pfn + j)
					break;
				j++;
			}
			/* compound size = 1 head + j tails */
			unsigned long compound = j + 1;
			int           order    = 0;

			while ((1UL << order) < compound)
				order++;
			if (order >= MAX_ORDERS)
				order = MAX_ORDERS - 1;
			v->order_pages[order]++;
			v->present_pages += compound;
			i += compound;

		} else if (flags & (1ULL << KPF_COMPOUND_TAIL)) {
			/* Stray tail — shouldn't happen; skip */
			i++;
		} else {
			v->order_pages[0]++;
			v->present_pages++;
			i++;
		}
	}
	free(pm);
	v->scanned = 1;
}

/* ── ASCII bar renderer ──────────────────────────────────────────────── */

static void print_bar(double frac, int width)
{
	int filled = (int)(frac * width + 0.5);

	if (filled > width)
		filled = width;
	for (int i = 0; i < filled; i++)
		fputs("█", stdout);
	for (int i = filled; i < width; i++)
		fputs("░", stdout);
}

/* ── Per-VMA display ─────────────────────────────────────────────────── */

static void print_vma(const struct vma *v, int show_all)
{
	if (!show_all && v->rss_kb < 64)
		return;
	/* skip read-only file-backed mappings unless --all */
	if (!show_all && v->perms[1] != 'w')
		return;

	const char *name = v->name[0] ? v->name : "[anonymous]";

	printf("  %-36s  RSS=%5lu KB  AnonHP=%5lu KB  THP=%s\n",
	       name,
	       v->rss_kb, v->anon_hp_kb,
	       v->thp_eligible ? "yes" : " no");

	if (v->scanned && v->present_pages > 0) {
		unsigned long total_kb = v->present_pages * 4;

		printf("    %-4s %-8s %-7s %-9s %-6s  %-*s\n",
		       "ord", "size", "pages", "memory", "cover",
		       BAR_WIDTH, "distribution");
		printf("    %s\n",
		       "────────────────────────────────────────────────────"
		       "──────────────────────");

		for (int o = 0; o < MAX_ORDERS; o++) {
			if (v->order_pages[o] == 0)
				continue;
			unsigned long pages  = v->order_pages[o];
			unsigned long mem_kb = pages * order_kb[o];
			double        frac   = total_kb ?
					(double)mem_kb / total_kb : 0.0;

			printf("    [%2d] %5lu KB  %6lu  %6lu KB  %5.1f%%  ",
			       o, order_kb[o], pages, mem_kb, frac * 100.0);
			print_bar(frac, BAR_WIDTH);

			if (o == 9)
				printf("  ← PMD");
			else if (o >= 1)
				printf("  ← mTHP");
			printf("\n");
		}
		printf("\n");

	} else if (!v->scanned) {
		/* Fallback: smaps AnonHugePages estimate */
		if (v->rss_kb == 0) {
			printf("\n");
			return;
		}
		unsigned long base_kb = (v->rss_kb > v->anon_hp_kb) ?
					  v->rss_kb - v->anon_hp_kb : 0;
		double frac_base = (double)base_kb / v->rss_kb;
		double frac_thp  = (double)v->anon_hp_kb / v->rss_kb;

		printf("    (smaps estimate — run as root for exact folio orders)\n");
		printf("    [0]    4 KB  ~%5lu  %6lu KB  %5.1f%%  ",
		       base_kb / 4, base_kb, frac_base * 100.0);
		print_bar(frac_base, BAR_WIDTH);
		printf("\n");

		if (v->anon_hp_kb > 0) {
			printf("    [9] 2048 KB  ~%5lu  %6lu KB  %5.1f%%  ",
			       v->anon_hp_kb / 2048, v->anon_hp_kb,
			       frac_thp * 100.0);
			print_bar(frac_thp, BAR_WIDTH);
			printf("  ← PMD\n");
		}
		printf("\n");
	}
}

/* ── System metrics display ──────────────────────────────────────────── */

static void print_system(const struct proc_snap *cur,
			  const struct proc_snap *base)
{
	long dcs = cur->compact_stall  - base->compact_stall;
	long dta = cur->thp_alloc      - base->thp_alloc;
	long dtf = cur->thp_fallback   - base->thp_fallback;
	long dds = cur->deferred_split - base->deferred_split;

	printf("  compact_stall      %8ld    delta: %+ld\n",
	       cur->compact_stall, dcs);
	printf("  PMD free blocks    %8d    (order-9; <10 → stall risk)\n",
	       cur->pmd_free);
	printf("  MemAvailable       %8ld MB\n",
	       cur->memavail_kb / 1024);
	printf("  deferred_split     %8ld    delta: %+ld\n",
	       cur->deferred_split, dds);
	printf("  thp_fault_alloc    %8ld    delta: %+ld\n",
	       cur->thp_alloc, dta);
	printf("  thp_fault_fallback %8ld    delta: %+ld\n",
	       cur->thp_fallback, dtf);
	if (dta + dtf > 0)
		printf("  THP hit rate (run) %8.1f%%   (%ld alloc / %ld fallback)\n",
		       100.0 * dta / (dta + dtf), dta, dtf);
}

/* ── CSV append ──────────────────────────────────────────────────────── */

static void csv_append(const char *path, const struct proc_snap *cur,
		       const struct proc_snap *base, int iter)
{
	FILE *f;
	int   is_new = (access(path, F_OK) != 0);

	f = fopen(path, "a");
	if (!f)
		return;

	if (is_new)
		fprintf(f, "iter,pid,comm,compact_stall,d_compact_stall,"
			"deferred_split,d_deferred_split,"
			"thp_alloc,thp_fallback,pmd_free,memavail_kb\n");

	fprintf(f, "%d,%d,%s,%ld,%ld,%ld,%ld,%ld,%ld,%d,%ld\n",
		iter, cur->pid, cur->comm,
		cur->compact_stall,  cur->compact_stall  - base->compact_stall,
		cur->deferred_split, cur->deferred_split - base->deferred_split,
		cur->thp_alloc, cur->thp_fallback,
		cur->pmd_free, cur->memavail_kb);
	fclose(f);
}

/* ── Take snapshot ───────────────────────────────────────────────────── */

static int take_snapshot(struct proc_snap *snap, int pm_fd, int use_pagemap,
			  int show_all)
{
	/* Read comm */
	char path[64];

	snprintf(path, sizeof(path), "/proc/%d/comm", snap->pid);
	FILE *cf = fopen(path, "r");

	if (cf) {
		if (fgets(snap->comm, sizeof(snap->comm), cf))
			snap->comm[strcspn(snap->comm, "\n")] = 0;
		fclose(cf);
	}

	/* System metrics */
	snap->compact_stall  = read_vmstat("compact_stall");
	snap->thp_alloc      = read_vmstat("thp_fault_alloc");
	snap->thp_fallback   = read_vmstat("thp_fault_fallback");
	snap->deferred_split = read_vmstat("nr_deferred_split_page");
	snap->pmd_free       = read_pmd_free();
	snap->memavail_kb    = read_memavail_kb();

	if (parse_smaps(snap) < 0)
		return -1;

	if (use_pagemap) {
		for (int i = 0; i < snap->nvmas; i++) {
			struct vma *v = &snap->vmas[i];

			if (v->rss_kb < 64 && !show_all)
				continue;
			if (v->perms[1] != 'w' && !show_all)
				continue;
			scan_pagemap(v, pm_fd);
		}
	}
	return 0;
}

/* ── Main ────────────────────────────────────────────────────────────── */

static volatile int g_stop;

static void on_signal(int sig)
{
	(void)sig;
	g_stop = 1;
}

int main(int argc, char *argv[])
{
	pid_t       pid      = 0;
	int         do_watch = 0;
	int         interval = 5;
	int         show_all = 0;
	const char *csv_path = NULL;
	const char *find_name = NULL;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--watch") == 0) {
			do_watch = 1;
		} else if (strcmp(argv[i], "--all") == 0) {
			show_all = 1;
		} else if (strcmp(argv[i], "--interval") == 0 && i + 1 < argc) {
			interval = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
			csv_path = argv[++i];
		} else if (strcmp(argv[i], "--name") == 0 && i + 1 < argc) {
			find_name = argv[++i];
		} else if (argv[i][0] != '-') {
			pid = (pid_t)atol(argv[i]);
		}
	}

	if (find_name) {
		pid = find_pid_by_comm(find_name);
		if (!pid) {
			fprintf(stderr, "No process found with comm='%s'\n",
				find_name);
			return 1;
		}
		fprintf(stderr, "Found PID %d for comm='%s'\n", pid, find_name);
	}

	if (pid <= 0) {
		fprintf(stderr,
			"Usage: %s <pid> [--watch] [--interval N] [--all] [--csv file]\n"
			"       %s --name <comm>\n"
			"\n"
			"  --watch        refresh every interval seconds\n"
			"  --interval N   set refresh period (default: 5)\n"
			"  --all          show all VMAs including tiny/read-only\n"
			"  --csv <file>   append one metrics row per refresh\n"
			"  --name <comm>  find PID by process name\n"
			"\n"
			"  Run as root for exact folio order distribution.\n"
			"  Without root: smaps-based AnonHugePages estimate only.\n",
			argv[0], argv[0]);
		return 1;
	}

	/* Open pagemap for this PID */
	char pm_path[64];

	snprintf(pm_path, sizeof(pm_path), "/proc/%d/pagemap", pid);
	int pm_fd = open(pm_path, O_RDONLY);

	kpageflags_fd = open("/proc/kpageflags", O_RDONLY);

	int use_pagemap = (pm_fd >= 0 && kpageflags_fd >= 0);

	if (!use_pagemap)
		fprintf(stderr,
			"Note: running without full root access — "
			"using smaps estimates.\n"
			"      Sudo for exact per-VMA folio order distribution.\n\n");

	signal(SIGINT, on_signal);
	signal(SIGTERM, on_signal);

	/* Baseline snapshot for delta calculations */
	struct proc_snap base = { .pid = pid };
	struct proc_snap cur  = { .pid = pid };

	take_snapshot(&base, pm_fd, use_pagemap, show_all);

	int iter = 0;

	do {
		cur.pid = pid;
		if (take_snapshot(&cur, pm_fd, use_pagemap, show_all) < 0) {
			fprintf(stderr, "Process %d exited.\n", pid);
			break;
		}

		if (do_watch)
			printf("\033[2J\033[H"); /* clear screen */

		/* Header */
		time_t now = time(NULL);
		char   tbuf[32];

		strftime(tbuf, sizeof(tbuf), "%H:%M:%S", localtime(&now));

		printf("══════════════════════════════════════════════════════════"
		       "═══════════════\n");
		printf("  mTHP VMA Inspector   PID=%-6d  comm=%-20s  %s\n",
		       pid, cur.comm, tbuf);
		printf("══════════════════════════════════════════════════════════"
		       "═══════════════\n\n");

		printf("System metrics (delta = change since inspector started):\n");
		printf("──────────────────────────────────────────────────────────\n");
		print_system(&cur, &base);
		printf("\n");

		printf("VMAs%s:\n",
		       show_all ? "" : " (writable, RSS ≥ 64 KB)");
		printf("──────────────────────────────────────────────────────────"
		       "────────────────\n");

		int shown = 0;

		for (int i = 0; i < cur.nvmas; i++) {
			print_vma(&cur.vmas[i], show_all);
			if (cur.vmas[i].rss_kb >= 64 ||
			    (show_all && cur.vmas[i].rss_kb > 0))
				shown++;
		}
		if (!shown)
			printf("  (no matching VMAs)\n");

		if (csv_path)
			csv_append(csv_path, &cur, &base, iter);

		iter++;

		if (!do_watch || g_stop)
			break;

		if (kill(pid, 0) < 0) {
			printf("\nProcess %d no longer running.\n", pid);
			break;
		}
		sleep((unsigned)interval);

	} while (!g_stop);

	if (pm_fd >= 0)
		close(pm_fd);
	if (kpageflags_fd >= 0)
		close(kpageflags_fd);
	return 0;
}
