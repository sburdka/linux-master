// SPDX-License-Identifier: GPL-2.0
/*
 * mm_fragmenter.c — Buddy allocator fragmentation tool for mTHP testing
 *
 * Creates artificial buddy fragmentation to simulate a real embedded device
 * after hours of operation, so mthp_bestfit benefits become measurable even
 * on a freshly booted test system.
 *
 * Why pre-fragmentation is necessary
 * ─────────────────────────────────────
 * compact_stall only fires when the buddy allocator has no 2 MB (PMD_ORDER)
 * contiguous free block and must physically move pages to create one.
 *
 * On a freshly booted system with 4 GB RAM, /proc/buddyinfo shows ~100 free
 * order-9 (2 MB) blocks — the buddy never needs to compact.
 *
 * After hours of real use, the order-9 count drops to 2-5 because:
 *   • Many anonymous VMAs of varying sizes have been mmap'd and munmap'd
 *   • The freed pages return as small blocks (orders 0-7) that don't
 *     coalesce back into 2 MB because adjacent blocks have different owners
 *   • This is the "swiss cheese" fragmentation pattern
 *
 * This program creates that fragmentation deliberately in seconds:
 *
 *   Phase 1 — Allocate:
 *     Allocates N "fragments" of sizes uniformly distributed between
 *     FRAG_MIN and FRAG_MAX.  These sizes are chosen to be sub-PMD so
 *     they consume order-7 and order-8 blocks from the buddy, leaving
 *     order-9 blocks depleted (can't be assembled from two order-8 blocks
 *     if the order-8 blocks are occupied).
 *
 *   Phase 2 — Create holes:
 *     Frees every other fragment.  This leaves alternating live/free
 *     blocks of varying sizes — the classic buddy fragmentation pattern.
 *     The free blocks are at orders 5-8, too small to form order-9 pairs.
 *
 *   Phase 3 — Hold:
 *     Keeps the remaining half live while the caller runs the workload.
 *     This prevents the kernel from coalescing the freed blocks back.
 *
 *   Phase 4 — Release:
 *     Frees all remaining fragments.
 *
 * Effect on /proc/buddyinfo (zone Normal, order 9):
 *   Before:  ~100 free 2 MB blocks
 *   After phase 2:  ~5-15 free 2 MB blocks
 *   → Now running TFLite arenas triggers compact_stall on the baseline
 *     kernel, while bestfit kernel selects order-8 and avoids the stall.
 *
 * Build:
 *   gcc -O2 -o mm_fragmenter mm_fragmenter.c
 *
 * Usage:
 *   ./mm_fragmenter [target_mb] [mode]
 *
 *   target_mb  amount of RAM to use for fragmentation (default: 25% of MemFree)
 *   mode       hold: stay resident until SIGINT (default, for parallel testing)
 *              oneshot: alloc, fragment, sleep 5s, free, exit
 *
 * Example (run in background while TFLite bench runs):
 *   ./mm_fragmenter 512 hold &
 *   FRAGMENTER_PID=$!
 *   ./mthp_bestfit_tflite_bench.sh --tag baseline ...
 *   kill $FRAGMENTER_PID
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>
#include <signal.h>

/* Fragment size range: sub-PMD to consume orders 6-8 from buddy */
#define FRAG_MIN_KB    96    /* order 5 (96/4096 ≈ 23 pages, rounds up to order 5) */
#define FRAG_MAX_KB   768   /* order 7 (768 KB > 512 KB, round up to order 8) */

/* Maximum number of fragment slots */
#define MAX_FRAGS    8192

static volatile int g_stop = 0;

static void handle_signal(int sig)
{
	(void)sig;
	g_stop = 1;
}

static inline long ns_now(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

/* Read MemFree from /proc/meminfo in KB */
static long read_memfree_kb(void)
{
	FILE *f = fopen("/proc/meminfo", "r");
	char  line[128];
	long  kb = 0;

	if (!f) return 0;
	while (fgets(line, sizeof(line), f)) {
		if (sscanf(line, "MemFree: %ld kB", &kb) == 1)
			break;
	}
	fclose(f);
	return kb;
}

/* Read order-9 (PMD) free block count from /proc/buddyinfo */
static int read_pmd_free(void)
{
	FILE *f = fopen("/proc/buddyinfo", "r");
	char  line[256];
	int   total = 0;

	if (!f) return -1;
	while (fgets(line, sizeof(line), f)) {
		/* Format: "Node N, zone NAME  cnt0 cnt1 ... cnt10" */
		char *p = strstr(line, "zone");
		if (!p) continue;
		/* Skip "zone NAME" */
		while (*p && *p != ' ') p++;
		while (*p == ' ') p++;
		while (*p && *p != ' ') p++;
		/* Now read 10 integers (orders 0-9) */
		int counts[11] = {0};
		int n = sscanf(p, "%d%d%d%d%d%d%d%d%d%d%d",
			       &counts[0], &counts[1], &counts[2], &counts[3],
			       &counts[4], &counts[5], &counts[6], &counts[7],
			       &counts[8], &counts[9], &counts[10]);
		if (n >= 10)
			total += counts[9]; /* order 9 = PMD (2 MB) */
	}
	fclose(f);
	return total;
}

int main(int argc, char *argv[])
{
	long target_mb   = 0;
	int  oneshot     = 0;
	int  i;

	if (argc > 1) target_mb = atol(argv[1]);
	if (argc > 2 && strcmp(argv[2], "oneshot") == 0) oneshot = 1;

	/* Default: use 25% of current MemFree */
	if (target_mb <= 0) {
		long memfree_kb = read_memfree_kb();
		target_mb = memfree_kb / 4 / 1024;
		if (target_mb < 64)  target_mb = 64;
		if (target_mb > 2048) target_mb = 2048;
	}

	printf("mm_fragmenter — buddy allocator fragmentation for mTHP testing\n");
	printf("Target: %ld MB   Mode: %s\n\n",
	       target_mb, oneshot ? "oneshot" : "hold (send SIGINT to release)");

	int pmd_before = read_pmd_free();
	printf("PMD-sized (2 MB) free blocks before: %d\n", pmd_before);

	/* ── Phase 1: Allocate fragments ─────────────────────────────────── */
	size_t frag_min = (size_t)FRAG_MIN_KB * 1024;
	size_t frag_max = (size_t)FRAG_MAX_KB * 1024;
	size_t total_target = (size_t)target_mb * 1024 * 1024;

	void  *frags[MAX_FRAGS];
	size_t sizes[MAX_FRAGS];
	int    nfrags = 0;
	size_t allocated = 0;

	srand((unsigned)time(NULL));
	printf("Phase 1: allocating fragments (%d-%d KB) ...\n",
	       FRAG_MIN_KB, FRAG_MAX_KB);

	while (allocated < total_target && nfrags < MAX_FRAGS) {
		/* Random size uniformly distributed in [frag_min, frag_max] */
		size_t sz = frag_min + (size_t)rand() % (frag_max - frag_min + 1);
		/* Round to page size */
		sz = (sz + 4095) & ~(size_t)4095;

		void *p = mmap(NULL, sz,
			       PROT_READ | PROT_WRITE,
			       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		if (p == MAP_FAILED)
			break;

		/* Touch every page to force physical allocation */
		volatile uint8_t *q = p;
		size_t j;
		for (j = 0; j < sz; j += 4096)
			q[j] = (uint8_t)(nfrags ^ j);

		frags[nfrags] = p;
		sizes[nfrags] = sz;
		nfrags++;
		allocated += sz;
	}

	printf("  Allocated %d fragments, %.1f MB total\n",
	       nfrags, allocated / 1048576.0);

	/* ── Phase 2: Free every other fragment (create holes) ───────────── */
	printf("Phase 2: freeing alternating fragments (creating holes) ...\n");
	int freed_count = 0;
	size_t freed_bytes = 0;

	for (i = 0; i < nfrags; i += 2) {
		munmap(frags[i], sizes[i]);
		freed_bytes += sizes[i];
		freed_count++;
		frags[i] = NULL;
	}

	printf("  Freed %d fragments (%.1f MB), %d remain live (%.1f MB)\n",
	       freed_count, freed_bytes / 1048576.0,
	       nfrags - freed_count,
	       (allocated - freed_bytes) / 1048576.0);

	int pmd_after = read_pmd_free();
	printf("\nPMD-sized (2 MB) free blocks after fragmentation: %d\n", pmd_after);
	if (pmd_before > 0)
		printf("  Reduction: %d → %d  (%.0f%% fewer PMD blocks)\n",
		       pmd_before, pmd_after,
		       100.0 * (pmd_before - pmd_after) / pmd_before);

	if (pmd_after <= 10)
		printf("  ✓ Good fragmentation — compact_stall will fire on next THP request\n");
	else if (pmd_after <= 30)
		printf("  △ Moderate fragmentation — may need higher target_mb\n");
	else
		printf("  ✗ Low fragmentation — increase target_mb or run on lower-RAM system\n");

	printf("\nFragmenter holding %d live fragments.\n", nfrags - freed_count);

	if (oneshot) {
		printf("Oneshot mode: sleeping 5s then releasing all...\n");
		sleep(5);
	} else {
		printf("Hold mode: send SIGINT (Ctrl+C) to release all fragments.\n");
		printf("  Run your benchmark now, then press Ctrl+C.\n\n");
		signal(SIGINT,  handle_signal);
		signal(SIGTERM, handle_signal);
		while (!g_stop) {
			sleep(1);
			/* Re-touch live fragments periodically so they aren't
			 * reclaimed by the kernel's background compaction */
			for (i = 1; i < nfrags; i += 2) {
				if (!frags[i]) continue;
				volatile uint8_t *q = frags[i];
				q[0] = (uint8_t)i;
			}
		}
		printf("\nReceived signal — releasing all fragments...\n");
	}

	/* ── Phase 4: Release remaining fragments ────────────────────────── */
	for (i = 1; i < nfrags; i += 2) {
		if (frags[i])
			munmap(frags[i], sizes[i]);
	}

	int pmd_final = read_pmd_free();
	printf("PMD-sized free blocks after release: %d\n", pmd_final);
	return 0;
}
