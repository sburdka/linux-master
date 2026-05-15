// SPDX-License-Identifier: GPL-2.0
/*
 * cnn_tensor_workload.c — Synthetic CNN tensor lifecycle benchmark
 *
 * Simulates the anonymous memory allocation pattern of real CNN inference
 * frameworks (ncnn, TFLite, PyTorch) to benchmark mTHP fragmentation.
 *
 * Why this workload exposes mTHP fragmentation
 * ─────────────────────────────────────────────
 * Each CNN layer produces an activation tensor whose byte size follows a
 * characteristic profile: large at the input (spatial dims = 224×224),
 * peaks after the first conv (feature maps expand), then shrinks as spatial
 * dims halve at each pooling layer.
 *
 * The tensor sizes for two common models (float32, batch=1):
 *
 *   ResNet-18:
 *     input      224×224×3    =   588 KB   ← order 7 (512 KB)
 *     conv1      112×112×64   = 3,136 KB   ← PMD    (2 MB) ← passes through
 *     res1_out    56×56×64    =   784 KB   ← order 8 (1 MB)
 *     res2_out    56×56×128   = 1,568 KB   ← order 8 (1 MB)
 *     res3_out    28×28×256   =   784 KB   ← order 8 (1 MB)
 *     res4_out    14×14×512   =   392 KB   ← order 7 (512 KB)
 *     res5_out     7×7×512    =    98 KB   ← order 5 (128 KB)
 *     pool_out     1×1×512    =     2 KB   ← 4 KB page
 *
 *   MobileNetV2:
 *     input      224×224×3    =   588 KB   ← order 7
 *     conv_stem  112×112×32   = 1,568 KB   ← order 8
 *     bottleneck  56×56×16    =   196 KB   ← order 6 (256 KB)
 *     bottleneck  56×56×24    =   294 KB   ← order 6
 *     bottleneck  28×28×32    =    98 KB   ← order 5
 *     bottleneck  28×28×96    =   294 KB   ← order 6
 *     bottleneck  14×14×64    =    49 KB   ← order 4 (64 KB)
 *     bottleneck  14×14×160   =   122 KB   ← order 5
 *     bottleneck   7×7×320    =    61 KB   ← order 4
 *     conv_head    7×7×1280   =   245 KB   ← order 6
 *
 * Without mthp_bestfit: the kernel tries PMD (2 MB) for every tensor ≥ 512 KB.
 *   → 784 KB tensor gets a 2 MB page  → 1.24 MB wasted → deferred_split later
 *   → 196 KB tensor gets a 2 MB page  → 1.81 MB wasted → deferred_split later
 *
 * With mthp_bestfit: each tensor gets the right-sized hugepage.
 *   → 784 KB tensor → order 8 (1 MB)  → 240 KB wasted only
 *   → 196 KB tensor → order 6 (256 KB)→ 60 KB wasted only
 *
 * Allocation strategy
 * ────────────────────
 * Uses mmap(MAP_PRIVATE|MAP_ANONYMOUS) for all tensors above 64 KB, matching
 * how real frameworks allocate large buffers (glibc malloc delegates allocs
 * > MMAP_THRESHOLD = 128 KB to mmap; ncnn and TFLite do explicit mmap for
 * large tensors).  mmap is the path through which mTHP policy applies.
 *
 * Each iteration:
 *   1. Allocate all layer activation tensors for both models simultaneously
 *      (simulates holding activations for gradient checkpointing / batch norm)
 *   2. Touch every page of every tensor (forces physical allocation, makes
 *      pages resident so THP must actually commit the hugepage)
 *   3. Free all tensors (creates fragmentation for the next iteration)
 *
 * This interleaved alloc/touch/free loop across tensors of wildly different
 * sizes is exactly what causes buddy-allocator fragmentation in real workloads.
 *
 * Build:
 *   gcc -O2 -o cnn_tensor_workload cnn_tensor_workload.c -lm
 *
 * Run:
 *   ./cnn_tensor_workload [iterations]   (default: 1000)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>

/* Tensor descriptor */
struct tensor {
	const char *name;
	size_t      bytes;
};

/* ── ResNet-18 activation tensor profile (float32, batch=1) ──────────────── */
static const struct tensor resnet18[] = {
	/* name              H    W    C   × 4 bytes */
	{ "input",     224*224*3  *4 },  /*   588 KB */
	{ "conv1",     112*112*64 *4 },  /* 3,136 KB — PMD territory */
	{ "res1_out",   56* 56*64 *4 },  /*   784 KB */
	{ "res2_out",   56* 56*128*4 },  /* 1,568 KB */
	{ "res3_out",   28* 28*256*4 },  /*   784 KB */
	{ "res4_out",   14* 14*512*4 },  /*   392 KB */
	{ "res5_out",    7*  7*512*4 },  /*    98 KB */
	{ "avgpool",     1*  1*512*4 },  /*     2 KB */
	{ "fc_weight", 512*1000  *4 },   /* 1,953 KB */
};
#define RESNET18_LAYERS  (sizeof(resnet18) / sizeof(resnet18[0]))

/* ── MobileNetV2 activation tensor profile (float32, batch=1) ────────────── */
static const struct tensor mobilenetv2[] = {
	{ "input",       224*224*3  *4 },  /*   588 KB */
	{ "conv_stem",   112*112*32 *4 },  /* 1,568 KB */
	{ "bottleneck1",  56* 56*16 *4 },  /*   196 KB */
	{ "bottleneck2",  56* 56*24 *4 },  /*   294 KB */
	{ "bottleneck3",  28* 28*32 *4 },  /*    98 KB */
	{ "bottleneck4",  28* 28*96 *4 },  /*   294 KB */
	{ "bottleneck5",  14* 14*64 *4 },  /*    49 KB */
	{ "bottleneck6",  14* 14*160*4 },  /*   122 KB */
	{ "bottleneck7",   7*  7*320*4 },  /*    61 KB */
	{ "conv_head",     7*  7*1280*4},  /*   245 KB */
	{ "avgpool",       1*  1*1280*4},  /*     5 KB */
};
#define MOBILENETV2_LAYERS (sizeof(mobilenetv2) / sizeof(mobilenetv2[0]))

/* ── SqueezeNet 1.1 activation profile ───────────────────────────────────── */
static const struct tensor squeezenet[] = {
	{ "input",       224*224*3  *4 },  /*   588 KB */
	{ "conv1",       111*111*64 *4 },  /* 3,021 KB */
	{ "pool1",        55* 55*64 *4 },  /*   748 KB */
	{ "fire2_out",    55* 55*128*4 },  /* 1,496 KB */
	{ "fire3_out",    55* 55*128*4 },  /* 1,496 KB */
	{ "pool3",        27* 27*128*4 },  /*   373 KB */
	{ "fire4_out",    27* 27*256*4 },  /*   746 KB */
	{ "fire5_out",    27* 27*256*4 },  /*   746 KB */
	{ "pool5",        13* 13*256*4 },  /*   169 KB */
	{ "fire6_out",    13* 13*384*4 },  /*   253 KB */
	{ "fire7_out",    13* 13*384*4 },  /*   253 KB */
	{ "fire8_out",    13* 13*512*4 },  /*   338 KB */
	{ "fire9_out",    13* 13*512*4 },  /*   338 KB */
	{ "conv10",       13* 13*1000*4},  /*   660 KB */
};
#define SQUEEZENET_LAYERS (sizeof(squeezenet) / sizeof(squeezenet[0]))

/* Minimum size to use mmap (mirrors glibc mmap_threshold) */
#define MMAP_THRESHOLD  (64 * 1024)

static inline long ns_now(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

/*
 * alloc_tensor - allocate a tensor buffer using mmap for large tensors.
 *
 * Forces MAP_POPULATE so the kernel resolves THP at allocation time rather
 * than lazily.  This makes the timing measurements meaningful and ensures
 * deferred_split events are attributed to this workload.
 */
static void *alloc_tensor(size_t bytes)
{
	void *p;

	if (bytes >= MMAP_THRESHOLD) {
		p = mmap(NULL, bytes,
			 PROT_READ | PROT_WRITE,
			 MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE,
			 -1, 0);
		if (p == MAP_FAILED) {
			fprintf(stderr, "mmap(%zu) failed: %s\n",
				bytes, strerror(errno));
			return NULL;
		}
	} else {
		p = malloc(bytes);
		if (p)
			memset(p, 0, bytes);
	}
	return p;
}

static void free_tensor(void *p, size_t bytes)
{
	if (!p)
		return;
	if (bytes >= MMAP_THRESHOLD)
		munmap(p, bytes);
	else
		free(p);
}

/*
 * touch_tensor - write every cacheline to force physical page backing.
 *
 * A simple sequential write is enough: the THP engine coalesces the
 * PTEs during the fault sequence triggered by MAP_POPULATE, but we
 * write a non-zero pattern to prevent the zero-page optimisation from
 * hiding real allocation.
 */
static void touch_tensor(volatile uint8_t *p, size_t bytes, uint8_t pattern)
{
	size_t i;
	/* Stride = 64 bytes (one cacheline) to touch every physical page */
	for (i = 0; i < bytes; i += 64)
		p[i] = pattern;
}

/*
 * run_model - allocate, touch, and free all activation tensors for one model.
 *
 * Interleaves allocation of all layers before touching any, to simulate
 * real frameworks that pipeline alloc and compute.
 */
static int run_model(const char *model_name,
		     const struct tensor *layers, size_t nlayers,
		     int iter)
{
	void **ptrs = calloc(nlayers, sizeof(void *));
	size_t i;
	int ok = 1;

	if (!ptrs)
		return 0;

	/* Phase 1: allocate all tensors */
	for (i = 0; i < nlayers; i++) {
		ptrs[i] = alloc_tensor(layers[i].bytes);
		if (!ptrs[i]) {
			fprintf(stderr, "[%s] iter %d: alloc failed at layer %s\n",
				model_name, iter, layers[i].name);
			ok = 0;
			break;
		}
	}

	/* Phase 2: touch all tensors (force physical backing) */
	if (ok) {
		for (i = 0; i < nlayers; i++)
			touch_tensor(ptrs[i], layers[i].bytes,
				     (uint8_t)(i ^ iter));
	}

	/* Phase 3: free all tensors */
	for (i = 0; i < nlayers; i++)
		free_tensor(ptrs[i], layers[i].bytes);

	free(ptrs);
	return ok;
}

int main(int argc, char *argv[])
{
	int iterations = 1000;
	int i;
	long t0, t1, elapsed_ms;
	unsigned long page_kb = sysconf(_SC_PAGESIZE) / 1024;

	if (argc > 1)
		iterations = atoi(argv[1]);
	if (iterations <= 0)
		iterations = 1000;

	printf("CNN tensor workload — synthetic mTHP fragmentation benchmark\n");
	printf("Models: ResNet-18, MobileNetV2, SqueezeNet-1.1\n");
	printf("Iterations: %d   Page size: %lu KB\n\n", iterations, page_kb);

	/* Print tensor size table on first run */
	printf("ResNet-18 tensor profile (key fragmentation drivers):\n");
	for (i = 0; i < (int)RESNET18_LAYERS; i++) {
		size_t kb = resnet18[i].bytes / 1024;
		int order = 0;
		size_t sz = 4096;
		while (sz < resnet18[i].bytes) { sz <<= 1; order++; }
		printf("  %-14s %6zu KB  → bestfit order %d (%zu KB hugepage)\n",
		       resnet18[i].name, kb, order, sz / 1024);
	}
	printf("\n");

	printf("MobileNetV2 tensor profile:\n");
	for (i = 0; i < (int)MOBILENETV2_LAYERS; i++) {
		size_t kb = mobilenetv2[i].bytes / 1024;
		printf("  %-14s %6zu KB\n", mobilenetv2[i].name, kb);
	}
	printf("\n");

	printf("Starting %d iterations...\n", iterations);
	fflush(stdout);

	t0 = ns_now();

	/*
	 * Main loop: alternate ResNet-18 → MobileNetV2 → SqueezeNet.
	 * Different size profiles between models = interleaved alloc/free of
	 * different-sized blocks = maximum buddy fragmentation.
	 */
	for (i = 0; i < iterations; i++) {
		if (!run_model("ResNet-18",    resnet18,    RESNET18_LAYERS,    i))
			break;
		if (!run_model("MobileNetV2",  mobilenetv2, MOBILENETV2_LAYERS, i))
			break;
		if (!run_model("SqueezeNet",   squeezenet,  SQUEEZENET_LAYERS,  i))
			break;

		if ((i + 1) % 100 == 0) {
			t1 = ns_now();
			printf("  %4d / %d  (%.1f iter/s)\n",
			       i + 1, iterations,
			       (i + 1) * 1e9 / (double)(t1 - t0));
			fflush(stdout);
		}
	}

	t1 = ns_now();
	elapsed_ms = (t1 - t0) / 1000000;

	printf("\nDone. %d iterations × 3 models in %ld ms  (%.1f iter/s)\n\n",
	       iterations, elapsed_ms,
	       iterations * 1e3 / (double)elapsed_ms);

	/*
	 * Total bytes allocated/freed per iteration (one pass of all three models):
	 *   ResNet-18:    sum of all layer bytes
	 *   MobileNetV2:  sum of all layer bytes
	 *   SqueezeNet:   sum of all layer bytes
	 *
	 * Print so the reader can reason about the churn rate.
	 */
	size_t total_per_iter = 0;
	for (i = 0; i < (int)RESNET18_LAYERS;    i++) total_per_iter += resnet18[i].bytes;
	for (i = 0; i < (int)MOBILENETV2_LAYERS; i++) total_per_iter += mobilenetv2[i].bytes;
	for (i = 0; i < (int)SQUEEZENET_LAYERS;  i++) total_per_iter += squeezenet[i].bytes;

	printf("Memory churn per iteration: %.1f MB\n",
	       total_per_iter / 1048576.0);
	printf("Total churn over run:       %.1f GB\n",
	       (double)total_per_iter * iterations / (1024.0 * 1024.0 * 1024.0));

	return 0;
}
