// SPDX-License-Identifier: GPL-2.0
/*
 * hexagon_cnn_workload.c — SNPE/QNN-style CNN tensor allocation benchmark
 *
 * Simulates the tensor buffer allocation pattern of Qualcomm SNPE and QNN
 * SDK runtimes targeting Snapdragon SoCs with Hexagon DSP (HTP).
 *
 * Why SNPE/QNN allocation differs from generic CNN frameworks
 * ───────────────────────────────────────────────────────────
 * SNPE CPU runtime uses posix_memalign(128, size) for all tensor buffers.
 * The 128-byte alignment is required for:
 *   • HVX (Hexagon Vector eXtensions) 128-byte vector register loads
 *   • DMA transfers between HLOS DDR and Hexagon TCM/L2 VTCM
 *   • Cache line alignment for prefetch efficiency on Kryo CPUs
 *
 * For tensors >= glibc mmap_threshold (128 KB by default), glibc implements
 * posix_memalign via mmap(NULL, size + 128, MAP_PRIVATE|MAP_ANONYMOUS).
 * This is the anonymous-memory path through which mTHP policy applies.
 *
 * Critical difference from run-time frameworks (ncnn/TFLite):
 *   SNPE PRE-ALLOCATES all layer activation buffers at model-init time,
 *   holds them simultaneously for the model lifetime, then frees all at
 *   model-destroy time.  This creates the worst-case fragmentation pattern:
 *   allocations of wildly different sizes (from int8 tensor 50 KB at the
 *   detection head up to fp32 stem at 3.2 MB) all coexist in the buddy
 *   allocator simultaneously.
 *
 *   Furthermore, SNPE uses ping-pong (double-buffer) for async execution:
 *   two copies of the input tensor are allocated so the CPU can stage the
 *   next inference while the DSP is processing the current one.
 *
 * Models chosen for this workload
 * ────────────────────────────────
 * All four are standard Qualcomm AI Hub / SNPE model zoo entries:
 *
 *   MobileNetV3-Small fp32  (224×224)  — SNPE classification benchmark
 *   EfficientDet-Lite0 fp32 (320×320)  — detection, large stem tensors
 *   Inception-v3 int8       (299×299)  — quantized, 4× smaller tensors
 *   MobileDet-DSP int8      (320×320)  — Hexagon HTP detection model
 *
 * Tensor size ranges by model type:
 *
 *   fp32 models: 62 KB – 3,200 KB  (orders 4–9, passes through PMD)
 *   int8 models: 12 KB –   800 KB  (orders 2–8, sub-PMD range)
 *
 * mTHP bestfit impact
 * ────────────────────
 * Without bestfit: kernel tries PMD (2 MB) first for every tensor ≥ 512 KB.
 *   EfficientDet stem (3.2 MB) → 1× PMD + 1.2 MB remainder → 800 KB waste
 *   EfficientDet MBConv2 exp (2.4 MB) → 1× PMD + 400 KB remainder → 1.6 MB waste
 *   → Both hugepages go on deferred_split list at model-destroy time
 *
 * With bestfit:
 *   EfficientDet stem (3.2 MB) → 1× PMD (correct size) + smaller for tail
 *   EfficientDet MBConv2 exp (2.4 MB) → 1× PMD for 2 MB + order-8 for 400 KB
 *   MobileNetV3 stem (784 KB) → order-7 (512 KB) instead of order-8 → exact
 *
 * Allocation strategy
 * ────────────────────
 * posix_memalign(SNPE_ALIGN=128, size) for all tensors ≥ MMAP_THRESHOLD.
 * malloc() for smaller tensors (< MMAP_THRESHOLD).
 *
 * Lifecycle per iteration:
 *   1. model_init  — alloc all layer activations + 2× input ping-pong buffer
 *   2. model_infer — touch every layer's activations in forward-pass order
 *                    (repeated INFER_REPS times to simulate warm inference)
 *   3. model_destroy — free all buffers
 *
 * Build:
 *   gcc -O2 -o hexagon_cnn_workload hexagon_cnn_workload.c -lm
 *
 * Run:
 *   ./hexagon_cnn_workload [iterations]   (default: 500)
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
#include <math.h>

/* SNPE / QNN HVX alignment requirement */
#define SNPE_ALIGN      128

/* glibc mmap_threshold default; below this posix_memalign uses brk heap */
#define MMAP_THRESHOLD  (128 * 1024)

/* Number of warm inference passes per model_infer() call */
#define INFER_REPS      3

/* Tensor descriptor */
struct tensor {
	const char *name;
	size_t      bytes;
};

/* ── MobileNetV3-Small fp32 (224×224×3, batch=1) ────────────────────────── */
/* Standard SNPE model-zoo benchmark; all sizes = H×W×C×4 bytes             */
static const struct tensor mobilenetv3_small[] = {
	{ "input",          224*224*3  *4 },  /*   588 KB → order 7 */
	{ "stem_conv",      112*112*16 *4 },  /*   784 KB → order 7 */
	{ "bneck2_exp",      56* 56*72 *4 },  /*   882 KB → order 7 */
	{ "bneck3_exp",      28* 28*88 *4 },  /*   269 KB → order 6 */
	{ "bneck4_exp",      28* 28*96 *4 },  /*   294 KB → order 6 */
	{ "bneck5_exp",      14* 14*240*4 },  /*   183 KB → order 5 */
	{ "bneck6_exp",      14* 14*120*4 },  /*    92 KB → order 5 */
	{ "bneck7_exp",      14* 14*240*4 },  /*   183 KB → order 5 */
	{ "bneck8_exp",      14* 14*200*4 },  /*   153 KB → order 5 */
	{ "bneck11_exp",      7*  7*672*4 },  /*   128 KB → order 5 */
	{ "conv_head",        7*  7*960*4 },  /*   183 KB → order 5 */
};
#define MOBILENETV3_LAYERS  (sizeof(mobilenetv3_small) / sizeof(mobilenetv3_small[0]))

/* ── EfficientDet-Lite0 fp32 (320×320×3, batch=1) ───────────────────────── */
/* EfficientNet-B0 backbone + BiFPN neck; 320×320 input → large stem tensors */
static const struct tensor efficientdet_lite0[] = {
	{ "input",          320*320*3  *4 },  /* 1,200 KB → order 8 */
	{ "stem",           160*160*32 *4 },  /* 3,200 KB → PMD territory */
	{ "mbconv1_out",    160*160*16 *4 },  /* 1,600 KB → order 8 */
	{ "mbconv2_exp",     80* 80*96 *4 },  /* 2,400 KB → PMD territory */
	{ "mbconv2_out",     80* 80*16 *4 },  /*   400 KB → order 7 */
	{ "mbconv3_exp",     80* 80*144*4 },  /* 3,600 KB → PMD territory */
	{ "mbconv3_out",     40* 40*24 *4 },  /*   150 KB → order 5 */
	{ "mbconv4_exp",     40* 40*144*4 },  /*   900 KB → order 8 */
	{ "mbconv4_out",     40* 40*24 *4 },  /*   150 KB → order 5 */
	{ "mbconv5_exp",     40* 40*240*4 },  /* 1,500 KB → order 8 */
	{ "mbconv5_out",     20* 20*40 *4 },  /*    62 KB → order 4 */
	{ "mbconv6_exp",     20* 20*240*4 },  /*   375 KB → order 7 */
	{ "mbconv6_out",     20* 20*40 *4 },  /*    62 KB → order 4 */
	{ "mbconv7_exp",     20* 20*480*4 },  /*   750 KB → order 7 */
	{ "mbconv7_out",     20* 20*80 *4 },  /*   125 KB → order 5 */
	{ "bifpn_p3",        40* 40*64 *4 },  /*   400 KB → order 7 */
	{ "bifpn_p4",        20* 20*64 *4 },  /*   100 KB → order 5 */
	{ "bifpn_p5",        10* 10*64 *4 },  /*    25 KB → order 2 */
	{ "box_pred",        40* 40*36 *4 },  /*   225 KB → order 6 */
	{ "cls_pred",        40* 40*90 *4 },  /*   562 KB → order 7 */
};
#define EFFICIENTDET_LAYERS (sizeof(efficientdet_lite0) / sizeof(efficientdet_lite0[0]))

/* ── Inception-v3 int8 (299×299×3, batch=1) ─────────────────────────────── */
/* Quantized to INT8 → element size = 1 byte; sizes = H×W×C×1               */
/* Classic SNPE demo model; int8 halves the order vs fp32 equivalent         */
static const struct tensor inception_v3_int8[] = {
	{ "input",          299*299*3  *1 },  /*   262 KB → order 6 */
	{ "conv1",          149*149*32 *1 },  /*   693 KB → order 7 */
	{ "conv2",          147*147*32 *1 },  /*   671 KB → order 7 */
	{ "conv3",          147*147*64 *1 },  /* 1,342 KB → order 8 */
	{ "pool",            73* 73*64 *1 },  /*   332 KB → order 6 */
	{ "mixed5b",         35* 35*256*1 },  /*   306 KB → order 6 */
	{ "mixed5c",         35* 35*288*1 },  /*   344 KB → order 6 */
	{ "mixed5d",         35* 35*288*1 },  /*   344 KB → order 6 */
	{ "mixed6a",         17* 17*768*1 },  /*   222 KB → order 6 */
	{ "mixed6b",         17* 17*768*1 },  /*   222 KB → order 6 */
	{ "mixed6e",         17* 17*768*1 },  /*   222 KB → order 6 */
	{ "mixed7a",          8*  8*1280*1},  /*    80 KB → order 4 */
	{ "mixed7b",          8*  8*2048*1},  /*   128 KB → order 5 */
	{ "mixed7c",          8*  8*2048*1},  /*   128 KB → order 5 */
	{ "avgpool",          1*  1*2048*1},  /*     2 KB */
};
#define INCEPTION_LAYERS    (sizeof(inception_v3_int8) / sizeof(inception_v3_int8[0]))

/* ── MobileDet-DSP int8 (320×320×3, batch=1) ────────────────────────────── */
/* Qualcomm's MobileDet architecture optimized for Hexagon HTP               */
/* int8 detection model; detection heads create many mid-size tensors        */
static const struct tensor mobiledet_dsp[] = {
	{ "input",          320*320*3  *1 },  /*   300 KB → order 6 */
	{ "stem_conv",      160*160*32 *1 },  /*   800 KB → order 8 */
	{ "fused_conv1",     80* 80*64 *1 },  /*   400 KB → order 7 */
	{ "fused_conv2",     80* 80*128*1 },  /*   800 KB → order 8 */
	{ "fused_conv3",     40* 40*128*1 },  /*   200 KB → order 6 */
	{ "fused_conv4",     40* 40*256*1 },  /*   400 KB → order 7 */
	{ "fused_conv5",     20* 20*256*1 },  /*   100 KB → order 5 */
	{ "fused_conv6",     20* 20*512*1 },  /*   200 KB → order 6 */
	{ "ssd_p5",          20* 20*256*1 },  /*   100 KB → order 5 */
	{ "ssd_p4",          40* 40*128*1 },  /*   200 KB → order 6 */
	{ "ssd_p3",          80* 80* 64*1 },  /*   400 KB → order 7 */
	{ "box_cls",         20* 20*324*1 },  /*   126 KB → order 5 */
	{ "box_loc",         20* 20*144*1 },  /*    56 KB → order 4 */
};
#define MOBILEDET_LAYERS    (sizeof(mobiledet_dsp) / sizeof(mobiledet_dsp[0]))

static inline long ns_now(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

/*
 * snpe_alloc - allocate a tensor buffer with SNPE's 128-byte alignment.
 *
 * posix_memalign(128, size) is what SNPE CPU runtime calls for every tensor
 * buffer.  For size >= MMAP_THRESHOLD glibc internally calls
 *   mmap(NULL, size + 128, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS)
 * which is the anonymous-memory path through which mTHP policy applies.
 */
static void *snpe_alloc(size_t bytes)
{
	void *p = NULL;

	if (bytes >= MMAP_THRESHOLD) {
		if (posix_memalign(&p, SNPE_ALIGN, bytes) != 0) {
			fprintf(stderr, "posix_memalign(%zu) failed: %s\n",
				bytes, strerror(errno));
			return NULL;
		}
		/* Write first cacheline to force physical backing */
		memset(p, 0, bytes < 64 ? bytes : 64);
	} else {
		p = aligned_alloc(SNPE_ALIGN, (bytes + SNPE_ALIGN - 1) & ~(size_t)(SNPE_ALIGN - 1));
		if (p)
			memset(p, 0, bytes);
	}
	return p;
}

static void snpe_free(void *p)
{
	free(p);
}

/*
 * touch_tensor - simulate inference forward-pass touching each layer output.
 * Pattern = (layer_index ^ iter) written every 64 bytes (one cacheline).
 */
static void touch_tensor(volatile uint8_t *p, size_t bytes,
			 uint8_t pattern)
{
	size_t i;
	for (i = 0; i < bytes; i += 64)
		p[i] = pattern;
}

/* Per-model runtime state (SNPE pre-allocates all buffers at init) */
struct model_rt {
	const char         *name;
	const char         *dtype;
	const struct tensor *layers;
	size_t              nlayers;
	void              **bufs;      /* one buffer per layer */
	void               *pingpong[2]; /* double-buffered input */
};

/*
 * model_init - pre-allocate all activation buffers (SNPE model load phase).
 *
 * SNPE loads the model and immediately allocates scratch memory for every
 * layer's activation tensor before the first inference call.  All buffers
 * coexist in the buddy allocator simultaneously — this is the root cause of
 * fragmentation: the allocator sees a sequence of posix_memalign calls for
 * tensors ranging from 25 KB to 3.2 MB with no frees in between.
 */
static int model_init(struct model_rt *rt)
{
	size_t i;

	rt->bufs = calloc(rt->nlayers, sizeof(void *));
	if (!rt->bufs)
		return 0;

	for (i = 0; i < rt->nlayers; i++) {
		rt->bufs[i] = snpe_alloc(rt->layers[i].bytes);
		if (!rt->bufs[i]) {
			fprintf(stderr, "[%s] alloc failed at layer %s\n",
				rt->name, rt->layers[i].name);
			return 0;
		}
	}

	/* Ping-pong input buffers: 2× input_size for async double-buffering */
	rt->pingpong[0] = snpe_alloc(rt->layers[0].bytes);
	rt->pingpong[1] = snpe_alloc(rt->layers[0].bytes);
	if (!rt->pingpong[0] || !rt->pingpong[1])
		return 0;

	return 1;
}

/*
 * model_infer - simulate INFER_REPS forward passes through the model.
 *
 * Touches every layer's activation buffer in order, swapping ping-pong
 * input buffers each pass.  This replicates the cache pressure of real
 * inference without needing actual compute kernels.
 */
static void model_infer(struct model_rt *rt, int iter)
{
	size_t i;
	int    rep;

	for (rep = 0; rep < INFER_REPS; rep++) {
		/* Input from ping-pong buffer (swapped each pass) */
		touch_tensor(rt->pingpong[rep & 1],
			     rt->layers[0].bytes,
			     (uint8_t)(iter ^ rep));

		/* Forward pass through all layers */
		for (i = 0; i < rt->nlayers; i++)
			touch_tensor(rt->bufs[i], rt->layers[i].bytes,
				     (uint8_t)(i ^ iter ^ rep));
	}
}

/* model_destroy - free all buffers (SNPE model unload phase) */
static void model_destroy(struct model_rt *rt)
{
	size_t i;

	snpe_free(rt->pingpong[0]);
	snpe_free(rt->pingpong[1]);
	for (i = 0; i < rt->nlayers; i++)
		snpe_free(rt->bufs[i]);
	free(rt->bufs);
	rt->bufs        = NULL;
	rt->pingpong[0] = NULL;
	rt->pingpong[1] = NULL;
}

/* Print tensor size table with bestfit order annotation */
static void print_profile(const char *name, const char *dtype,
			   const struct tensor *layers, size_t nlayers)
{
	size_t i, total = 0;

	printf("%s [%s] tensor profile:\n", name, dtype);
	for (i = 0; i < nlayers; i++) {
		size_t bytes = layers[i].bytes;
		size_t kb    = bytes / 1024;
		int    bf_order = 0;
		size_t pg = bytes / 4096;

		if (pg >= 1) {
			bf_order = 0;
			while ((1u << (bf_order + 1)) <= pg)
				bf_order++;
			if (bf_order > 9) bf_order = 9; /* PMD_ORDER */
		}

		printf("  %-16s %6zu KB  → bestfit order %d (%zu KB)\n",
		       layers[i].name, kb, bf_order,
		       (size_t)(1 << (12 + bf_order)) / 1024);
		total += bytes;
	}
	printf("  total activations: %.1f MB\n\n", total / 1048576.0);
}

int main(int argc, char *argv[])
{
	int   iterations = 500;
	int   i;
	long  t0, t1, elapsed_ms;

	if (argc > 1)
		iterations = atoi(argv[1]);
	if (iterations <= 0)
		iterations = 500;

	printf("Hexagon CNN workload — SNPE/QNN-style mTHP fragmentation benchmark\n");
	printf("Models: MobileNetV3-Small (fp32), EfficientDet-Lite0 (fp32),\n");
	printf("        Inception-v3 (int8), MobileDet-DSP (int8)\n");
	printf("Allocation: posix_memalign(%d, size)  [SNPE CPU runtime pattern]\n",
	       SNPE_ALIGN);
	printf("Iterations: %d   Infer reps/iter: %d\n\n",
	       iterations, INFER_REPS);

	print_profile("MobileNetV3-Small", "fp32",
		      mobilenetv3_small, MOBILENETV3_LAYERS);
	print_profile("EfficientDet-Lite0", "fp32",
		      efficientdet_lite0, EFFICIENTDET_LAYERS);
	print_profile("Inception-v3", "int8",
		      inception_v3_int8, INCEPTION_LAYERS);
	print_profile("MobileDet-DSP", "int8",
		      mobiledet_dsp, MOBILEDET_LAYERS);

	printf("Starting %d iterations...\n", iterations);
	fflush(stdout);

	struct model_rt models[] = {
		{ "MobileNetV3-Small", "fp32",
		  mobilenetv3_small, MOBILENETV3_LAYERS, NULL, {NULL, NULL} },
		{ "EfficientDet-Lite0", "fp32",
		  efficientdet_lite0, EFFICIENTDET_LAYERS, NULL, {NULL, NULL} },
		{ "Inception-v3", "int8",
		  inception_v3_int8, INCEPTION_LAYERS, NULL, {NULL, NULL} },
		{ "MobileDet-DSP", "int8",
		  mobiledet_dsp, MOBILEDET_LAYERS, NULL, {NULL, NULL} },
	};
	int nmodels = (int)(sizeof(models) / sizeof(models[0]));

	t0 = ns_now();

	for (i = 0; i < iterations; i++) {
		int m;

		/*
		 * SNPE pre-allocation pattern: all four models are initialised
		 * (buffers allocated) before any inference runs.  This maximises
		 * concurrent live allocations in the buddy allocator — worst-case
		 * fragmentation scenario.
		 */
		for (m = 0; m < nmodels; m++) {
			if (!model_init(&models[m])) {
				fprintf(stderr, "model_init failed iter %d model %s\n",
					i, models[m].name);
				goto done;
			}
		}

		/*
		 * Inference phase: run each model INFER_REPS times.
		 * All activation buffers still live → maximum fragmentation pressure.
		 */
		for (m = 0; m < nmodels; m++)
			model_infer(&models[m], i);

		/* Destroy all models (mass free) → feeds buddy with mixed-order blocks */
		for (m = 0; m < nmodels; m++)
			model_destroy(&models[m]);

		if ((i + 1) % 50 == 0) {
			t1 = ns_now();
			printf("  %4d / %d  (%.1f iter/s)\n",
			       i + 1, iterations,
			       (i + 1) * 1e9 / (double)(t1 - t0));
			fflush(stdout);
		}
	}

done:
	t1 = ns_now();
	elapsed_ms = (t1 - t0) / 1000000;

	printf("\nDone. %d iterations in %ld ms  (%.1f iter/s)\n\n",
	       iterations, elapsed_ms,
	       iterations * 1e3 / (double)elapsed_ms);

	/* Report memory churn per iteration */
	size_t per_iter = 0;
	int    m;

	for (m = 0; m < nmodels; m++) {
		size_t model_bytes = 0;
		size_t j;
		for (j = 0; j < models[m].nlayers; j++)
			model_bytes += models[m].layers[j].bytes;
		/* +2× input for ping-pong */
		model_bytes += 2 * models[m].layers[0].bytes;
		per_iter += model_bytes;
	}

	printf("Concurrent live memory (all 4 models allocated simultaneously):\n");
	printf("  Peak allocation per iteration: %.1f MB\n",
	       per_iter / 1048576.0);
	printf("  Total churn over run:          %.1f GB\n",
	       (double)per_iter * iterations / (1024.0 * 1024.0 * 1024.0));

	return 0;
}
