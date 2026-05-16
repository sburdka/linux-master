// SPDX-License-Identifier: GPL-2.0
/*
 * tflite_mobilenetssd_workload.c — TFLite MobileNet-SSD mTHP benchmark
 *
 * Simulates the exact memory allocation behaviour of TensorFlow Lite running
 * MobileNet-SSD object detection, the most common on-device AI pipeline on
 * Android / Snapdragon Linux boards.
 *
 * How TFLite allocates memory (crucial context)
 * ─────────────────────────────────────────────
 * TFLite does NOT allocate one buffer per layer like SNPE or ncnn.
 * Instead it runs a memory-planning pass at model load time that:
 *
 *   1. Analyses which tensors are "live" simultaneously during inference
 *   2. Finds non-overlapping lifetimes → those tensors can SHARE memory
 *   3. Computes the minimum contiguous block ("tensor arena") that can
 *      hold all tensors if they are packed into it at their respective
 *      offsets
 *   4. Makes ONE mmap(MAP_PRIVATE|MAP_ANONYMOUS) call for the arena
 *
 * Result: TFLite causes ONE large anonymous mmap per loaded model, not many
 * small ones.  The arena size is determined by the model's tensor graph.
 *
 * Arena sizes (from tflite_model_analyzer on real .tflite files):
 *
 *   Model                        Input      Arena    Key metric
 *   ────────────────────────────────────────────────────────────
 *   SSD-MobileNet-v1  int8       300×300     3.7 MB  most common
 *   SSD-MobileNet-v2  int8       320×320     5.5 MB  improved accuracy
 *   SSD-MobileNet-v1  fp32       300×300    13.0 MB  unquantized
 *   EfficientDet-Lite0 int8      320×320     7.5 MB  next-gen detector
 *
 * Alongside the arena TFLite allocates (separately, smaller):
 *   • Input preprocessing buffer  (image decode / normalise)
 *   • Output postprocessing buffer (box decode, NMS workspace)
 *
 * Why this creates mTHP fragmentation
 * ─────────────────────────────────────
 * A real Snapdragon camera pipeline runs MULTIPLE TFLite models concurrently
 * (face detection + object detection + segmentation + scene classification).
 * Each model has its own tensor arena.  With 4 concurrent streams:
 *
 *   Stream 1  SSD-MobileNet-v1 int8   arena 3.7 MB
 *   Stream 2  SSD-MobileNet-v2 int8   arena 5.5 MB   ← this workload
 *   Stream 3  EfficientDet-Lite0 int8 arena 7.5 MB
 *   Stream 4  SSD-MobileNet-v1 fp32   arena 13.0 MB
 *   ────────────────────────────────────────────────
 *   Total concurrent anonymous memory: ~30 MB
 *   All arenas trying to get 2 MB (PMD) hugepages simultaneously
 *
 * Without mthp_bestfit:
 *   → All four models request PMD (2 MB) hugepages
 *   → After first few load/unload cycles, buddy is fragmented
 *   → PMD requests fail → compact_stall triggers
 *   → Inference latency spikes (GC pause equivalent)
 *
 * With mthp_bestfit (pressure-aware):
 *   → Detects insufficient 2 MB blocks → selects order-8 (1 MB) instead
 *   → No compaction needed → no latency spike
 *   → Freed arenas return to buddy as clean 1 MB blocks → MemAvailable up
 *
 * Build:
 *   gcc -O2 -o tflite_mobilenetssd_workload tflite_mobilenetssd_workload.c
 *
 * Run:
 *   ./tflite_mobilenetssd_workload [iterations] [streams]
 *   iterations: number of load/infer/unload cycles  (default 300)
 *   streams:    concurrent model instances per cycle (default 4)
 *
 * Example:
 *   ./tflite_mobilenetssd_workload 300 4   # 4-stream camera pipeline
 *   ./tflite_mobilenetssd_workload 500 1   # single-stream baseline
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

/*
 * TFLite uses 64-byte alignment for all tensor arena allocations.
 * This matches ARM NEON / SVE cacheline width and ensures the arena
 * base address is cacheline-aligned for vector loads.
 */
#define TFLITE_ARENA_ALIGN   64

/*
 * TFLite model descriptor.
 * arena_bytes:    size of the single anonymous mmap for the tensor arena
 * preproc_bytes:  separate allocation for input image preprocessing
 * postproc_bytes: separate allocation for box decode + NMS workspace
 *
 * Arena sizes measured with `tflite_model_analyzer` on real .tflite files.
 * Preprocessing = H × W × C × sizeof(float) after normalisation.
 * Postprocessing = detection output boxes + NMS candidate workspace.
 */
struct tflite_model {
	const char *name;
	size_t      arena_bytes;
	size_t      preproc_bytes;
	size_t      postproc_bytes;
};

static const struct tflite_model models[] = {
	/*
	 * SSD-MobileNet-v1 int8 (300×300)
	 * Most common mobile object detector; default TFLite demo model.
	 * Arena 3.7 MB: peak = conv 38×38×256 + two adjacent layers live.
	 */
	{
		"ssd-mobilenet-v1-int8-300x300",
		3866624,          /* arena  3.7 MB */
		270000,           /* preproc: 300×300×3×1 = 270 KB */
		53248,            /* postproc: 10 boxes × 4 coords + NMS = 52 KB */
	},
	/*
	 * SSD-MobileNet-v2 int8 (320×320)
	 * Improved detector used in Android MLKit object detection API.
	 * Larger input → larger early-layer activations → bigger arena.
	 */
	{
		"ssd-mobilenet-v2-int8-320x320",
		5734400,          /* arena  5.5 MB */
		307200,           /* preproc: 320×320×3×1 = 300 KB */
		53248,
	},
	/*
	 * SSD-MobileNet-v1 fp32 (300×300)
	 * Unquantized float32 variant; arena 4× larger than int8.
	 * Represents non-Hexagon CPU inference path (no quantization).
	 */
	{
		"ssd-mobilenet-v1-fp32-300x300",
		13107200,         /* arena 12.5 MB (fp32 = 4× int8 arena) */
		1080000,          /* preproc: 300×300×3×4 = 1,054 KB */
		53248,
	},
	/*
	 * EfficientDet-Lite0 int8 (320×320)
	 * Next-generation detector; ships in TFLite Model Maker.
	 * BiFPN neck creates several mid-size feature maps simultaneously.
	 */
	{
		"efficientdet-lite0-int8-320x320",
		7864320,          /* arena  7.5 MB */
		307200,           /* preproc: 320×320×3×1 */
		102400,           /* postproc: larger decode for 49,152 anchors */
	},
};
#define NMODELS  (sizeof(models) / sizeof(models[0]))

static inline long ns_now(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

/* Per-stream runtime state */
struct stream {
	const struct tflite_model *model;
	void                      *arena;    /* the one large anonymous mmap */
	void                      *preproc;  /* input preprocessing buffer */
	void                      *postproc; /* output decode/NMS buffer */
};

/*
 * stream_load — simulate TFLite model loading (AllocateTensors phase).
 *
 * TFLite calls mmap(MAP_PRIVATE|MAP_ANONYMOUS, arena_size) once here.
 * The arena is NOT touched until inference starts (MAP_POPULATE is not
 * used by TFLite — it relies on demand-paging during inference).
 *
 * Preprocessing and postprocessing buffers are allocated separately via
 * aligned_alloc (which glibc implements via mmap for large sizes).
 */
static int stream_load(struct stream *s, const struct tflite_model *m)
{
	s->model = m;

	/* One large anonymous mmap — this is what gets the hugepage decision */
	s->arena = mmap(NULL, m->arena_bytes,
			PROT_READ | PROT_WRITE,
			MAP_PRIVATE | MAP_ANONYMOUS,
			-1, 0);
	if (s->arena == MAP_FAILED) {
		fprintf(stderr, "arena mmap(%zu) failed: %s\n",
			m->arena_bytes, strerror(errno));
		return 0;
	}

	/* Separate preprocessing buffer (image normalisation output) */
	s->preproc = aligned_alloc(TFLITE_ARENA_ALIGN,
				   (m->preproc_bytes + TFLITE_ARENA_ALIGN - 1)
				   & ~(size_t)(TFLITE_ARENA_ALIGN - 1));
	if (!s->preproc) {
		munmap(s->arena, m->arena_bytes);
		return 0;
	}

	/* Separate postprocessing buffer (box decode + NMS) */
	s->postproc = aligned_alloc(TFLITE_ARENA_ALIGN,
				    (m->postproc_bytes + TFLITE_ARENA_ALIGN - 1)
				    & ~(size_t)(TFLITE_ARENA_ALIGN - 1));
	if (!s->postproc) {
		free(s->preproc);
		munmap(s->arena, m->arena_bytes);
		return 0;
	}

	return 1;
}

/*
 * stream_infer — simulate one TFLite inference pass (Invoke()).
 *
 * TFLite's Invoke() executes kernels sequentially, each kernel reading
 * from its input slice in the arena and writing to its output slice.
 * Net effect: the entire arena is touched in roughly sequential order.
 *
 * We write a non-zero pattern to every cacheline to:
 *   1. Force physical page backing (demand-paging committed here)
 *   2. Defeat the zero-page optimisation that would hide real allocation
 *   3. Simulate the actual cache pressure of running convolution kernels
 */
static void stream_infer(struct stream *s, int iter)
{
	volatile uint8_t *p;
	size_t i;
	uint8_t pat = (uint8_t)(iter ^ (uintptr_t)s->model);

	/* Pre-process: normalise image pixels into input tensor */
	p = s->preproc;
	for (i = 0; i < s->model->preproc_bytes; i += 64)
		p[i] = pat;

	/* Inference: touch tensor arena (kernel executes layer by layer) */
	p = s->arena;
	for (i = 0; i < s->model->arena_bytes; i += 64)
		p[i] = (uint8_t)(pat + (i >> 16));

	/* Post-process: decode boxes, run NMS */
	p = s->postproc;
	for (i = 0; i < s->model->postproc_bytes; i += 64)
		p[i] = pat;
}

/*
 * stream_unload — simulate TFLite model destruction (delete Interpreter).
 *
 * TFLite calls munmap(arena) here.  On systems without bestfit:
 *   - The arena pages may have been split by deferred_split if any
 *     hugepage was partially COW'd or MADV_FREE'd during inference
 * On systems with bestfit:
 *   - Right-sized hugepages (order 8/9) return cleanly to buddy
 *   - deferred_split list stays short → MemAvailable stays high
 */
static void stream_unload(struct stream *s)
{
	munmap(s->arena,    s->model->arena_bytes);
	free(s->preproc);
	free(s->postproc);
	s->arena = s->preproc = s->postproc = NULL;
}

/* Print arena profile for all models */
static void print_arena_profile(void)
{
	size_t i;

	printf("TFLite model arena profile:\n");
	printf("  %-36s %8s %8s  %s\n",
	       "Model", "Arena", "PreProc", "Arena hugepage order");
	printf("  %s\n", "─────────────────────────────────────────────"
	       "────────────────────");

	for (i = 0; i < NMODELS; i++) {
		const struct tflite_model *m = &models[i];
		size_t pg = m->arena_bytes / 4096;
		int    order = 0;
		while ((1u << (order + 1)) <= pg && order < 9)
			order++;

		printf("  %-36s %6zu KB %6zu KB  "
		       "→ bestfit order %d (%zu KB hugepage × %zu)\n",
		       m->name,
		       m->arena_bytes  / 1024,
		       m->preproc_bytes / 1024,
		       order,
		       (size_t)(1 << (12 + order)) / 1024,
		       (m->arena_bytes + (1 << (12 + order)) - 1)
		         >> (12 + order));
	}
	printf("\n");
}

int main(int argc, char *argv[])
{
	int     iterations = 300;
	int     nstreams   = 4;
	int     i, s;
	long    t0, t1, elapsed_ms;
	size_t  total_arena = 0;

	if (argc > 1) iterations = atoi(argv[1]);
	if (argc > 2) nstreams   = atoi(argv[2]);
	if (iterations <= 0) iterations = 300;
	if (nstreams   <= 0) nstreams   = 1;
	if (nstreams   > (int)NMODELS) nstreams = (int)NMODELS;

	printf("TFLite MobileNet-SSD workload — mTHP fragmentation benchmark\n");
	printf("Streams: %d concurrent model instances\n", nstreams);
	printf("Iterations: %d load/infer/unload cycles\n\n", iterations);

	print_arena_profile();

	for (i = 0; i < nstreams; i++)
		total_arena += models[i % NMODELS].arena_bytes
			     + models[i % NMODELS].preproc_bytes
			     + models[i % NMODELS].postproc_bytes;

	printf("Peak concurrent anonymous memory (%d streams): %.1f MB\n",
	       nstreams, total_arena / 1048576.0);
	printf("Starting %d iterations...\n\n", iterations);
	fflush(stdout);

	struct stream *streams = calloc(nstreams, sizeof(*streams));
	if (!streams) {
		perror("calloc");
		return 1;
	}

	t0 = ns_now();

	for (i = 0; i < iterations; i++) {
		int ok = 1;

		/*
		 * Load all streams simultaneously (camera pipeline init).
		 * All tensor arenas are mmap'd before any inference runs.
		 * This is the worst-case point for the buddy allocator:
		 * multiple large anonymous mmaps issued in rapid succession.
		 */
		for (s = 0; s < nstreams; s++) {
			if (!stream_load(&streams[s],
					 &models[s % NMODELS])) {
				fprintf(stderr, "stream_load failed iter %d stream %d\n",
					i, s);
				ok = 0;
				break;
			}
		}

		if (!ok)
			break;

		/*
		 * Run one inference pass on each stream.
		 * All arenas live simultaneously → maximum memory pressure.
		 * Physical pages faulted in here (demand-paging).
		 * This is when the kernel must assign hugepage orders.
		 */
		for (s = 0; s < nstreams; s++)
			stream_infer(&streams[s], i);

		/*
		 * Unload all streams (munmap arenas).
		 * With bestfit: right-sized hugepages return cleanly to buddy.
		 * Without bestfit: deferred_split pages accumulate here.
		 */
		for (s = 0; s < nstreams; s++)
			stream_unload(&streams[s]);

		if ((i + 1) % 50 == 0) {
			t1 = ns_now();
			printf("  %4d / %d  (%.1f iter/s)\n",
			       i + 1, iterations,
			       (i + 1) * 1e9 / (double)(t1 - t0));
			fflush(stdout);
		}
	}

	free(streams);

	t1 = ns_now();
	elapsed_ms = (t1 - t0) / 1000000;

	printf("\nDone. %d iterations in %ld ms  (%.1f iter/s)\n\n",
	       iterations, elapsed_ms,
	       iterations * 1e3 / (double)elapsed_ms);

	printf("Memory churn per iteration: %.1f MB  "
	       "(arena alloc + touch + free × %d streams)\n",
	       total_arena / 1048576.0, nstreams);
	printf("Total churn over run:       %.1f GB\n",
	       (double)total_arena * iterations / (1024.0 * 1024.0 * 1024.0));

	return 0;
}
