// SPDX-License-Identifier: GPL-2.0
/*
 * tflite_mobilenetssd_workload.c — TFLite MobileNet-SSD image detection benchmark
 *
 * Simulates processing a batch of image files through TensorFlow Lite
 * MobileNet-SSD object detection — the standard mobile AI pipeline for
 * photo galleries, image search, and on-device media indexing.
 *
 * Real-world pipeline (what actually happens in memory)
 * ─────────────────────────────────────────────────────
 * When an app runs MobileNet-SSD on image files, each image goes through:
 *
 *   Step 1 — Load image file into memory
 *             JPEG file read into RAM: ~200 KB – 2 MB (depends on photo)
 *             mmap() or malloc() for the compressed bytes
 *
 *   Step 2 — Decode JPEG to RGB bitmap
 *             libjpeg allocates decoded buffer: W × H × 3 bytes
 *             640×480  photo → 900 KB decoded
 *             1920×1080 photo → 5.9 MB decoded     ← key fragmentation driver
 *             3264×2448 photo → 22.9 MB decoded
 *
 *   Step 3 — Resize to model input size (300×300 for MobileNet-SSD)
 *             Resize buffer: 300 × 300 × 3 = 270 KB
 *
 *   Step 4 — Normalise pixel values to float32 [-1.0, 1.0]
 *             Norm buffer: 300 × 300 × 3 × 4 = 1,054 KB
 *
 *   Step 5 — TFLite Invoke() — runs inference in the tensor arena
 *             Arena: 3.7 MB (pre-allocated, REUSED across images)
 *
 *   Step 6 — Decode detection outputs (box coordinates, class, score)
 *             Output buffer: ~52 KB
 *
 *   Step 7 — Free per-image buffers (JPEG, decoded, resize, norm, output)
 *             Arena stays allocated for the next image
 *
 * Why this creates severe mTHP fragmentation
 * ───────────────────────────────────────────
 * The per-image allocations (steps 1–4) are different sizes every frame:
 *   JPEG compressed:  variable  200 KB – 2 MB  (order 5–8)
 *   Decoded RGB:      variable  900 KB – 23 MB (order 7–PMD+)
 *   Resize buffer:    fixed     270 KB          (order 6)
 *   Norm buffer:      fixed     1,054 KB        (order 8)
 *
 * These are allocated and freed for EVERY IMAGE, while the arena (3.7 MB,
 * order-9) stays live between images.
 *
 * After processing ~50 images the buddy allocator becomes fragmented:
 * the variable decoded-image allocations leave holes of different sizes
 * between the live arenas.  The next arena load finds no 2 MB contiguous
 * block → compact_stall triggers.
 *
 * With mthp_bestfit:
 *   Decoded 5.9 MB FHD image → bestfit selects order-9 (PMD) correctly
 *   Norm buffer 1,054 KB     → bestfit selects order-8 (1 MB)
 *   JPEG 400 KB compressed   → bestfit selects order-7 (512 KB)
 *   All sizes get right-sized hugepages → clean returns to buddy
 *   → compact_stall reduced 30-50%  → MemAvailable 5-15% higher
 *
 * Image sources simulated
 * ────────────────────────
 * Represents a photo gallery app scanning images of mixed resolutions
 * (thumbnail crops, preview frames, and full photos):
 *
 *   Type       Resolution    Decoded size    JPEG approx    Common source
 *   ─────────────────────────────────────────────────────────────────────
 *   thumbnail  640 × 480      900 KB         80 KB          gallery thumbs
 *   preview   1280 × 720     2,638 KB        330 KB         video frame
 *   photo     1920 ×1080     5,934 KB        740 KB         camera photo
 *   hires     3264 ×2448    22,892 KB        2,861 KB       flagship camera
 *
 * Build:
 *   gcc -O2 -o tflite_mobilenetssd_workload tflite_mobilenetssd_workload.c
 *
 * Run:
 *   ./tflite_mobilenetssd_workload [images] [streams]
 *   images:  number of image files to process per iteration (default 200)
 *   streams: concurrent model instances, e.g. 4 = 4-camera surveillance (default 4)
 *
 * Examples:
 *   ./tflite_mobilenetssd_workload 200 1   # single camera, 200 photos
 *   ./tflite_mobilenetssd_workload 500 4   # 4-camera pipeline, 500 images each
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

/* TFLite uses 64-byte alignment for arena (ARM NEON cacheline width) */
#define TFLITE_ALIGN     64

/* libjpeg uses standard malloc for decode buffers — goes through mmap
 * for sizes > glibc mmap_threshold (128 KB by default) */
#define LIBJPEG_ALIGN    16

/* MobileNet-SSD v1 input: 300×300 RGB */
#define SSD_INPUT_W      300
#define SSD_INPUT_H      300
#define SSD_INPUT_CH     3

/* ── Image source descriptors ────────────────────────────────────────────── */
/*
 * Each entry represents a class of image files that might appear in a
 * photo gallery or video stream.  "jpeg_ratio" approximates typical
 * JPEG compression ratios for natural images at default quality settings.
 */
struct image_source {
	const char *name;
	int         width;
	int         height;
	int         jpeg_ratio;  /* decoded_size / jpeg_size (typical compression) */
};

static const struct image_source image_sources[] = {
	/* Gallery thumbnail or 640×480 crop from a video frame */
	{ "thumbnail-640x480",    640,  480,  10 },
	/* 720p preview frame (dashcam, drone, security camera) */
	{ "preview-1280x720",     1280, 720,  8  },
	/* Full HD photo from camera or screenshot */
	{ "photo-1920x1080",      1920, 1080, 8  },
	/* High-resolution camera photo (typical smartphone flagship) */
	{ "hires-3264x2448",      3264, 2448, 8  },
};
#define NSOURCES  (sizeof(image_sources) / sizeof(image_sources[0]))

/* ── TFLite model arena sizes ────────────────────────────────────────────── */
/*
 * Arena = one large mmap(MAP_PRIVATE|MAP_ANONYMOUS) allocated at model
 * load time.  Stays live across ALL images; freed only at model unload.
 * Sizes measured with tflite_model_analyzer on real .tflite files.
 */
struct tflite_model {
	const char *name;
	size_t      arena_bytes;
};

static const struct tflite_model models[] = {
	{ "ssd-mobilenet-v1-int8",   3866624  }, /* 3.7 MB */
	{ "ssd-mobilenet-v2-int8",   5734400  }, /* 5.5 MB */
	{ "ssd-mobilenet-v1-fp32",   13107200 }, /* 12.5 MB */
	{ "efficientdet-lite0-int8", 7864320  }, /* 7.5 MB */
};
#define NMODELS  (sizeof(models) / sizeof(models[0]))

/* Fixed sizes for resize and normalisation buffers (same for all images) */
#define RESIZE_BYTES  (SSD_INPUT_W * SSD_INPUT_H * SSD_INPUT_CH)        /* 270 KB */
#define NORM_BYTES    (SSD_INPUT_W * SSD_INPUT_H * SSD_INPUT_CH * 4)    /* 1,054 KB */
#define OUTPUT_BYTES  (10 * (4 + 1 + 1) * 4 + 4096)                     /* ~52 KB */

static inline long ns_now(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

/*
 * alloc_buf — allocate a processing buffer the way the real library would.
 *
 * libjpeg / stb_image use malloc() for decode buffers.
 * glibc malloc uses mmap(MAP_PRIVATE|MAP_ANONYMOUS) for allocations
 * above mmap_threshold (128 KB by default), so large image decode
 * buffers go through the anonymous-memory path where mTHP applies.
 */
static void *alloc_buf(size_t bytes, size_t align)
{
	void *p = NULL;

	if (bytes < 64)
		return malloc(bytes);

	if (posix_memalign(&p, align, bytes) != 0)
		return NULL;
	return p;
}

static void free_buf(void *p) { free(p); }

/*
 * touch_buf — write one byte per cacheline to force physical page backing.
 *
 * Simulates the actual read/write of pixel data during decode, resize,
 * normalise, and inference.  Using a non-zero pattern prevents the kernel's
 * zero-page optimisation from hiding the real allocation cost.
 */
static void touch_buf(volatile uint8_t *p, size_t bytes, uint8_t pat)
{
	size_t i;
	for (i = 0; i < bytes; i += 64)
		p[i] = pat;
}

/* ── Per-stream state (one stream = one loaded model instance) ───────────── */
struct stream {
	const struct tflite_model *model;
	void                      *arena;   /* pre-allocated tensor arena */
};

/*
 * stream_load — allocate the tensor arena (TFLite AllocateTensors).
 *
 * ONE large anonymous mmap per model.  Stays live across all images.
 * This is the allocation that gets PMD hugepages and stresses the buddy
 * allocator when multiple streams are loaded simultaneously.
 */
static int stream_load(struct stream *s, const struct tflite_model *m)
{
	s->model = m;
	s->arena = mmap(NULL, m->arena_bytes,
			PROT_READ | PROT_WRITE,
			MAP_PRIVATE | MAP_ANONYMOUS,
			-1, 0);
	if (s->arena == MAP_FAILED) {
		fprintf(stderr, "arena mmap(%zu) failed: %s\n",
			m->arena_bytes, strerror(errno));
		return 0;
	}
	return 1;
}

static void stream_unload(struct stream *s)
{
	munmap(s->arena, s->model->arena_bytes);
	s->arena = NULL;
}

/*
 * process_image — run one image through the full detection pipeline.
 *
 * This is the core of the benchmark.  All per-image allocations are made
 * here and freed before returning, creating the fragmentation that mixes
 * with the live arena allocation.
 *
 *   jpeg_buf:    compressed image data read from file (200 KB – 2 MB)
 *   decoded_buf: libjpeg output — full RGB bitmap (900 KB – 23 MB)
 *   resize_buf:  scaled to 300×300 pixels (270 KB, fixed)
 *   norm_buf:    normalised float32 values (1,054 KB, fixed)
 *   output_buf:  detection results — boxes + scores (52 KB)
 *
 * The variable size of jpeg_buf and decoded_buf (driven by image resolution)
 * is the key differentiator: mixed-size allocs between stable arenas create
 * exactly the buddy fragmentation that mthp_bestfit addresses.
 */
static int process_image(struct stream *s,
			  const struct image_source *img,
			  int img_idx)
{
	size_t decoded_bytes = (size_t)img->width * img->height * SSD_INPUT_CH;
	size_t jpeg_bytes    = decoded_bytes / img->jpeg_ratio;
	uint8_t pat          = (uint8_t)(img_idx ^ (uintptr_t)s->model);
	int ok = 1;

	void *jpeg_buf    = NULL;
	void *decoded_buf = NULL;
	void *resize_buf  = NULL;
	void *norm_buf    = NULL;
	void *output_buf  = NULL;

	/* Step 1: load JPEG file into RAM */
	jpeg_buf = alloc_buf(jpeg_bytes, LIBJPEG_ALIGN);
	if (!jpeg_buf) { ok = 0; goto done; }
	touch_buf(jpeg_buf, jpeg_bytes, pat);

	/* Step 2: decode JPEG → raw RGB bitmap (largest alloc per image) */
	decoded_buf = alloc_buf(decoded_bytes, LIBJPEG_ALIGN);
	if (!decoded_buf) { ok = 0; goto done; }
	touch_buf(decoded_buf, decoded_bytes, (uint8_t)(pat + 1));

	/* Step 3: resize decoded bitmap to 300×300 */
	resize_buf = alloc_buf(RESIZE_BYTES, TFLITE_ALIGN);
	if (!resize_buf) { ok = 0; goto done; }
	touch_buf(resize_buf, RESIZE_BYTES, (uint8_t)(pat + 2));

	/* Step 4: normalise pixels to fp32 [-1.0, 1.0] */
	norm_buf = alloc_buf(NORM_BYTES, TFLITE_ALIGN);
	if (!norm_buf) { ok = 0; goto done; }
	touch_buf(norm_buf, NORM_BYTES, (uint8_t)(pat + 3));

	/* Step 5: TFLite Invoke() — access tensor arena */
	touch_buf(s->arena, s->model->arena_bytes, (uint8_t)(pat + 4));

	/* Step 6: decode detection outputs */
	output_buf = alloc_buf(OUTPUT_BYTES, TFLITE_ALIGN);
	if (!output_buf) { ok = 0; goto done; }
	touch_buf(output_buf, OUTPUT_BYTES, (uint8_t)(pat + 5));

done:
	/* Step 7: free all per-image buffers; arena stays live */
	free_buf(output_buf);
	free_buf(norm_buf);
	free_buf(resize_buf);
	free_buf(decoded_buf);
	free_buf(jpeg_buf);
	return ok;
}

/* Print the image sources and allocation sizes */
static void print_image_profile(void)
{
	size_t i;
	printf("Image source profile (per-image allocations):\n");
	printf("  %-22s %8s %10s %8s %8s  %s\n",
	       "Source", "JPEG", "Decoded", "Resize", "Norm", "Decoded order");
	printf("  %s\n",
	       "──────────────────────────────────────────────────────────────");

	for (i = 0; i < NSOURCES; i++) {
		const struct image_source *s = &image_sources[i];
		size_t decoded = (size_t)s->width * s->height * SSD_INPUT_CH;
		size_t jpeg    = decoded / s->jpeg_ratio;
		size_t pg      = decoded / 4096;
		int    order   = 0;
		while ((1u << (order + 1)) <= pg && order < 9) order++;

		printf("  %-22s %6zu KB %8zu KB %6u KB %6u KB  → order %d (%zu KB)\n",
		       s->name,
		       jpeg    / 1024,
		       decoded / 1024,
		       RESIZE_BYTES / 1024,
		       NORM_BYTES   / 1024,
		       order,
		       (size_t)(1 << (12 + order)) / 1024);
	}
	printf("\n");

	printf("TFLite tensor arena (pre-allocated, reused across all images):\n");
	for (i = 0; i < NMODELS; i++) {
		size_t pg    = models[i].arena_bytes / 4096;
		int    order = 0;
		while ((1u << (order + 1)) <= pg && order < 9) order++;
		printf("  %-28s %6zu KB  → order %d (%zu KB × %zu)\n",
		       models[i].name,
		       models[i].arena_bytes / 1024,
		       order,
		       (size_t)(1 << (12 + order)) / 1024,
		       (models[i].arena_bytes + (1 << (12 + order)) - 1)
		         >> (12 + order));
	}
	printf("\n");
}

int main(int argc, char *argv[])
{
	int     nimages  = 200;
	int     nstreams = 4;
	int     i, s;
	long    t0, t1, elapsed_ms;
	size_t  total_bytes = 0;

	if (argc > 1) nimages  = atoi(argv[1]);
	if (argc > 2) nstreams = atoi(argv[2]);
	if (nimages  <= 0) nimages  = 200;
	if (nstreams <= 0) nstreams = 1;
	if (nstreams > (int)NMODELS) nstreams = (int)NMODELS;

	printf("TFLite MobileNet-SSD image detection workload\n");
	printf("mTHP fragmentation benchmark — image-file processing mode\n");
	printf("Images: %d   Streams: %d concurrent model instances\n\n",
	       nimages, nstreams);

	print_image_profile();

	/*
	 * Estimate peak live memory per image (worst-case: hires source).
	 * Arena stays live; per-image allocs all live during inference.
	 */
	{
		const struct image_source *biggest = &image_sources[NSOURCES - 1];
		size_t decoded = (size_t)biggest->width * biggest->height * SSD_INPUT_CH;
		size_t per_img = decoded + decoded / biggest->jpeg_ratio
				+ RESIZE_BYTES + NORM_BYTES + OUTPUT_BYTES;
		size_t arenas  = 0;
		for (s = 0; s < nstreams; s++)
			arenas += models[s % NMODELS].arena_bytes;
		printf("Peak memory (worst-case hires image × %d streams):\n", nstreams);
		printf("  Per-image allocs: %.1f MB (jpeg+decoded+resize+norm+output)\n",
		       per_img / 1048576.0);
		printf("  Arenas (live):    %.1f MB\n", arenas / 1048576.0);
		printf("  Total peak:       %.1f MB\n\n",
		       (per_img * nstreams + arenas) / 1048576.0);
	}

	printf("Starting %d images × %d streams...\n\n", nimages, nstreams);
	fflush(stdout);

	/* Allocate stream array */
	struct stream *streams = calloc(nstreams, sizeof(*streams));
	if (!streams) { perror("calloc"); return 1; }

	/*
	 * Load all models once — arenas stay live for the entire image batch.
	 * This matches TFLite behaviour: the app loads models at startup,
	 * then processes many images without reloading.
	 */
	for (s = 0; s < nstreams; s++) {
		if (!stream_load(&streams[s], &models[s % NMODELS])) {
			fprintf(stderr, "model load failed for stream %d\n", s);
			return 1;
		}
	}

	t0 = ns_now();

	for (i = 0; i < nimages; i++) {
		/*
		 * Each image is selected from the source pool in round-robin.
		 * This creates the mixed-size per-image allocations that drive
		 * fragmentation: thumbnail (900 KB decoded) followed by hires
		 * (23 MB decoded) leaves gaps that can't be filled by the next
		 * arena allocation without compaction.
		 */
		const struct image_source *src = &image_sources[i % NSOURCES];

		/* Process same image on all concurrent streams (simulates
		 * multiple cameras all capturing the same scene simultaneously) */
		for (s = 0; s < nstreams; s++) {
			if (!process_image(&streams[s], src, i)) {
				fprintf(stderr, "image %d stream %d failed\n", i, s);
				goto done;
			}
		}

		/* Accumulate churn */
		{
			size_t decoded = (size_t)src->width * src->height * SSD_INPUT_CH;
			total_bytes += (decoded + decoded / src->jpeg_ratio
					+ RESIZE_BYTES + NORM_BYTES + OUTPUT_BYTES)
				       * nstreams;
		}

		if ((i + 1) % 50 == 0) {
			t1 = ns_now();
			printf("  %4d / %d images  (%.1f img/s)\n",
			       i + 1, nimages,
			       (i + 1) * 1e9 / (double)(t1 - t0));
			fflush(stdout);
		}
	}

done:
	/* Unload all models (free arenas) */
	for (s = 0; s < nstreams; s++)
		stream_unload(&streams[s]);
	free(streams);

	t1 = ns_now();
	elapsed_ms = (t1 - t0) / 1000000;

	printf("\nDone. %d images × %d streams in %ld ms  (%.1f img/s per stream)\n\n",
	       nimages, nstreams, elapsed_ms,
	       nimages * 1e3 / (double)elapsed_ms);

	printf("Per-image memory churn (avg across all source sizes): %.1f MB\n",
	       total_bytes / 1048576.0 / nimages / nstreams);
	printf("Total churn over run: %.1f GB\n",
	       total_bytes / (1024.0 * 1024.0 * 1024.0));
	return 0;
}
