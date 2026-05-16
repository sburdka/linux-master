// SPDX-License-Identifier: GPL-2.0
/*
 * tflite_mobilenetssd_workload.c — TFLite MobileNet-SSD mTHP benchmark
 *
 * Simulates the memory allocation pipeline of TensorFlow Lite running
 * MobileNet-SSD object detection on image files extracted from video.
 *
 * Two operating modes
 * ────────────────────
 * Synthetic mode (default, no extra files needed):
 *   Simulates processing images of four standard resolutions in round-robin,
 *   representing a mixed-resolution image gallery or multi-camera feed.
 *
 * Frame mode (--frames <dir>, recommended for real demo):
 *   Reads actual PPM frame files extracted from a video (e.g. Big Buck Bunny
 *   by Blender Foundation, CC BY 3.0).  Uses the real per-frame dimensions
 *   to drive the JPEG decode buffer allocations, giving accurate memory sizes.
 *
 *   Prepare frames with ffmpeg:
 *     mkdir -p frames/1080p frames/720p frames/480p
 *     ffmpeg -i bigbuckbunny.mp4 -r 1 -vf scale=1920:1080 frames/1080p/f%04d.ppm
 *     ffmpeg -i bigbuckbunny.mp4 -r 1 -vf scale=1280:720  frames/720p/f%04d.ppm
 *     ffmpeg -i bigbuckbunny.mp4 -r 2 -vf scale=640:480   frames/480p/f%04d.ppm
 *
 *   Then run:
 *     ./tflite_mobilenetssd_workload --frames frames/ --streams 4
 *
 * Memory pipeline per frame
 * ─────────────────────────
 * Each frame goes through (real TFLite + libjpeg pipeline):
 *
 *   jpeg_buf    compressed frame from disk       ~ 80 – 2,900 KB  (order 4–9)
 *   decoded_buf libjpeg RGB decode output        ~ 900 – 23,000 KB (order 7–9+)
 *   resize_buf  scaled to 300×300 (fixed)          263 KB           (order 6)
 *   norm_buf    float32 normalised input (fixed)  1,054 KB           (order 8)
 *   arena       TFLite tensor arena (REUSED)      3,700 KB           (order 9)
 *   output_buf  detection results (fixed)            52 KB
 *
 *   All six are live simultaneously during inference.
 *   jpeg_buf, decoded_buf, resize_buf, norm_buf, output_buf freed after each frame.
 *   arena stays live for the lifetime of the loaded model.
 *
 * The decoded_buf for a 1920×1080 frame is 5.9 MB (order-9 = PMD territory).
 * After ~50 frames the buddy allocator has many 6 MB holes from freed decode
 * buffers.  The next arena allocation (3.7 MB) finds no 2 MB contiguous block
 * → compact_stall fires → inference latency spikes 15–40 ms.
 *
 * With mthp_bestfit: each size class gets its own best-fit hugepage order →
 * clean returns to buddy → no fragmentation → no compact_stall.
 *
 * Build:
 *   gcc -O2 -o tflite_mobilenetssd_workload tflite_mobilenetssd_workload.c
 *
 * Run (synthetic):
 *   ./tflite_mobilenetssd_workload [frames] [streams]
 *
 * Run (real video frames):
 *   ./tflite_mobilenetssd_workload --frames <dir> [--streams N] [--repeat N]
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
#include <dirent.h>
#include <sys/stat.h>

/* TFLite uses 64-byte alignment (ARM NEON cacheline) */
#define TFLITE_ALIGN      64
/* libjpeg internal alignment */
#define LIBJPEG_ALIGN     16
/* Typical JPEG compression ratio for natural video frames */
#define JPEG_RATIO        8

/* MobileNet-SSD v1 fixed input dimensions */
#define SSD_W             300
#define SSD_H             300
#define SSD_C             3
#define RESIZE_BYTES      ((size_t)(SSD_W) * SSD_H * SSD_C)         /* 263 KB */
#define NORM_BYTES        ((size_t)(SSD_W) * SSD_H * SSD_C * 4)     /* 1,054 KB */
#define OUTPUT_BYTES      ((size_t)10 * (4 + 2) * 4 + 4096)         /* ~52 KB */

/* ── TFLite tensor arena sizes ───────────────────────────────────────────── */
struct tflite_model {
	const char *name;
	size_t      arena_bytes;
};
static const struct tflite_model models[] = {
	{ "ssd-mobilenet-v1-int8",    3866624  }, /* 3.7 MB */
	{ "ssd-mobilenet-v2-int8",    5734400  }, /* 5.5 MB */
	{ "ssd-mobilenet-v1-fp32",    13107200 }, /* 12.5 MB */
	{ "efficientdet-lite0-int8",  7864320  }, /* 7.5 MB */
};
#define NMODELS  (sizeof(models) / sizeof(models[0]))

/* ── Synthetic fallback image sources ────────────────────────────────────── */
struct image_source {
	const char *name;
	int         width, height;
	int         jpeg_ratio;
};
static const struct image_source synth_sources[] = {
	{ "thumbnail-640x480",    640,  480,  10 },
	{ "preview-1280x720",     1280, 720,   8 },
	{ "photo-1920x1080",      1920, 1080,  8 },
	{ "hires-3264x2448",      3264, 2448,  8 },
};
#define NSOURCES  (sizeof(synth_sources) / sizeof(synth_sources[0]))

/* ── Frame descriptor (used in frame-dir mode) ───────────────────────────── */
struct frame {
	char   path[768];
	int    width, height;   /* from PPM header; 0 = use file-size heuristic */
	size_t file_bytes;      /* actual file size on disk */
};

static inline long ns_now(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

static void *alloc_buf(size_t bytes, size_t align)
{
	void *p = NULL;
	if (bytes < 128)
		return malloc(bytes);
	if (posix_memalign(&p, align, bytes) != 0)
		return NULL;
	return p;
}
static void free_buf(void *p) { free(p); }

static void touch_buf(volatile uint8_t *p, size_t bytes, uint8_t pat)
{
	size_t i;
	for (i = 0; i < bytes; i += 64)
		p[i] = pat;
}

/* ── Per-stream runtime state ────────────────────────────────────────────── */
struct stream {
	const struct tflite_model *model;
	void                      *arena;
};

static int stream_load(struct stream *s, const struct tflite_model *m)
{
	s->model = m;
	s->arena = mmap(NULL, m->arena_bytes,
			PROT_READ | PROT_WRITE,
			MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
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
 * process_frame - simulate one frame through the full detection pipeline.
 *
 * jpeg_bytes:    file size of the compressed frame on disk
 * decoded_bytes: W × H × 3 after JPEG decode (largest per-frame allocation)
 */
static int process_frame(struct stream *s,
			  size_t jpeg_bytes, size_t decoded_bytes,
			  int frame_idx)
{
	uint8_t pat = (uint8_t)(frame_idx ^ (uintptr_t)s->model);
	void *jpeg_buf = NULL, *decoded_buf = NULL;
	void *resize_buf = NULL, *norm_buf = NULL, *output_buf = NULL;
	int ok = 1;

	/* Step 1: read compressed frame from disk into RAM */
	jpeg_buf = alloc_buf(jpeg_bytes, LIBJPEG_ALIGN);
	if (!jpeg_buf) { ok = 0; goto done; }
	touch_buf(jpeg_buf, jpeg_bytes, pat);

	/* Step 2: JPEG decode → full RGB bitmap (largest alloc) */
	decoded_buf = alloc_buf(decoded_bytes, LIBJPEG_ALIGN);
	if (!decoded_buf) { ok = 0; goto done; }
	touch_buf(decoded_buf, decoded_bytes, (uint8_t)(pat + 1));

	/* Step 3: resize to 300×300 */
	resize_buf = alloc_buf(RESIZE_BYTES, TFLITE_ALIGN);
	if (!resize_buf) { ok = 0; goto done; }
	touch_buf(resize_buf, RESIZE_BYTES, (uint8_t)(pat + 2));

	/* Step 4: normalise pixels to fp32 */
	norm_buf = alloc_buf(NORM_BYTES, TFLITE_ALIGN);
	if (!norm_buf) { ok = 0; goto done; }
	touch_buf(norm_buf, NORM_BYTES, (uint8_t)(pat + 3));

	/* Step 5: TFLite Invoke() using the pre-allocated arena */
	touch_buf(s->arena, s->model->arena_bytes, (uint8_t)(pat + 4));

	/* Step 6: decode detection output boxes */
	output_buf = alloc_buf(OUTPUT_BYTES, TFLITE_ALIGN);
	if (!output_buf) { ok = 0; goto done; }
	touch_buf(output_buf, OUTPUT_BYTES, (uint8_t)(pat + 5));

done:
	/* Step 7: free all per-frame buffers; arena stays live */
	free_buf(output_buf);
	free_buf(norm_buf);
	free_buf(resize_buf);
	free_buf(decoded_buf);
	free_buf(jpeg_buf);
	return ok;
}

/* ── PPM header parser ────────────────────────────────────────────────────── */
/*
 * Reads width and height from a PPM file header (P6 format).
 * Format: "P6\n[# comment\n]<W> <H>\n<MAXVAL>\n<raw data>"
 * Returns 1 on success, 0 on failure.
 */
static int ppm_dimensions(const char *path, int *w, int *h)
{
	FILE *f = fopen(path, "rb");
	char  line[256];
	int   ok = 0;

	if (!f)
		return 0;

	/* Magic number */
	if (!fgets(line, sizeof(line), f) || strncmp(line, "P6", 2) != 0)
		goto out;

	/* Skip comment lines */
	while (fgets(line, sizeof(line), f)) {
		if (line[0] != '#')
			break;
	}

	/* Width and height on same line */
	if (sscanf(line, "%d %d", w, h) == 2 && *w > 0 && *h > 0)
		ok = 1;
out:
	fclose(f);
	return ok;
}

/* ── Frame directory loader ───────────────────────────────────────────────── */
/*
 * Scans <dirpath> recursively (one level of subdirectories) for .ppm and
 * .jpg files.  For PPM files, reads the exact frame dimensions.  For JPEG
 * files, estimates decoded size from the file size × JPEG_RATIO.
 *
 * Returns a heap-allocated array of frame descriptors, sets *nframes.
 * Caller frees with free().
 */
static struct frame *load_frame_list(const char *dirpath, int *nframes)
{
	struct frame *list   = NULL;
	int           count  = 0;
	int           cap    = 256;
	DIR          *top;
	struct dirent *de;

	list = malloc(cap * sizeof(*list));
	if (!list)
		return NULL;

	top = opendir(dirpath);
	if (!top) {
		perror(dirpath);
		free(list);
		return NULL;
	}

	/*
	 * Two-level scan: top directory + one level of subdirectories.
	 * This matches the typical layout:
	 *   frames/
	 *     1080p/f0001.ppm  f0002.ppm ...
	 *     720p/f0001.ppm   ...
	 *     480p/f0001.ppm   ...
	 */
	while ((de = readdir(top)) != NULL) {
		char subpath[512];
		struct stat st;

		if (de->d_name[0] == '.')
			continue;

		snprintf(subpath, sizeof(subpath), "%s/%s", dirpath, de->d_name);
		if (stat(subpath, &st) != 0)
			continue;

		/* Process files in the top directory */
		if (S_ISREG(st.st_mode)) {
			const char *ext = strrchr(de->d_name, '.');
			if (!ext || (strcmp(ext, ".ppm") != 0 &&
				     strcmp(ext, ".jpg") != 0 &&
				     strcmp(ext, ".jpeg") != 0))
				continue;

			if (count >= cap) {
				cap *= 2;
				list = realloc(list, cap * sizeof(*list));
				if (!list) { closedir(top); return NULL; }
			}

			struct frame *fr = &list[count];
			snprintf(fr->path, sizeof(fr->path), "%s", subpath);
			fr->file_bytes = (size_t)st.st_size;
			fr->width = fr->height = 0;

			if (strcmp(ext, ".ppm") == 0)
				ppm_dimensions(subpath, &fr->width, &fr->height);

			count++;

		/* Recurse one level into subdirectories */
		} else if (S_ISDIR(st.st_mode)) {
			DIR *sub = opendir(subpath);
			struct dirent *sde;
			if (!sub) continue;

			while ((sde = readdir(sub)) != NULL) {
				char fpath[768];
				struct stat fst;

				if (sde->d_name[0] == '.') continue;
				snprintf(fpath, sizeof(fpath), "%s/%s",
					 subpath, sde->d_name);
				if (stat(fpath, &fst) != 0 || !S_ISREG(fst.st_mode))
					continue;

				const char *ext2 = strrchr(sde->d_name, '.');
				if (!ext2 || (strcmp(ext2, ".ppm") != 0 &&
					      strcmp(ext2, ".jpg") != 0 &&
					      strcmp(ext2, ".jpeg") != 0))
					continue;

				if (count >= cap) {
					cap *= 2;
					list = realloc(list, cap * sizeof(*list));
					if (!list) {
						closedir(sub);
						closedir(top);
						return NULL;
					}
				}

				struct frame *fr = &list[count];
				snprintf(fr->path, sizeof(fr->path), "%s", fpath);
				fr->file_bytes = (size_t)fst.st_size;
				fr->width = fr->height = 0;
				if (strcmp(ext2, ".ppm") == 0)
					ppm_dimensions(fpath, &fr->width, &fr->height);
				count++;
			}
			closedir(sub);
		}
	}
	closedir(top);

	*nframes = count;
	return list;
}

/* ── Print helpers ───────────────────────────────────────────────────────── */
static void print_model_arenas(int nstreams)
{
	int i;
	printf("TFLite arena (pre-allocated, reused across all frames):\n");
	for (i = 0; i < nstreams && i < (int)NMODELS; i++) {
		size_t pg = models[i].arena_bytes / 4096;
		int order = 0;
		while ((1u << (order + 1)) <= pg && order < 9) order++;
		printf("  Stream %d  %-28s %6zu KB  → order %d (%zu KB × %zu)\n",
		       i, models[i].name,
		       models[i].arena_bytes / 1024,
		       order,
		       (size_t)(1 << (12 + order)) / 1024,
		       (models[i].arena_bytes + (1 << (12 + order)) - 1)
		         >> (12 + order));
	}
	printf("\n");
}

/* ── Synthetic mode ──────────────────────────────────────────────────────── */
static void run_synthetic(int nimages, int nstreams)
{
	int s, i;
	long t0, t1;
	size_t total_bytes = 0;

	printf("Image source profile (synthetic round-robin):\n");
	for (i = 0; i < (int)NSOURCES; i++) {
		const struct image_source *src = &synth_sources[i];
		size_t decoded = (size_t)src->width * src->height * SSD_C;
		size_t pg = decoded / 4096;
		int order = 0;
		while ((1u << (order + 1)) <= pg && order < 9) order++;
		printf("  %-22s decoded %6zu KB → bestfit order %d\n",
		       src->name, decoded / 1024, order);
	}
	printf("\n");
	print_model_arenas(nstreams);

	struct stream *streams = calloc(nstreams, sizeof(*streams));
	if (!streams) { perror("calloc"); return; }

	for (s = 0; s < nstreams; s++) {
		if (!stream_load(&streams[s], &models[s % NMODELS])) {
			fprintf(stderr, "model load failed\n");
			free(streams);
			return;
		}
	}

	t0 = ns_now();
	for (i = 0; i < nimages; i++) {
		const struct image_source *src = &synth_sources[i % NSOURCES];
		size_t decoded = (size_t)src->width * src->height * SSD_C;
		size_t jpeg    = decoded / src->jpeg_ratio;

		for (s = 0; s < nstreams; s++) {
			if (!process_frame(&streams[s], jpeg, decoded, i)) {
				fprintf(stderr, "frame %d stream %d failed\n", i, s);
				goto done_synth;
			}
		}
		total_bytes += (jpeg + decoded + RESIZE_BYTES + NORM_BYTES
				+ OUTPUT_BYTES) * nstreams;

		if ((i + 1) % 50 == 0) {
			t1 = ns_now();
			printf("  %4d / %d  (%.1f img/s)\n",
			       i + 1, nimages,
			       (i + 1) * 1e9 / (double)(t1 - t0));
			fflush(stdout);
		}
	}
done_synth:
	for (s = 0; s < nstreams; s++) stream_unload(&streams[s]);
	free(streams);
	t1 = ns_now();
	printf("\nDone. %d images × %d streams in %ld ms  (%.1f img/s)\n",
	       nimages, nstreams, (t1 - t0) / 1000000,
	       nimages * 1e3 / ((t1 - t0) / 1e6));
	printf("Total memory churn: %.1f GB\n",
	       total_bytes / (1024.0 * 1024.0 * 1024.0));
}

/* ── Frame-directory mode ────────────────────────────────────────────────── */
static void run_from_frames(const char *dirpath, int nstreams, int repeat)
{
	int nframes = 0, s, i, r;
	long t0 = 0, t1;
	size_t total_bytes = 0;
	size_t largest_decoded = 0;

	printf("Loading frame list from: %s\n", dirpath);
	struct frame *frames = load_frame_list(dirpath, &nframes);
	if (!frames || nframes == 0) {
		fprintf(stderr, "No .ppm/.jpg frames found in %s\n", dirpath);
		free(frames);
		return;
	}
	printf("Found %d frames\n\n", nframes);

	/* Print frame size distribution */
	printf("Frame size distribution:\n");
	{
		size_t min_d = SIZE_MAX, max_d = 0, total_d = 0;
		for (i = 0; i < nframes; i++) {
			size_t d;
			if (frames[i].width > 0 && frames[i].height > 0)
				d = (size_t)frames[i].width * frames[i].height * SSD_C;
			else
				d = frames[i].file_bytes * JPEG_RATIO;
			if (d < min_d) min_d = d;
			if (d > max_d) max_d = d;
			if (d > largest_decoded) largest_decoded = d;
			total_d += d;
		}
		printf("  Min decoded: %zu KB   Max decoded: %zu KB   Avg: %zu KB\n",
		       min_d / 1024, max_d / 1024, total_d / 1024 / nframes);

		int pg = (int)(max_d / 4096);
		int order = 0;
		while ((1 << (order + 1)) <= pg && order < 9) order++;
		printf("  Largest frame order: %d (%zu KB hugepage × %d)\n\n",
		       order, (size_t)(1 << (12 + order)) / 1024,
		       (int)((max_d + (1 << (12 + order)) - 1) >> (12 + order)));
	}

	print_model_arenas(nstreams);

	printf("Peak live memory (all streams + largest frame):\n");
	{
		size_t arenas = 0;
		for (s = 0; s < nstreams && s < (int)NMODELS; s++)
			arenas += models[s].arena_bytes;
		size_t per_frame = largest_decoded + largest_decoded / JPEG_RATIO
				   + RESIZE_BYTES + NORM_BYTES + OUTPUT_BYTES;
		printf("  Arenas:       %.1f MB\n", arenas / 1048576.0);
		printf("  Per-frame:    %.1f MB\n", per_frame / 1048576.0);
		printf("  Total peak:   %.1f MB\n\n",
		       (arenas + per_frame * nstreams) / 1048576.0);
	}

	printf("Processing %d frames × %d streams × %d repeat(s)...\n\n",
	       nframes, nstreams, repeat);
	fflush(stdout);

	struct stream *streams = calloc(nstreams, sizeof(*streams));
	if (!streams) { perror("calloc"); free(frames); return; }

	for (s = 0; s < nstreams; s++) {
		if (!stream_load(&streams[s], &models[s % NMODELS])) {
			fprintf(stderr, "model load failed for stream %d\n", s);
			goto out;
		}
	}

	t0 = ns_now();

	for (r = 0; r < repeat; r++) {
		for (i = 0; i < nframes; i++) {
			struct frame *fr = &frames[i];
			size_t decoded;
			size_t jpeg = fr->file_bytes;

			if (fr->width > 0 && fr->height > 0)
				decoded = (size_t)fr->width * fr->height * SSD_C;
			else
				decoded = fr->file_bytes * JPEG_RATIO;

			for (s = 0; s < nstreams; s++) {
				if (!process_frame(&streams[s], jpeg, decoded,
						   r * nframes + i)) {
					fprintf(stderr, "frame %d/%d failed\n",
						i, nframes);
					goto done_frames;
				}
			}

			total_bytes += (jpeg + decoded + RESIZE_BYTES
					+ NORM_BYTES + OUTPUT_BYTES) * nstreams;

			if ((r * nframes + i + 1) % 50 == 0) {
				t1 = ns_now();
				int done = r * nframes + i + 1;
				int total = repeat * nframes;
				printf("  %4d / %d frames  (%.1f fps)\n",
				       done, total,
				       done * 1e9 / (double)(t1 - t0));
				fflush(stdout);
			}
		}
	}

done_frames:
	for (s = 0; s < nstreams; s++) stream_unload(&streams[s]);
out:
	free(streams);
	free(frames);
	t1 = ns_now();
	long ms = (t1 - t0) / 1000000;
	printf("\nDone. %d total frames in %ld ms  (%.1f fps)\n",
	       repeat * nframes, ms,
	       repeat * nframes * 1e3 / (double)ms);
	printf("Total memory churn: %.1f GB\n",
	       total_bytes / (1024.0 * 1024.0 * 1024.0));
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
	const char *frames_dir = NULL;
	int         nimages    = 200;
	int         nstreams   = 4;
	int         repeat     = 3;
	int         i;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
			frames_dir = argv[++i];
		} else if (strcmp(argv[i], "--streams") == 0 && i + 1 < argc) {
			nstreams = atoi(argv[++i]);
		} else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
			repeat = atoi(argv[++i]);
		} else if (argv[i][0] != '-' && nimages == 200) {
			nimages = atoi(argv[i]);
		} else if (argv[i][0] != '-') {
			nstreams = atoi(argv[i]);
		}
	}

	if (nstreams <= 0) nstreams = 1;
	if (nstreams > (int)NMODELS) nstreams = (int)NMODELS;
	if (repeat   <= 0) repeat   = 1;

	printf("TFLite MobileNet-SSD object detection — mTHP benchmark\n");
	printf("Streams: %d concurrent models\n\n", nstreams);

	if (frames_dir)
		run_from_frames(frames_dir, nstreams, repeat);
	else
		run_synthetic(nimages, nstreams);

	return 0;
}
