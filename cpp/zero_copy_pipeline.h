/*
 * zero_copy_pipeline.h - DMA-BUF zero-copy video pipeline for RK3588
 *
 * Pipeline:
 *   V4L2(ISP) → dma_fd → RGA(import fd) → dma_fd → NPU(rknn_create_mem_from_fd)
 *        │                                                  │
 *        └── dma_fd → CPU(mmap draw boxes) → MPP(import fd) → H.264 → UDP
 *
 * All frame data stays in DMA-BUF. Only dma_buf fd pointers are passed.
 */
#ifndef ZERO_COPY_PIPELINE_H
#define ZERO_COPY_PIPELINE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* opaque context */
typedef struct zcp_s zcp_t;

/* Callback: when H.264 encoded data is ready. data is valid until callback returns. */
typedef void (*zcp_h264_callback)(const uint8_t* data, size_t len, uint64_t pts_us, void* user);

/* --- Pipeline configuration --- */
struct zcp_cfg {
    const char* video_dev;    /* "/dev/video53" */
    const char* text_model;   /* "model/clip_text_fp16.rknn" */
    const char* text_file;    /* "model/detect_classes.txt" */
    const char* yolo_model;   /* "model/yolo_world_v2s_i8.rknn" */
    int width;                /* 1920 */
    int height;               /* 1080 */
    int fps;                  /* 30 */
    int bitrate;              /* 8000000 */
    int npu_workers;          /* 3 */
    const char* target_ip;    /* "192.168.1.25" or NULL for file output */
    int target_port;          /* 1235 */
    const char* output_path;  /* file path if target_ip is NULL */
};

/* --- API --- */

/**
 * Initialize the full pipeline.
 * Returns NULL on failure (prints reason to stderr).
 */
zcp_t* zcp_create(const zcp_cfg* cfg);

/**
 * Print current fps/queue stats to stderr.
 */
void zcp_print_stats(zcp_t* p);

/**
 * Destroy pipeline, release all resources.
 */
void zcp_destroy(zcp_t* p);

#ifdef __cplusplus
}
#endif
#endif
