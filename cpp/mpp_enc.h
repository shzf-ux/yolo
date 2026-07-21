#ifndef MPP_ENC_H
#define MPP_ENC_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* H.264 encoder handle */
typedef struct mpp_enc_s mpp_enc_t;

/**
 * Create MPP H.264 encoder.
 * @w, @h: frame dimensions; @fps: frame rate; @bitrate: target bps (e.g. 8000000)
 * @out_fd: file descriptor to write encoded H.264 Annex-B stream, or -1 for none
 */
mpp_enc_t* mpp_enc_create(int w, int h, int fps, int bitrate, int out_fd);

/**
 * Encode one NV12 frame. Encoded data is written to out_fd set in create().
 * @nv12, @sz: NV12 frame data and size (must equal w*h*3/2)
 * Returns 0 on success, -1 on error (caller may drop the frame).
 */
int mpp_enc_encode(mpp_enc_t* e, const uint8_t* nv12, size_t sz);

/** Flush and destroy encoder */
void mpp_enc_destroy(mpp_enc_t* e);

#ifdef __cplusplus
}
#endif
#endif
