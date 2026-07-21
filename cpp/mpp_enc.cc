#include "mpp_enc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>

#include "rk_mpi.h"
#include "mpp_frame.h"
#include "mpp_packet.h"
#include "mpp_buffer.h"
#include "rk_venc_cmd.h"

struct mpp_enc_s {
    MppCtx ctx;
    MppApi* mpi;
    MppBufferGroup grp;
    MppBuffer buf_pool[4];
    int pool_idx;
    int w, h, stride;
    size_t frame_sz;
    int out_fd;
    uint64_t fcnt;
};

mpp_enc_t* mpp_enc_create(int w, int h, int fps, int bitrate, int out_fd)
{
    mpp_enc_t* e = (mpp_enc_t*)calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->w = w; e->h = h; e->stride = w;
    e->frame_sz = (size_t)w * h * 3 / 2;
    e->out_fd = out_fd;
    e->fcnt = 0;

    MPP_RET ret = mpp_create(&e->ctx, &e->mpi);
    if (ret) { fprintf(stderr, "[MPP] create failed %d\n", ret); goto fail; }

    ret = mpp_init(e->ctx, MPP_CTX_ENC, MPP_VIDEO_CodingAVC);
    if (ret) { fprintf(stderr, "[MPP] enc init failed %d\n", ret); goto fail; }

    /* prep config */
    MppEncPrepCfg prep;
    memset(&prep, 0, sizeof(prep));
    prep.change = MPP_ENC_PREP_CFG_CHANGE_INPUT | MPP_ENC_PREP_CFG_CHANGE_FORMAT;
    prep.width = w; prep.height = h;
    prep.hor_stride = w; prep.ver_stride = h;
    prep.format = MPP_FMT_YUV420SP;
    if (e->mpi->control(e->ctx, MPP_ENC_SET_PREP_CFG, &prep)) goto fail;

    /* rate control */
    MppEncRcCfg rc;
    memset(&rc, 0, sizeof(rc));
    rc.change = MPP_ENC_RC_CFG_CHANGE_ALL;
    rc.rc_mode = MPP_ENC_RC_MODE_VBR;
    rc.bps_target = bitrate;
    rc.bps_max = (int)(bitrate * 1.5);
    rc.bps_min = (int)(bitrate * 0.5);
    rc.fps_in_flex = 0;    rc.fps_in_denom = 1;  rc.fps_in_num = fps;
    rc.fps_out_flex = 0;   rc.fps_out_denom = 1; rc.fps_out_num = fps;
    rc.gop = fps * 2;
    if (e->mpi->control(e->ctx, MPP_ENC_SET_RC_CFG, &rc)) goto fail;

    /* codec config: H.264 High profile level 5.0 */
    MppEncCodecCfg codec;
    memset(&codec, 0, sizeof(codec));
    codec.coding = MPP_VIDEO_CodingAVC;
    codec.h264.change = MPP_ENC_H264_CFG_CHANGE_PROFILE;
    codec.h264.profile = 100;
    codec.h264.level = 50;
    if (e->mpi->control(e->ctx, MPP_ENC_SET_CODEC_CFG, &codec)) goto fail;

    /* pre-allocate DRM buffers */
    if (mpp_buffer_group_get_internal(&e->grp, MPP_BUFFER_TYPE_DRM)) goto fail;
    for (int i = 0; i < 4; i++)
        if (mpp_buffer_get(e->grp, &e->buf_pool[i], e->frame_sz)) goto fail;

    fprintf(stderr, "[MPP] encoder ready: %dx%d %dfps %dbps\n", w, h, fps, bitrate);
    return e;

fail:
    mpp_enc_destroy(e);
    return NULL;
}

int mpp_enc_encode(mpp_enc_t* e, const uint8_t* nv12, size_t sz)
{
    if (!e || !e->ctx || sz < e->frame_sz) return -1;

    /* copy NV12 into pre-allocated buffer */
    memcpy(mpp_buffer_get_ptr(e->buf_pool[e->pool_idx]), nv12, e->frame_sz);

    MppFrame frame = NULL;
    mpp_frame_init(&frame);
    mpp_frame_set_width(frame, e->w);
    mpp_frame_set_height(frame, e->h);
    mpp_frame_set_hor_stride(frame, e->stride);
    mpp_frame_set_ver_stride(frame, e->h);
    mpp_frame_set_fmt(frame, MPP_FMT_YUV420SP);
    mpp_frame_set_pts(frame, e->fcnt * 1000000LL / 30);
    mpp_frame_set_buffer(frame, e->buf_pool[e->pool_idx]);

    MPP_RET ret = e->mpi->encode_put_frame(e->ctx, frame);
    mpp_frame_deinit(&frame);
    e->pool_idx = (e->pool_idx + 1) % 4;
    if (ret) return -1;

    /* drain encoded packets and write to out_fd */
    MppPacket pkt = NULL;
    int wrote = 0;
    while (1) {
        ret = e->mpi->encode_get_packet(e->ctx, &pkt);
        if (ret != MPP_OK || !pkt) break;
        void* data = mpp_packet_get_data(pkt);
        size_t len = mpp_packet_get_length(pkt);
        if (len > 0 && data && e->out_fd >= 0) {
            ssize_t n = write(e->out_fd, data, len);
            if (n > 0) wrote++;
        }
        mpp_packet_deinit(&pkt);
    }
    e->fcnt++;
    return wrote ? 0 : -1;
}

void mpp_enc_destroy(mpp_enc_t* e)
{
    if (!e) return;
    /* flush */
    if (e->ctx && e->mpi) {
        e->mpi->encode_put_frame(e->ctx, NULL);
        MppPacket pkt = NULL;
        while (e->mpi->encode_get_packet(e->ctx, &pkt) == MPP_OK && pkt)
            mpp_packet_deinit(&pkt);
    }
    for (int i = 0; i < 4; i++) if (e->buf_pool[i]) mpp_buffer_put(e->buf_pool[i]);
    if (e->grp) mpp_buffer_group_put(e->grp);
    if (e->ctx) mpp_destroy(e->ctx);
    free(e);
    fprintf(stderr, "[MPP] encoder destroyed\n");
}
