# DMA-BUF Zero-Copy Video Pipeline for RK3588 YOLO-World

## Task

Rewrite `cpp/realtime_main.cc` to implement a **single-process, DMA-BUF zero-copy video pipeline** for RK3588. All video frames must stay in DMA-BUF/dmabuf fd — only file descriptors are passed between hardware modules. No CPU data copies for video frames.

## Hardware Architecture

```
RK3588 on ATK-DLRK3588 board:
  Sensor: IMX415 (MIPI CSI, 3864x2192 native)
  ISP: RKISP (outputs NV12)
  RGA3: 2 cores, feature=0x2045, supports im2d API with dma-buf fd
  NPU: 3 cores (CORE_0, CORE_1, CORE_2), supports rknn_create_mem_from_fd()
  MPP/VEPU: H.264 hardware encoder, supports mpp_buffer_import() with EXT_DMA
  Network: gigabit Ethernet
  Kernel: 5.10.209, Linux buildroot
```

## Target Pipeline

```
V4L2 → VIDIOC_EXPBUF → dma_fd → RGA(importbuffer_fd, wrapbuffer_fd) → output dma_fd → NPU(rknn_create_mem_from_fd)
  │                                                                       │
  └── same dma_fd → CPU(mmap, draw boxes on Y-plane) → MPP(mpp_buffer_import EXT_DMA) → H.264 → TS/UDP
```

All frame data in DMA-BUF. Only `int dma_fd` passed between stages.

## Environment

### Cross-compilation
```makefile
CROSS_COMPILE = /home/zzy/rk3588/rk3588_linux_sdk/buildroot/output/rockchip_atk_dlrk3588/host/bin/aarch64-buildroot-linux-gnu-
CC  = $(CROSS_COMPILE)gcc
CXX = $(CROSS_COMPILE)g++ -O2 -fPIC -std=c++11
```

### Include paths
```
RGA:    /home/zzy/rk3588/yolov8_runner_pool/3rdparty/librga/include
RKNN:   /home/zzy/rk3588/rknn/rknn-toolkit2-master/rknpu2/include
RKNN2:  /home/zzy/rk3588/rknn/rknn-toolkit2-master/rknpu2/runtime/Linux/librknn_api/include
MPP:    /home/zzy/rk3588/rk3588_linux_sdk/buildroot/output/rockchip_atk_dlrk3588/host/aarch64-buildroot-linux-gnu/sysroot/usr/include/rockchip
SYSROOT:/home/zzy/rk3588/rk3588_linux_sdk/buildroot/output/rockchip_atk_dlrk3588/host/aarch64-buildroot-linux-gnu/sysroot/usr/include
```

### Link libraries
```
-lrknnrt -lrga -lrockchip_mpp -ljpeg -lturbojpeg -ldl -lpthread
-L/home/zzy/rk3588/RK3588-yolo-world--main/deploy/rknn_yolo_world_demo/lib
-L.../sysroot/usr/lib
```

### Board paths
```
Binary:  /root/rknn_yolo_world_demo/rknn_yolo_world_realtime
Models:  /root/rknn_yolo_world_demo/model/clip_text_fp16.rknn
         /root/rknn_yolo_world_demo/model/yolo_world_v2s_i8.rknn
         /root/rknn_yolo_world_demo/model/detect_classes.txt
Library: /root/rknn_yolo_world_demo/lib/librknnrt.so
         /root/rknn_yolo_world_demo/lib/librga.so
Camera:  /dev/video53
Target:  192.168.1.25:1235 (Windows PC)
```

### Build and deploy
```bash
cd /home/zzy/rk3588/RK3588-yolo-world--main/build
make clean && make -j$(nproc)
# Then scp rknn_yolo_world_realtime to root@192.168.1.32:/root/rknn_yolo_world_demo/
# (password: root)
```

## Key APIs (verified available in this SDK)

### 1. V4L2 DMA-BUF Export
```c
// Open camera, set NV12 1920x1080, request MMAP buffers (count=4)
// Then for each buffer:
struct v4l2_exportbuffer expbuf;
expbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
expbuf.index = i;
expbuf.plane = 0;
ioctl(video_fd, VIDIOC_EXPBUF, &expbuf);
// expbuf.fd is the dma_buf fd for this buffer
```

### 2. RGA im2d API (zero-copy with fd)
```c
#include "im2d.h"
#include "RgaUtils.h"

// Import external dma-buf as RGA handle:
rga_buffer_handle_t handle = importbuffer_fd(fd, width, height, RK_FORMAT_YCbCr_420_SP);

// Wrap handle to rga_buffer_t for processing:
rga_buffer_t src = wrapbuffer_handle(handle, width, height, RK_FORMAT_YCbCr_420_SP);

// For output: create a new DMA buffer via im2d allocation or import
// Wrap as fd:
rga_buffer_t dst = wrapbuffer_fd_t(output_fd, dst_w, dst_h, dst_w, dst_h, RK_FORMAT_RGBA_8888);

// Hardware resize (zero CPU):
IM_STATUS st = imresize(src, dst);  // synchronous

// Release handles when done
releasebuffer_handle(handle);
```

### 3. NPU DMA-BUF Input
```c
#include "rknn_api.h"

// Create tensor memory from dma-buf fd (zero-copy for NPU to read):
rknn_tensor_mem* mem = rknn_create_mem_from_fd(ctx, fd, virt_addr, size, offset);

// Set as input:
rknn_set_io_mem(ctx, mem, &input_attrs[0]);

// Important for model: this YOLO-World model has 2 inputs:
// input_attrs[0] = images  (NHWC, INT8, [1,640,640,3], size=1228800)
// input_attrs[1] = texts   (raw float, [1,80,512], size=40960)

// Set text input via normal API (text is small, no DMA needed):
rknn_input inputs[1];
inputs[0].index = 1;  // "texts"
inputs[0].type = RKNN_TENSOR_FLOAT32;
inputs[0].buf = text_output_data;
inputs[0].size = text_size * sizeof(float);
rknn_inputs_set(ctx, 1, inputs);

// Then run:
rknn_run(ctx);

// Get outputs
rknn_output outputs[io_num.n_output];
rknn_outputs_get(ctx, io_num.n_output, outputs);
```

### 4. MPP DMA-BUF Import for Encoding
```c
#include "rk_mpi.h"

// Import external dma-buf:
MppBufferInfo info;
memset(&info, 0, sizeof(info));
info.type = MPP_BUFFER_TYPE_EXT_DMA;
info.fd = dup(capture_dmabuf_fd);  // dup so MPP owns its reference
info.size = width * height * 3 / 2;
MppBuffer buf;
mpp_buffer_import(&buf, &info);

// Attach to frame:
MppFrame frame;
mpp_frame_init(&frame);
mpp_frame_set_width(frame, width);
mpp_frame_set_height(frame, height);
mpp_frame_set_hor_stride(frame, width);
mpp_frame_set_ver_stride(frame, height);
mpp_frame_set_fmt(frame, MPP_FMT_YUV420SP);
mpp_frame_set_buffer(frame, buf);
// No data copy — MPP reads directly from the dma-buf

// Encode:
mpi->encode_put_frame(ctx, frame);
// Drain:
while (mpi->encode_get_packet(ctx, &pkt) == MPP_OK && pkt) {
    // send pkt->data, pkt->length over UDP/TS
}
```

### 5. CPU Draw on DMA-BUF
```c
// Map the dma-buf fd to userspace (read/write):
size_t frame_size = 1920 * 1080 * 3 / 2;
void* ptr = mmap(NULL, frame_size, PROT_READ | PROT_WRITE, MAP_SHARED, capture_fd, 0);

// Draw boxes on Y-plane (first w*h bytes):
// ptr[0..w*h-1] is Y plane, ptr[w*h..] is UV plane

// IMPORTANT: after drawing, flush CPU cache to make data visible to MPP:
// Option A: use __builtin___clear_cache or DMA-BUF sync API
// Option B: munmap and use MS_SYNC before encoding
// The simplest working approach from SDK examples: just munmap, changes are
// automatically visible because the buffer is MAP_SHARED with coherent DMA

munmap(ptr, frame_size);
```

## Existing Code Structure (DO NOT BREAK)

### Model loading (keep as-is):
```cpp
// 1. Build text embedding (run once at startup):
std::vector<float> text_output;
int text_size = 0;
build_text_embedding(cfg, &text_output, &text_size);

// 2. Create NPU workers (3 workers, one per core):
struct YoloWorker { int id; rknn_core_mask core_mask; rknn_app_context_t ctx; };
for (int i = 0; i < 3; i++) {
    init_yolo_world_model(&workers[i].ctx, "model/yolo_world_v2s_i8.rknn");
    rknn_set_core_mask(workers[i].ctx.rknn_ctx, worker_core_mask(i));
}
```

### The YOLO-World model requires 2 inputs:
- input_attrs[0]: "images" - INT8 NHWC [1,640,640,3], 1228800 bytes
- input_attrs[1]: "texts" - INT8 [1,80,512], 40960 bytes (pre-computed, same every frame)

### Post-processing:
```cpp
// After rknn_run + rknn_outputs_get:
post_process(&ctx, outputs, &letterbox, 0.25f, 0.45f, &od_results);
// od_results.results[i]: box.left/top/right/bottom, cls_id, prop
```

## Implementation Requirements

### Core main loop (sequential, per-frame):
```
1. v4l2_dqbuf(video_fd) → get buffer_index + capture_dmabuf_fd
2. RGA: importbuffer_fd(capture_dmabuf_fd) → imresize to 640x640 → output dma_fd
3. NPU: rknn_create_mem_from_fd(rga_output_fd) → set as input_attrs[0]
        → rknn_inputs_set for text (input_attrs[1])
        → rknn_run()
        → rknn_outputs_get() → post_process() → get boxes
4. CPU: mmap(capture_dmabuf_fd) → draw boxes on Y-plane → munmap
5. MPP: mpp_buffer_import(capture_dmabuf_fd) → encode_put_frame → drain packets
6. For each MPP packet: send over UDP with simple TS encapsulation (or raw H.264)
7. v4l2_qbuf(video_fd, buffer_index) — return buffer to driver
```

### Must handle:
- **Double/triple buffering**: While frame N is being NPU-processed, frame N+1 is being captured. RGA output buffers and NPU input mem objects must be per-worker or pooled.
- **DMA-BUF lifecycle**: dup() fds when passing ownership to RGA/MPP modules. close() unused fds.
- **NPU text input**: The text embedding is 80 categories × 512 float = 40960 floats. Pre-compute once, reuse every frame via rknn_inputs_set.
- **TS/UDP**: Wrap H.264 NAL units in 188-byte MPEG-TS packets (simple version: just split into 184-byte payloads with 4-byte TS header, PID=0x100 for video). Or even simpler: write raw Annex-B H.264 to a pipe and let FFmpeg mux (but goal is single-process).
- **No external FFmpeg process** — encode + send entirely in-process.

### Output format for Windows playback:
Option 1 (simpler): TCP server on port 1235, send raw H.264 byte stream. Windows: `ffplay -f h264 tcp://192.168.1.32:1235`
Option 2: UDP + MPEG-TS. Windows: `ffplay -f mpegts udp://0.0.0.0:1235`

## Files to Modify

1. **MAIN TARGET**: Rewrite `cpp/realtime_main.cc` — replace the entire main() and render_loop. Keep the model loading (build_text_embedding, init_yolo_world_model) but replace the pipeline.

2. **CREATE**: New file `cpp/dmabuf_pipeline.cc` — contain the DMA-BUF pipeline implementation:
   - `v4l2_dmabuf_init/start/dqbuf/qbuf/close`
   - `rga_dmabuf_init/resize/close`
   - `mpp_dmabuf_init/encode/close`
   - `ts_udp_init/send_h264/close`

3. **UPDATE**: `build/Makefile` — add new .cc files and `-lrockchip_mpp` flag.

## Known Pitfalls

- MPP encode_put_frame + encode_get_packet drain should be in a tight loop per frame. Don't call encode_put_frame multiple times without draining in between.
- RGA imresize is synchronous. For NV12→RGBA conversion, use imcvtcolor + imresize.
- V4L2 VIDIOC_EXPBUF may fail if kernel config lacks DMA_BUF support. Fallback: use mmap'd virtual address for RGA (via importbuffer_virtualaddr) — this adds one copy but is robust.
- NPU output tensors are INT8 quantized. post_process() dequantizes automatically.
- After mmap draw, no explicit cache flush needed on RK3588 (coherent DMA).
- MPP buffer import with EXT_DMA: you must dup(fd) before passing to MPP.

## Success Criteria

- Pipeline runs at 25-30fps streaming (vs current 18fps)
- NPU utilization > 70% (vs current 54%)
- No pipe, no FFmpeg subprocess, no data copies between modules
- Single binary: `rknn_yolo_world_realtime`
- Windows can receive and play the stream

## Testing

1. Copy binary to board: `scp build/rknn_yolo_world_realtime root@192.168.1.32:/root/rknn_yolo_world_demo/`
2. SSH to board: `ssh root@192.168.1.32` (password: root)
3. Run: `cd /root/rknn_yolo_world_demo && echo 4194304 > /proc/sys/fs/pipe-max-size && ./rknn_yolo_world_realtime model/clip_text_fp16.rknn model/detect_classes.txt model/yolo_world_v2s_i8.rknn --device /dev/video53 --width 1920 --height 1080 --fps 30 --stream-fps 30`
4. Windows: `ffplay -f h264 tcp://192.168.1.32:1235`

## Git

Repo: `git@github.com:shzf-ux/yolo.git`
Commit and push when done: `git add -A && git commit -m "..." && git push`
