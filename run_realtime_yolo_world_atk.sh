#!/bin/sh
#
# 正点原子 ATK-DLRK3588 专用启动脚本
# 适配 YOLO-World 实时检测推流
#

set -eu

# ========== 用户可配置参数 ==========
# 摄像头设备（正点原子 3 MIPI: video44, video53(主摄), video62, video71）
DEVICE="${DEVICE:-/dev/video53}"
# Windows 接收端 IP
UDP_TARGET="${UDP_TARGET:-192.168.1.25}"
UDP_PORT="${UDP_PORT:-1235}"
# 采集分辨率（video53 支持 3840x2160，但 1920x1080 性能更均衡）
WIDTH="${WIDTH:-1920}"
HEIGHT="${HEIGHT:-1080}"
CAPTURE_FPS="${CAPTURE_FPS:-30}"
STREAM_FPS="${STREAM_FPS:-30}"
# 推流码率
BITRATE="${BITRATE:-8M}"
# 推理线程数（RK3588 3 个 NPU core，默认 3）
WORKERS="${WORKERS:-3}"
# 预处理线程数
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-6}"
# ===================================

TEXT_MODEL="${TEXT_MODEL:-model/clip_text_fp16.rknn}"
YOLO_MODEL="${YOLO_MODEL:-model/yolo_world_v2s_i8.rknn}"
TEXT_FILE="${TEXT_FILE:-model/detect_classes.txt}"
PIPE="${PIPE:-/tmp/yolo_world_realtime_nv12.pipe}"
V4L2_BUFFERS="${V4L2_BUFFERS:-8}"
RAW_QUEUE="${RAW_QUEUE:-2}"
INFER_QUEUE="${INFER_QUEUE:-1}"
RESULT_QUEUE="${RESULT_QUEUE:-1}"

UDP_URL="udp://${UDP_TARGET}:${UDP_PORT}?pkt_size=1316"

# 正点原子 Buildroot 系统库路径
# librknnrt.so → 用 deploy.zip 带的 v2.3.2（./lib/）
# librga.so    → 用系统自带的 v2.1.0（内核驱动 v1.3.3 更兼容）
export LD_LIBRARY_PATH=./lib:${LD_LIBRARY_PATH:-}
# 用有 h264_rkmpp 支持的 FFmpeg
FFMPEG="${FFMPEG:-/root/yolov8/ffmpeg_rkmpp/ffmpeg}"

cleanup() {
  if [ -n "${FFMPEG_PID:-}" ]; then
    kill "$FFMPEG_PID" 2>/dev/null || true
    wait "$FFMPEG_PID" 2>/dev/null || true
  fi
  rm -f "$PIPE"
}

trap cleanup INT TERM EXIT

rm -f "$PIPE"
mkfifo "$PIPE"

echo "=================================="
echo " YOLO-World Realtime Detection"
echo "=================================="
echo " Camera:      $DEVICE"
echo " Capture:     ${WIDTH}x${HEIGHT} @ ${CAPTURE_FPS}fps"
echo " Stream:      ${STREAM_FPS}fps"
echo " Target:      $UDP_URL"
echo " Workers:     $WORKERS (infer) / $PREPROCESS_WORKERS (pre)"
echo "=================================="

# 唤醒 RGA 硬件加速（防止内核电源管理自动休眠）
echo on > /sys/devices/platform/fdb60000.rga/power/control 2>/dev/null || true
echo on > /sys/devices/platform/fdb70000.rga/power/control 2>/dev/null || true
echo on > /sys/devices/platform/fdb80000.rga/power/control 2>/dev/null || true

echo ""

# 启动 FFmpeg 硬件编码推流（从 FIFO 读取 NV12 → h264_rkmpp → UDP MPEG-TS）
$FFMPEG -hide_banner -loglevel warning \
  -fflags nobuffer \
  -flags low_delay \
  -f rawvideo \
  -pix_fmt nv12 \
  -s "${WIDTH}x${HEIGHT}" \
  -r "$STREAM_FPS" \
  -i "$PIPE" \
  -an \
  -c:v h264_rkmpp \
  -b:v "$BITRATE" \
  -g 10 \
  -bf 0 \
  -f mpegts \
  -flush_packets 1 \
  -muxdelay 0 \
  -muxpreload 0 \
  "$UDP_URL" &
FFMPEG_PID=$!

sleep 1

# 启动实时检测程序
./rknn_yolo_world_realtime \
  "$TEXT_MODEL" \
  "$TEXT_FILE" \
  "$YOLO_MODEL" \
  --device "$DEVICE" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --fps "$CAPTURE_FPS" \
  --stream-fps "$STREAM_FPS" \
  --buffers "$V4L2_BUFFERS" \
  --skip 5 \
  --preprocess-workers "$PREPROCESS_WORKERS" \
  --workers "$WORKERS" \
  --raw-queue "$RAW_QUEUE" \
  --infer-queue "$INFER_QUEUE" \
  --result-queue "$RESULT_QUEUE" \
  --output "$PIPE"
