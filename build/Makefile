CROSS_COMPILE ?= /home/zzy/rk3588/rk3588_linux_sdk/buildroot/output/rockchip_atk_dlrk3588/host/bin/aarch64-buildroot-linux-gnu-
CC = $(CROSS_COMPILE)gcc
CXX = $(CROSS_COMPILE)g++
AR = $(CROSS_COMPILE)ar

CFLAGS = -O2 -fPIC
CXXFLAGS = -O2 -fPIC -std=c++11
LDFLAGS = -L. -ldl -lpthread

SRC_DIR = ../cpp
UTILS_DIR = /home/zzy/rk3588/yolov8_runner_pool/utils
RGA_INC = /home/zzy/rk3588/yolov8_runner_pool/3rdparty/librga/include
STB_INC = /home/zzy/rk3588/rknn/rknn-toolkit2-master/rknpu2/examples/3rdparty/stb
JPEG_INC = /home/zzy/rk3588/rk3588_linux_sdk/buildroot/output/rockchip_atk_dlrk3588/host/aarch64-buildroot-linux-gnu/sysroot/usr/include

INC = -I$(SRC_DIR) \
      -I$(SRC_DIR)/tokenizer \
      -I$(SRC_DIR)/rknpu2/clip_text \
      -I$(SRC_DIR)/rknpu2/yolo_world \
      -I$(UTILS_DIR) \
      -I$(RGA_INC) \
      -I$(STB_INC) \
      -I$(JPEG_INC) \
      -DLIBRGA_IM2D_HANDLE

# RKNN include path
RKNN_INC = -I/home/zzy/rk3588/rknn/rknn-toolkit2-master/rknpu2/include \
           -I/home/zzy/rk3588/rknn/rknn-toolkit2-master/rknpu2/runtime/Linux/librknn_api/include

.PHONY: all clean

all: rknn_yolo_world_realtime

# === Utils static libs ===
UTILS_OBJS = file_utils.o image_drawing.o image_utils.o

file_utils.o: $(UTILS_DIR)/file_utils.c
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

image_drawing.o: $(UTILS_DIR)/image_drawing.c
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

image_utils.o: $(UTILS_DIR)/image_utils.c
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

# === YOLO World objects ===
YOLO_OBJS = postprocess.o clip_text.o yolo_world.o clip_tokenizer.o realtime_main.o

postprocess.o: $(SRC_DIR)/postprocess.cc
	$(CXX) $(CXXFLAGS) $(INC) $(RKNN_INC) -c $< -o $@

clip_text.o: $(SRC_DIR)/rknpu2/clip_text/clip_text.cc
	$(CXX) $(CXXFLAGS) $(INC) $(RKNN_INC) -c $< -o $@

yolo_world.o: $(SRC_DIR)/rknpu2/yolo_world/yolo_world.cc
	$(CXX) $(CXXFLAGS) $(INC) $(RKNN_INC) -c $< -o $@

clip_tokenizer.o: $(SRC_DIR)/tokenizer/clip_tokenizer.cpp
	$(CXX) $(CXXFLAGS) $(INC) $(RKNN_INC) -c $< -o $@

realtime_main.o: $(SRC_DIR)/realtime_main.cc
	$(CXX) $(CXXFLAGS) $(INC) $(RKNN_INC) -c $< -o $@

# === Link ===
rknn_yolo_world_realtime: $(YOLO_OBJS) $(UTILS_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ \
		-L/home/zzy/rk3588/RK3588-yolo-world--main/deploy/rknn_yolo_world_demo/lib \
		-L/home/zzy/rk3588/rk3588_linux_sdk/buildroot/output/rockchip_atk_dlrk3588/host/aarch64-buildroot-linux-gnu/sysroot/usr/lib \
		-lrknnrt -lrga -ljpeg -lturbojpeg $(LDFLAGS)

clean:
	rm -f *.o rknn_yolo_world_realtime
