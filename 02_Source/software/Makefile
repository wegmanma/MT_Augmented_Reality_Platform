VULKAN_LIB_PATH = /usr/lib/aarch64-linux-gnu
VULKAN_INC_PATH = /usr/include/vulkan
VULKAN_LAY_PATH = /etc/vulkan
CUDA_INCLUDE_PATH = /usr/local/cuda-10.2/targets/aarch64-linux/include/
CUDA_PATH ?= /usr/local/cuda
FFMPEG_PATH := /opt/ffmpeg_isztldav

INCLUDE :=include
SRC := source
NVINCLUDE :=include
NVINCLUDE2 +=include/cuda 

CFLAGS = -std=c++17 -O2 -I$(STB_INCLUDE_PATH) -I$(CUDA_INCLUDE_PATH) -I$(NVINCLUDE2) -I/opt/ffmpeg_isztldav/include -Wall
LDFLAGS = -L/usr/lib/aarch64-linux-gnu -L"$(CUDA_PATH)/lib64" -L"/usr/lib/aarch64-linux-gnu/tegra" `pkg-config --static --libs glfw3` -lvulkan -lpthread -L/opt/ffmpeg_isztldav/lib -lavdevice -lm -lxcb -lxcb-shape -lxcb-xfixes -lasound -lavfilter -lpthread -lm -lavformat -lm -lz -lavcodec -lpthread -lm -lz -lswresample -lm -lswscale -lm -lavutil -lpthread -lm
NVFLAGS = -ccbin g++ -Xcudafe="--diag_suppress=2886" -I$(CUDA_INCLUDE_PATH) -I$(NVINCLUDE2) -I/opt/ffmpeg_isztldav/include  -I../../common/inc `pkg-config --static --cflags glfw3` -I/usr/include  -m64    --std=c++11  -gencode arch=compute_72,code=sm_72

# Flags for FFMPEG Framework
ifeq ($(VERBOSE), 1)
AT =
else
AT = @
endif
CC             = $(AT) gcc
CCFLAGS :=
CCFLAGS += -std=c99 -Wall -ggdb 
COMMONFLAGS :=

COMMONFLAGS += \
	-g -O0 \
	-I"/usr/include/libdrm" \
	-I"/$(CUDA_PATH)/include" \
	`PKG_CONFIG_PATH=$(FFMPEG_PATH)/lib/pkgconfig pkg-config --cflags libavdevice libavformat libavfilter libavcodec libswresample libswscale libavutil` -c


all: gyro ar_demo

build: gyro ar_demo

bin/main.o: source/main.cpp include/VulkanFramework.hpp
	@ mkdir -p bin
	nvcc -c $(NVFLAGS) -c $(COMMONFLAGS) -I$(NVINCLUDE) -I$(INCLUDE) -o bin/main.o source/main.cpp $(LDFLAGS)
bin/gyro.o: source/gyro.cpp
	@ mkdir -p bin
	g++ -c $(CFLAGS) -I$(INCLUDE) -o bin/gyro.o source/gyro.cpp $(LDFLAGS)
bin/BMI160.o: source/BMI160.cpp include/BMI160.hpp
	@ mkdir -p bin
	g++  -c $(CFLAGS) -I$(INCLUDE) -o bin/BMI160.o source/BMI160.cpp $(LDFLAGS)
bin/VulkanFramework.o: source/VulkanFramework.cpp include/VulkanFramework.hpp
	@ mkdir -p bin
	nvcc  $(NVFLAGS) -c -I$(NVINCLUDE) -I$(INCLUDE) -o bin/VulkanFramework.o source/VulkanFramework.cpp $(LDFLAGS)
bin/VulkanHelper.o: source/VulkanHelper.cpp include/VulkanHelper.hpp
	@ mkdir -p bin
	g++  -c $(CFLAGS) -c -I$(INCLUDE) -o bin/VulkanHelper.o source/VulkanHelper.cpp $(LDFLAGS)
bin/ProjectedSurface.o: source/ProjectedSurface.cpp include/ProjectedSurface.hpp
	@ mkdir -p bin
	nvcc  $(NVFLAGS) -c -I$(NVINCLUDE) -I$(NVINCLUDE2) -I$(INCLUDE) -o bin/ProjectedSurface.o source/ProjectedSurface.cpp $(LDFLAGS)
bin/MainCamera.o: source/MainCamera.cpp include/MainCamera.hpp
	@ mkdir -p bin
	g++ -c -I../../common/inc $(CFLAGS) -c -I$(INCLUDE) -o bin/MainCamera.o source/MainCamera.cpp $(LDFLAGS)
bin/CudaCapture.o: source/CudaCapture.cpp include/CudaCapture.hpp
	@ mkdir -p bin
	nvcc  $(NVFLAGS) -c  -I$(NVINCLUDE) -I$(INCLUDE) -o bin/CudaCapture.o source/CudaCapture.cpp $(LDFLAGS)
bin/TCPFrameCapture.o: source/TCPFrameCapture.cpp include/TCPFrameCapture.hpp
	@ mkdir -p bin
	nvcc  $(NVFLAGS) -c  -I$(NVINCLUDE)  -I$(NVINCLUDE2) -I$(INCLUDE) -o bin/TCPFrameCapture.o source/TCPFrameCapture.cpp $(LDFLAGS)
bin/PositionEstimate.o: source/PositionEstimate.cpp include/PositionEstimate.hpp
	@ mkdir -p bin
	nvcc  $(NVFLAGS) -c -I$(NVINCLUDE) -I$(NVINCLUDE2) -I$(INCLUDE) -o bin/PositionEstimate.o source/PositionEstimate.cpp $(LDFLAGS)
bin/computation.o: source/computation.cu include/computation.cuh
	@ mkdir -p bin
	nvcc  $(NVFLAGS) -c -I$(NVINCLUDE) -I$(NVINCLUDE2)  -o bin/computation.o source/computation.cu
bin/scaler.o: source/scaler.c
	$(CC) $(CCFLAGS) -c $(COMMONFLAGS) -I$(INCLUDE) -o $@ $<
bin/framebuffer.o: source/framebuffer.c
	$(CC) $(CCFLAGS) -c $(COMMONFLAGS) -I$(INCLUDE) -o $@ $<	
bin/setup_camera.o: source/setup_camera.c
	@echo "Compiling: $<"
	$(CC) -I/usr/local/include -I"/usr/include/aarch64-linux-gnu" -I"/usr/src/linux-headers-4.9.253-tegra-ubuntu18.04_aarch64/nvidia/include/" -c -I$(INCLUDE) -o $@ $<



gyro: bin/BMI160.o bin/gyro.o 
	g++ $(CFLAGS) -I$(INCLUDE) $^ -o $@ $(LDFLAGS)

ar_demo: bin/main.o bin/framebuffer.o bin/BMI160.o  bin/VulkanHelper.o bin/ProjectedSurface.o bin/MainCamera.o bin/CudaCapture.o bin/TCPFrameCapture.o bin/VulkanFramework.o bin/PositionEstimate.o bin/computation.o bin/setup_camera.o bin/scaler.o
	nvcc -I$(INCLUDE) -I$(NVINCLUDE) $^ -o $@ $(LDFLAGS) -lXext -lxcb -lX11 -lnvbuf_utils
	
.PHONY: directories test clean 


clean:
	rm -rf bin gyro