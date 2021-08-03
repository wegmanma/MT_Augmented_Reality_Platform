VULKAN_LIB_PATH = /usr/lib/aarch64-linux-gnu
VULKAN_INC_PATH = /usr/include/vulkan
VULKAN_LAY_PATH = /etc/vulkan
CUDA_INCLUDE_PATH = /usr/local/cuda/targets/aarch64-linux/include
CUDA_PATH ?= /usr/local/cuda


INCLUDE :=include
SRC := source
NVINCLUDE :=include
NVINCLUDE2 +=include/cuda 

CFLAGS = -std=c++17 -O2 -I$(STB_INCLUDE_PATH) -I$(CUDA_INCLUDE_PATH) -I$(NVINCLUDE2) 
LDFLAGS = -L/usr/lib/aarch64-linux-gnu -L"$(CUDA_PATH)/lib64" -L"/usr/lib/aarch64-linux-gnu/tegra" `pkg-config --static --libs glfw3` -lvulkan -lpthread
NVFLAGS = -ccbin g++ -Xcudafe="--diag_suppress=2886" -I$(CUDA_INCLUDE_PATH) -I$(NVINCLUDE2) -I../../common/inc `pkg-config --static --cflags glfw3` -I/usr/include  -m64    --std=c++11 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75

all: gyro ar_demo

build: gyro ar_demo

bin/main.o: source/main.cpp include/VulkanFramework.hpp
	@ mkdir -p bin
	g++ -c $(CFLAGS) -c -I$(INCLUDE) -o bin/main.o source/main.cpp $(LDFLAGS)
bin/gyro.o: source/gyro.cpp
	@ mkdir -p bin
	g++ -c $(CFLAGS) -I$(INCLUDE) -o bin/gyro.o source/gyro.cpp $(LDFLAGS)
bin/BMI160.o: source/BMI160.cpp include/BMI160.hpp
	@ mkdir -p bin
	g++  -c $(CFLAGS) -I$(INCLUDE) -o bin/BMI160.o source/BMI160.cpp $(LDFLAGS)
bin/VulkanFramework.o: source/VulkanFramework.cpp include/VulkanFramework.hpp
	@ mkdir -p bin
	g++  -c $(CFLAGS) -c -I$(INCLUDE) -o bin/VulkanFramework.o source/VulkanFramework.cpp $(LDFLAGS)
bin/VulkanHelper.o: source/VulkanHelper.cpp include/VulkanHelper.hpp
	@ mkdir -p bin
	g++  -c $(CFLAGS) -c -I$(INCLUDE) -o bin/VulkanHelper.o source/VulkanHelper.cpp $(LDFLAGS)
bin/ProjectedSurface.o: source/ProjectedSurface.cpp include/ProjectedSurface.hpp
	@ mkdir -p bin
	g++ -c -I../../common/inc $(CFLAGS) -c -I$(INCLUDE) -o bin/ProjectedSurface.o source/ProjectedSurface.cpp $(LDFLAGS)
bin/MainCamera.o: source/MainCamera.cpp include/MainCamera.hpp
	@ mkdir -p bin
	g++ -c -I../../common/inc $(CFLAGS) -c -I$(INCLUDE) -o bin/MainCamera.o source/MainCamera.cpp $(LDFLAGS)
bin/FrameCapture.o: source/FrameCapture.cpp include/FrameCapture.hpp
	@ mkdir -p bin
	g++  -c $(CFLAGS) -c -I$(INCLUDE) -o bin/FrameCapture.o source/FrameCapture.cpp $(LDFLAGS)
bin/TCPFrameCapture.o: source/TCPFrameCapture.cpp include/TCPFrameCapture.hpp
	@ mkdir -p bin
	g++  -c $(CFLAGS) -c -I$(INCLUDE) -o bin/TCPFrameCapture.o source/TCPFrameCapture.cpp $(LDFLAGS)
bin/PositionEstimate.o: source/PositionEstimate.cpp include/PositionEstimate.hpp
	@ mkdir -p bin
	nvcc  $(NVFLAGS) -c -I$(NVINCLUDE) -I$(NVINCLUDE2) -o bin/PositionEstimate.o source/PositionEstimate.cpp $(LDFLAGS)
bin/computation.o: source/computation.cu include/computation.cuh
	@ mkdir -p bin
	nvcc  $(NVFLAGS) -c -I$(NVINCLUDE) -I$(NVINCLUDE2)  -o bin/computation.o source/computation.cu


gyro: bin/BMI160.o bin/gyro.o 
	nvcc $(NVFLAGS) $^ -o $@ $(LDFLAGS)

ar_demo: bin/main.o bin/BMI160.o  bin/VulkanHelper.o bin/ProjectedSurface.o bin/MainCamera.o bin/FrameCapture.o bin/TCPFrameCapture.o bin/VulkanFramework.o  bin/PositionEstimate.o bin/computation.o
	nvcc $(NVFLAGS) $^ -o $@ $(LDFLAGS) -lnvbuf_utils
	
.PHONY: directories test clean 


clean:
	rm -rf bin gyro