VULKAN_LIB_PATH = /usr/lib/aarch64-linux-gnu
VULKAN_INC_PATH = /usr/include/vulkan
VULKAN_LAY_PATH = /etc/vulkan

INCLUDE :=include
SRC := source

CFLAGS = -std=c++17 -O2 -I$(STB_INCLUDE_PATH) 
LDFLAGS = `pkg-config --static --libs glfw3` -lvulkan -lpthread

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
bin/FrameCapture.o: source/FrameCapture.cpp include/FrameCapture.hpp
	@ mkdir -p bin
	g++  -c $(CFLAGS) -c -I$(INCLUDE) -o bin/FrameCapture.o source/FrameCapture.cpp $(LDFLAGS)
bin/PositionEstimate.o: source/PositionEstimate.cpp include/PositionEstimate.hpp
	@ mkdir -p bin
	g++  -c $(CFLAGS) -c -I$(INCLUDE) -o bin/PositionEstimate.o source/PositionEstimate.cpp $(LDFLAGS)


gyro: bin/BMI160.o bin/gyro.o 
	g++ $(NVFLAGS) $^ -o $@ $(LDFLAGS)

ar_demo: bin/main.o bin/BMI160.o  bin/VulkanHelper.o bin/ProjectedSurface.o bin/FrameCapture.o bin/VulkanFramework.o  bin/PositionEstimate.o
	g++ $(NVFLAGS) $^ -o $@ $(LDFLAGS)
	
.PHONY: directories test clean 


clean:
	rm -rf bin gyro