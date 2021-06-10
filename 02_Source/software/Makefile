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
bin/VideoDisplay.o: source/VideoDisplay.cpp include/VideoDisplay.hpp
	@ mkdir -p bin
	g++ -c -I../../common/inc $(CFLAGS) -c -I$(INCLUDE) -o bin/VideoDisplay.o source/VideoDisplay.cpp $(LDFLAGS)
bin/FrameCapture.o: source/FrameCapture.cpp include/FrameCapture.hpp
	@ mkdir -p bin
	g++  -c $(CFLAGS) -c -I$(INCLUDE) -o bin/FrameCapture.o source/FrameCapture.cpp $(LDFLAGS)

gyro: bin/BMI160.o bin/gyro.o 
	g++ $(NVFLAGS) $^ -o $@ $(LDFLAGS)

ar_demo: bin/main.o bin/BMI160.o  bin/VulkanHelper.o bin/VideoDisplay.o bin/FrameCapture.o bin/VulkanFramework.o 
	g++ $(NVFLAGS) $^ -o $@ $(LDFLAGS)
	
.PHONY: directories test clean 


clean:
	rm -rf bin gyro