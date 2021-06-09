INCLUDE :=include
SRC := source

CFLAGS = -std=c++17 -O2 -I$(STB_INCLUDE_PATH) 
LDFLAGS = 

all: gyro

build: gyro

bin/gyro.o: source/gyro.cpp
	@ mkdir -p bin
	g++ -c $(CFLAGS) -I$(INCLUDE) -o bin/gyro.o source/gyro.cpp $(LDFLAGS)
bin/BMI160.o: source/BMI160.cpp include/BMI160.hpp
	@ mkdir -p bin
	g++  -c $(CFLAGS) -I$(INCLUDE) -o bin/BMI160.o source/BMI160.cpp $(LDFLAGS)
gyro: bin/BMI160.o bin/gyro.o 
	g++ $(NVFLAGS) $^ -o $@ $(LDFLAGS)

	
.PHONY: directories test clean 


clean:
	rm -rf bin gyro