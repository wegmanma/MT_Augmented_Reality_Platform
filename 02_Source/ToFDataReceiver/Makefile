SHELL = /bin/sh

OBJS =  client.o
CFLAG = -Wall -g
CC = gcc
CCPP = g++
INCLUDE = 
LIBS =

all: ToFDataReceiver

build: ToFDataReceiver

ToFDataReceiver: main.cpp
	${CCPP} ${CFLAG} ${INCLUDE} -o ToFDataReceiver main.cpp $(LIBS)


clean:
	-rm -f *.o core *.core