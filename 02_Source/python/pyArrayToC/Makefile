SHELL = /bin/sh

OBJS =  client.o
CFLAG = -Wall -g
CC = gcc
CCPP = g++
INCLUDE = 
LIBS = 

all: pyToC

build: pyToC

pyToC: main.c
	${CC} ${CFLAG} ${INCLUDE} -o pyToC main.c $(LIBS)


clean:
	-rm -f *.o core *.core