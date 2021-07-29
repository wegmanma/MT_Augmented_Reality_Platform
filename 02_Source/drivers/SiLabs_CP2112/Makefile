 
MODULE_MAME = cp2112_mod

SRCS = cp2112.c

OBJS =  $(SRCS:.c=.o)

obj-m += $(MODULE_MAME).o
$(MODULE_MAME)-y = $(OBJS)

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean