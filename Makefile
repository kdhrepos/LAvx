CC = gcc

OBJS = main.o \
       avx_256.o
SRCS = $(OBJS:.o = .c)
MAIN = main

# AVXFLAGS = -mavx512vnni -mavx512vl -mavx512f 

help:
	@echo "help!"

$(MAIN) : $(OBJS)
	$(CC) -o $(MAIN) $(OBJS)

test:
	gcc -o test test.c -march=native

clean :
	rm $(OBJS)
