CC = gcc

OBJS = main.o \
       avx_256.o
SRCS = $(OBJS:.o = .c)
MAIN = main

# AVXFLAGS = -mavx512vnni -mavx512vl -mavx512f 
FLAGS = -std=c11 -march=native -O3
LIB_FLAGS = -lm

help:
	@echo "help!"

$(MAIN) : $(OBJS)
	$(CC) -o $(MAIN) $(OBJS)

test:
	gcc -o test test.c $(FLAGS) $(LIB_FLAGS)

clean :
	rm $(OBJS)
