CC = gcc
SRCS = cache.c gemm.c kernel.c pack.c test.c #$(wildcard *.c)

FLAGS = -std=c11 -march=native -O3 -fopenmp -Wall
LIB = -lm

_test:
	$(CC) -o _test test_main.c $(SRCS) $(FLAGS) $(LIB)

clean :
	rm $(TEST_EXE)