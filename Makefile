CC = gcc
SRCS = cache.c gemm.c kernel.c pack.c test.c test_main.c #$(wildcard *.c)

FLAGS = -std=c11 -march=native -O2 -fopenmp -Wall
LIB = -lm

_test:
	$(CC) -o _test $(SRCS) $(FLAGS) $(LIB)

clean :
	rm $(TEST_EXE)
