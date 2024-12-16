CC = gcc
SRCS = opt.c gemm.c kernel.c pack.c ops.c test.c test_main.c #$(wildcard *.c)

FLAGS = -std=c11 -march=native -O2 -fopenmp -Wall
LIB = -lm

tt:
	$(CC) -o tt $(SRCS) $(FLAGS) $(LIB)

clean :
	rm $(TEST_EXE)
