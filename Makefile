CC = gcc
SRCS = opt.c gemm.c kernel.c pack.c ops.c test.c test_main.c #$(wildcard *.c)
HDRS = gemm.h sse.h util.h # Header files

FLAGS = -std=c11 -march=native -O2 -fopenmp -Wall
LIB = -lm

tt:
	$(CC) -o tt $(SRCS) $(HDRS) $(FLAGS) $(LIB)

.PHONY: tt

clean :
	rm $(TEST_EXE)
