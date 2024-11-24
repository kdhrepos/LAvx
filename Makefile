CC = gcc
SRCS = $(wildcard *.c)

TEST_DIR = test/
TEST_EXE = gemm_t kernel_t gemm_harness

FLAGS = -std=c11 -march=native -O3 -fopenmp -Wall
LIB = -lm

matmul:
	gcc -O2 -mno-avx512f -march=native -fopenmp $(TEST_DIR)matmul_parallel.c -o matmul	

sgemm:
	$(CC) -o sgemm_t $(TEST_DIR)sgemm_t.c $(SRCS) $(FLAGS) $(LIB)

dgemm:
	$(CC) -o dgemm_t $(TEST_DIR)dgemm_t.c $(SRCS) $(FLAGS) $(LIB)

clean :
	rm $(TEST_EXE)
