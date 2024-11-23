CC = gcc
SRCS = $(wildcard *.c)

TEST_DIR = test/
TEST_EXE = gemm_t kernel_t gemm_harness

FLAGS = -std=c11 -march=native -O3 -fopenmp -Wall
LIB = -lm

matmul:
	gcc -O2 -mno-avx512f -march=native -fopenmp $(TEST_DIR)matmul_parallel.c -o matmul	

gemm_t:
	$(CC) -o gemm_t $(TEST_DIR)gemm_t.c $(SRCS) $(FLAGS) $(LIB)

clean :
	rm $(TEST_EXE)
