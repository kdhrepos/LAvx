CC = gcc

# OBJS = main.o \
#        avx_256.o
# SRCS = $(OBJS:.o = .c)

TEST_DIR = test/
TEST_EXE = gemm_t kernel_t gemm_harness

# AVXFLAGS = -mavx512vnni -mavx512vl -mavx512f 
FLAGS = -std=c11 -march=native -O2 -fopenmp 
# FLAGS = -std=c11 -march=native
LIB = -lm

help:
	# @echo "FP32 GEMM Test : make fp32"
	# @echo "FP64 GEMM Test : make fp64"
	# @echo "INT32 GEMM Test : make int32"

#$(MAIN) : $(OBJS)
#	$(CC) -o $(MAIN) $(OBJS)

# int32:
# 	gcc -o int32 int32.c $(FLAGS) $(LIB)

# fp32:
# 	gcc -o fp32 fp32.c $(FLAGS) $(LIB)

# fp64:
# 	gcc -o fp64 fp64.c $(FLAGS) $(LIB)

matmul:
#	g++ -o goto_gemm goto_gemm.cpp -std=c++17 -ffast-math -mavx2 -O3
#	gcc -o -O3 -mno-avx512f -march=native -fopenmp matmul_parallel.c -o matmul
	gcc -O2 -mno-avx512f -march=native -fopenmp matmul_parallel.c -o matmul	
gemm_harness:
	$(CC) -o gemm_harness $(TEST_DIR)gemm_harness.c $(FLAGS) $(LIB)

gemm_t:
	$(CC) -o gemm_t $(TEST_DIR)gemm_t.c $(FLAGS) $(LIB)

clean :
	rm $(TEST_EXE)
