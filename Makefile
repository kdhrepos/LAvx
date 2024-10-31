CC = gcc

OBJS = main.o \
       avx_256.o
SRCS = $(OBJS:.o = .c)

TEST_DIR = test/
# TEST = 

# AVXFLAGS = -mavx512vnni -mavx512vl -mavx512f 
FLAGS = -std=c11 -march=native -O3
LIB_FLAGS = -lm

help:
	@echo "FP32 GEMM Test : make fp32"
	@echo "FP64 GEMM Test : make fp64"
	@echo "INT32 GEMM Test : make int32"

$(MAIN) : $(OBJS)
	$(CC) -o $(MAIN) $(OBJS)

int32:
	gcc -o int32 int32.c $(FLAGS) $(LIB_FLAGS)

fp32:
	gcc -o fp32 fp32.c $(FLAGS) $(LIB_FLAGS)

fp64:
	gcc -o fp64 fp64.c $(FLAGS) $(LIB_FLAGS)

# clean_t:
# 	rm 

clean :
	rm $(OBJS)
