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
	@echo "Matrix addition test			: 'make mat_add_t'"
	@echo "Matrix multiplication test		: 'make mat_mul_t'"
	@echo "Matrix Scalar Multiplication test	: 'make Scalar_mul_t'"
	@echo "Matrix transpose test			: 'make transpose_t'"
	@echo "L1 Norm test				: 'make l1_norm_t'"
	@echo "L2 Norm test				: 'make l2_norm_t'"
	@echo "Dot Product test			: 'make transpose_t'"

$(MAIN) : $(OBJS)
	$(CC) -o $(MAIN) $(OBJS)

fp32:
	gcc -o fp32 fp32.c $(FLAGS) $(LIB_FLAGS)

# clean_t:
# 	rm 

clean :
	rm $(OBJS)
