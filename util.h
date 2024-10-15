#ifndef UTIL_H
#define UTIL_H

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <stdint.h>

typedef short BOOL;
#define TRUE 1
#define FALSE 0

#define AVX_FP64 4
#define AVX_FP32 8

#define FP32_VEC_DIM(vec) ((sizeof(vec))/(sizeof(float)))
#define FP64_VEC_DIM(vec) ((sizeof(vec))/(sizeof(double)))

/* This was for vector */
// #define GET_RANDOM(vec, dim)                \
//     do {                                    \
//         for (int i = 0; i < (dim); i++)     \
//             (vec)[i] = ((rand()) % (20));   \
//     } while (0)                             \

#define GET_RANDOM(mat, row, col)                   \
    do {                                            \
        for (int (r) = 0; (r) < (row); (r)++)       \
            for (int (c) = 0; (c) < (col); (c)++)   \
                    (mat)[(r)][(c)] = ((rand()) % (20)); \
    } while (0)                                     \

typedef int OP;
enum {ADD, MUL, DP};

void test(int row_a, int col_a, int row_b, int col_b,
            float mat_a[row_a][col_a], float mat_b[row_b][col_b], float mat_c[row_a][col_a],
            OP op) {
    /* matrix dimension check */
    assert((row_a > 0 && row_b > 0 && col_a > 0 && col_b > 0) 
            && "Dimensions must be greater than zero");
    assert(((row_a == row_b) && (col_a == col_b)) 
            && "Dimensions of two vector are not valid");

    switch(op) {
        case ADD: {
            for(int r=0; r<row_a; r++) 
                for(int c=0; c<col_a; c++) 
                    assert(((mat_a[r][c] + mat_b[r][c]) == mat_c[r][c]) 
                    && "Matrix operation is not correct");
            break;
        }
        case MUL: {

            break;
        }
    }


}

#endif