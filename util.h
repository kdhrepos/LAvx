#ifndef UTIL_H
#define UTIL_H 1

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <memory.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <argp.h>

typedef short BOOL;
#define TRUE 1
#define FALSE 0

#define ROW 10
#define COL 10

#define FP32_VEC_DIM(vec) ((sizeof(vec))/(sizeof(float)))
#define FP64_VEC_DIM(vec) ((sizeof(vec))/(sizeof(double)))

// #define ROW(mat) ((sizeof(mat[0]))/(sizeof(mat[0][0])))
// #define COL(mat) ((sizeof(mat[0]))/(sizeof(mat[0][0])))

/* This was for 1-dim array, vector */
// #define GET_RANDOM(vec, dim)                \
//     do {                                    \
//         for (int i = 0; i < (dim); i++)     \
//             (vec)[i] = ((rand()) % (20));   \
//     } while (0)                             \

#define RANDOM_MATRIX(row, col, mat, bound)                           \
    do {                                                    \
        for (int (r) = 0; (r) < (row); (r)++)               \
            for (int (c) = 0; (c) < (col); (c)++)           \
                    (mat)[(r)][(c)] = ((rand()) % (bound));    \
    } while (0)                                             \

#define print(row, col, mat)    \
    _Generic((mat), \ 
            float:fp32_print, \
            double:fp64_print)  \
            (row, col, mat)   \

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

#define mat_print(row, col, mat) \
    _Generic((mat), \
        int32_t (*)[(col)]: int32_print, \
        float (*)[(col)]: fp32_print, \
        double (*)[(col)]: fp64_print \
    )(row, col, mat) \

void int32_print(int row, int col, int32_t mat[row][col]) {
    printf("INT32 Print\n");
    for(int r=0; r<row; r++) {
        for (int c=0; c<col; c++) 
            printf("%d ", mat[r][c]);
        printf("\n");
    }
    printf("\n");
}

void fp32_print(int row, int col, float mat[row][col]) {
    printf("FP32 Print\n");
    for(int r=0; r<row; r++) {
        for (int c=0; c<col; c++) 
            printf("%5.2f ", mat[r][c]);
        printf("\n");
    }
    printf("\n");
}

void fp64_print(int row, int col, double mat[row][col]) {
    printf("FP64 Print\n");
    for(int r=0; r<row; r++) {
        for (int c=0; c<col; c++) 
            printf("%5.2lf ", mat[r][c]);
        printf("\n");
    }
    printf("\n");
}
#endif