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

#define ROW 1000
#define COL 1000

#define GET_ROWS(mat) (sizeof(mat) / sizeof(mat[0]))
#define GET_COLS(mat) (sizeof(mat[0]) / sizeof(mat[0][0]))

#define RANDOM_MATRIX(row, col, mat, bound)                           \
    do {                                                    \
        for (int (r) = 0; (r) < (row); (r)++)               \
            for (int (c) = 0; (c) < (col); (c)++)           \
                    (mat)[(r)][(c)] = ((rand()) % (bound));    \
    } while (0)                                             \

/********************************************************
 *                                                     
 *          GEMM Validation Check                               
 *                                                     
*********************************************************/

#define GEMM_VALID_DIM_CHECK(Ar, Ac, Br, Bc)  \
    do {    \
        assert((((Ar) == (Br)) && ((Ac) == (Bc)))   \ 
        && "Dimensions of two matrices are not valid");\
    } while(0)  \

#define GEMM_ZERO_DIM_CHECK(Ar, Ac, Br, Bc)  \
    do {    \
        assert((Ar > 0 && Br > 0 && Ac > 0 && Bc > 0) \
        && "Dimensions must be greater than zero");\
    } while(0)  \

/********************************************************
 *                                                      
 *          Matrix Print                                
 *                                                      
*********************************************************/
#define print(row, col, mat)    \
    _Generic((mat), \ 
            float:fp32_print, \
            double:fp64_print)  \
            (row, col, mat)   \

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