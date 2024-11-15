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
#include <omp.h>

typedef short BOOL;
#define TRUE 1
#define FALSE 0

#define min(a,b) ((a) < (b) ? (a) : (b))

#define DEBUG FALSE

#define GET_ROWS(mat) (sizeof(mat) / sizeof(mat[0]))
#define GET_COLS(mat) (sizeof(mat[0]) / sizeof(mat[0][0]))

void int32_get_rand_mat(int row, int col, int32_t* mat, int bound) {
    for (int r = 0; r < (row); r++)
        for (int c = 0; c < (col); c++)
            mat[r*col + c] = ((rand()) % (bound));
}

void fp32_get_rand_mat(int row, int col, float* mat, int bound) {
    for (int r = 0; r < row; r++)
        for (int c = 0; c < col; c++)
            mat[r * col + c] = (rand() % bound);
}

void fp64_get_rand_mat(int row, int col, double* mat, int bound) {
    for (int r = 0; r < (row); r++)
        for (int c = 0; c < (col); c++)
            mat[r*col + c] = ((rand()) % (bound));
}

// #define RANDOM_MATRIX(row, col, mat, bound)                           \
//     do {                                                    \
//         for (int r = 0; (r) < (row); (r)++)               \
//             for (int (c) = 0; (c) < (col); (c)++)           \
//                     (mat)[(r)][(c)] = ((rand()) % (bound));    \
//     } while (0)                                             \

/********************************************************
 *                                                     
 *          GEMM Input & Output Validation Check                               
 *                                                     
*********************************************************/

#define GEMM_INPUT_VALID_CHECK(Ar, Ac, Br, Bc)  \
    do {    \
        assert((Ac == Br)   \ 
        && "Dimensions of two matrices are not valid");\
    } while(0)  \

#define GEMM_INPUT_ZERO_CHECK(Ar, Ac, Br, Bc)  \
    do {    \
        assert((Ar > 0 && Br > 0 && Ac > 0 && Bc > 0) \
        && "Dimensions must be greater than zero");\
    } while(0)  \

void int32_scalar_gemm(const int Ar, const int Ac, const int Br, const int Bc,
            const int32_t* A, const int32_t* B, int32_t* C) {
    GEMM_INPUT_VALID_CHECK(Ar, Ac, Br, Bc);
    GEMM_INPUT_ZERO_CHECK(Ar, Ac, Br, Bc);

    for(int r=0; r<Ar; r++){
        for(int c=0; c<Ar; c++) {
            for(int k=0; k<Bc; k++) 
                C[r*Bc + c] += (A[r*Ac + k] * B[k*Br + c]);
        }
    }
}

void fp64_scalar_gemm(const int Ar, const int Ac, const int Br, const int Bc,
            const double* A, const double* B, double* C) {
    GEMM_INPUT_VALID_CHECK(Ar, Ac, Br, Bc);
    GEMM_INPUT_ZERO_CHECK(Ar, Ac, Br, Bc);

    for(int r=0; r<Ar; r++){
        for(int c=0; c<Ar; c++) {
            for(int k=0; k<Bc; k++) 
                C[r*Bc + c] += (A[r*Ac + k] * B[k*Br + c]);
        }
    }
}

BOOL int32_gemm_result_check(int row, int col, int32_t* T, int32_t* C) {
    for(int r=0; r<row; r++)
        for(int c=0; c<col; c++)
            if(T[r*col + c] != C[r*col + c]) 
                return FALSE;
    return TRUE;
}

BOOL fp32_gemm_result_check(int row, int col, float* T, float* C) {
    for(int r=0; r<row; r++)
        for(int c=0; c<col; c++)
            if(T[r*col + c] != C[r*col + c]) 
                return FALSE;
    return TRUE;
}

BOOL fp64_gemm_result_check(int row, int col, double* T, double* C) {
    for(int r=0; r<row; r++)
        for(int c=0; c<col; c++)
            if(T[r*col + c] != C[r*col + c]) 
                return FALSE;
    return TRUE;
}

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

void fp32_print(int row, int col, float* mat) {
    printf("FP32 Print\n");
    for(int r=0; r<row; r++) {
        for (int c=0; c<col; c++) 
            printf("%5.2f ", mat[r * col + c]);
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