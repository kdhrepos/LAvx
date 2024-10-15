/**
 * @name Linear Algebra Functions
 */

#ifndef LIN_ALG_H
#define LIN_ALG_H

#pragma once

#include "util.h"
#include "simd.h"

/*****************************************************************************
*
*          Matrix operations
*
*****************************************************************************/

void int32_mat_add(int32_t** mat_a, int32_t** mat_b, int32_t** mat_c,
                   int row_a, int col_a, int row_b, int col_b) {
    /* matrix dimension check */
    assert((row_a > 0 && row_b > 0 && col_a > 0 && col_b > 0) 
            && "Dimensions must be greater than zero");
    assert(((row_a == row_b) && (col_a == col_b)) 
            && "Dimensions of two matrices are not valid");

    #if INSTLEVEL >= 6 /* AVX: No integer addition */
    for(int r=0; r<row_a; r+=8) {
        for(int c=0; c<col_a; c+=8) {
            
        }
    }
    #endif
}

void fp32_mat_add(int row_a, int col_a, int row_b, int col_b,
            float mat_a[row_a][col_a], float mat_b[row_b][col_b], float mat_c[row_a][col_a]) {
    /* matrix dimension check */
    assert((row_a > 0 && row_b > 0 && col_a > 0 && col_b > 0) 
            && "Dimensions must be greater than zero");
    assert(((row_a == row_b) && (col_a == col_b)) 
            && "Dimensions of two matrices are not valid");

    int num = row_a * col_a;
    int quotient = col_a / AVX_FP32;
    int remained = col_a % AVX_FP32;

    #if INSTLEVEL == 6 /* AVX */
    /* Optimization */
    // int num = rol_a * col_a;
    // for(int i=0; i<num; i++) {
        /* use [mm_blend_ps] */
    // }

    /* Simple version */
    for(int r=0; r<row_a; r++) {
        for(int c=0; c<col_a - remained; c+=AVX_FP32) {
            __m256 avx_a = _mm256_loadu_ps(&mat_a[r][c]);
            __m256 avx_b = _mm256_loadu_ps(&mat_b[r][c]);
            // __m256 avx_a = _mm256_set_ps(mat_a[r][c], mat_a[r][c+1], mat_a[r][c+2], mat_a[r][c+3],
            //                              mat_a[r][c+4], mat_a[r][c+5], mat_a[r][c+6], mat_a[r][c+7]);

            // __m256 avx_b = _mm256_set_ps(mat_b[r][c], mat_b[r][c+1], mat_b[r][c+2], mat_b[r][c+3],
            //                              mat_b[r][c+4], mat_b[r][c+5], mat_b[r][c+6], mat_b[r][c+7]);

            __m256 avx_result = _mm256_add_ps(avx_a, avx_b);
            
            _mm256_storeu_ps(&mat_c[r][c], avx_result);
        }
    }
    for(int r=0; r<row_a; r++) {
        int last_c = quotient * AVX_FP32;
        for (int c=last_c; c<last_c+remained; c++)
            mat_c[r][c] = mat_a[r][c] + mat_b[r][c];
    }
    #endif
}

/* TODO: Error correction (Segmentation fault) on MAT_SIZE >= 600 */
void fp64_mat_add(int row_a, int col_a, int row_b, int col_b,
            double mat_a[row_a][col_a], double mat_b[row_b][col_b], double mat_c[row_a][col_a]) {
    /* matrix dimension check */
    assert((row_a > 0 && row_b > 0 && col_a > 0 && col_b > 0) 
            && "Dimensions must be greater than zero");
    assert(((row_a == row_b) && (col_a == col_b)) 
            && "Dimensions of two matrices are not valid");

    int num = row_a * col_a;
    int quotient = col_a / AVX_FP64;
    int remained = col_a % AVX_FP64;

    #if INSTLEVEL >= 6 /* AVX */
    for(int r=0; r<row_a; r++) {
        for(int c=0; c<col_a - remained; c+=AVX_FP64) {
            __m256d avx_a = _mm256_loadu_pd(&mat_a[r][c]);
            __m256d avx_b = _mm256_loadu_pd(&mat_b[r][c]);

            __m256d avx_result = _mm256_add_pd(avx_a, avx_b);
            
            _mm256_storeu_pd(&mat_c[r][c], avx_result);
        }
    }
    for(int r=0; r<row_a; r++) {
        int last_c = quotient * AVX_FP64;
        for (int c=last_c; c<last_c+remained; c++)
            mat_c[r][c] = mat_a[r][c] + mat_b[r][c];
    }
    #endif
}

double** fp32_mat_product(int row_a, int col_a, int row_b, int col_b,
            float mat_a[row_a][col_a], float mat_b[row_b][col_b], float mat_c[row_a][col_a]) {
    /* matrix dimension check */
    assert((row_a > 0 && row_b > 0 && col_a > 0 && col_b > 0) 
            && "Dimensions must be greater than zero");
    assert(((row_a == row_b) && (col_a == col_b)) 
            && "Dimensions of two matrices are not valid");

    
}

/** 
 * Matrix multiplication for two matrices, 64-bit floating point, double.
 * 
 * @ref - https://codereview.stackexchange.com/questions/177616/avx-simd-in-matrix-multiplication
 * @todo - Optimization for matrix multiplication using algorithms such as [Winograd, Strassen]
 */
double** fp64_mat_product(int row_a, int col_a, int row_b, int col_b,
            double mat_a[row_a][col_a], double mat_b[row_b][col_b]) {
    /* matrix dimension check */
    assert((row_a > 0 && row_b > 0 && col_a > 0 && col_b > 0) 
            && "Dimensions must be greater than zero");
    assert((col_a == row_b)
            && "Dimensions of two matrices are not valid");
    
    int num_a = row_a * col_a, num_b = row_b * col_b;
    int quotient = col_a / AVX_FP64;
    int remained = col_a % AVX_FP64;

    /* allocation of new matrix */
    // TODO: Need to do initialize mat_c with 0
    double** mat_c = (double **)malloc(sizeof(double) * (row_a));
    for(int row=0; row<row_b; row++) 
        mat_c[row] = (double *)malloc(sizeof(double) * (col_b));

    #if INSTLEVEL >= 100 /* AVX */
    for(int r=0; r<row_a; r++) {
        for(int c=0; c<col_b; c+=AVX_FP64) {
            __m256d sum = _mm256_setzero_pd();
            for(int k=0; k<col_a; k++) {
                __m256d scalar = _mm256_set1_pd(mat_a[r][k]);
                __m256d vector = _mm256_loadu_pd(&mat_b[k][c]);
                __m256d result = _mm256_mul_pd(scalar, vector);
                sum = _mm256_add_pd(sum, result);
            }
            _mm256_storeu_pd(&mat_c[r][c], sum);
        }
    }
    return mat_c;
    #else
    for(int r=0; r<row_a; r++) 
        for(int c=0; c<col_b; c++)
            for(int k=0; k<col_a; k++) 
                mat_c[r][c] += mat_a[r][k] * mat_b[k][c];
    return mat_c;
    #endif
}

void fp32_vec_dp(float* vec_a, float* vec_b, float* result, int dim_a, int dim_b) {
    /* vector dimension check */
    assert((dim_a > 0 && dim_b > 0) && "Dimensions must be greater than zero");
    assert(((dim_a == dim_b)) && "Dimensions of two vector are not valid");

    #if INSTLEVEL >= 6
    for (int i = 0; i < dim_a; i += 8) 
    {
        __m256 avx_a = _mm256_loadu_ps(&vec_a[i]);
        __m256 avx_b = _mm256_loadu_ps(&vec_b[i]);
        
        __m256 avx_result = _mm256_dp_ps(avx_a, avx_b, 0xFF);

        float dp_result = (((float*)(&avx_result))[0] +((float*)(&avx_result))[4]);

        (*result) += dp_result;
    }

    // __m256 avx_a = _mm256_loadu_ps(vec_a);
    // __m256 avx_b = _mm256_loadu_ps(vec_b);
    
    // __m256 avx_result = _mm256_dp_ps(avx_a, avx_b, 0xFF);

    // float dp_result = (((float*)(&avx_result))[0] +((float*)(&avx_result))[4]);

    // (*result) += dp_result;
    // #elif INSTLEVEL <= 0
    // #error No SIMD instruction set is available
    #endif
}
// void cross_product();
// void l1_norm();
// void l2_norm();

#endif