/**
 * 
 */

#ifndef GEMM_H
#define GEMM_H 1

#pragma once

#include "util.h"
#include "simd.h"

#define ROW 300
#define COL 300

#define mat_mul(row_a, col_a, row_b, col_b, mat_a, mat_b)   \
    _Generic((mat_a),\ 
        float (*)[col_a] : fp32_mat_mul,    \
        double (*)[col_a]: fp64_mat_mul)   \
        (row_a, col_a, row_b, col_b, mat_a, mat_b) \

float** fp32_mat_mul(int row_a, int col_a, int row_b, int col_b,
            float mat_a[row_a][col_a], float mat_b[row_b][col_b]) {
    /* matrix dimension check */
    assert((row_a > 0 && row_b > 0 && col_a > 0 && col_b > 0) 
            && "Dimensions must be greater than zero");
    assert(((row_a == row_b) && (col_a == col_b)) 
            && "Dimensions of two matrices are not valid");

    /* allocation of new matrix */
    // TODO: Need to optimize memory allocation logic
    float** mat_c = (float **)malloc(sizeof(float*) * (row_a));
    for(int row=0; row<row_a; row++) 
        mat_c[row] = (float *)malloc(sizeof(float) * (col_b));
    
    int quotient = col_b / AVX_FP32;
    int remainder = col_b % AVX_FP32;
    
    #if INSTLEVEL >= 6 /* AVX */
    __m256i mask;
    __m256 sum;
    if(remainder == 0)      mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0);
    else if(remainder == 1) mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1);
    else if(remainder == 2) mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
    else if(remainder == 3) mask = _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1);
    else if(remainder == 4) mask = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1);
    else if(remainder == 5) mask = _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1);
    else if(remainder == 6) mask = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1);
    else if(remainder == 7) mask = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1);

    for(int r=0; r<row_a; r++) {
        int c=0;
        for(; c<(col_b - remainder); c+=AVX_FP64) {
            sum = _mm256_setzero_ps();
            for(int k=0; k<col_a; k++) {
                __m256 scalar = _mm256_set1_ps(mat_a[r][k]); /* one element of first matrix */
                __m256 vector = _mm256_loadu_ps(&mat_b[k][c]); /* one row of second matrix */
                __m256 result = _mm256_mul_ps(scalar, vector); /* scalar * vector multiplication */
                sum = _mm256_add_ps(sum, result); /* stacking partial sum */
            }
            _mm256_storeu_ps(&mat_c[r][c], sum);
        }
        // TODO: Optimization for remained elements
        sum = _mm256_setzero_ps();
        for(int k=0; k<col_a; k++) {
            __m256 scalar = _mm256_set1_ps(mat_a[r][k]); /* one element of first matrix */
            __m256 vector = _mm256_maskload_ps(&mat_b[k][c], mask); /* one row of second matrix */
            __m256 result = _mm256_mul_ps(scalar, vector); /* scalar * vector multiplication */
            sum = _mm256_add_ps(sum, result); /* stacking partial sum */
        }
        _mm256_maskstore_ps(&mat_c[r][c], mask, sum);
    }
    return mat_c;
    #else /* No SIMD Extension -> Scalar */
    for(int r=0; r<row_a; r++) 
        for(int c=0; c<col_b; c++)
            for(int k=0; k<col_a; k++) 
                mat_c[r][c] += (mat_a[r][k] * mat_b[k][c]);
    return mat_c;
    #endif
}

double** fp64_mat_mul(int row_a, int col_a, int row_b, int col_b,
            double mat_a[row_a][col_a], double mat_b[row_b][col_b]) {
    /* matrix dimension check */
    assert((row_a > 0 && row_b > 0 && col_a > 0 && col_b > 0) 
            && "Dimensions must be greater than zero");
    assert((col_a == row_b)
            && "Dimensions of two matrices are not valid");
    
    /* allocation of new matrix */
    // TODO: Need to do initialize mat_c with 0
    double** mat_c = (double **)malloc(sizeof(double)* (row_a));
    for(int r=0; r<row_a; r++) 
        mat_c[r] = (double *)malloc(sizeof(double)* (col_b));

    int quotient = col_b / AVX_FP64;
    int remained = col_b % AVX_FP64;

    #if INSTLEVEL >= 6 /* AVX */
    __m256i mask;
    __m256d sum;
    if(remained == 0) mask = _mm256_set_epi64x(0, 0, 0, 0);
    else if(remained == 1) mask = _mm256_set_epi64x(0, 0, 0, -1);
    else if(remained == 2) mask = _mm256_set_epi64x(0, 0, -1, -1);
    else if(remained == 3) mask = _mm256_set_epi64x(0, -1, -1, -1);

    for(int r=0; r<row_a; r++) {
        int c=0;
        for(; c<(col_b - remained); c+=AVX_FP64) {
            sum = _mm256_setzero_pd();
            for(int k=0; k<col_a; k++) {
                __m256d scalar = _mm256_set1_pd(mat_a[r][k]); /* one element of first matrix */
                __m256d vector = _mm256_loadu_pd(&mat_b[k][c]); /* one row of second matrix */
                __m256d result = _mm256_mul_pd(scalar, vector); /* scalar * vector multiplication */
                sum = _mm256_add_pd(sum, result); /* stacking partial sum */
            }
            _mm256_storeu_pd(&mat_c[r][c], sum);
        }
        // TODO: Optimization for remained elements
        sum = _mm256_setzero_pd();
        for(int k=0; k<col_a; k++) {
            __m256d scalar = _mm256_set1_pd(mat_a[r][k]); /* one element of first matrix */
            __m256d vector = _mm256_maskload_pd(&mat_b[k][c], mask); /* one row of second matrix */
            __m256d result = _mm256_mul_pd(scalar, vector); /* scalar * vector multiplication */
            sum = _mm256_add_pd(sum, result); /* stacking partial sum */
        }
        _mm256_maskstore_pd(&mat_c[r][c], mask, sum);
    }

    /** Loop unrolled code
     * @todo : Error handling
     * malloc(): corrupted top size
     * Aborted
     * 
     */
    // for(int r=0; r<row_a; r++) {
    //     for(int c=0; c<col_b; c+=(AVX_FP64 * 2)) {
    //         __m256d sum = _mm256_setzero_pd();
    //         __m256d sum2 = _mm256_setzero_pd();
    //         for(int k=0; k<col_a; k++) {
    //             __m256d scalar = _mm256_set1_pd(mat_a[r][k]); /* one element of first matrix */
    //             __m256d vector = _mm256_loadu_pd(&mat_b[k][c]); /* one row of second matrix */
    //             __m256d vector2 = _mm256_loadu_pd(&mat_b[k][c+AVX_FP64]); /* one row of second matrix */
    //             __m256d result = _mm256_mul_pd(scalar, vector); /* scalar * vector multiplication */
    //             __m256d result2 = _mm256_mul_pd(scalar, vector2);
    //             sum = _mm256_add_pd(sum, result); /* stacking partial sum */
    //             sum2 = _mm256_add_pd(sum2, result2); /* stacking partial sum */
    //         }
    //         _mm256_storeu_pd(&mat_c[r][c], sum);
    //         _mm256_storeu_pd(&mat_c[r][c+AVX_FP64], sum2);
    //     }
    // }
    return mat_c;
    #else /* No SIMD Extension -> Scalar */
    for(int r=0; r<row_a; r++) 
        for(int c=0; c<col_b; c++)
            for(int k=0; k<col_a; k++) 
                mat_c[r][c] += (mat_a[r][k] * mat_b[k][c]);
    return mat_c;
    #endif
}

#endif