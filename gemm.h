/**
 * gemm.h - General Matrix Multiplication with Intel SSE
 * 
 * @todo   
 *  - [ ] right handling of 2d array as a parameter
 *  - [ ] optimization for memory allocation
 *  - [ ] optimization for remained elements
 *  - [ ] new algorithm such as strassen, winograd
 */

#ifndef GEMM_H
#define GEMM_H 1

#pragma once

#include "util.h"
#include "sse.h"

#if INSTLEVEL >= 6
#define B32_YMM 8 /* 256-ymm / 32-bit */
#define B64_YMM 4 /* 256-ymm / 64-bit */
#endif

#define mat_mul(row_a, col_a, row_b, col_b, mat_a, mat_b)   \
    _Generic((mat_a),\ 
        int32_t (*)[col_a] : int32_mat_mul,    \
        float (*)[col_a] : fp32_mat_mul,    \
        double (*)[col_a]: fp64_mat_mul)   \
        (row_a, col_a, row_b, col_b, mat_a, mat_b) \

int16_t int16_mat_mul(int row_a, int col_a, int row_b, int col_b,
            int16_t mat_a[row_a][col_a], int16_t mat_b[row_b][col_b]) {
    /* matrix dimension check */
    assert((row_a > 0 && row_b > 0 && col_a > 0 && col_b > 0) 
            && "Dimensions must be greater than zero");
    assert(((row_a == row_b) && (col_a == col_b)) 
            && "Dimensions of two matrices are not valid");

    
}

int32_t** int32_mat_mul(int row_a, int col_a, int row_b, int col_b,
            int32_t mat_a[row_a][col_a], int32_t mat_b[row_b][col_b]) {
    /* matrix dimension check */
    assert((row_a > 0 && row_b > 0 && col_a > 0 && col_b > 0) 
            && "Dimensions must be greater than zero");
    assert(((row_a == row_b) && (col_a == col_b)) 
            && "Dimensions of two matrices are not valid");

    int32_t** mat_c = (int32_t **)malloc(sizeof(int32_t*) * (row_a));
    for(int row=0; row<row_a; row++) 
        mat_c[row] = (int32_t *)malloc(sizeof(int32_t) * (col_b));

    #if INSTLEVEL >= 7 /* AVX2 */
    int quotient = col_b / B32_YMM;
    int remainder = col_b % B32_YMM;
    __m256i mask, sum;
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
        for(; c<(col_b - remainder); c+=B32_YMM) {
            sum = _mm256_setzero_si256();
            for(int k=0; k<col_a; k++) {
                __m256i scalar = _mm256_set1_epi32(mat_a[r][k]); /* one element of first matrix */
                __m256i vector = _mm256_loadu_si256(&mat_b[k][c]); /* one row of second matrix */
                __m256i result = _mm256_mullo_epi32(scalar, vector); /* scalar * vector multiplication */
                sum = _mm256_add_epi32(sum, result); /* stacking partial sum */
            }
            _mm256_storeu_si256(&mat_c[r][c], sum);
        }
        // TODO: Optimization for remainder elements
        sum = _mm256_setzero_si256();
        for(int k=0; k<col_a; k++) {
            __m256i scalar = _mm256_set1_epi32(mat_a[r][k]); /* one element of first matrix */
            __m256i vector = _mm256_maskload_epi32(&mat_b[k][c], mask); /* one row of second matrix */
            __m256i result = _mm256_mullo_epi32(scalar, vector); /* scalar * vector multiplication */
            sum = _mm256_add_epi32(sum, result); /* stacking partial sum */
        }
        _mm256_maskstore_epi32(&mat_c[r][c], mask, sum);
    }
    return mat_c;
    #elif INSTLEVEL >= 5 /* AVX, SSE4 */

    #endif
}

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
    
    // #if INSTLEVEL >= 7 /* AVX2 */

    // return mat_c;
    #if INSTLEVEL >= 6
    // #elif INSTLEVEL >= 6 /* AVX */
    int quotient = col_b / B32_YMM;
    int remainder = col_b % B32_YMM;
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
        for(; c<(col_b - remainder); c+=B32_YMM) {
            sum = _mm256_setzero_ps();
            for(int k=0; k<col_a; k++) {
                __m256 scalar = _mm256_set1_ps(mat_a[r][k]); /* one element of first matrix */
                __m256 vector = _mm256_loadu_ps(&mat_b[k][c]); /* one row of second matrix */
                __m256 result = _mm256_mul_ps(scalar, vector); /* scalar * vector multiplication */
                sum = _mm256_add_ps(sum, result); /* stacking partial sum */
            }
            _mm256_storeu_ps(&mat_c[r][c], sum);
        }
        // TODO: Optimization for remainder elements
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
    // TODO: Need to optimize memory allocation logic
    double** mat_c = (double **)malloc(sizeof(double)* (row_a));
    for(int r=0; r<row_a; r++) 
        mat_c[r] = (double *)malloc(sizeof(double)* (col_b));

    #if INSTLEVEL >= 7

    return mat_c;
    #elif INSTLEVEL >= 6 /* AVX */
    int quotient = col_b / B64_YMM;
    int remainder = col_b % B64_YMM;
    __m256i mask;
    __m256d sum;
    if(remainder == 0) mask = _mm256_set_epi64x(0, 0, 0, 0);
    else if(remainder == 1) mask = _mm256_set_epi64x(0, 0, 0, -1);
    else if(remainder == 2) mask = _mm256_set_epi64x(0, 0, -1, -1);
    else if(remainder == 3) mask = _mm256_set_epi64x(0, -1, -1, -1);

    for(int r=0; r<row_a; r++) {
        int c=0;
        for(; c<(col_b - remainder); c+=B64_YMM) {
            sum = _mm256_setzero_pd();
            for(int k=0; k<col_a; k++) {
                __m256d scalar = _mm256_set1_pd(mat_a[r][k]); /* one element of first matrix */
                __m256d vector = _mm256_loadu_pd(&mat_b[k][c]); /* one row of second matrix */
                __m256d result = _mm256_mul_pd(scalar, vector); /* scalar * vector multiplication */
                sum = _mm256_add_pd(sum, result); /* stacking partial sum */
            }
            _mm256_storeu_pd(&mat_c[r][c], sum);
        }
        // TODO: Optimization for remainder elements
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
    //     for(int c=0; c<col_b; c+=(B64_YMM * 2)) {
    //         __m256d sum = _mm256_setzero_pd();
    //         __m256d sum2 = _mm256_setzero_pd();
    //         for(int k=0; k<col_a; k++) {
    //             __m256d scalar = _mm256_set1_pd(mat_a[r][k]); /* one element of first matrix */
    //             __m256d vector = _mm256_loadu_pd(&mat_b[k][c]); /* one row of second matrix */
    //             __m256d vector2 = _mm256_loadu_pd(&mat_b[k][c+B64_YMM]); /* one row of second matrix */
    //             __m256d result = _mm256_mul_pd(scalar, vector); /* scalar * vector multiplication */
    //             __m256d result2 = _mm256_mul_pd(scalar, vector2);
    //             sum = _mm256_add_pd(sum, result); /* stacking partial sum */
    //             sum2 = _mm256_add_pd(sum2, result2); /* stacking partial sum */
    //         }
    //         _mm256_storeu_pd(&mat_c[r][c], sum);
    //         _mm256_storeu_pd(&mat_c[r][c+B64_YMM], sum2);
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