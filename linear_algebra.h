/**
 * @name Linear Algebra Functions
 */

#ifndef LIN_ALG_H
#define LIN_ALG_H 1

#pragma once

#include "util.h"
#include "simd.h"

/*****************************************************************************
*
*          Matrix Addition
*
*****************************************************************************/

#define mat_add(row_a, col_a, row_b, col_b, mat_a, mat_b) \ 
    _Generic((mat_a), \
    float   (*)[(col)]: fp32_mat_add,   \
    double  (*)[(col)]: fp64_mat_add    \
    ) (row_a, col_a, row_b, col_b, mat_a, mat_b)   \

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

/* TODO: Error (Segmentation fault) correction */
float** fp32_mat_add(int row_a, int col_a, int row_b, int col_b,
            float mat_a[row_a][col_a], float mat_b[row_b][col_b]) {
    /* matrix dimension check */
    assert((row_a > 0 && row_b > 0 && col_a > 0 && col_b > 0) 
            && "Dimensions must be greater than zero");
    assert(((row_a == row_b) && (col_a == col_b)) 
            && "Dimensions of two matrices are not valid");

    float** mat_c = (float **)malloc(sizeof(float) * (row_a));
    for(int r=0; r<row_a; r++) 
        mat_c[r] = (float *)malloc(sizeof(float) * (col_a));

    // int num = row_a * col_a;
    int quotient = col_a / AVX_FP32;
    int remainder = col_a % AVX_FP32;

    #if INSTLEVEL == 6 /* AVX */
    /* Optimization */
    // int num = rol_a * col_a;
    // for(int i=0; i<num; i++) {
        /* use [mm_blend_ps] */
    // }
    
    __m256i mask;
    if(remainder == 0)      mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0);
    else if(remainder == 1) mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1);
    else if(remainder == 2) mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
    else if(remainder == 3) mask = _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1);
    else if(remainder == 4) mask = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1);
    else if(remainder == 5) mask = _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1);
    else if(remainder == 6) mask = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1);
    else if(remainder == 7) mask = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1);
    
    for(int r=0; r<row_a; r++) {
        for(int c=0; c<col_a-remainder; c+=AVX_FP32) {
            // printf("R: %d | C: %d\n", r, c);
            __m256 avx_a = _mm256_loadu_ps(&mat_a[r][c]);
            __m256 avx_b = _mm256_loadu_ps(&mat_b[r][c]);

            __m256 avx_result = _mm256_add_ps(avx_a, avx_b);
            
            _mm256_storeu_ps(&mat_c[r][c], avx_result);
        }
        // TODO: Optimization for remainder elements
        int c = (quotient * AVX_FP32);
        if(remainder == 0) 
            continue; /* No remainder */
        __m256 avx_a = _mm256_maskload_ps(&mat_a[r][c], mask);
        __m256 avx_b = _mm256_maskload_ps(&mat_b[r][c], mask);
        __m256 avx_result = _mm256_add_ps(avx_a, avx_b);
        _mm256_maskstore_ps(&mat_c[r][c], mask, avx_result);
    }
    return mat_c;
    #endif
}

/* TODO: [v] Error correction (Segmentation fault) on MAT_SIZE >= 600 */
double** fp64_mat_add(int row_a, int col_a, int row_b, int col_b,
            double mat_a[row_a][col_a], double mat_b[row_b][col_b]) {
    /* matrix dimension check */
    assert((row_a > 0 && row_b > 0 && col_a > 0 && col_b > 0) 
            && "Dimensions must be greater than zero");
    assert(((row_a == row_b) && (col_a == col_b)) 
            && "Dimensions of two matrices are not valid");

    /* allocation of new matrix */
    // TODO: Need to do initialize mat_c with 0
    double** mat_c = (double **)malloc(sizeof(double) * (row_a));
    for(int row=0; row<row_a; row++) 
        mat_c[row] = (double *)malloc(sizeof(double) * (col_b));
        
    int num = row_a * col_a;
    int quotient = col_a / AVX_FP64;
    int remained = col_a % AVX_FP64;

    #if INSTLEVEL >= 6 /* AVX */
    __m256i mask;
    if(remained == 0) mask = _mm256_set_epi64x(-1, -1, -1, -1);
    else if(remained == 1) mask = _mm256_set_epi64x(0, 0, 0, -1);
    else if(remained == 2) mask = _mm256_set_epi64x(0, 0, -1, -1);
    else if(remained == 3) mask = _mm256_set_epi64x(0, -1, -1, -1);

    for(int r=0; r<row_a; r++) {
        for(int c=0; c<col_a - remained; c+=AVX_FP64) {
            __m256d avx_a = _mm256_loadu_pd(&mat_a[r][c]);
            __m256d avx_b = _mm256_loadu_pd(&mat_b[r][c]);

            __m256d avx_result = _mm256_add_pd(avx_a, avx_b);
            
            _mm256_storeu_pd(&mat_c[r][c], avx_result);
        }
        // TODO: Optimization for remained elements
        if(remained == 0) 
            continue; /* No remainder */
        int c = (quotient * AVX_FP64);
        __m256d avx_a = _mm256_maskload_pd(&mat_a[r][c], mask);
        __m256d avx_b = _mm256_maskload_pd(&mat_b[r][c], mask);
        __m256d avx_result = _mm256_add_pd(avx_a, avx_b);
        _mm256_maskstore_pd(&mat_c[r][c], mask, avx_result);
    }
    return mat_c;
    #endif
}

/*****************************************************************************
*
*          Scalar Multiplication
*
*****************************************************************************/

#define scalar_mul(row, col, mat, scalar) \
    _Generic((mat), \
    float   (*)[(col)]:  fp32_scalar_mul, \
    double  (*)[(col)]:  fp64_scalar_mul) \
    (row, col, mat, scalar)    \

void fp32_scalar_mul(int row, int col, float mat[row][col], float scalar) {
    assert((row > 0 && col > 0) && "Dimensions must be greater than zero");
    
    int quotient = col / AVX_FP32;
    int remainder = col % AVX_FP32;

    #if INSTLEVEL >= 6 /* AVX */
    __m256i mask;
    if(remainder == 0)      mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0);
    else if(remainder == 1) mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1);
    else if(remainder == 2) mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
    else if(remainder == 3) mask = _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1);
    else if(remainder == 4) mask = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1);
    else if(remainder == 5) mask = _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1);
    else if(remainder == 6) mask = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1);
    else if(remainder == 7) mask = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1);

    __m256 ymm_scalar = _mm256_set1_ps(scalar);
    __m256 ymm_vector;
    __m256 ymm_rslt;
    for(int r=0; r<row; r++) {
        int c=0;
        for(; c<(col-remainder); c+=AVX_FP32) {
            ymm_vector  = _mm256_loadu_ps(&mat[r][c]);
            ymm_rslt    = _mm256_mul_ps(ymm_vector, ymm_scalar);
            _mm256_storeu_ps(&mat[r][c], ymm_rslt);
        }
        ymm_vector = _mm256_maskload_ps(&mat[r][c], mask);
        _mm256_maskstore_ps(&mat[r][c], mask, ymm_rslt);
    }
    #endif
}

void fp64_scalar_mul(int row, int col, double mat[row][col], double scalar) {
    assert((row > 0 && col > 0) && "Dimensions must be greater than zero");
    
    int quotient = col / AVX_FP64;
    int remainder = col % AVX_FP64;

    #if INSTLEVEL >= 6 /* AVX */
    __m256i mask;
    if(remainder == 0) mask = _mm256_set_epi64x(0, 0, 0, 0);
    else if(remainder == 1) mask = _mm256_set_epi64x(0, 0, 0, -1);
    else if(remainder == 2) mask = _mm256_set_epi64x(0, 0, -1, -1);
    else if(remainder == 3) mask = _mm256_set_epi64x(0, -1, -1, -1);

    __m256d ymm_scalar = _mm256_set1_pd(scalar);
    __m256d ymm_vector;
    __m256d ymm_rslt;

    for(int r=0; r<row; r++) {
        int c=0;
        for(; c<(col-remainder); c+=AVX_FP64) {
            ymm_vector  = _mm256_loadu_pd(&mat[r][c]);
            ymm_rslt    = _mm256_mul_pd(ymm_vector, ymm_scalar);
            _mm256_storeu_pd(&mat[r][c], ymm_rslt);
        }
        ymm_vector = _mm256_maskload_pd(&mat[r][c], mask);
        _mm256_maskstore_pd(&mat[r][c], mask, ymm_rslt);
    }
    #endif
}

/*****************************************************************************
*
*          Matrix Multiplication
*
*****************************************************************************/

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
    // TODO: Need to do initialize mat_c with 0
    float** mat_c = (float **)malloc(sizeof(float) * (row_a));
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

/** 
 * Matrix multiplication for two matrices, 64-bit floating point, double.
 * 
 * @ref - https://codereview.stackexchange.com/questions/177616/avx-simd-in-matrix-multiplication
 * @todo:
 *  [ ] Optimization for matrix multiplication using algorithms such as [Winograd, Strassen]
 *  [ ] Optimization with loop unrolling & prefetching
 *  [v] Correctness for various matrix dimension
 */
double** fp64_mat_mul(int row_a, int col_a, int row_b, int col_b,
            double mat_a[row_a][col_a], double mat_b[row_b][col_b]) {
    /* matrix dimension check */
    assert((row_a > 0 && row_b > 0 && col_a > 0 && col_b > 0) 
            && "Dimensions must be greater than zero");
    assert((col_a == row_b)
            && "Dimensions of two matrices are not valid");
    
    /* allocation of new matrix */
    // TODO: Need to do initialize mat_c with 0
    double** mat_c = (double **)malloc(sizeof(double) * (row_a));
    for(int row=0; row<row_a; row++) 
        mat_c[row] = (double *)malloc(sizeof(double) * (col_b));

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
        for(int c=0; c<(col_b - remained); c+=AVX_FP64) {
            __m256d sum = _mm256_setzero_pd();
            for(int k=0; k<col_a; k++) {
                __m256d scalar = _mm256_set1_pd(mat_a[r][k]); /* one element of first matrix */
                __m256d vector = _mm256_loadu_pd(&mat_b[k][c]); /* one row of second matrix */
                __m256d result = _mm256_mul_pd(scalar, vector); /* scalar * vector multiplication */
                sum = _mm256_add_pd(sum, result); /* stacking partial sum */
            }
            _mm256_storeu_pd(&mat_c[r][c], sum);
        }
        // TODO: Optimization for remained elements
        int c = quotient * AVX_FP64;
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

/*****************************************************************************
*
*          Vector Norm
*
*****************************************************************************/

#define l1_norm(row, col, vec)   \
    _Generic((vec),\ 
        float (*)[(col)]: fp32_l1_norm, \
        double (*)[(col)]: fp64_l1_norm)   \
        (row, col, vec) \

float fp32_l1_norm(int row, int col, float vec[row][col]) {
    /* Vector dimension check */
    assert((row > 0 && col > 0) && "Dimensions must be greater than zero");
    assert((row == 1 && col >= 1) && "Only row vectors are allowed");

    int quotient = col / AVX_FP32;
    int remainder = col % AVX_FP32;
    float l1_norm = 0.0;
    
    #if INSTLEVEL >= 6 /* AVX */
    __m256 ymm_sign = _mm256_set1_ps(-0.0f); /* 1000000000...000 */

    for(int r=0; r<row; r++) {
        int c=0;
        for(; c<(col-remainder); c+=AVX_FP32) {
            __m256 ymm_vector = _mm256_loadu_ps(&vec[0][c]);            
            __m256 ymm_rslt   = _mm256_andnot_ps(ymm_sign, ymm_vector); /* !ymm_sign: 0111...111 */
                                                                        /* ymm_vector stays same */
            __m256  sum = _mm256_hadd_ps(ymm_rslt, ymm_rslt);
                    sum = _mm256_hadd_ps(sum, sum);
            __m128 sum_high = _mm256_extractf128_ps(sum, 1); /* extract values at highest index */
            __m128 result = _mm_add_ps(sum_high, _mm256_castps256_ps128(sum));
        
            l1_norm += _mm_cvtss_f32(result); /* extract lowest value from register */;
        }
        for(; c<col; c++) 
            l1_norm += (vec[0][c] > 0 ? vec[0][c] : -vec[0][c]); /* add leftovers (remained values) */
    }
    return l1_norm;
    #endif
}

double fp64_l1_norm(int row, int col, double vec[row][col]) {
    /* Vector dimension check */
    assert((row > 0 && col > 0) && "Dimensions must be greater than zero");
    assert((row == 1 && col >= 1) && "Only row vectors are allowed");

    int quotient = col / AVX_FP64;
    int remainder = col % AVX_FP64;
    double l1_norm = 0.0;
    
    #if INSTLEVEL >= 6 /* AVX */
    __m256d ymm_sign = _mm256_set1_pd(-0.0f); /* 1000000000...000 */

    for(int r=0; r<row; r++) {
        int c=0;
        for(; c<(col-remainder); c+=AVX_FP64) {
            __m256d ymm_vector = _mm256_loadu_pd(&vec[0][c]);            
            __m256d ymm_rslt   = _mm256_andnot_pd(ymm_sign, ymm_vector); /* !ymm_sign: 0111...111 */
                                                                        /* ymm_vector stays same */
            __m256d  sum = _mm256_hadd_pd(ymm_rslt, ymm_rslt);
            __m128d sum_high = _mm256_extractf128_pd(sum, 1); /* extract values at highest index */
            __m128d result = _mm_add_pd(sum_high, _mm256_castpd256_pd128(sum));
        
            l1_norm += _mm_cvtsd_f64(result); /* extract lowest value from register */;
        }
        for(; c<col; c++) 
            l1_norm += (vec[0][c] > 0 ? vec[0][c] : -vec[0][c]); /* add leftovers (remained values) */
    }
    return l1_norm;
    #endif
}

#define l2_norm(row, col, vec)   \
    _Generic((vec),\ 
        float (*)[(col)]: fp32_l2_norm, \
        double (*)[(col)]: fp64_l2_norm)   \
        (row, col, vec) \

double fp32_l2_norm(int row, int col, float vec[row][col]) {
    /* Vector dimension check */
    assert((row > 0 && col > 0) && "Dimensions must be greater than zero");
    assert((row == 1 && col >= 1) && "Only row vectors are allowed");

    int quotient = col / AVX_FP32;
    int remainder = col % AVX_FP32;
    double l2_norm = 0.0;
    
    #if INSTLEVEL >= 6 /* AVX */
    int c=0;
    for(; c<(col - remainder); c+=AVX_FP32) {
        __m256 avx = _mm256_loadu_ps(&vec[0][c]);
        /* reduce sum */
        __m256 sum = _mm256_mul_ps(avx, avx);   
               sum = _mm256_hadd_ps(sum, sum);
               sum = _mm256_hadd_ps(sum, sum);

        __m128 sum_high = _mm256_extractf128_ps(sum, 1); /* extract values at highest index */
        __m128 result = _mm_add_ps(sum_high, _mm256_castps256_ps128(sum));

        l2_norm += _mm_cvtss_f32(result); /* extract lowest value from register */
    }
    for(; c<col; c++) l2_norm += (pow(vec[0][c], 2.0)); /* add leftovers (remained values) */

    return sqrt(l2_norm);
    #endif
}

long double fp64_l2_norm(int row, int col, double vec[row][col]) {
    /* Vector dimension check */
    assert((row > 0 && col > 0) && "Dimensions must be greater than zero");
    assert((row == 1 && col >= 1) && "Only row vectors are allowed");

    int quotient = col / AVX_FP64;
    int remainder = col % AVX_FP64;
    long double l2_norm = 0.0;
    
    #if INSTLEVEL >= 6 /* AVX */
    int c=0;
    for(; c<(col - remainder); c+=AVX_FP64) {
        __m256d avx = _mm256_loadu_pd(&vec[0][c]);
        /* reduce sum */
        __m256d sum = _mm256_mul_pd(avx, avx);
                sum = _mm256_hadd_pd(sum, sum);

        __m128d sum_high = _mm256_extractf128_pd(sum, 1); /* extract values at highest index */
        __m128d result = _mm_add_pd(sum_high, _mm256_castpd256_pd128(sum));

        l2_norm += _mm_cvtsd_f64(result); /* extract lowest value from register */
    }
    for(; c<col; c++) l2_norm += (pow(vec[0][c], 2.0)); /* add leftovers (remained values) */

    return sqrt(l2_norm);
    #endif
}

/*****************************************************************************
*
*          Vector Dot Product
*
*****************************************************************************/

double fp32_vec_dp(int row_a, int col_a, int row_b, int col_b, 
                    float vec_a[row_a][col_a], float vec_b[row_b][col_b]) {
    /* vector dimension check */
    assert((row_a > 0 && col_a > 0) && "Dimensions must be greater than zero");
    assert((row_a >= 1 && col_a == 1) && "Only column vectors are allowed");
    assert((row_b >= 1 && col_b == 1) && "Only column vectors are allowed");
    assert((row_a == row_b && col_a == col_b) && "Two vectors must have same dimension");

    int quotient    = row_a / AVX_FP32;
    int remainder   = row_a % AVX_FP32;
    double result = 0.0;

    #if INSTLEVEL >= 6
    int r=0;
    for (; r < (row_a - remainder); r += AVX_FP32) 
    {
        __m256 avx_a = _mm256_set_ps(vec_a[r][0], vec_a[r+1][0], vec_a[r+2][0], vec_a[r+3][0], 
                                     vec_a[r+4][0], vec_a[r+5][0], vec_a[r+6][0], vec_a[r+7][0]);
        __m256 avx_b = _mm256_set_ps(vec_b[r][0], vec_b[r+1][0], vec_b[r+2][0], vec_b[r+3][0], 
                                     vec_b[r+4][0], vec_b[r+5][0], vec_b[r+6][0], vec_b[r+7][0]);
        
        __m256 avx_result = _mm256_dp_ps(avx_a, avx_b, 0xFF);

        result += (((float*)(&avx_result))[0] +((float*)(&avx_result))[4]);
    }
    for(; r < row_a; r++)
        result += (vec_a[r][0] * vec_b[r][0]);
    return result;
    #endif
}

// void cross_product();

#endif