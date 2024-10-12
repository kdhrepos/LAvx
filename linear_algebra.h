/**
 * @name Linear Algebra Functions
 */

#ifndef L_A_H
#define L_A_H

#pragma once

#include "util.h"
#include "simd.h"

// int8_t** b8_mat_mul(int8_t ** mat_a, int8_t ** mat_b);
// int16_t** b16_mat_mul(int16_t ** mat_a, int16_t ** mat_b);
// int32_t** b32_mat_mul(int32_t ** mat_a, int32_t ** mat_b, int row_a, int col_a, int row_b, int col_b);
// int64_t** b64_mat_mul(int64_t ** mat_a, int64_t ** mat_b);

// int8_t** b8_mat_add(int8_t ** mat_a, int8_t ** mat_b);
// int16_t** b16_mat_add(int16_t ** mat_a, int16_t ** mat_b);
// int32_t** b32_mat_add(int32_t ** mat_a, int32_t ** mat_b);
// int64_t** b64_mat_add(int64_t ** mat_a, int64_t ** mat_b);

// int8_t* b8_dp(int8_t * vec_a, int8_t * vec_b);
// int16_t* b16_dp(int16_t * vec_a, int16_t * vec_b);
// int32_t* b32_dp(int32_t * vec_a, int32_t * vec_b);
// int64_t* b64_dp(int64_t * vec_a, int64_t * vec_b);

// int8_t* b8_vec_add(int8_t * vec_a, int8_t * vec_b);
// int16_t* b16_vec_add(int16_t * vec_a, int16_t * vec_b);

// int8_t* b8_scalar_vec_mul(int8_t * vec, int8_t scalar);
// int16_t* b16_scalar_vec_mul(int16_t * vec, int16_t scalar);
// int32_t* b32_scalar_vec_mul(int32_t * vec, int32_t scalar);
// int64_t* b64_scalar_vec_mul(int64_t * vec, int64_t scalar);

/*****************************************************************************
*
*          Vector operations
*
*****************************************************************************/

void int32_vec_add(int32_t* vec_a, int32_t* vec_b, int32_t* vec_c, int dim_a, int dim_b) {
    /* vector dimension check */
    assert((dim_a > 0 && dim_b > 0) && "Dimensions must be greater than zero");
    assert(((dim_a == dim_b)) && "Dimensions of two vector are not valid");

    #if INSTLEVEL >= 6 /* AVX: No integer addtion */
    #endif
}

void int64_vec_add(int64_t* vec_a, int64_t* vec_b, int64_t* vec_c, int dim_a, int dim_b) {
    /* vector dimension check */
    assert((dim_a > 0 && dim_b > 0) && "Dimensions must be greater than zero");
    assert(((dim_a == dim_b)) && "Dimensions of two vector are not valid");

    #if INSTLEVEL >=6 /* AVX: No integer addtion */
    #endif
}

void fp32_vec_add(float* vec_a, float* vec_b, float* vec_c, int dim_a, int dim_b) {
    /* vector dimension check */
    assert((dim_a > 0 && dim_b > 0) && "Dimensions must be greater than zero");
    assert(((dim_a == dim_b)) && "Dimensions of two vector are not valid");

    #if INSTLEVEL >= 6 /* AVX */
    __m256 avx_a = _mm256_loadu_ps(vec_a);
    __m256 avx_b = _mm256_loadu_ps(vec_b);

    __m256 avx_result = _mm256_add_ps(avx_a, avx_b);
    
    _mm256_storeu_ps(vec_c, avx_result);
    #endif
}

void fp64_vec_add(double* vec_a, double* vec_b, double* vec_c, int dim_a, int dim_b) {
    /* vector dimension check */
    assert((dim_a > 0 && dim_b > 0) && "Dimensions must be greater than zero");
    assert(((dim_a == dim_b)) && "Dimensions of two vector are not valid");

    #if INSTLEVEL >= 6 /* AVX */
    __m256d avx_a = _mm256_loadu_pd(vec_a);
    __m256d avx_b = _mm256_loadu_pd(vec_b);

    __m256d avx_result = _mm256_add_pd(avx_a, avx_b);
    
    _mm256_storeu_pd(vec_c, avx_result);
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