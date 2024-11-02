/**
 * gemm.h - General Matrix Multiplication with Intel SSE
 * 
 * @todo   
 *  - [ ] right handling of 2d aBray as a parameter
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

#define mat_mul(Ar, Ac, Br, Bc, A, B)   \
    _Generic((A),\ 
        int32_t (*)[Ac] : int32_mat_mul,    \
        float (*)[Ac] : fp32_mat_mul,    \
        double (*)[Ac]: fp64_mat_mul)   \
        (Ar, Ac, Br, Bc, A, B) \

/**
 * Layered Approach
 * 
 * @ref - Anatomy of High-Performance Matrix Multiplication, Goto et. al.
 */
void gepp();    /* Panel x Panel */
void gemp();    /* Matrix x Panel */
void gepm();    /* Panel x Matrix */
void gebp();    /* Block x Panel */
void gepb();    /* Panel x Block */
void gepdot();  /* Panel x Panel = Dot*/
void gebb();    /* Block x Block */

int16_t int16_mat_mul(int Ar, int Ac, int Br, int Bc,
            int16_t A[Ar][Ac], int16_t B[Br][Bc]) {
    GEMM_INPUT_VALID_CHECK(Ar, Ac, Br, Bc);
    GEMM_INPUT_ZERO_CHECK(Ar, Ac, Br, Bc);

    
}

void int32_gemm(const int Ar, const int Ac, const int Br, const int Bc,
            const int32_t* A, const int32_t* B, int32_t* C) {
    GEMM_INPUT_VALID_CHECK(Ar, Ac, Br, Bc);
    GEMM_INPUT_ZERO_CHECK(Ar, Ac, Br, Bc);

    #if INSTLEVEL >= 7 /* AVX2 */

    // #elif INSTLEVEL >= 6 /* AVX */
    int quotient = Bc / B32_YMM;
    int remainder = Bc % B32_YMM;
    __m256i mask, C_sum;
    if(remainder == 0)      mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0);
    else if(remainder == 1) mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1);
    else if(remainder == 2) mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
    else if(remainder == 3) mask = _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1);
    else if(remainder == 4) mask = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1);
    else if(remainder == 5) mask = _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1);
    else if(remainder == 6) mask = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1);
    else if(remainder == 7) mask = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1);

    for(int r=0; r<Ar; r++) {
        int c=0;
        for(; c<(Bc - remainder); c+=B32_YMM) {
            C_sum = _mm256_setzero_si256();
            for(int k=0; k<Ac; k++) {
                __m256i A_val = _mm256_set1_epi32(A[r*Ac + k]); /* one element of first matrix */
                __m256i B_vec = _mm256_loadu_si256(&B[k*Bc + c]); /* one row of second matrix */
                __m256i C_vec = _mm256_mullo_epi32(A_val, B_vec); /* scalar * vector multiplication */
                C_sum = _mm256_add_epi32(C_sum, C_vec); /* stacking partial sum */
            }
            _mm256_storeu_si256(&C[r*Br + c], C_sum);
        }
        // TODO: Optimization for remainder elements
        C_sum = _mm256_setzero_si256();
        for(int k=0; k<Ac; k++) {
            __m256i A_val = _mm256_set1_epi32(A[r*Ac + k]); /* one element of first matrix */
            __m256i B_vec = _mm256_maskload_epi32(&B[k*Bc + c], mask); /* one row of second matrix */
            __m256i C_vec = _mm256_mullo_epi32(A_val, B_vec); /* scalar * vector multiplication */
            C_sum = _mm256_add_epi32(C_sum, C_vec); /* stacking partial sum */
        }
        _mm256_maskstore_epi32(&C[r*Br + c], mask, C_sum);
    }
    #elif INSTLEVEL >= 5 /* AVX, SSE4 */

    #endif
}

// #define gemm(Ar, Ac, Br, Bc, A, B, C) _Generic((A)  \
//     float* : fp32_gemm \
//     )(Ar, Ac, Br, Bc, A, B, C)  \

void fp32_gemm(const int Ar, const int Ac, const int Br, const int Bc,
            const float* A, const float* B, float* C) {
    GEMM_INPUT_VALID_CHECK(Ar, Ac, Br, Bc);
    GEMM_INPUT_ZERO_CHECK(Ar, Ac, Br, Bc);

    #if INSTLEVEL >= 7 /* AVX2 */

    // #elif INSTLEVEL >= 6 /* AVX */
    int quotient = Bc / B32_YMM;
    int remainder = Bc % B32_YMM;
    __m256i mask;
    __m256 C_sum;
    if(remainder == 0)      mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0);
    else if(remainder == 1) mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1);
    else if(remainder == 2) mask = _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1);
    else if(remainder == 3) mask = _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1);
    else if(remainder == 4) mask = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1);
    else if(remainder == 5) mask = _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1);
    else if(remainder == 6) mask = _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1);
    else if(remainder == 7) mask = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1);

    for(int r=0; r<Ar; r++) {
        int c=0;
        for(; c<(Bc - remainder); c+=B32_YMM) {
            C_sum = _mm256_setzero_ps();
            for(int k=0; k<Ac; k++) {
                __m256 A_val = _mm256_set1_ps(A[r*Ac + k]); /* one element of first matrix */
                __m256 B_vec = _mm256_loadu_ps(&B[k*Bc + c]); /* one row of second matrix */
                __m256 C_vec = _mm256_mul_ps(A_val, B_vec); /* scalar * vector multiplication */
                C_sum = _mm256_add_ps(C_sum, C_vec); /* stacking partial sum */
            }
            _mm256_storeu_ps(&C[r*Br + c], C_sum);
        }
        // TODO: Optimization for remainder elements
        C_sum = _mm256_setzero_ps();
        for(int k=0; k<Ac; k++) {
            __m256 A_val = _mm256_set1_ps(A[r*Ac + k]); /* one element of first matrix */
            __m256 B_vec = _mm256_maskload_ps(&B[k*Bc + c], mask); /* one row of second matrix */
            __m256 C_vec = _mm256_mul_ps(A_val, B_vec); /* scalar * vector multiplication */
            C_sum = _mm256_add_ps(C_sum, C_vec); /* stacking partial sum */
        }
        _mm256_maskstore_ps(&C[r*Br + c], mask, C_sum);
    }
    #else /* No SIMD Extension -> Scalar */
    // for(int r=0; r<Ar; r++) 
    //     for(int c=0; c<Bc; c++)
    //         for(int k=0; k<Ac; k++) 
    //             C[r][c] += (A[r][k] * B[k][c]);
    #endif
}

void fp64_gemm(const int Ar, const int Ac, const int Br, const int Bc,
            const double* A, const double* B, double* C) {
    GEMM_INPUT_VALID_CHECK(Ar, Ac, Br, Bc);
    GEMM_INPUT_ZERO_CHECK(Ar, Ac, Br, Bc);
    
    #if INSTLEVEL >= 7 /* AVX2 */

    // #elif INSTLEVEL >= 6 /* AVX */
    int quotient = Bc / B64_YMM;
    int remainder = Bc % B64_YMM;
    __m256i mask;
    __m256d C_sum;
    if(remainder == 0) mask = _mm256_set_epi64x(0, 0, 0, 0);
    else if(remainder == 1) mask = _mm256_set_epi64x(0, 0, 0, -1);
    else if(remainder == 2) mask = _mm256_set_epi64x(0, 0, -1, -1);
    else if(remainder == 3) mask = _mm256_set_epi64x(0, -1, -1, -1);

    for(int r=0; r<Ar; r++) {
        int c=0;
        for(; c<(Bc - remainder); c+=B64_YMM) {
            C_sum = _mm256_setzero_pd();
            for(int k=0; k<Ac; k++) {
                __m256d A_val = _mm256_set1_pd(A[r*Ac + k]); /* one element of first matrix */
                __m256d B_vec = _mm256_loadu_pd(&B[k*Bc + c]); /* one row of second matrix */
                __m256d C_vec = _mm256_mul_pd(A_val, B_vec); /* scalar * vector multiplication */
                C_sum = _mm256_add_pd(C_sum, C_vec); /* stacking partial sum */
            }
            _mm256_storeu_pd(&C[r*Br + c], C_sum);
        }
        // TODO: Optimization for remainder elements
        C_sum = _mm256_setzero_pd();
        for(int k=0; k<Ac; k++) {
            __m256d A_val = _mm256_set1_pd(A[r*Ac + k]); /* one element of first matrix */
            __m256d B_vec = _mm256_maskload_pd(&B[k*Bc + c], mask); /* one row of second matrix */
            __m256d C_vec = _mm256_mul_pd(A_val, B_vec); /* scalar * vector multiplication */
            C_sum = _mm256_add_pd(C_sum, C_vec); /* stacking partial sum */
        }
        _mm256_maskstore_pd(&C[r*Br + c], mask, C_sum);
    }

    /** Loop unrolled code
     * @todo : EBror handling
     * malloc(): coBrupted top size
     * Aborted
     * 
     */
    // for(int r=0; r<Ar; r++) {
    //     for(int c=0; c<Bc; c+=(B64_YMM * 2)) {
    //         __m256d sum = _mm256_setzero_pd();
    //         __m256d sum2 = _mm256_setzero_pd();
    //         for(int k=0; k<Ac; k++) {
    //             __m256d scalar = _mm256_set1_pd(A[r][k]); /* one element of first matrix */
    //             __m256d vector = _mm256_loadu_pd(&B[k][c]); /* one row of second matrix */
    //             __m256d vector2 = _mm256_loadu_pd(&B[k][c+B64_YMM]); /* one row of second matrix */
    //             __m256d result = _mm256_mul_pd(scalar, vector); /* scalar * vector multiplication */
    //             __m256d result2 = _mm256_mul_pd(scalar, vector2);
    //             sum = _mm256_add_pd(sum, result); /* stacking partial sum */
    //             sum2 = _mm256_add_pd(sum2, result2); /* stacking partial sum */
    //         }
    //         _mm256_storeu_pd(&C[r][c], sum);
    //         _mm256_storeu_pd(&C[r][c+B64_YMM], sum2);
    //     }
    // }
    #else /* No SIMD Extension -> Scalar */
    // for(int r=0; r<Ar; r++) 
    //     for(int c=0; c<Bc; c++)
    //         for(int k=0; k<Ac; k++) 
    //             C[r][c] += (A[r][k] * B[k][c]);
    #endif
}

#endif