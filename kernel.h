#ifndef KERNEL_H
#define KERNEL_H

#pragma once

#include "sse.h"
#include "util.h"

#if INSTLEVEL >= 8 /* AVX512F */
// No need mask array
#elif INSTLEVEL >= 6 /* AVX, AVX2 */
static int32_t mask[32] __attribute__((aligned(32))) = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
};
#endif

/**
 * Micro kernel for GEMM, implemented with Intel SSE intrinsic
 * 
 * INSTLEVEL >= 8 for AVX512F
 *  14x32 kernel vs 31x16 kernel
 *  32 ZMM registers
 * INSTLEVEL >= 7 for AVX2
 *  6x16 kernel
 *  16 YMM registers
 *  FMA
 * INSTLEVEL >= 6 for AVX
 *  6x16 kernel
 *  16 YMM registers
 *  No FMA
 */
void u_kernel(const float* packed_blockA, const float* packed_blockB, float* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N) {
#if INSTLEVEL >= 8 /* AVX512F */ /* 14x32 micro kernel */
    __m512 packed_C[14][2]; /* 14x32 */
    __m512 b0_blockB, b1_blockB, a_blockA;
    __mmask16 packed_mask_0 = (n < 16)  ? 0xFFFF >> (16 - n) : 0xFFFF;
    __mmask16 packed_mask_1 = (n >= 16) ? 0xFFFF >> (32 - n) : 0xFFFF;

    for (int r = 0; r < m; r++) {
        packed_C[r][0] = _mm512_maskz_loadu_ps(packed_mask_0, &C[r * N + 0]);
        packed_C[r][1] = _mm512_maskz_loadu_ps(packed_mask_1, &C[r * N + 16]);
    }
    for(int k = 0; k < kc; k++) {
        b0_blockB = _mm512_load_ps(packed_blockB + 0);
        b1_blockB = _mm512_load_ps(packed_blockB + 16);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 0]); /* scalar */
        packed_C[0][0] = _mm512_fmadd_ps(b0_blockB, a_blockA, packed_C[0][0]);
        packed_C[0][1] = _mm512_fmadd_ps(b1_blockB, a_blockA, packed_C[0][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 1]); /* scalar */
        packed_C[1][0] = _mm512_fmadd_ps(b0_blockB, a_blockA, packed_C[1][0]);
        packed_C[1][1] = _mm512_fmadd_ps(b1_blockB, a_blockA, packed_C[1][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 2]); /* scalar */
        packed_C[2][0] = _mm512_fmadd_ps(b0_blockB, a_blockA, packed_C[2][0]);
        packed_C[2][1] = _mm512_fmadd_ps(b1_blockB, a_blockA, packed_C[2][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 3]); /* scalar */
        packed_C[3][0] = _mm512_fmadd_ps(b0_blockB, a_blockA, packed_C[3][0]);
        packed_C[3][1] = _mm512_fmadd_ps(b1_blockB, a_blockA, packed_C[3][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 4]); /* scalar */
        packed_C[4][0] = _mm512_fmadd_ps(b0_blockB, a_blockA, packed_C[4][0]);
        packed_C[4][1] = _mm512_fmadd_ps(b1_blockB, a_blockA, packed_C[4][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 5]); /* scalar */
        packed_C[5][0] = _mm512_fmadd_ps(b0_blockB, a_blockA, packed_C[5][0]);
        packed_C[5][1] = _mm512_fmadd_ps(b1_blockB, a_blockA, packed_C[5][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 6]); /* scalar */
        packed_C[6][0] = _mm512_fmadd_ps(b0_blockB, a_blockA, packed_C[6][0]);
        packed_C[6][1] = _mm512_fmadd_ps(b1_blockB, a_blockA, packed_C[6][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 7]); /* scalar */
        packed_C[7][0] = _mm512_fmadd_ps(b0_blockB, a_blockA, packed_C[7][0]);
        packed_C[7][1] = _mm512_fmadd_ps(b1_blockB, a_blockA, packed_C[7][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 8]); /* scalar */
        packed_C[8][0] = _mm512_fmadd_ps(b0_blockB, a_blockA, packed_C[8][0]);
        packed_C[8][1] = _mm512_fmadd_ps(b1_blockB, a_blockA, packed_C[8][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 9]); /* scalar */
        packed_C[9][0] = _mm512_fmadd_ps(b0_blockB, a_blockA, packed_C[9][0]);
        packed_C[9][1] = _mm512_fmadd_ps(b1_blockB, a_blockA, packed_C[9][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 10]); /* scalar */
        packed_C[10][0] = _mm512_fmadd_ps(b0_blockB, a_blockA, packed_C[10][0]);
        packed_C[10][1] = _mm512_fmadd_ps(b1_blockB, a_blockA, packed_C[10][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 11]); /* scalar */
        packed_C[11][0] = _mm512_fmadd_ps(b0_blockB, a_blockA, packed_C[11][0]);
        packed_C[11][1] = _mm512_fmadd_ps(b1_blockB, a_blockA, packed_C[11][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 12]); /* scalar */
        packed_C[12][0] = _mm512_fmadd_ps(b0_blockB, a_blockA, packed_C[12][0]);
        packed_C[12][1] = _mm512_fmadd_ps(b1_blockB, a_blockA, packed_C[12][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 13]); /* scalar */
        packed_C[13][0] = _mm512_fmadd_ps(b0_blockB, a_blockA, packed_C[13][0]);
        packed_C[13][1] = _mm512_fmadd_ps(b1_blockB, a_blockA, packed_C[13][1]);

        packed_blockA += 1; /* next column */
        packed_blockB += NC; /* next 32 elements*/
    }
    for(int r = 0; r < m; r++) {
        _mm512_mask_storeu_ps(&C[r * N + 0], packed_mask_0, packed_C[r][0]);
        _mm512_mask_storeu_ps(&C[r * N + 16], packed_mask_1, packed_C[r][1]);
    }
#elif INSTLEVEL >= 7 /* AVX2 */ /* 6x16 micro kernel */
    __m256 packed_C[6][2]; /* 6x16 */
    __m256 b0_blockB, b1_blockB, a_blockA;
    __m256i packed_mask[2];
    packed_mask[0] = _mm256_loadu_si256(&mask[16 - n + 0]);
    packed_mask[1] = _mm256_loadu_si256(&mask[16 - n + 8]);
    for (int r = 0; r < m; r++) {
        packed_C[r][0] = _mm256_maskload_ps(&C[r * N + 0], packed_mask[0]);
        packed_C[r][1] = _mm256_maskload_ps(&C[r * N + 8], packed_mask[1]);
    }
    for(int k = 0; k < kc; k++) {
        b0_blockB = _mm256_loadu_ps(packed_blockB + 0);
        b1_blockB = _mm256_loadu_ps(packed_blockB + 8);

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 0)); /* scalar */
        packed_C[0][0] = _mm256_fmadd_ps(b0_blockB, a_blockA, packed_C[0][0]);
        packed_C[0][1] = _mm256_fmadd_ps(b1_blockB, a_blockA, packed_C[0][1]);

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 1)); /* scalar */
        packed_C[1][0] = _mm256_fmadd_ps(b0_blockB, a_blockA, packed_C[1][0]);
        packed_C[1][1] = _mm256_fmadd_ps(b1_blockB, a_blockA, packed_C[1][1]);

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 2)); /* scalar */
        packed_C[2][0] = _mm256_fmadd_ps(b0_blockB, a_blockA, packed_C[2][0]);
        packed_C[2][1] = _mm256_fmadd_ps(b1_blockB, a_blockA, packed_C[2][1]);

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 3)); /* scalar */
        packed_C[3][0] = _mm256_fmadd_ps(b0_blockB, a_blockA, packed_C[3][0]);
        packed_C[3][1] = _mm256_fmadd_ps(b1_blockB, a_blockA, packed_C[3][1]);

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 4)); /* scalar */
        packed_C[4][0] = _mm256_fmadd_ps(b0_blockB, a_blockA, packed_C[4][0]);
        packed_C[4][1] = _mm256_fmadd_ps(b1_blockB, a_blockA, packed_C[4][1]);

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 5)); /* scalar */
        packed_C[5][0] = _mm256_fmadd_ps(b0_blockB, a_blockA, packed_C[5][0]);
        packed_C[5][1] = _mm256_fmadd_ps(b1_blockB, a_blockA, packed_C[5][1]);

        packed_blockA += 1; /* next column */
        packed_blockB += NC; /* next 16 elements*/
    }
    for(int r = 0; r < m; r++) {
        _mm256_maskstore_ps(&C[r * N + 0], packed_mask[0], packed_C[r][0]);
        _mm256_maskstore_ps(&C[r * N + 8], packed_mask[1], packed_C[r][1]);
    }
#elif INSTLEVEL >= 6 /* AVX */ /* 6x16 micro kernel */
    __m256 packed_C[6][2]; /* 6x16 */
    __m256 b0_blockB, b1_blockB, a_blockA;
    __m256i packed_mask[2];
    packed_mask[0] = _mm256_loadu_si256(&mask[16 - n + 0]);
    packed_mask[1] = _mm256_loadu_si256(&mask[16 - n + 8]);
    for (int r = 0; r < m; r++) {
        packed_C[r][0] = _mm256_maskload_ps(&C[r * N + 0], packed_mask[0]);
        packed_C[r][1] = _mm256_maskload_ps(&C[r * N + 8], packed_mask[1]);
    }
    for(int k = 0; k < kc; k++) {
        b0_blockB = _mm256_loadu_ps(packed_blockB + 0);
        b1_blockB = _mm256_loadu_ps(packed_blockB + 8);
        
        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 0)); /* scalar */
        packed_C[0][0] = _mm256_add_ps(packed_C[0][0], _mm256_mul_ps(b0_blockB, a_blockA)); /* FMA */
        packed_C[0][1] = _mm256_add_ps(packed_C[0][1], _mm256_mul_ps(b1_blockB, a_blockA)); /* FMA */

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 1)); /* scalar */
        packed_C[1][0] = _mm256_add_ps(packed_C[1][0], _mm256_mul_ps(b0_blockB, a_blockA)); /* FMA */
        packed_C[1][1] = _mm256_add_ps(packed_C[1][1], _mm256_mul_ps(b1_blockB, a_blockA)); /* FMA */

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 2)); /* scalar */
        packed_C[2][0] = _mm256_add_ps(packed_C[2][0], _mm256_mul_ps(b0_blockB, a_blockA)); /* FMA */
        packed_C[2][1] = _mm256_add_ps(packed_C[2][1], _mm256_mul_ps(b1_blockB, a_blockA)); /* FMA */

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 3)); /* scalar */
        packed_C[3][0] = _mm256_add_ps(packed_C[3][0], _mm256_mul_ps(b0_blockB, a_blockA)); /* FMA */
        packed_C[3][1] = _mm256_add_ps(packed_C[3][1], _mm256_mul_ps(b1_blockB, a_blockA)); /* FMA */

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 4)); /* scalar */
        packed_C[4][0] = _mm256_add_ps(packed_C[4][0], _mm256_mul_ps(b0_blockB, a_blockA)); /* FMA */
        packed_C[4][1] = _mm256_add_ps(packed_C[4][1], _mm256_mul_ps(b1_blockB, a_blockA)); /* FMA */

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 5)); /* scalar */
        packed_C[5][0] = _mm256_add_ps(packed_C[5][0], _mm256_mul_ps(b0_blockB, a_blockA)); /* FMA */
        packed_C[5][1] = _mm256_add_ps(packed_C[5][1], _mm256_mul_ps(b1_blockB, a_blockA)); /* FMA */

        packed_blockA += 1; /* next column */
        packed_blockB += NC; /* next 16 elements*/
    }
#if DEBUG
    for(int r = 0; r < m; r++) {
        printf("%5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f ", 
        packed_C[r][0][0], packed_C[r][0][1], packed_C[r][0][2], packed_C[r][0][3],
        packed_C[r][0][4], packed_C[r][0][5], packed_C[r][0][6], packed_C[r][0][7]);

        printf("%5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f\n", 
        packed_C[r][1][0], packed_C[r][1][1], packed_C[r][1][2], packed_C[r][1][3],
        packed_C[r][1][4], packed_C[r][1][5], packed_C[r][1][6], packed_C[r][1][7]);
    }
#endif

    for(int r = 0; r < m; r++) {
        _mm256_maskstore_ps(&C[r * N + 0], packed_mask[0], packed_C[r][0]);
        _mm256_maskstore_ps(&C[r * N + 8], packed_mask[1], packed_C[r][1]);
    }
#endif
}

// void kernel_14x8() {
    
// }

#endif