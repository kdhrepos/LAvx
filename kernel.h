#ifndef KERNEL_H
#define KERNEL_H

#pragma once

#include "sse.h"
#include "util.h"

void u_kernel(const float* packed_blockA, const float* packed_blockB, float* C,
                const int m, const int kc, const int KC, 
                const int n, const int N, const int NC);
// void kernel_14x8();

#if INSTLEVEL >= 6 /* AVX */
static int32_t mask[32] __attribute__((aligned(32))) = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};
#endif

void u_kernel(const float* packed_blockA, const float* packed_blockB, float* C,
                const int m, const int kc, const int KC, 
                const int n, const int NC, const int N) {
#if INSTLEVEL >= 7 /* AVX2 */
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
#elif INSTLEVEL >= 6 /* AVX */
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