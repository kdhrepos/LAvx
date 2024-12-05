#include "gemm.h"

/**
 * Micro kernel for GEMM, implemented with Intel SSE intrinsic
 * 
 * INSTLEVEL >= 8 for AVX512F
 *  14x32 kernel vs 31x16 kernel
 *  32 ZMM registers
 *  Use FMA
 * INSTLEVEL >= 7 for AVX2
 *  6x16 kernel
 *  16 YMM registers
 *  Use FMA
 * INSTLEVEL >= 6 for AVX
 *  6x16 kernel
 *  16 YMM registers
 *  No FMA
 */
void s_kernel(const float* packed_blockA, const float* packed_blockB, float* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N) {
#if INSTLEVEL >= 8 /* AVX512F */ /* 14x32 kernel */
    __m512 packed_C[14][2]; /* 14x32 */
    __m512 a_blockA, b0_blockB, b1_blockB;
    __mmask16 packed_mask_0 = (n < 16)  ? 0xFFFF >> (16 - n) : 0xFFFF;
    __mmask16 packed_mask_1 = (n >= 16) ? 0xFFFF >> (32 - n) : 0x0000;

    for (int r = 0; r < m; r++) {
        packed_C[r][0] = _mm512_maskz_loadu_ps(packed_mask_0, &C[r * N + 0]);
        packed_C[r][1] = _mm512_maskz_loadu_ps(packed_mask_1, &C[r * N + 16]);
    }
    for(int k = 0; k < kc; k++) {
        b0_blockB = _mm512_load_ps(packed_blockB + 0);
        b1_blockB = _mm512_load_ps(packed_blockB + 16);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 0]); 
        packed_C[0][0] = sfma(a_blockA, b0_blockB, packed_C[0][0]);
        packed_C[0][1] = sfma(a_blockA, b1_blockB, packed_C[0][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 1]); 
        packed_C[1][0] = sfma(a_blockA, b0_blockB, packed_C[1][0]);
        packed_C[1][1] = sfma(a_blockA, b1_blockB, packed_C[1][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 2]); 
        packed_C[2][0] = sfma(a_blockA, b0_blockB, packed_C[2][0]);
        packed_C[2][1] = sfma(a_blockA, b1_blockB, packed_C[2][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 3]); 
        packed_C[3][0] = sfma(a_blockA, b0_blockB, packed_C[3][0]);
        packed_C[3][1] = sfma(a_blockA, b1_blockB, packed_C[3][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 4]); 
        packed_C[4][0] = sfma(a_blockA, b0_blockB, packed_C[4][0]);
        packed_C[4][1] = sfma(a_blockA, b1_blockB, packed_C[4][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 5]); 
        packed_C[5][0] = sfma(a_blockA, b0_blockB, packed_C[5][0]);
        packed_C[5][1] = sfma(a_blockA, b1_blockB, packed_C[5][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 6]); 
        packed_C[6][0] = sfma(a_blockA, b0_blockB, packed_C[6][0]);
        packed_C[6][1] = sfma(a_blockA, b1_blockB, packed_C[6][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 7]); 
        packed_C[7][0] = sfma(a_blockA, b0_blockB, packed_C[7][0]);
        packed_C[7][1] = sfma(a_blockA, b1_blockB, packed_C[7][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 8]); 
        packed_C[8][0] = sfma(a_blockA, b0_blockB, packed_C[8][0]);
        packed_C[8][1] = sfma(a_blockA, b1_blockB, packed_C[8][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 9]); 
        packed_C[9][0] = sfma(a_blockA, b0_blockB, packed_C[9][0]);
        packed_C[9][1] = sfma(a_blockA, b1_blockB, packed_C[9][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 10]); 
        packed_C[10][0] = sfma(a_blockA, b0_blockB, packed_C[10][0]);
        packed_C[10][1] = sfma(a_blockA, b1_blockB, packed_C[10][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 11]); 
        packed_C[11][0] = sfma(a_blockA, b0_blockB, packed_C[11][0]);
        packed_C[11][1] = sfma(a_blockA, b1_blockB, packed_C[11][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 12]); 
        packed_C[12][0] = sfma(a_blockA, b0_blockB, packed_C[12][0]);
        packed_C[12][1] = sfma(a_blockA, b1_blockB, packed_C[12][1]);

        a_blockA = _mm512_set1_ps(packed_blockA[KC * 13]); 
        packed_C[13][0] = sfma(a_blockA, b0_blockB, packed_C[13][0]);
        packed_C[13][1] = sfma(a_blockA, b1_blockB, packed_C[13][1]);

        packed_blockA += 1; /* next column */
        packed_blockB += NC; /* next 32 elements*/
    }
    for(int r = 0; r < m; r++) {
        _mm512_mask_storeu_ps(&C[r * N + 0], packed_mask_0, packed_C[r][0]);
        _mm512_mask_storeu_ps(&C[r * N + 16], packed_mask_1, packed_C[r][1]);
    }
#elif INSTLEVEL >= 7 /* AVX2 */ /* 6x16 kernel */
    __m256 packed_C[6][2]; /* 6x16 */
    __m256 b0_blockB, b1_blockB, a_blockA;
    __m256i packed_mask[2];

    static int32_t mask[32] __attribute__((aligned(32))) = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
    };
    
    packed_mask[0] = _mm256_loadu_si256((__m256i_u*)&mask[16 - n + 0]);
    packed_mask[1] = _mm256_loadu_si256((__m256i_u*)&mask[16 - n + 8]);
    
    for (int r = 0; r < m; r++) {
        packed_C[r][0] = _mm256_maskload_ps(&C[r * N + 0], packed_mask[0]);
        packed_C[r][1] = _mm256_maskload_ps(&C[r * N + 8], packed_mask[1]);
    }
    for(int k = 0; k < kc; k++) {
        b0_blockB = _mm256_loadu_ps(packed_blockB + 0);
        b1_blockB = _mm256_loadu_ps(packed_blockB + 8);

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 0)); 
        packed_C[0][0] = sfma(a_blockA, b0_blockB, packed_C[0][0]);
        packed_C[0][1] = sfma(a_blockA, b1_blockB, packed_C[0][1]);

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 1)); 
        packed_C[1][0] = sfma(a_blockA, b0_blockB, packed_C[1][0]);
        packed_C[1][1] = sfma(a_blockA, b1_blockB, packed_C[1][1]);

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 2)); 
        packed_C[2][0] = sfma(a_blockA, b0_blockB, packed_C[2][0]);
        packed_C[2][1] = sfma(a_blockA, b1_blockB, packed_C[2][1]);

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 3)); 
        packed_C[3][0] = sfma(a_blockA, b0_blockB, packed_C[3][0]);
        packed_C[3][1] = sfma(a_blockA, b1_blockB, packed_C[3][1]);

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 4)); 
        packed_C[4][0] = sfma(a_blockA, b0_blockB, packed_C[4][0]);
        packed_C[4][1] = sfma(a_blockA, b1_blockB, packed_C[4][1]);

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 5)); 
        packed_C[5][0] = sfma(a_blockA, b0_blockB, packed_C[5][0]);
        packed_C[5][1] = sfma(a_blockA, b1_blockB, packed_C[5][1]);

        packed_blockA += 1; /* next column */
        packed_blockB += NC; /* next 16 elements*/
    }
    for(int r = 0; r < m; r++) {
        _mm256_maskstore_ps(&C[r * N + 0], packed_mask[0], packed_C[r][0]);
        _mm256_maskstore_ps(&C[r * N + 8], packed_mask[1], packed_C[r][1]);
    }
#elif INSTLEVEL >= 6 /* AVX */ /* 6x16 kernel */
    __m256 packed_C[6][2]; /* 6x16 */
    __m256 b0_blockB, b1_blockB, a_blockA;
    __m256i packed_mask[2];

    static int32_t mask[32] __attribute__((aligned(32))) = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
    };

    packed_mask[0] = _mm256_loadu_si256((__m256i_u*)&mask[16 - n + 0]);
    packed_mask[1] = _mm256_loadu_si256((__m256i_u*)&mask[16 - n + 8]);
    
    for (int r = 0; r < m; r++) {
        packed_C[r][0] = _mm256_maskload_ps(&C[r * N + 0], packed_mask[0]);
        packed_C[r][1] = _mm256_maskload_ps(&C[r * N + 8], packed_mask[1]);
    }
    for(int k = 0; k < kc; k++) {
        b0_blockB = _mm256_loadu_ps(packed_blockB + 0);
        b1_blockB = _mm256_loadu_ps(packed_blockB + 8);
        
        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 0)); 
        packed_C[0][0] = sfma(a_blockA, b0_blockB, packed_C[0][0]);
        packed_C[0][1] = sfma(a_blockA, b1_blockB, packed_C[0][1]);

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 1)); 
        packed_C[1][0] = sfma(a_blockA, b0_blockB, packed_C[1][0]);
        packed_C[1][1] = sfma(a_blockA, b1_blockB, packed_C[1][1]);

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 2)); 
        packed_C[2][0] = sfma(a_blockA, b0_blockB, packed_C[2][0]);
        packed_C[2][1] = sfma(a_blockA, b1_blockB, packed_C[2][1]);

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 3)); 
        packed_C[3][0] = sfma(a_blockA, b0_blockB, packed_C[3][0]);
        packed_C[3][1] = sfma(a_blockA, b1_blockB, packed_C[3][1]);

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 4)); 
        packed_C[4][0] = sfma(a_blockA, b0_blockB, packed_C[4][0]);
        packed_C[4][1] = sfma(a_blockA, b1_blockB, packed_C[4][1]);

        a_blockA = _mm256_broadcast_ss(packed_blockA + (KC * 5)); 
        packed_C[5][0] = sfma(a_blockA, b0_blockB, packed_C[5][0]);
        packed_C[5][1] = sfma(a_blockA, b1_blockB, packed_C[5][1]);

        packed_blockA += 1; /* next column */
        packed_blockB += NC; /* next 16 elements*/
    }
    for(int r = 0; r < m; r++) {
        _mm256_maskstore_ps(&C[r * N + 0], packed_mask[0], packed_C[r][0]);
        _mm256_maskstore_ps(&C[r * N + 8], packed_mask[1], packed_C[r][1]);
    }
#endif
}

void d_kernel(const double* packed_blockA, const double* packed_blockB, double* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N) {
#if INSTLEVEL >= 8 /* AVX512F */ /* 6x16 kernel */
    __m512d packed_C[6][4]; /* 6x16 */
    __m512d a_blockA, b0_blockB, b1_blockB;
    __mmask16 packed_mask_0 = (n < 8)  ? 0xFFFF >> (8 - n)  : 0xFFFF;
    __mmask16 packed_mask_1 = (n >= 8) ? 0xFFFF >> (16 - n) : 0x0000;

    for (int r = 0; r < m; r++) {
        packed_C[r][0] = _mm512_maskz_loadu_pd(packed_mask_0, &C[r * N + 0]);
        packed_C[r][1] = _mm512_maskz_loadu_pd(packed_mask_1, &C[r * N + 8]);
    }
    for(int k = 0; k < kc; k++) {
        b0_blockB = _mm512_load_pd(packed_blockB + 0);
        b1_blockB = _mm512_load_pd(packed_blockB + 8);

        a_blockA = _mm512_set1_pd(packed_blockA[KC * 0]); 
        packed_C[0][0] = dfma(a_blockA, b0_blockB, packed_C[0][0]);
        packed_C[0][1] = dfma(a_blockA, b1_blockB, packed_C[0][1]);

        a_blockA = _mm512_set1_pd(packed_blockA[KC * 1]); 
        packed_C[1][0] = dfma(a_blockA, b0_blockB, packed_C[1][0]);
        packed_C[1][1] = dfma(a_blockA, b1_blockB, packed_C[1][1]);
        
        a_blockA = _mm512_set1_pd(packed_blockA[KC * 2]); 
        packed_C[2][0] = dfma(a_blockA, b0_blockB, packed_C[2][0]);
        packed_C[2][1] = dfma(a_blockA, b1_blockB, packed_C[2][1]);

        a_blockA = _mm512_set1_pd(packed_blockA[KC * 3]); 
        packed_C[3][0] = dfma(a_blockA, b0_blockB, packed_C[3][0]);
        packed_C[3][1] = dfma(a_blockA, b1_blockB, packed_C[3][1]);

        a_blockA = _mm512_set1_pd(packed_blockA[KC * 4]); 
        packed_C[4][0] = dfma(a_blockA, b0_blockB, packed_C[4][0]);
        packed_C[4][1] = dfma(a_blockA, b1_blockB, packed_C[4][1]);

        a_blockA = _mm512_set1_pd(packed_blockA[KC * 5]); 
        packed_C[5][0] = dfma(a_blockA, b0_blockB, packed_C[5][0]);
        packed_C[5][1] = dfma(a_blockA, b1_blockB, packed_C[5][1]);

        packed_blockA += 1;  /* next column */
        packed_blockB += NC; /* next 16 elements*/
    }
    for(int r = 0; r < m; r++) {
        _mm512_mask_storeu_pd(&C[r * N + 0], packed_mask_0, packed_C[r][0]);
        _mm512_mask_storeu_pd(&C[r * N + 8], packed_mask_1, packed_C[r][1]);
    }
#elif INSTLEVEL >= 7 /* AVX2 */ /* 6x8 kernel */
    __m256d packed_C[6][2]; /* 6x8 */
    __m256d a_blockA, b0_blockB, b1_blockB;
    __m256i packed_mask[2];

    static int64_t mask[16] = {
        -1, -1, -1, -1, -1, -1, -1, -1, 
        0,  0,  0,  0,  0,  0,  0,  0
    };

    packed_mask[0] = _mm256_loadu_si256((__m256i_u*)&mask[8 - n + 0]);
    packed_mask[1] = _mm256_loadu_si256((__m256i_u*)&mask[8 - n + 4]);

    for (int r = 0; r < m; r++) {
        packed_C[r][0] = _mm256_maskload_pd(&C[r * N + 0],  packed_mask[0]);
        packed_C[r][1] = _mm256_maskload_pd(&C[r * N + 4],  packed_mask[1]);
    }
    for(int k = 0; k < kc; k++) {
        b0_blockB = _mm256_load_pd(packed_blockB + 0);
        b1_blockB = _mm256_load_pd(packed_blockB + 4);

        a_blockA = _mm256_broadcast_sd(packed_blockA + (KC * 0));
        packed_C[0][0] = dfma(a_blockA, b0_blockB, packed_C[0][0]);
        packed_C[0][1] = dfma(a_blockA, b1_blockB, packed_C[0][1]);

        a_blockA = _mm256_broadcast_sd(packed_blockA + (KC * 1));
        packed_C[1][0] = dfma(a_blockA, b0_blockB, packed_C[1][0]);
        packed_C[1][1] = dfma(a_blockA, b1_blockB, packed_C[1][1]);

        a_blockA = _mm256_broadcast_sd(packed_blockA + (KC * 2));
        packed_C[2][0] = dfma(a_blockA, b0_blockB, packed_C[2][0]);
        packed_C[2][1] = dfma(a_blockA, b1_blockB, packed_C[2][1]);

        a_blockA = _mm256_broadcast_sd(packed_blockA + (KC * 3));
        packed_C[3][0] = dfma(a_blockA, b0_blockB, packed_C[3][0]);
        packed_C[3][1] = dfma(a_blockA, b1_blockB, packed_C[3][1]);

        a_blockA = _mm256_broadcast_sd(packed_blockA + (KC * 4));
        packed_C[4][0] = dfma(a_blockA, b0_blockB, packed_C[4][0]);
        packed_C[4][1] = dfma(a_blockA, b1_blockB, packed_C[4][1]);

        a_blockA = _mm256_broadcast_sd(packed_blockA + (KC * 5));
        packed_C[5][0] = dfma(a_blockA, b0_blockB, packed_C[5][0]);
        packed_C[5][1] = dfma(a_blockA, b1_blockB, packed_C[5][1]);

        packed_blockA += 1;  /* next column */
        packed_blockB += NC; /* next 8 elements*/
    }
    for(int r = 0; r < m; r++) {
        _mm256_maskstore_pd(&C[r * N + 0],  packed_mask[0], packed_C[r][0]);
        _mm256_maskstore_pd(&C[r * N + 4],  packed_mask[1], packed_C[r][1]);
    }
#elif INSTLEVEL >= 6 /* AVX */ /* 6x8 kernel */
    __m256d packed_C[6][2]; /* 6x8 */
    __m256d a_blockA, b0_blockB, b1_blockB;
    __m256i packed_mask[2];

    static int64_t mask[16] = {
        -1, -1, -1, -1, -1, -1, -1, -1, 
        0,  0,  0,  0,  0,  0,  0,  0
    };

    packed_mask[0] = _mm256_loadu_si256((__m256i_u*)&mask[8 - n + 0]);
    packed_mask[1] = _mm256_loadu_si256((__m256i_u*)&mask[8 - n + 4]);
    for (int r = 0; r < m; r++) {
        packed_C[r][0] = _mm256_maskload_pd(&C[r * N + 0], packed_mask[0]);
        packed_C[r][1] = _mm256_maskload_pd(&C[r * N + 4], packed_mask[1]);
    }
    for(int k = 0; k < kc; k++) {
        b0_blockB = _mm256_loadu_pd(packed_blockB + 0);
        b1_blockB = _mm256_loadu_pd(packed_blockB + 4);
        
        a_blockA = _mm256_broadcast_sd(packed_blockA + (KC * 0));
        packed_C[0][0] = dfma(a_blockA, b0_blockB, packed_C[0][0]);
        packed_C[0][1] = dfma(a_blockA, b1_blockB, packed_C[0][1]);

        a_blockA = _mm256_broadcast_sd(packed_blockA + (KC * 1));
        packed_C[1][0] = dfma(a_blockA, b0_blockB, packed_C[1][0]);
        packed_C[1][1] = dfma(a_blockA, b1_blockB, packed_C[1][1]);

        a_blockA = _mm256_broadcast_sd(packed_blockA + (KC * 2));
        packed_C[2][0] = dfma(a_blockA, b0_blockB, packed_C[2][0]);
        packed_C[2][1] = dfma(a_blockA, b1_blockB, packed_C[2][1]);

        a_blockA = _mm256_broadcast_sd(packed_blockA + (KC * 3));
        packed_C[3][0] = dfma(a_blockA, b0_blockB, packed_C[3][0]);
        packed_C[3][1] = dfma(a_blockA, b1_blockB, packed_C[3][1]);

        a_blockA = _mm256_broadcast_sd(packed_blockA + (KC * 4));
        packed_C[4][0] = dfma(a_blockA, b0_blockB, packed_C[4][0]);
        packed_C[4][1] = dfma(a_blockA, b1_blockB, packed_C[4][1]);

        a_blockA = _mm256_broadcast_sd(packed_blockA + (KC * 5));
        packed_C[5][0] = dfma(a_blockA, b0_blockB, packed_C[5][0]);
        packed_C[5][1] = dfma(a_blockA, b1_blockB, packed_C[5][1]);

        packed_blockA += 1; /* next column */
        packed_blockB += NC; /* next 16 elements*/
    }
    for(int r = 0; r < m; r++) {
        _mm256_maskstore_pd(&C[r * N + 0], packed_mask[0], packed_C[r][0]);
        _mm256_maskstore_pd(&C[r * N + 4], packed_mask[1], packed_C[r][1]);
    }
#endif // d_kernel
}

void i_kernel(const int* packed_blockA, const int* packed_blockB, int* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N) {
#if INSTLEVEL >= 8      /* AVX512F */   /* 14x32 kernel */
    __m512i packed_C[14][2]; /* 14x32 */
    __m512i a_blockA, b0_blockB, b1_blockB;
    __mmask16 packed_mask_0 = (n < 16)  ? 0xFFFF >> (16 - n) : 0xFFFF;
    __mmask16 packed_mask_1 = (n >= 16) ? 0xFFFF >> (32 - n) : 0x0000;

    for (int r = 0; r < m; r++) {
        packed_C[r][0] = _mm512_maskz_loadu_epi32(packed_mask_0, &C[r * N + 0]);
        packed_C[r][1] = _mm512_maskz_loadu_epi32(packed_mask_1, &C[r * N + 16]);
    }
    for(int k = 0; k < kc; k++) {
        b0_blockB = _mm512_load_epi32(packed_blockB + 0);
        b1_blockB = _mm512_load_epi32(packed_blockB + 16);

        a_blockA = _mm512_set1_epi32(packed_blockA[KC * 0]); 
        packed_C[0][0] = _mm512_add_epi32(packed_C[0][0], _mm512_mullo_epi32(b0_blockB, a_blockA));
        packed_C[0][1] = _mm512_add_epi32(packed_C[0][1], _mm512_mullo_epi32(b1_blockB, a_blockA));

        a_blockA = _mm512_set1_epi32(packed_blockA[KC * 1]);  
        packed_C[1][0] = _mm512_add_epi32(packed_C[1][0], _mm512_mullo_epi32(b0_blockB, a_blockA));
        packed_C[1][1] = _mm512_add_epi32(packed_C[1][1], _mm512_mullo_epi32(b1_blockB, a_blockA));

        a_blockA = _mm512_set1_epi32(packed_blockA[KC * 2]); 
        packed_C[2][0] = _mm512_add_epi32(packed_C[2][0], _mm512_mullo_epi32(b0_blockB, a_blockA));
        packed_C[2][1] = _mm512_add_epi32(packed_C[2][1], _mm512_mullo_epi32(b1_blockB, a_blockA));

        a_blockA = _mm512_set1_epi32(packed_blockA[KC * 3]); 
        packed_C[3][0] = _mm512_add_epi32(packed_C[3][0], _mm512_mullo_epi32(b0_blockB, a_blockA));
        packed_C[3][1] = _mm512_add_epi32(packed_C[3][1], _mm512_mullo_epi32(b1_blockB, a_blockA));

        a_blockA = _mm512_set1_epi32(packed_blockA[KC * 4]); 
        packed_C[4][0] = _mm512_add_epi32(packed_C[4][0], _mm512_mullo_epi32(b0_blockB, a_blockA));
        packed_C[4][1] = _mm512_add_epi32(packed_C[4][1], _mm512_mullo_epi32(b1_blockB, a_blockA));

        a_blockA = _mm512_set1_epi32(packed_blockA[KC * 5]); 
        packed_C[5][0] = _mm512_add_epi32(packed_C[5][0], _mm512_mullo_epi32(b0_blockB, a_blockA));
        packed_C[5][1] = _mm512_add_epi32(packed_C[5][1], _mm512_mullo_epi32(b1_blockB, a_blockA));

        a_blockA = _mm512_set1_epi32(packed_blockA[KC * 6]); 
        packed_C[6][0] = _mm512_add_epi32(packed_C[6][0], _mm512_mullo_epi32(b0_blockB, a_blockA));
        packed_C[6][1] = _mm512_add_epi32(packed_C[6][1], _mm512_mullo_epi32(b1_blockB, a_blockA));

        a_blockA = _mm512_set1_epi32(packed_blockA[KC * 7]); 
        packed_C[7][0] = _mm512_add_epi32(packed_C[7][0], _mm512_mullo_epi32(b0_blockB, a_blockA));
        packed_C[7][1] = _mm512_add_epi32(packed_C[7][1], _mm512_mullo_epi32(b1_blockB, a_blockA));

        a_blockA = _mm512_set1_epi32(packed_blockA[KC * 8]); 
        packed_C[8][0] = _mm512_add_epi32(packed_C[8][0], _mm512_mullo_epi32(b0_blockB, a_blockA));
        packed_C[8][1] = _mm512_add_epi32(packed_C[8][1], _mm512_mullo_epi32(b1_blockB, a_blockA));

        a_blockA = _mm512_set1_epi32(packed_blockA[KC * 9]); 
        packed_C[9][0] = _mm512_add_epi32(packed_C[9][0], _mm512_mullo_epi32(b0_blockB, a_blockA));
        packed_C[9][1] = _mm512_add_epi32(packed_C[9][1], _mm512_mullo_epi32(b1_blockB, a_blockA));

        a_blockA = _mm512_set1_epi32(packed_blockA[KC * 10]); 
        packed_C[10][0] = _mm512_add_epi32(packed_C[10][0], _mm512_mullo_epi32(b0_blockB, a_blockA));
        packed_C[10][1] = _mm512_add_epi32(packed_C[10][1], _mm512_mullo_epi32(b1_blockB, a_blockA));

        a_blockA = _mm512_set1_epi32(packed_blockA[KC * 11]); 
        packed_C[11][0] = _mm512_add_epi32(packed_C[11][0], _mm512_mullo_epi32(b0_blockB, a_blockA));
        packed_C[11][1] = _mm512_add_epi32(packed_C[11][1], _mm512_mullo_epi32(b1_blockB, a_blockA));

        a_blockA = _mm512_set1_epi32(packed_blockA[KC * 12]); 
        packed_C[12][0] = _mm512_add_epi32(packed_C[12][0], _mm512_mullo_epi32(b0_blockB, a_blockA));
        packed_C[12][1] = _mm512_add_epi32(packed_C[12][1], _mm512_mullo_epi32(b1_blockB, a_blockA));

        a_blockA = _mm512_set1_epi32(packed_blockA[KC * 13]); 
        packed_C[13][0] = _mm512_add_epi32(packed_C[13][0], _mm512_mullo_epi32(b0_blockB, a_blockA));
        packed_C[13][1] = _mm512_add_epi32(packed_C[13][1], _mm512_mullo_epi32(b1_blockB, a_blockA));

        packed_blockA += 1;     /* next column */
        packed_blockB += NC;    /* next 32 elements*/
    }
    for(int r = 0; r < m; r++) {
        _mm512_mask_storeu_epi32(&C[r * N + 0],  packed_mask_0, packed_C[r][0]);
        _mm512_mask_storeu_epi32(&C[r * N + 16], packed_mask_1, packed_C[r][1]);
    }
#elif INSTLEVEL >= 7    /* AVX2 */      /* 6x16 kernel */
    __m256i packed_C[6][2]; /* 6x16 */
    __m256i a_blockA, b0_blockB, b1_blockB;
    __m256i packed_mask[2];

    static int32_t mask[32] __attribute__((aligned(32))) = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
    };
    
    packed_mask[0] = _mm256_loadu_si256((__m256i_u*)&mask[16 - n + 0]);
    packed_mask[1] = _mm256_loadu_si256((__m256i_u*)&mask[16 - n + 8]);
    
    for (int r = 0; r < m; r++) {
        packed_C[r][0] = _mm256_maskload_epi32(&C[r * N + 0], packed_mask[0]);
        packed_C[r][1] = _mm256_maskload_epi32(&C[r * N + 8], packed_mask[1]);
    }
    for(int k = 0; k < kc; k++) {
        b0_blockB = _mm256_loadu_si256((__m256i_u*)(packed_blockB + 0));
        b1_blockB = _mm256_loadu_si256((__m256i_u*)(packed_blockB + 8));

        a_blockA = _mm256_set1_epi32(packed_blockA[KC * 0]);
        packed_C[0][0] = _mm256_add_epi32(packed_C[0][0], _mm256_mullo_epi32(b0_blockB, a_blockA)); /* FMA */
        packed_C[0][1] = _mm256_add_epi32(packed_C[0][1], _mm256_mullo_epi32(b1_blockB, a_blockA)); /* FMA */

        a_blockA = _mm256_set1_epi32(packed_blockA[KC * 1]);
        packed_C[1][0] = _mm256_add_epi32(packed_C[1][0], _mm256_mullo_epi32(b0_blockB, a_blockA)); /* FMA */
        packed_C[1][1] = _mm256_add_epi32(packed_C[1][1], _mm256_mullo_epi32(b1_blockB, a_blockA)); /* FMA */

        a_blockA = _mm256_set1_epi32(packed_blockA[KC * 2]);
        packed_C[2][0] = _mm256_add_epi32(packed_C[2][0], _mm256_mullo_epi32(b0_blockB, a_blockA)); /* FMA */
        packed_C[2][1] = _mm256_add_epi32(packed_C[2][1], _mm256_mullo_epi32(b1_blockB, a_blockA)); /* FMA */

        a_blockA = _mm256_set1_epi32(packed_blockA[KC * 3]);
        packed_C[3][0] = _mm256_add_epi32(packed_C[3][0], _mm256_mullo_epi32(b0_blockB, a_blockA)); /* FMA */
        packed_C[3][1] = _mm256_add_epi32(packed_C[3][1], _mm256_mullo_epi32(b1_blockB, a_blockA)); /* FMA */

        a_blockA = _mm256_set1_epi32(packed_blockA[KC * 4]);
        packed_C[4][0] = _mm256_add_epi32(packed_C[4][0], _mm256_mullo_epi32(b0_blockB, a_blockA)); /* FMA */
        packed_C[4][1] = _mm256_add_epi32(packed_C[4][1], _mm256_mullo_epi32(b1_blockB, a_blockA)); /* FMA */

        a_blockA = _mm256_set1_epi32(packed_blockA[KC * 5]);
        packed_C[5][0] = _mm256_add_epi32(packed_C[5][0], _mm256_mullo_epi32(b0_blockB, a_blockA)); /* FMA */
        packed_C[5][1] = _mm256_add_epi32(packed_C[5][1], _mm256_mullo_epi32(b1_blockB, a_blockA)); /* FMA */

        packed_blockA += 1;     /* next column */
        packed_blockB += NC;    /* next 16 elements*/
    }
    for(int r = 0; r < m; r++) {
        _mm256_maskstore_epi32(&C[r * N + 0], packed_mask[0], packed_C[r][0]);
        _mm256_maskstore_epi32(&C[r * N + 8], packed_mask[1], packed_C[r][1]);
    }
#endif // i_kernel
}