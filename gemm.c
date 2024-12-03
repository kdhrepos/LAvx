#include "gemm.h"

void sgemm(const float* A, const float* B, float* C,
        const int M, const int N, const int K) {

#if INSTLEVEL >= 8 /* AVX512F */
    int MR = 14, NR = 32;
#elif INSTLEVEL >= 6  /* AVX, AVX2 */
    int MR = 6,  NR = 16;
#endif

    int MC, KC, NC;
    int NTHREADS = 8;
    cache_opt(NTHREADS, MR, NR, &MC, &KC, &NC, D_FP32);

    /* packing for TLB efficiency */
    float* packed_A = (float* )aligned_alloc(MEM_ALIGN, sizeof(float)* (MC * KC));
    float* packed_B = (float* )aligned_alloc(MEM_ALIGN, sizeof(float)* (KC * NC));

    for(int Bm_col = 0; Bm_col < N; Bm_col += NC) { /* 5th loop */
        const int nc = min(NC, N - Bm_col);         
        for(int k = 0; k < K; k += KC) {            /* 4th loop */
            const int kc = min(KC, K - k);     
            spack_blockB(&B[k * N + Bm_col], packed_B, NR, nc, NC, N, kc);
            for(int Am_row = 0; Am_row < M; Am_row += MC) { /* 3rd loop */
                const int mc = min(MC, M - Am_row);
                spack_blockA(&A[(Am_row * K) + k], packed_A, MR, mc, kc, KC, K);
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
                for(int Ab_row = 0; Ab_row < mc; Ab_row += MR) {    /* 1st loop */
                    for(int Bb_col = 0; Bb_col < nc; Bb_col += NR) {    /* 2nd loop */
                        const int nr = min(NR, nc - Bb_col);
                        const int mr = min(MR, mc - Ab_row);
                        s_kernel(&packed_A[Ab_row * KC], &packed_B[Bb_col], 
                        &C[((Am_row + Ab_row) * N) + (Bm_col + Bb_col)], mr, kc, KC, nr, NC, N);
                    }
                }
            }
        }
    }

    free(packed_A);
    free(packed_B);
}

void dgemm(const double* A, const double* B, double* C,
        const int M, const int N, const int K) {

#if INSTLEVEL >= 8 /* AVX512F */
    int MR = 6, NR = 16;
#elif INSTLEVEL >= 6  /* AVX, AVX2 */
    int MR = 6, NR = 8;
#endif

    int MC, KC, NC;
    int NTHREADS = 8;
    cache_opt(NTHREADS, MR, NR, &MC, &KC, &NC, D_FP64);

    /* packing for TLB efficiency */
    double* packed_A = (double* )aligned_alloc(MEM_ALIGN, sizeof(double)* (MC * KC));
    double* packed_B = (double* )aligned_alloc(MEM_ALIGN, sizeof(double)* (KC * NC));

    for(int Bm_col = 0; Bm_col < N; Bm_col += NC) { /* 5th loop */
        const int nc = min(NC, N - Bm_col);         
        for(int k = 0; k < K; k += KC) {            /* 4th loop */
            const int kc = min(KC, K - k);     
            dpack_blockB(&B[k * N + Bm_col], packed_B, NR, nc, NC, N, kc);
            for(int Am_row = 0; Am_row < M; Am_row += MC) { /* 3rd loop */
                const int mc = min(MC, M - Am_row);
                dpack_blockA(&A[(Am_row * K) + k], packed_A, MR, mc, kc, KC, K);
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
                for(int Ab_row = 0; Ab_row < mc; Ab_row += MR) {    /* 1st loop */
                    for(int Bb_col = 0; Bb_col < nc; Bb_col += NR) {    /* 2nd loop */
                        const int nr = min(NR, nc - Bb_col);
                        const int mr = min(MR, mc - Ab_row);
                        d_kernel(&packed_A[Ab_row * KC], &packed_B[Bb_col], 
                        &C[((Am_row + Ab_row) * N) + (Bm_col + Bb_col)], mr, kc, KC, nr, NC, N);
                    }
                }
            }
        }
    }

    free(packed_A);
    free(packed_B);
}

void igemm(const int* A, const int* B, int* C,
           const int M, const int N, const int K) {

#if INSTLEVEL >= 8 /* AVX512F */
    int MR = 14, NR = 32;
#elif INSTLEVEL >= 6  /* AVX, AVX2 */
    int MR = 6,  NR = 16;
#endif

    int MC, KC, NC;
    int NTHREADS = 8;
    cache_opt(NTHREADS, MR, NR, &MC, &KC, &NC, D_INT32);

    /* packing for TLB efficiency */
    int* packed_A = (int* )aligned_alloc(MEM_ALIGN, sizeof(int)* (MC * KC));
    int* packed_B = (int* )aligned_alloc(MEM_ALIGN, sizeof(int)* (KC * NC));

    for(int Bm_col = 0; Bm_col < N; Bm_col += NC) {                         /* 5th loop */
        const int nc = min(NC, N - Bm_col);         
        for(int k = 0; k < K; k += KC) {                                    /* 4th loop */
            const int kc = min(KC, K - k);     
            ipack_blockB(&B[k * N + Bm_col], packed_B, NR, nc, NC, N, kc);
            for(int Am_row = 0; Am_row < M; Am_row += MC) {                 /* 3rd loop */
                const int mc = min(MC, M - Am_row);
                ipack_blockA(&A[(Am_row * K) + k], packed_A, MR, mc, kc, KC, K);
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
                for(int Ab_row = 0; Ab_row < mc; Ab_row += MR) {            /* 2nd loop */
                    for(int Bb_col = 0; Bb_col < nc; Bb_col += NR) {        /* 1st loop */
                        const int nr = min(NR, nc - Bb_col);
                        const int mr = min(MR, mc - Ab_row);
                        i_kernel(&packed_A[Ab_row * KC], &packed_B[Bb_col], 
                        &C[((Am_row + Ab_row) * N) + (Bm_col + Bb_col)], mr, kc, KC, nr, NC, N);
                    }
                }
            }
        }
    }
    
    free(packed_A);
    free(packed_B);
}

void qgemm(const int16_t* A, const int16_t* B, int16_t* C,
           const int M, const int N, const int K) {

#if INSTLEVEL >= 9 /* AVX512BW */
    int MR = 30, NR = 32;
#elif INSTLEVEL >= 7  /* AVX2 */
    int MR = 6,  NR = 16;
#endif

    int MC, KC, NC;
    int NTHREADS = 8;
    cache_opt(NTHREADS, MR, NR, &MC, &KC, &NC, D_INT16);

    /* packing for TLB efficiency */
    int16_t* packed_A = (int16_t* )aligned_alloc(MEM_ALIGN, sizeof(int16_t)* (MC * KC));
    int16_t* packed_B = (int16_t* )aligned_alloc(MEM_ALIGN, sizeof(int16_t)* (KC * NC));

    for(int Bm_col = 0; Bm_col < N; Bm_col += NC) {                         /* 5th loop */
        const int nc = min(NC, N - Bm_col);         
        for(int k = 0; k < K; k += KC) {                                    /* 4th loop */
            const int kc = min(KC, K - k);     
            qpack_blockB(&B[k * N + Bm_col], packed_B, NR, nc, NC, N, kc);
            for(int Am_row = 0; Am_row < M; Am_row += MC) {                 /* 3rd loop */
                const int mc = min(MC, M - Am_row);
                qpack_blockA(&A[(Am_row * K) + k], packed_A, MR, mc, kc, KC, K);
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
                for(int Ab_row = 0; Ab_row < mc; Ab_row += MR) {            /* 2nd loop */
                    for(int Bb_col = 0; Bb_col < nc; Bb_col += NR) {        /* 1st loop */
                        const int nr = min(NR, nc - Bb_col);
                        const int mr = min(MR, mc - Ab_row);
                        q_kernel(&packed_A[Ab_row * KC], &packed_B[Bb_col], 
                        &C[((Am_row + Ab_row) * N) + (Bm_col + Bb_col)], mr, kc, KC, nr, NC, N);
                    }
                }
            }
        }
    }
    
    free(packed_A);
    free(packed_B);
}

void hqgemm(const int8_t* A, const int8_t* B, int8_t* C,
           const int M, const int N, const int K) {

#if INSTLEVEL >= 9 /* AVX512BW */
    int MR = 30, NR = 32;
#elif INSTLEVEL >= 7  /* AVX2 */
    int MR = 6,  NR = 16;
#endif

    int MC, KC, NC;
    int NTHREADS = 8;
    cache_opt(NTHREADS, MR, NR, &MC, &KC, &NC, D_INT8);

    /* packing for TLB efficiency */
    int8_t* packed_A = (int8_t* )aligned_alloc(MEM_ALIGN, sizeof(int8_t)* (MC * KC));
    int8_t* packed_B = (int8_t* )aligned_alloc(MEM_ALIGN, sizeof(int8_t)* (KC * NC));

    for(int Bm_col = 0; Bm_col < N; Bm_col += NC) {                         /* 5th loop */
        const int nc = min(NC, N - Bm_col);         
        for(int k = 0; k < K; k += KC) {                                    /* 4th loop */
            const int kc = min(KC, K - k);     
            hqpack_blockB(&B[k * N + Bm_col], packed_B, NR, nc, NC, N, kc);
            for(int Am_row = 0; Am_row < M; Am_row += MC) {                 /* 3rd loop */
                const int mc = min(MC, M - Am_row);
                hqpack_blockA(&A[(Am_row * K) + k], packed_A, MR, mc, kc, KC, K);
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
                for(int Ab_row = 0; Ab_row < mc; Ab_row += MR) {            /* 2nd loop */
                    for(int Bb_col = 0; Bb_col < nc; Bb_col += NR) {        /* 1st loop */
                        const int nr = min(NR, nc - Bb_col);
                        const int mr = min(MR, mc - Ab_row);
                        hq_kernel(&packed_A[Ab_row * KC], &packed_B[Bb_col], 
                        &C[((Am_row + Ab_row) * N) + (Bm_col + Bb_col)], mr, kc, KC, nr, NC, N);
                    }
                }
            }
        }
    }
    
    free(packed_A);
    free(packed_B);
}