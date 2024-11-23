#include "gemm.h"

void gemm(const float* A, const float* B, float* C,
        const int M, const int N, const int K) {

    int MC, KC, NC;
    int NTHREADS = 8;
    cache_opt(NTHREADS, &MC, &KC, &NC);

    /* packing for TLB efficiency */
    float* packed_A = (float* )aligned_alloc(MEM_ALIGN, sizeof(float)* (MC * KC));
    float* packed_B = (float* )aligned_alloc(MEM_ALIGN, sizeof(float)* (KC * NC));

    for(int Bm_col = 0; Bm_col < N; Bm_col += NC) { /* 5th loop */
        const int nc = min(NC, N - Bm_col);         
        for(int k = 0; k < K; k += KC) {            /* 4th loop */
            const int kc = min(KC, K - k);     
            pack_blockB(&B[k * N + Bm_col], packed_B, nc, NC, N, kc);
            for(int Am_row = 0; Am_row < M; Am_row += MC) { /* 3rd loop */
                const int mc = min(MC, M - Am_row);
                pack_blockA(&A[(Am_row * K) + k], packed_A, mc, kc, KC, K);
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
                for(int Ab_row = 0; Ab_row < mc; Ab_row += MR) {    /* 1st loop */
                    for(int Bb_col = 0; Bb_col < nc; Bb_col += NR) {    /* 2nd loop */
                        const int nr = min(NR, nc - Bb_col);
                        const int mr = min(MR, mc - Ab_row);
                        kernel(&packed_A[Ab_row * KC], &packed_B[Bb_col], 
                        &C[((Am_row + Ab_row) * N) + (Bm_col + Bb_col)], mr, kc, KC, nr, NC, N);
                    }
                }
            }
        }
    }

    free(packed_A);
    free(packed_B);
}