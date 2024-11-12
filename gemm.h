#ifndef GEMM_H
#define GEMM_H

#include "../sse.h"
#include "../util.h"

#include "pack.h"
#include "kernel.h"

#define NTHREADS 8

#define MR 6
#define NR 16

#define NC NR * NTHREADS * 1
#define MC MR * NTHREADS * 80
#define KC 200

#define MEM_ALIGN 32

#define min(a,b) ((a) < (b) ? (a) : (b))

void gemm(const float* A, const float* B, float* C,
        const int M, const int N, const int K) {

    /* packing for TLB efficiency */
    float* packed_A = (float* )aligned_alloc(MEM_ALIGN, sizeof(float)* (MC * KC));
    float* packed_B = (float* )aligned_alloc(MEM_ALIGN, sizeof(float)* (KC * NC));

    for(int Bm_col = 0; Bm_col < N; Bm_col += NC) {            /* 5th loop */
        const int nc = min(NC, N - Bm_col);         /* prevent wrong memory access */
        for(int k = 0; k < K; k += KC) {        /* 4th loop */
            const int kc = min(KC, K - k);     /* prevent wrong memory access */
            /**
             * pack B
             * k is row iterator for Bj, 
             * Bm_col is col iterator for Bj.
            */
            pack_blockB(&B[k * N + Bm_col], packed_B, nc, NR, kc, N);
            for(int Am_row = 0; Am_row < M; Am_row += MC) { /* 3rd loop */
                const int mc = min(MC, M - Am_row);
                /**
                 * pack A
                 * Am_row is row iterator for Aj, 
                 * k is col iterator for Aj.
                 */
                pack_blockA(&A[Am_row * K + k], packed_A, mc, MR, kc, K);
                for(int Ab_row = 0; Ab_row < mc; Ab_row += MR) { /* 1st loop */
                    for(int Bb_col = 0; Bb_col < nc; Bb_col += NR) { /* 2nd loop */
                        const int nr = min(NR, nc - Bb_col);
                        const int mr = min(MR, mc - Ab_row);
                        kernel_6x16(&packed_A[Ab_row * KC], &packed_B[Bb_col], 
                        &C[((Am_row + Ab_row) * N) + (Bm_col + Bb_col)], mr, nr, kc, N);
                    }
                }
            }
        }
    }

    free(packed_A);
    free(packed_B);
}

#endif