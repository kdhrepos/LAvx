#include "gemm.h"

void pack_blockB(const float* B, float* packed_B, const int nc, 
                 const int NC, const int N, const int kc) {
    int NTHREADS = 8; 
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
    for(int Bb_row = 0; Bb_row < nc; Bb_row += NR) {
        int nr = min(NR, nc - Bb_row);
        pack_panelB(&B[Bb_row], &packed_B[Bb_row], nr, NC, N, kc);
    }
#if DEBUG
    printf("Packed B\n");
    for(int r = 0; r < kc; r++) {
        for (int c = 0; c < NR; c++) {
            printf("%5.2f ", packed_B[r * NC + c]);
            if(B[r * N + c] != packed_B[r * NC + c])
                printf("ERROR\n");
        }
        printf("\n");
    }
#endif
}

void pack_panelB(const float* B, float* packed_B, 
                const int nr, const int NC, const int N, const int kc) {
    for(int Bp_row = 0; Bp_row < kc; Bp_row++) {            /* row access */
        for(int Bp_col = 0; Bp_col < nr; Bp_col++) {     /* Bp_col access */
            packed_B[Bp_row * NC + Bp_col] = B[Bp_row * N + Bp_col];
        }
        // for (int Bp_col = nr; Bp_col < 16; Bp_col++) {
            // *packed_B++ = 0;
        // }
    }
}

void pack_blockA(const float* A, float* packed_A, const int mc, 
                 const int kc, const int KC, const int K) {
    int NTHREADS = 8; 
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
    for(int Ab_row = 0; Ab_row < mc; Ab_row += MR) { /* split block to small panels */
        int mr = min(MR, mc - Ab_row);
        pack_panelA(&A[Ab_row * K], &packed_A[Ab_row * KC], mr, kc, KC, K);
    }
#if DEBUG
    printf("Packed A\n");
    for(int r = 0; r < MR; r++) {
        for (int c = 0; c < kc; c++) {
            printf("%5.2f ", packed_A[r * KC + c]);
            if(A[r * K + c] != packed_A[r * KC + c])
                printf("ERROR\n");
        }
        printf("\n");
    }
#endif
}

void pack_panelA(const float* A, float* packed_A, 
                const int mr, const int kc, const int KC, const int K) {
    for(int Ap_row = 0; Ap_row < mr; Ap_row++) {    /* row access */
        for(int Ap_col = 0; Ap_col < kc; Ap_col++) {   /* col access */
            packed_A[Ap_row * KC + Ap_col] = A[Ap_row * K + Ap_col];
        }
        // for (int i = mr; i < 6; i++) {
            // *packed_A++ = 0;
        // }
    }
}