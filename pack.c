#include "gemm.h"

void spack_blockB(const float* B, float* packed_B, const int NR, 
                 const int nc, const int NC, const int N, 
                 const int kc, const int NTHREADS) {
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
    for(int Bb_row = 0; Bb_row < nc; Bb_row += NR) {
        int nr = min(NR, nc - Bb_row);
        spack_panelB(&B[Bb_row], &packed_B[Bb_row], nr, NC, N, kc);
    }
}

void spack_blockA(const float* A, float* packed_A, const int MR,
                const int mc, const int kc, const int KC, 
                const int K, const int NTHREADS) {
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
    for(int Ab_row = 0; Ab_row < mc; Ab_row += MR) { /* split block to small panels */
        int mr = min(MR, mc - Ab_row);
        spack_panelA(&A[Ab_row * K], &packed_A[Ab_row * KC], mr, kc, KC, K);
    }
}

void spack_panelB(const float* B, float* packed_B, const int nr, 
                  const int NC, const int N, const int kc) {
    for(int Bp_row = 0; Bp_row < kc; Bp_row++) {
        for(int Bp_col = 0; Bp_col < nr; Bp_col++) {
            packed_B[Bp_row * NC + Bp_col] = B[Bp_row * N + Bp_col];
        }
        // for (int Bp_col = nr; Bp_col < 16; Bp_col++) {
            // *packed_B++ = 0;
        // }
    }
}

void spack_panelA(const float* A, float* packed_A, const int mr, 
                const int kc, const int KC, const int K) {
    for(int Ap_row = 0; Ap_row < mr; Ap_row++) {    /* row access */
        for(int Ap_col = 0; Ap_col < kc; Ap_col++) {   /* col access */
            packed_A[Ap_row * KC + Ap_col] = A[Ap_row * K + Ap_col];
        }
        // for (int i = mr; i < 6; i++) {
            // *packed_A++ = 0;
        // }
    }
}
void dpack_blockB(const double* B, double* packed_B, const int NR, 
                const int nc, const int NC, const int N, 
                const int kc, const int NTHREADS) {
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
    for(int Bb_row = 0; Bb_row < nc; Bb_row += NR) {
        int nr = min(NR, nc - Bb_row);
        dpack_panelB(&B[Bb_row], &packed_B[Bb_row], nr, NC, N, kc);
    }
}

void dpack_blockA(const double* A, double* packed_A, const int MR,
                const int mc, const int kc, const int KC, 
                const int K, const int NTHREADS) {
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
    for(int Ab_row = 0; Ab_row < mc; Ab_row += MR) { /* split block to small panels */
        int mr = min(MR, mc - Ab_row);
        dpack_panelA(&A[Ab_row * K], &packed_A[Ab_row * KC], mr, kc, KC, K);
    }
}

void dpack_panelB(const double* B, double* packed_B, const int nr, 
                const int NC, const int N, const int kc) {
    for(int Bp_row = 0; Bp_row < kc; Bp_row++) {            /* row access */
        for(int Bp_col = 0; Bp_col < nr; Bp_col++) {     /* Bp_col access */
            packed_B[Bp_row * NC + Bp_col] = B[Bp_row * N + Bp_col];
        }
        // for (int Bp_col = nr; Bp_col < 16; Bp_col++) {
            // *packed_B++ = 0;
        // }
    }
}

void dpack_panelA(const double* A, double* packed_A, const int mr, 
                const int kc, const int KC, const int K) {
    for(int Ap_row = 0; Ap_row < mr; Ap_row++) {    /* row access */
        for(int Ap_col = 0; Ap_col < kc; Ap_col++) {   /* col access */
            packed_A[Ap_row * KC + Ap_col] = A[Ap_row * K + Ap_col];
        }
        // for (int i = mr; i < 6; i++) {
            // *packed_A++ = 0;
        // }
    }
}

void ipack_blockB(const int* B, int* packed_B, const int NR, 
                  const int nc, const int NC, const int N, 
                  const int kc, const int NTHREADS) {
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
    for(int Bb_row = 0; Bb_row < nc; Bb_row += NR) {
        int nr = min(NR, nc - Bb_row);
        ipack_panelB(&B[Bb_row], &packed_B[Bb_row], nr, NC, N, kc);
    }
}

void ipack_blockA(const int* A, int* packed_A, const int MR,
                  const int mc, const int kc, const int KC, 
                  const int K, const int NTHREADS) {
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
    for(int Ab_row = 0; Ab_row < mc; Ab_row += MR) { /* split block to small panels */
        int mr = min(MR, mc - Ab_row);
        ipack_panelA(&A[Ab_row * K], &packed_A[Ab_row * KC], mr, kc, KC, K);
    }
}

void ipack_panelB(const int* B, int* packed_B, const int nr, 
                  const int NC, const int N, const int kc) {
    for(int Bp_row = 0; Bp_row < kc; Bp_row++) {
        for(int Bp_col = 0; Bp_col < nr; Bp_col++) {
            packed_B[Bp_row * NC + Bp_col] = B[Bp_row * N + Bp_col];
        }
        // for (int Bp_col = nr; Bp_col < 16; Bp_col++) {
            // *packed_B++ = 0;
        // }
    }
}

void ipack_panelA(const int* A, int* packed_A, const int mr, 
                  const int kc, const int KC, const int K) {
    for(int Ap_row = 0; Ap_row < mr; Ap_row++) {    /* row access */
        for(int Ap_col = 0; Ap_col < kc; Ap_col++) {   /* col access */
            packed_A[Ap_row * KC + Ap_col] = A[Ap_row * K + Ap_col];
        }
        // for (int i = mr; i < 6; i++) {
            // *packed_A++ = 0;
        // }
    }
}

void qpack_blockB(const int16_t* B, int16_t* packed_B, const int NR, 
                  const int nc, const int NC, const int N, const int kc) {
    int NTHREADS = 8; 
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
    for(int Bb_row = 0; Bb_row < nc; Bb_row += NR) {
        int nr = min(NR, nc - Bb_row);
        qpack_panelB(&B[Bb_row], &packed_B[Bb_row], nr, NC, N, kc);
    }
}

void qpack_blockA(const int16_t* A, int16_t* packed_A, const int MR,
                  const int mc, const int kc, const int KC, const int K) {
    int NTHREADS = 8; 
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
    for(int Ab_row = 0; Ab_row < mc; Ab_row += MR) { /* split block to small panels */
        int mr = min(MR, mc - Ab_row);
        qpack_panelA(&A[Ab_row * K], &packed_A[Ab_row * KC], mr, kc, KC, K);
    }
}

void qpack_panelB(const int16_t* B, int16_t* packed_B, const int nr, 
                  const int NC, const int N, const int kc) {
    for(int Bp_row = 0; Bp_row < kc; Bp_row++) {
        for(int Bp_col = 0; Bp_col < nr; Bp_col++) {
            packed_B[Bp_row * NC + Bp_col] = B[Bp_row * N + Bp_col];
        }
        // for (int Bp_col = nr; Bp_col < 16; Bp_col++) {
            // *packed_B++ = 0;
        // }
    }
}

void qpack_panelA(const int16_t* A, int16_t* packed_A, const int mr, 
                  const int kc, const int KC, const int K) {
    for(int Ap_row = 0; Ap_row < mr; Ap_row++) {        /* row access */
        for(int Ap_col = 0; Ap_col < kc; Ap_col++) {    /* col access */
            packed_A[Ap_row * KC + Ap_col] = A[Ap_row * K + Ap_col];
        }
        // for (int i = mr; i < 6; i++) {
            // *packed_A++ = 0;
        // }
    }
}

void hqpack_blockB(const int8_t* B, int8_t* packed_B, const int NR, 
                  const int nc, const int NC, const int N, const int kc) {
              int NTHREADS = 8; 
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
    for(int Bb_row = 0; Bb_row < nc; Bb_row += NR) {
        int nr = min(NR, nc - Bb_row);
        hqpack_panelB(&B[Bb_row], &packed_B[Bb_row], nr, NC, N, kc);
    }          
}
            
void hqpack_blockA(const int8_t* A, int8_t* packed_A, const int MR,
                  const int mc, const int kc, const int KC, const int K) {
                 int NTHREADS = 8; 
#pragma omp parallel for num_threads(NTHREADS) schedule(static)
    for(int Ab_row = 0; Ab_row < mc; Ab_row += MR) { /* split block to small panels */
        int mr = min(MR, mc - Ab_row);
        hqpack_panelA(&A[Ab_row * K], &packed_A[Ab_row * KC], mr, kc, KC, K);
    }       
}
            
void hqpack_panelB(const int8_t* B, int8_t* packed_B, const int nr, 
                  const int NC, const int N, const int kc) {
    for(int Bp_row = 0; Bp_row < kc; Bp_row++) {
        for(int Bp_col = 0; Bp_col < nr; Bp_col++) {
            packed_B[Bp_row * NC + Bp_col] = B[Bp_row * N + Bp_col];
        }
        // for (int Bp_col = nr; Bp_col < 16; Bp_col++) {
            // *packed_B++ = 0;
        // }
    }              
}
            
void hqpack_panelA(const int8_t* A, int8_t* packed_A, const int mr, 
                  const int kc, const int KC, const int K) {
              for(int Ap_row = 0; Ap_row < mr; Ap_row++) {        /* row access */
        for(int Ap_col = 0; Ap_col < kc; Ap_col++) {    /* col access */
            packed_A[Ap_row * KC + Ap_col] = A[Ap_row * K + Ap_col];
        }
        // for (int i = mr; i < 6; i++) {
            // *packed_A++ = 0;
        // }
    }          
}
            
