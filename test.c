#include "test.h"

void sgemm_test(const int M, const int N, const int K, const int niter,
                const int range, const int bound, FILE* file, BOOL console_flag) {
    int error_num = 0;

    for(int m = M; m < (M + range); m++) {
    for(int n = N; n < (N + range); n++) {
    for(int k = K; k < (K + range); k++) {
        BOOL is_valid_gemm = FALSE;
        double exec_times[3]  = {0, __FLT_MIN__, __FLT_MAX__};    /* avg max min */
        double gflops[3]      = {0, __FLT_MIN__, __FLT_MAX__};    /* avg max min */

        float* A = (float *)malloc(m * k * sizeof(float));
        float* B = (float *)malloc(k * n * sizeof(float));
        float* C = (float *)malloc(m * n * sizeof(float));

        for(int i = 0; i < niter; i++) {
            memset(A, 0, sizeof(float) * m * k);
            memset(B, 0, sizeof(float) * k * n);
            memset(C, 0, sizeof(float) * m * n);

            fp32_get_rand_mat(m, k, A, bound);
            fp32_get_rand_mat(k, n, B, bound);

            double FLOP = 2 * (double)m * n * k;

            uint64_t start = timer();
            sgemm(A, B, C, m, n, k);
            uint64_t end = timer();
            double elapsed = (end - start) * 1e-9;
            double FLOPS = FLOP / elapsed;

            if(i == 0) is_valid_gemm = naive_sgemm(A, B, C, m, n, k);

            // if range is not 0, don't print each results
            if(range == 0) {
                printf("Exec. time = %.3lfms\n", elapsed * 1000);
                printf("GFLOPS = %.3lf\n", FLOPS / 1e9);
            }

            exec_times[0] += (elapsed * 1000);
            exec_times[1] = (exec_times[1] < (elapsed * 1000) ? (elapsed * 1000) : exec_times[1]);
            exec_times[2] = (exec_times[2] > (elapsed * 1000) ? (elapsed * 1000) : exec_times[2]);
            
            gflops[0] += (FLOPS / 1e9);
            gflops[1] = (gflops[1] < (FLOPS / 1e9) ? (FLOPS / 1e9) : gflops[1]);
            gflops[2] = (gflops[2] > (FLOPS / 1e9) ? (FLOPS / 1e9) : gflops[2]);

        }
        free(A);
        free(B);
        free(C);

        if(!is_valid_gemm) error_num++;
        if(console_flag) print_console(m, k, n, niter, exec_times, gflops, is_valid_gemm);
        if(file != NULL) print_file(m, k, n, niter, exec_times, gflops, is_valid_gemm, file);
    }
    }
    }
}

void dgemm_test(const int M, const int N, const int K, const int niter,
                const int range, const int bound, FILE* file, BOOL console_flag) {
    int error_num = 0;

    for(int m = M; m < (M + range); m++) {
    for(int n = N; n < (N + range); n++) {
    for(int k = K; k < (K + range); k++) {
        BOOL is_valid_gemm = FALSE;
        double exec_times[3]  = {0, __FLT_MIN__, __FLT_MAX__};    /* avg max min */
        double gflops[3]      = {0, __FLT_MIN__, __FLT_MAX__};    /* avg max min */

        double* A = (double *)malloc(m * k * sizeof(double));
        double* B = (double *)malloc(k * n * sizeof(double));
        double* C = (double *)malloc(m * n * sizeof(double));

        for(int i = 0; i < niter; i++) {
            memset(A, 0, sizeof(double) * m * k);
            memset(B, 0, sizeof(double) * k * n);
            memset(C, 0, sizeof(double) * m * n);

            fp64_get_rand_mat(m, k, A, bound);
            fp64_get_rand_mat(k, n, B, bound);

            double FLOP = 2 * (double)m * n * k;

            uint64_t start = timer();
            dgemm(A, B, C, m, n, k);
            uint64_t end = timer();
            double elapsed = (end - start) * 1e-9;
            double FLOPS = FLOP / elapsed;

            if(i==0) is_valid_gemm = naive_dgemm(A, B, C, m, n, k);

            // if range is not 0, don't print each results
            if(range == 0) {
                printf("Exec. time = %.3lfms\n", elapsed * 1000);
                printf("GFLOPS = %.3lf\n", FLOPS / 1e9);
            }

            exec_times[0] += (elapsed * 1000);
            exec_times[1] = (exec_times[1] < (elapsed * 1000) ? (elapsed * 1000) : exec_times[1]);
            exec_times[2] = (exec_times[2] > (elapsed * 1000) ? (elapsed * 1000) : exec_times[2]);
            
            gflops[0] += (FLOPS / 1e9);
            gflops[1] = (gflops[1] < (FLOPS / 1e9) ? (FLOPS / 1e9) : gflops[1]);
            gflops[2] = (gflops[2] > (FLOPS / 1e9) ? (FLOPS / 1e9) : gflops[2]);
        }
        free(A);
        free(B);
        free(C);

        if(!is_valid_gemm) error_num++;
        if(console_flag) print_console(m, k, n, niter, exec_times, gflops, is_valid_gemm);
        if(file != NULL) print_file(m, k, n, niter, exec_times, gflops, is_valid_gemm, file);
    }
    }
    }
}

void igemm_test(const int M, const int N, const int K, const int niter,
                const int range, const int bound, FILE* file, BOOL console_flag) {
    int error_num = 0;

    for(int m = M; m < (M + range); m++) {
    for(int n = N; n < (N + range); n++) {
    for(int k = K; k < (K + range); k++) {
        BOOL is_valid_gemm = FALSE;
        double exec_times[3]  = {0, __FLT_MIN__, __FLT_MAX__};    /* avg max min */
        double gflops[3]      = {0, __FLT_MIN__, __FLT_MAX__};    /* avg max min */

        int* A = (int *)malloc(m * k * sizeof(int));
        int* B = (int *)malloc(k * n * sizeof(int));
        int* C = (int *)malloc(m * n * sizeof(int));

        for(int i = 0; i < niter; i++) {
            memset(A, 0, sizeof(int) * m * k);
            memset(B, 0, sizeof(int) * k * n);
            memset(C, 0, sizeof(int) * m * n);

            int32_get_rand_mat(m, k, A, bound);
            int32_get_rand_mat(k, n, B, bound);

            double FLOP = 2 * (double)m * n * k;

            uint64_t start = timer();
            igemm(A, B, C, m, n, k);
            uint64_t end = timer();
            double elapsed = (end - start) * 1e-9;
            double FLOPS = FLOP / elapsed;

            if(i==0) is_valid_gemm = naive_igemm(A, B, C, m, n, k);

            // if range is not 0, don't print each results
            if(range == 0) {
                printf("Exec. time = %.3lfms\n", elapsed * 1000);
                printf("GFLOPS = %.3lf\n", FLOPS / 1e9);
            }

            exec_times[0] += (elapsed * 1000);
            exec_times[1] = (exec_times[1] < (elapsed * 1000) ? (elapsed * 1000) : exec_times[1]);
            exec_times[2] = (exec_times[2] > (elapsed * 1000) ? (elapsed * 1000) : exec_times[2]);
            
            gflops[0] += (FLOPS / 1e9);
            gflops[1] = (gflops[1] < (FLOPS / 1e9) ? (FLOPS / 1e9) : gflops[1]);
            gflops[2] = (gflops[2] > (FLOPS / 1e9) ? (FLOPS / 1e9) : gflops[2]);
        }
        free(A);
        free(B);
        free(C);

        if(!is_valid_gemm) error_num++;
        if(console_flag) print_console(m, k, n, niter, exec_times, gflops, is_valid_gemm);
        if(file != NULL) print_file(m, k, n, niter, exec_times, gflops, is_valid_gemm, file);
    }
    }
    }
}

void qgemm_test(const int M, const int N, const int K, const int niter,
                const int range, const int bound, FILE* file, BOOL console_flag) {
int error_num = 0;

    for(int m = M; m < (M + range); m++) {
    for(int n = N; n < (N + range); n++) {
    for(int k = K; k < (K + range); k++) {
        BOOL is_valid_gemm = FALSE;
        double exec_times[3]  = {0, __FLT_MIN__, __FLT_MAX__};    /* avg max min */
        double gflops[3]      = {0, __FLT_MIN__, __FLT_MAX__};    /* avg max min */

        int16_t* A = (int16_t *)malloc(m * k * sizeof(int16_t));
        int16_t* B = (int16_t *)malloc(k * n * sizeof(int16_t));
        int16_t* C = (int16_t *)malloc(m * n * sizeof(int16_t));

        for(int i = 0; i < niter; i++) {
            memset(A, 0, sizeof(int16_t) * m * k);
            memset(B, 0, sizeof(int16_t) * k * n);
            memset(C, 0, sizeof(int16_t) * m * n);

            int16_get_rand_mat(m, k, A, bound);
            int16_get_rand_mat(k, n, B, bound);

            double FLOP = 2 * (double)m * n * k;

            uint64_t start = timer();
            qgemm(A, B, C, m, n, k);
            uint64_t end = timer();
            double elapsed = (end - start) * 1e-9;
            double FLOPS = FLOP / elapsed;

            if(i==0) is_valid_gemm = naive_qgemm(A, B, C, m, n, k);

            // if range is not 0, don't print each results
            if(range == 0) {
                printf("Exec. time = %.3lfms\n", elapsed * 1000);
                printf("GFLOPS = %.3lf\n", FLOPS / 1e9);
            }

            exec_times[0] += (elapsed * 1000);
            exec_times[1] = (exec_times[1] < (elapsed * 1000) ? (elapsed * 1000) : exec_times[1]);
            exec_times[2] = (exec_times[2] > (elapsed * 1000) ? (elapsed * 1000) : exec_times[2]);
            
            gflops[0] += (FLOPS / 1e9);
            gflops[1] = (gflops[1] < (FLOPS / 1e9) ? (FLOPS / 1e9) : gflops[1]);
            gflops[2] = (gflops[2] > (FLOPS / 1e9) ? (FLOPS / 1e9) : gflops[2]);
        }
        free(A);
        free(B);
        free(C);

        if(!is_valid_gemm) error_num++;
        if(console_flag) print_console(m, k, n, niter, exec_times, gflops, is_valid_gemm);
        if(file != NULL) print_file(m, k, n, niter, exec_times, gflops, is_valid_gemm, file);
    }
    }
    }
}

void hqgemm_test(const int M, const int N, const int K, const int niter,
                const int range, const int bound, FILE* file, BOOL console_flag) {
int error_num = 0;

    for(int m = M; m < (M + range); m++) {
    for(int n = N; n < (N + range); n++) {
    for(int k = K; k < (K + range); k++) {
        BOOL is_valid_gemm = FALSE;
        double exec_times[3]  = {0, __FLT_MIN__, __FLT_MAX__};    /* avg max min */
        double gflops[3]      = {0, __FLT_MIN__, __FLT_MAX__};    /* avg max min */

        int8_t* A = (int8_t *)malloc(m * k * sizeof(int8_t));
        int8_t* B = (int8_t *)malloc(k * n * sizeof(int8_t));
        int8_t* C = (int8_t *)malloc(m * n * sizeof(int8_t));

        for(int i = 0; i < niter; i++) {
            memset(A, 0, sizeof(int8_t) * m * k);
            memset(B, 0, sizeof(int8_t) * k * n);
            memset(C, 0, sizeof(int8_t) * m * n);

            int8_get_rand_mat(m, k, A, bound);
            int8_get_rand_mat(k, n, B, bound);

            double FLOP = 2 * (double)m * n * k;

            uint64_t start = timer();
            hqgemm(A, B, C, m, n, k);
            uint64_t end = timer();
            double elapsed = (end - start) * 1e-9;
            double FLOPS = FLOP / elapsed;

            if(i==0) is_valid_gemm = naive_hqgemm(A, B, C, m, n, k);

            // if range is not 0, don't print each results
            if(range == 0) {
                printf("Exec. time = %.3lfms\n", elapsed * 1000);
                printf("GFLOPS = %.3lf\n", FLOPS / 1e9);
            }

            exec_times[0] += (elapsed * 1000);
            exec_times[1] = (exec_times[1] < (elapsed * 1000) ? (elapsed * 1000) : exec_times[1]);
            exec_times[2] = (exec_times[2] > (elapsed * 1000) ? (elapsed * 1000) : exec_times[2]);
            
            gflops[0] += (FLOPS / 1e9);
            gflops[1] = (gflops[1] < (FLOPS / 1e9) ? (FLOPS / 1e9) : gflops[1]);
            gflops[2] = (gflops[2] > (FLOPS / 1e9) ? (FLOPS / 1e9) : gflops[2]);
        }
        free(A);
        free(B);
        free(C);

        if(!is_valid_gemm) error_num++;
        if(console_flag) print_console(m, k, n, niter, exec_times, gflops, is_valid_gemm);
        if(file != NULL) print_file(m, k, n, niter, exec_times, gflops, is_valid_gemm, file);
    }
    }
    }
}

uint64_t timer() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

/********************************************************
 *                                                      
 *          Naive GEMM
 *                                                      
*********************************************************/
BOOL naive_sgemm(const float* A, const float* B, const float* C,
                const int M, const int N, const int K) {
    for(int r = 0; r < M; r++) {
        for(int c = 0; c < N; c++) {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += (A[r * K + k] * B[k * N + c]);
            if(sum != C[r * N + c])
                return FALSE;
        }
    }
    return TRUE;
}

BOOL naive_dgemm(const double* A, const double* B, const double* C,
                const int M, const int N, const int K) {
    for(int r = 0; r < M; r++) {
        for(int c = 0; c < N; c++) {
            double sum = 0;
            for (int k = 0; k < K; k++)
                sum += (A[r * K + k] * B[k * N + c]);
            if(sum != C[r * N + c])
                return FALSE;
        }
    }
    return TRUE;
}

BOOL naive_igemm(const int* A, const int* B, const int* C,
                 const int M, const int N, const int K) {
    for(int r = 0; r < M; r++) {
        for(int c = 0; c < N; c++) {
            int sum = 0;
            for (int k = 0; k < K; k++)
                sum += (A[r * K + k] * B[k * N + c]);
            if(sum != C[r * N + c])
                return FALSE;
        }
    }
    return TRUE;
}

BOOL naive_qgemm(const int16_t* A, const int16_t* B, const int16_t* C,
                const int M, const int N, const int K) {
    for(int r = 0; r < M; r++) {
        for(int c = 0; c < N; c++) {
            int16_t sum = 0;
            for (int k = 0; k < K; k++)
                sum += (A[r * K + k] * B[k * N + c]);
            if(sum != C[r * N + c])
                return FALSE;
        }
    }
    return TRUE;
}

BOOL naive_hqgemm(const int8_t* A, const int8_t* B, const int8_t* C,
                const int M, const int N, const int K) {
    for(int r = 0; r < M; r++) {
        for(int c = 0; c < N; c++) {
            int8_t sum = 0;
            for (int k = 0; k < K; k++)
                sum += (A[r * K + k] * B[k * N + c]);
            if(sum != C[r * N + c])
                return FALSE;
        }
    }
    return TRUE;                 
}
            
/********************************************************
 *                                                      
 *          Generate Random Matrix                                 
 *                                                      
*********************************************************/
void int32_get_rand_mat(int row, int col, int32_t* mat, int bound) {
    srand(time(NULL));
    for (int r = 0; r < row; r++)
        for (int c = 0; c < col; c++)
            mat[r * col + c] = (rand() % bound);
}

void fp32_get_rand_mat(int row, int col, float* mat, int bound) {
    srand(time(NULL));
    for (int r = 0; r < row; r++)
        for (int c = 0; c < col; c++)
            mat[r * col + c] = (rand() % bound);
}

void fp64_get_rand_mat(int row, int col, double* mat, int bound) {
    srand(time(NULL));
    for (int r = 0; r < row; r++)
        for (int c = 0; c < col; c++)
            mat[r * col + c] = (rand() % bound);
}

void int16_get_rand_mat(int row, int col, int16_t* mat, int bound) {
    srand(time(NULL));
    for (int r = 0; r < row; r++)
        for (int c = 0; c < col; c++)
            mat[r * col + c] = (rand() % bound);
}

void int8_get_rand_mat(int row, int col, int8_t* mat, int bound) {
    srand(time(NULL));
    for (int r = 0; r < row; r++)
        for (int c = 0; c < col; c++)
            mat[r * col + c] = (rand() % bound);
}

/********************************************************
 *                                                      
 *          Matrix Print                                
 *                                                      
*********************************************************/
void int32_print(int row, int col, int32_t* mat) {
    printf("INT32 Matrix Print\n");
    for(int r=0; r<row; r++) {
        for (int c=0; c<col; c++) 
            printf("%3d ", mat[r *col + c]);
        printf("\n");
    }
    printf("\n");
}

void fp32_print(int row, int col, float* mat) {
    printf("FP32 Matrix Print\n");
    for(int r=0; r<row; r++) {
        for (int c=0; c<col; c++) 
            printf("%5.2f ", mat[r * col + c]);
        printf("\n");
    }
    printf("\n");
}

void fp64_print(int row, int col, double* mat) {
    printf("FP64 Matrix Print\n");
    for(int r=0; r<row; r++) {
        for (int c=0; c<col; c++) 
            printf("%5.2lf ", mat[r * col + c]);
        printf("\n");
    }
    printf("\n");
}

void print_console(const int M, const int K, const int N, const int niter, 
                   const double* exec_time, const double* gflops, const BOOL is_valid_gemm) {
    printf("════════════════════════════════════════════════\n\n");
    printf("A: %dx%d B: %dx%d C: %dx%d\n\n", M, K, K, N, M, N);
    printf("DGEMM Test on %d Iterations\n", niter);
    printf("%s\n\n", (is_valid_gemm == TRUE) ? "[Valid GEMM]" : "[Invalid GEMM!]");

    printf("AVG Exec. Time: %5.3lfms\n",     exec_time[0] / niter);
    printf("MAX Exec. Time: %5.3lfms\n",     exec_time[1]);
    printf("MIN Exec. Time: %5.3lfms\n\n",   exec_time[2]);

    printf("AVG GFLOPS : %5.3lf\n",   gflops[0] / niter);
    printf("MAX GFLOPS : %5.3lf\n",   gflops[1]);
    printf("MIN GFLOPS : %5.3lf\n\n", gflops[2]);
    printf("════════════════════════════════════════════════\n");
}

void print_file(const int M, const int K, const int N, const int niter,
                   const double* exec_time, const double* gflops, 
                   const BOOL is_valid_gemm, FILE* file) {
    fprintf(file, "C = A x B\n");
    fprintf(file, "M:%d, K:%d, N:%d\n", M, K, N);
    fprintf(file, "Exec. Time: AVX\t MAX\t MIN\n");
    fprintf(file, "%7.3lf %7.3lf %7.3lf\n", exec_time[0] / niter, exec_time[1] ,exec_time[2]);
    fprintf(file, "GFPS      : AVX\t MAX\t MIN\n");
    fprintf(file, "%7.3lf %7.3lf %7.3lf", gflops[0] / niter, gflops[1] ,gflops[2]);
}