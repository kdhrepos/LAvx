#ifndef TEST_H
#define TEST_H

#include "gemm.h"

#define GEMM_H_TESTER_VERSION "1.0.0"

void sgemm_test(const int M, const int N, const int K, const int niter,
                const int range, const int bound, FILE* file, BOOL console_flag);
void dgemm_test(const int M, const int N, const int K, const int niter,
                const int range, const int bound, FILE* file, BOOL console_flag);
void igemm_test(const int M, const int N, const int K, const int niter,
                const int range, const int bound, FILE* file, BOOL console_flag);
void qgemm_test(const int M, const int N, const int K, const int niter,
                const int range, const int bound, FILE* file, BOOL console_flag);
void hqgemm_test(const int M, const int N, const int K, const int niter,
                const int range, const int bound, FILE* file, BOOL console_flag);

uint64_t timer();

BOOL naive_sgemm(const float* A, const float* B, const float* C,
                const int M, const int N, const int K);
BOOL naive_dgemm(const double* A, const double* B, const double* C,
                const int M, const int N, const int K);
BOOL naive_igemm(const int* A, const int* B, const int* C,
                const int M, const int N, const int K);
BOOL naive_qgemm(const int16_t* A, const int16_t* B, const int16_t* C,
                const int M, const int N, const int K);
BOOL naive_hqgemm(const int8_t* A, const int8_t* B, const int8_t* C,
                const int M, const int N, const int K);

void fp32_get_rand_mat(int row, int col, float* mat, int bound);
void fp64_get_rand_mat(int row, int col, double* mat, int bound);
void int32_get_rand_mat(int row, int col, int32_t* mat, int bound);
void int16_get_rand_mat(int row, int col, int16_t* mat, int bound);
void int8_get_rand_mat(int row, int col, int8_t* mat, int bound);

/********************************************************
 *                                                      
 *          Print                                
 *                                                      
*********************************************************/
void fp32_print(int row, int col, float* mat);
void fp64_print(int row, int col, double* mat);
void int32_print(int row, int col, int32_t* mat);

void print_console(const int M, const int K, const int N, const int niter, 
                   const double* exec_time, const double* gflops, const BOOL is_valid_gemm);
void print_file(const int M, const int K, const int N, const int niter,
                const double* exec_time, const double* gflops, 
                const BOOL is_valid_gemm, FILE* file);

#endif // TEST_H
