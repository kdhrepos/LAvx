#ifndef GEMM_H
#define GEMM_H 1

#pragma once 

#include "util.h"
#include "sse.h"

#define MEM_ALIGN 64

/********************************************************
 *                                                      
 *          GEMM                              
 *                                                      
*********************************************************/

void sgemm(const float* A, const float* B, float* C,
           const int M, const int N, const int K);
void dgemm(const double* A, const double* B, double* C,
           const int M, const int N, const int K);
void igemm(const int* A, const int* B, int* C,
           const int M, const int N, const int K);
void qgemm(const int16_t* A, const int16_t* B, int16_t* C,
           const int M, const int N, const int K);
void hqgemm(const int8_t* A, const int8_t* B, int8_t* C,
           const int M, const int N, const int K);

/********************************************************
 *                                                      
 *          Kernel
 *                                                      
*********************************************************/

void s_kernel(const float* packed_blockA, const float* packed_blockB, float* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N);
void d_kernel(const double* packed_blockA, const double* packed_blockB, double* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N);
void i_kernel(const int* packed_blockA, const int* packed_blockB, int* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N);
void q_kernel(const int16_t* packed_blockA, const int16_t* packed_blockB, int16_t* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N);
void hq_kernel(const int8_t* packed_blockA, const int8_t* packed_blockB, int8_t* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N);

/********************************************************
 *                                                      
 *          FMA
 *                                                      
*********************************************************/
#if defined (__FMA__)
#define S_FMA(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#else 
#define S_FMA(a, b, c)  _mm256_add_ps((c), (_mm256_mul_ps((a), (b))))
#endif // S_FMA

#if defined (__FMA__)
#define D_FMA(a, b, c) _mm256_fmadd_pd((a), (b), (c))
#else 
#define D_FMA(a, b, c)  _mm256_add_pd((c), (_mm256_mul_pd((a), (b))))
#endif // D_FMA


/********************************************************
 *                                                      
 *          Matrix Pack
 *                                                      
*********************************************************/

void spack_blockB(const float* B, float* packed_B, const int NR, 
                  const int nc, const int NC, const int N, const int kc);
void spack_blockA(const float* A, float* packed_A, const int MR,
                  const int mc, const int kc, const int KC, const int K);
void spack_panelB(const float* B, float* packed_B, 
                  const int nr, const int NC, const int N, const int kc);
void spack_panelA(const float* A, float* packed_A, 
                  const int mr, const int kc, const int KC, const int K);

void dpack_blockB(const double* B, double* packed_B, const int NR, 
                  const int nc, const int NC, const int N, const int kc);
void dpack_blockA(const double* A, double* packed_A, const int MR,
                  const int mc, const int kc, const int KC, const int K);
void dpack_panelB(const double* B, double* packed_B, const int nr, 
                  const int NC, const int N, const int kc);
void dpack_panelA(const double* A, double* packed_A, const int mr, 
                  const int kc, const int KC, const int K);

void ipack_blockB(const int* B, int* packed_B, const int NR, 
                  const int nc, const int NC, const int N, const int kc);
void ipack_blockA(const int* A, int* packed_A, const int MR,
                  const int mc, const int kc, const int KC, const int K);
void ipack_panelB(const int* B, int* packed_B, const int nr, 
                  const int NC, const int N, const int kc);
void ipack_panelA(const int* A, int* packed_A, const int mr, 
                  const int kc, const int KC, const int K);

void qpack_blockB(const int16_t* B, int16_t* packed_B, const int NR, 
                  const int nc, const int NC, const int N, const int kc);
void qpack_blockA(const int16_t* A, int16_t* packed_A, const int MR,
                  const int mc, const int kc, const int KC, const int K);
void qpack_panelB(const int16_t* B, int16_t* packed_B, const int nr, 
                  const int NC, const int N, const int kc);
void qpack_panelA(const int16_t* A, int16_t* packed_A, const int mr, 
                  const int kc, const int KC, const int K);

void hqpack_blockB(const int8_t* B, int8_t* packed_B, const int NR, 
                  const int nc, const int NC, const int N, const int kc);
void hqpack_blockA(const int8_t* A, int8_t* packed_A, const int MR,
                  const int mc, const int kc, const int KC, const int K);
void hqpack_panelB(const int8_t* B, int8_t* packed_B, const int nr, 
                  const int NC, const int N, const int kc);
void hqpack_panelA(const int8_t* A, int8_t* packed_A, const int mr, 
                  const int kc, const int KC, const int K);

/********************************************************
 *                                                      
 *          Cache Optimization
 *                                                      
*********************************************************/

/**
 * Get cache information especially cache size.
 * This only cares about data cache and also L1 to L3.
 */
void show_cache(size_t* cache_size);
void get_cache_size(size_t* cache_size);
void set_block_size(size_t* cache_size, const int NTHREADS,
                    const int MR, const int NR,
                    int* MC, int* KC, int* NC, D_TYPE d_type);
void cache_opt(const int NTHREADS, const int MR, const int NR,
               int* MC, int* KC, int* NC, D_TYPE d_type) ;

#endif // GEMM_H