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

/********************************************************
 *                                                      
 *          Fused Multiply-Add
 *                                                      
*********************************************************/
#if INSTLEVEL >= 8 /* AVX512 */
 #if defined (__AVX512__) || defined (__AVX512F__)
  inline __m512  sfma(__m512 a, __m512 b, __m512 c)    { return _mm512_fmadd_ps(a, b, c); }
  inline __m512d dfma(__m512d a, __m512d b, __m512d c) { return _mm512_fmadd_pd(a, b, c); }
  inline __m512i ifma(__m512i a, __m512i b, __m512i c) { return _mm512_add_epi32(c, _mm512_mullo_epi32(a, b)); }
 #endif // AVX512
#elif INSTLEVEL >= 6 /* AVX */
 #if defined (__FMA__)
  inline __m256  sfma(__m256 a, __m256 b, __m256 c)    { return _mm256_fmadd_ps(a, b, c); }
  inline __m256d dfma(__m256d a, __m256d b, __m256d c) { return _mm256_fmadd_pd(a, b, c); }
  inline __m256i ifma(__m256i a, __m256i b, __m256i c) { return _mm256_add_epi32(c, _mm256_mullo_epi32(a, b)); }
 #else
  inline __m256  sfma(__m256 a, __m256 b, __m256 c)    { return _mm256_add_ps(c, _mm256_mul_ps(a, b)); }
  inline __m256d dfma(__m256d a, __m256d b, __m256d c) { return _mm256_add_pd(c, _mm256_mul_pd(a, b)); }
  inline __m256i ifma(__m256i a, __m256i b, __m256i c) { return _mm256_add_epi32(c, _mm256_mullo_epi32(a, b)); }
 #endif // AVX, FMA
#endif // INSTLEVEL

/********************************************************
 *                                                      
 *          Matrix Pack
 *                                                      
*********************************************************/

void spack_blockB(const float* B, float* packed_B, const int NR, 
                  const int nc, const int NC, const int N, 
                  const int kc, const int NTHREADS);
void spack_blockA(const float* A, float* packed_A, const int MR,
                  const int mc, const int kc, const int KC, 
                  const int K, const int NTHREADS);
void spack_panelB(const float* B, float* packed_B, 
                  const int nr, const int NC, const int N, const int kc);
void spack_panelA(const float* A, float* packed_A, 
                  const int mr, const int kc, const int KC, const int K);

void dpack_blockB(const double* B, double* packed_B, const int NR, 
                  const int nc, const int NC, const int N, 
                  const int kc, const int NTHREADS);
void dpack_blockA(const double* A, double* packed_A, const int MR,
                  const int mc, const int kc, const int KC, 
                  const int K, const int NTHREADS);
void dpack_panelB(const double* B, double* packed_B, const int nr, 
                  const int NC, const int N, const int kc);
void dpack_panelA(const double* A, double* packed_A, const int mr, 
                  const int kc, const int KC, const int K);

void ipack_blockB(const int* B, int* packed_B, const int NR, 
                  const int nc, const int NC, const int N, 
                  const int kc, const int NTHREADS);
void ipack_blockA(const int* A, int* packed_A, const int MR,
                  const int mc, const int kc, const int KC, 
                  const int K, const int NTHREADS);
void ipack_panelB(const int* B, int* packed_B, const int nr, 
                  const int NC, const int N, const int kc);
void ipack_panelA(const int* A, int* packed_A, const int mr, 
                  const int kc, const int KC, const int K);

/********************************************************
 *                                                      
 *          Hardware Optimization
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
               int* MC, int* KC, int* NC, D_TYPE d_type);
int get_core_num();

#endif // GEMM_H