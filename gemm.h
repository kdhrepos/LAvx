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

/********************************************************
 *                                                      
 *          Cache Optimization
 *                                                      
*********************************************************/
/**
 * Get cache information especially cache size.
 * This only cares about data cache and L1 to L3.
 */
void show_cache(size_t* cache_size);
void get_cache_size(size_t* cache_size);
void set_block_size(size_t* cache_size, const int NTHREADS,
                    const int MR, const int NR,
                    int* MC, int* KC, int* NC, D_TYPE d_type);
void cache_opt(const int NTHREADS, const int MR, const int NR,
               int* MC, int* KC, int* NC, D_TYPE d_type) ;

#endif // GEMM_H