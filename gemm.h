#ifndef GEMM_H
#define GEMM_H 1

#pragma once 

#include "util.h"
#include "sse.h"

#if INSTLEVEL >= 8 /* AVX512F */
#define NR 32
#define MR 14
#elif INSTLEVEL >= 6  /* AVX, AVX2 */
#define NR 16
#define MR 6
#endif

#define MEM_ALIGN 64

void gemm(const float* A, const float* B, float* C,
        const int M, const int N, const int K);

void kernel(const float* packed_blockA, const float* packed_blockB, float* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N);

void pack_blockB(const float* B, float* packed_B, const int nc, 
                const int NC, const int N, const int kc);
void pack_blockA(const float* A, float* packed_A, const int mc, 
                const int kc, const int KC, const int K);
void pack_panelB(const float* B, float* packed_B, 
                const int nr, const int NC, const int N, const int kc);
void pack_panelA(const float* A, float* packed_A, 
                const int mr, const int kc, const int KC, const int K);

/**
 * Get cache information especially cache size.
 * This only cares about data cache and L1 to L3.
 */
void show_cache(size_t* cache_size);
void get_cache_size(size_t* cache_size);
void set_block_size(size_t* cache_size, const int NTHREADS,
                    int* MC, int* KC, int* NC);
void cache_opt(const int NTHREADS, int* MC, int* KC, int* NC);

#endif // GEMM_H