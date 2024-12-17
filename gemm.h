/**********************************************************************************************
 * File   : gemm.h
 * Author : kdh     
 * Github : https://github.com/kdhrepos/gemm.h
 * 
 * Description: 
 *      Header file that declares almost all of the necessary functions for GEMM operation.
 *      All functions are separated based on their own features.
 *                                                    
**********************************************************************************************/

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
void hqgemm(const int16_t* A, const int16_t* B, int16_t* C,
           const int M, const int N, const int K);
void qgemm(const int8_t* A, const int8_t* B, int8_t* C,
           const int M, const int N, const int K);

/********************************************************
 *                                                      
 *          Kernel
 *                                                      
*********************************************************/
void skernel(const float* packed_blockA, const float* packed_blockB, float* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N);
void dkernel(const double* packed_blockA, const double* packed_blockB, double* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N);
void ikernel(const int* packed_blockA, const int* packed_blockB, int* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N);
void hqkernel(const int16_t* packed_blockA, const int16_t* packed_blockB, int16_t* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N);
void qkernel(const int8_t* packed_blockA, const int8_t* packed_blockB, int8_t* C,
              const int m, const int kc, const int KC, 
              const int n, const int NC, const int N);

/********************************************************
 *                                                      
 *          Arithmetic Operation
 *                                                      
*********************************************************/
#if INSTLEVEL >= 9 /* AVX512BW */
#if (defined (__AVX512__) || defined (__AVX512F__))
__m512i qmul(__m512i a, __m512i b);
#endif // AVX512F
#endif          /* INSTLEVEL */

/* FMA */
#if INSTLEVEL >= 9   /* AVX512BW */
#if defined (__AVX512BW__) && (defined (__AVX512__) || defined (__AVX512F__))
__m512  sfma(__m512 a, __m512 b, __m512 c)   ;
__m512d dfma(__m512d a, __m512d b, __m512d c);
__m512i ifma(__m512i a, __m512i b, __m512i c);
__m512i qfma(__m512i a, __m512i b, __m512i c);
#endif
#elif INSTLEVEL >= 8 /* AVX512 */
#if defined (__AVX512__) || defined (__AVX512F__)
__m512  sfma(__m512 a, __m512 b, __m512 c)   ;
__m512d dfma(__m512d a, __m512d b, __m512d c);
__m512i ifma(__m512i a, __m512i b, __m512i c);
#endif // AVX512F
#elif INSTLEVEL >= 7 /* AVX2 */
#if defined (__FMA__)
__m256  sfma(__m256 a, __m256 b, __m256 c)   ;
__m256d dfma(__m256d a, __m256d b, __m256d c);
__m256i ifma(__m256i a, __m256i b, __m256i c);
#else  // No FMA
__m256  sfma(__m256 a, __m256 b, __m256 c)   ;
__m256d dfma(__m256d a, __m256d b, __m256d c);
__m256i ifma(__m256i a, __m256i b, __m256i c);
#endif // AVX, FMA
#elif INSTLEVEL >= 6 /* AVX*/
#if defined (__FMA__)
__m256  sfma(__m256 a, __m256 b, __m256 c)   ;
__m256d dfma(__m256d a, __m256d b, __m256d c);
__m128i ifma(__m128i a, __m128i b, __m128i c);
#else  // No FMA
__m256  sfma(__m256 a, __m256 b, __m256 c)   ;
__m256d dfma(__m256d a, __m256d b, __m256d c);
__m128i ifma(__m128i a, __m128i b, __m128i c);
#endif // AVX, FMA
#endif              /* INSTLEVEL */

/********************************************************
 *                                                      
 *          Memory operation
 *                                                      
*********************************************************/
#if INSTLEVEL >= 9   /* AVX512BW */
#elif INSTLEVEL >= 8 /* AVX512 */
#elif INSTLEVEL >= 7 /* AVX2 */
#elif INSTLEVEL >= 6 /* AVX*/
__m128i maskload(int* C, int8_t mask);
void maskstore(int* C, int8_t mask, __m128i packed_C);
#endif              /* INSTLEVEL */

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

void hqpack_blockB(const int16_t* B, int16_t* packed_B, const int NR, 
                  const int nc, const int NC, const int N, const int kc);
void hqpack_blockA(const int16_t* A, int16_t* packed_A, const int MR,
                  const int mc, const int kc, const int KC, const int K);
void hqpack_panelB(const int16_t* B, int16_t* packed_B, const int nr, 
                  const int NC, const int N, const int kc);
void hqpack_panelA(const int16_t* A, int16_t* packed_A, const int mr, 
                  const int kc, const int KC, const int K);

void qpack_blockB(const int8_t* B, int8_t* packed_B, const int NR, 
                  const int nc, const int NC, const int N, const int kc);
void qpack_blockA(const int8_t* A, int8_t* packed_A, const int MR,
                  const int mc, const int kc, const int KC, const int K);
void qpack_panelB(const int8_t* B, int8_t* packed_B, const int nr, 
                  const int NC, const int N, const int kc);
void qpack_panelA(const int8_t* A, int8_t* packed_A, const int mr, 
                  const int kc, const int KC, const int K);

/********************************************************
 *                                                      
 *          Hardware Optimization
 *                                                      
*********************************************************/
void show_cache(size_t* cache_size);
void get_cache_size(size_t* cache_size);
void set_block_size(size_t* cache_size, const int NTHREADS,
                    const int MR, const int NR,
                    int* MC, int* KC, int* NC, D_TYPE d_type);
void cache_opt(const int NTHREADS, const int MR, const int NR,
               int* MC, int* KC, int* NC, D_TYPE d_type);
int get_core_num();

#endif // GEMM_H