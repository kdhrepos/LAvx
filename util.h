#ifndef UTIL_H
#define UTIL_H

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <memory.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <getopt.h>
#include <limits.h>
#include <omp.h>

typedef short BOOL;
#define TRUE 1
#define FALSE 0

#define min(a,b) ((a) < (b) ? (a) : (b))

typedef enum {D_FP32, D_FP64, D_INT32} D_TYPE;

#define DEBUG FALSE

void int32_get_rand_mat(int row, int col, int32_t* mat, int bound);
void fp32_get_rand_mat(int row, int col, float* mat, int bound);
void fp64_get_rand_mat(int row, int col, double* mat, int bound);
void int32_scalar_gemm(const int Ar, const int Ac, const int Br, const int Bc,
            const int32_t* A, const int32_t* B, int32_t* C);
void fp64_scalar_gemm(const int Ar, const int Ac, const int Br, const int Bc,
            const double* A, const double* B, double* C);

BOOL int32_gemm_result_check(int row, int col, int32_t* T, int32_t* C);
BOOL fp32_gemm_result_check(int row, int col, float* T, float* C);
BOOL fp64_gemm_result_check(int row, int col, double* T, double* C);

/********************************************************
 *                                                      
 *          Matrix Print                                
 *                                                      
*********************************************************/

void int32_print(int row, int col, int32_t mat[row][col]);
void fp32_print(int row, int col, float* mat);
void fp64_print(int row, int col, double* mat);
#endif // UTIL_H