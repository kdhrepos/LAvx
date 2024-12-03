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
#include <ctype.h>
// #include <unistd.h>
#include <getopt.h>
#include <limits.h>
#include <omp.h>

typedef short BOOL;
#define TRUE 1
#define FALSE 0
#define min(a,b) ((a) < (b) ? (a) : (b))

typedef enum {D_ALL, D_FP32, D_FP64, D_INT32, D_INT8, D_INT16} D_TYPE;

#define DEBUG FALSE

#endif // UTIL_H