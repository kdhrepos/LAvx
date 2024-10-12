#ifndef UTIL_H
#define UTIL_H

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <stdint.h>

typedef short BOOL;
#define TRUE 1
#define FALSE 0

#define FP32_VEC_DIM(vec) ((sizeof(vec))/(sizeof(float)))
#define FP64_VEC_DIM(vec) ((sizeof(vec))/(sizeof(double)))

#define GET_RANDOM(vec, dim)                \
    do {                                    \
        for (int i = 0; i < (dim); i++)     \
            (vec)[i] = ((rand()) % (20));   \
    } while (0)                             \


#endif