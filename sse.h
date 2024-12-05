#ifndef SSE_H
#define SSE_H

#pragma once

#include <immintrin.h>
 #if defined(_MSC_VER)
     /* Microsoft C/C++-compatible compiler */
     #include <intrin.h>
 #elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
     /* GCC-compatible compiler, targeting x86/x86-64 */
     #include <x86intrin.h>
// #elif defined(__GNUC__) && defined(__ARM_NEON__)
//      /* GCC-compatible compiler, targeting ARM with NEON */
//      #include <arm_neon.h>
// #elif defined(__GNUC__) && defined(__IWMMXT__)
//      /* GCC-compatible compiler, targeting ARM with WMMX */
     // #include <mmintrin.h>
// #elif (defined(__GNUC__) || defined(__xlC__)) && (defined(__VEC__) || defined(__ALTIVEC__))
//      /* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
//      #include <altivec.h>
// #elif defined(__GNUC__) && defined(__SPE__)
//      /* GCC-compatible compiler, targeting PowerPC with SPE */
//      #include <spe.h>
 #endif

// #if defined (__AVX512VL__)
// #if defined (__AVX512F__) || defined ( __AVX512__ )
// #if defined ( __AVX512VNNI__ )

// TODO: Define another macros (or not...?)
 #ifndef INSTLEVEL
  #if defined (__AVX512BW__) /* && defined (__AVX512VL__) */
     #define INSTLEVEL 9
  #elif defined (__AVX512F__) || defined (__AVX512__)
     #define INSTLEVEL 8
  #elif defined (__AVX2__)
     #define INSTLEVEL 7
  #elif defined (__AVX__)
     #define INSTLEVEL 6
  #elif defined (__SSE4_2__)
     #define INSTLEVEL 5
  #elif defined (__SSE4_1__)
     #define INSTLEVEL 4
  #elif defined (__SSE3__)
     #define INSTLEVEL 3
  #elif defined (__SSE2__)
     #define INSTLEVEL 2
  #elif defined (__SSE__)
     #define INSTLEVEL 1
  #else 
     #define INSTLEVEL 0
  #endif // INSTLEVEL <LEVEL>
 #endif // INSTLEVEL
#endif // SSE_H