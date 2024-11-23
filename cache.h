#ifndef CACHE_H
#define CACHE_H

#pragma once

#include "util.h"

/**
 * Get cache information especially cache size.
 * This only cares about data cache and L1 to L3.
 */
void get_cache_size(size_t* cache_size);
void set_block_size(size_t* cache_size, const int NTHREADS, 
                    const int MR, const int NR,
                    int* MC, int* KC, int* NC);
void show_cache(size_t* cache_size)

void get_cache_size(size_t* cache_size) { 
    memset(cache_size, 0, sizeof(size_t) * 32);

    for (int c_id = 0; c_id < 32; c_id++) {
        uint32_t eax, ebx, ecx, edx;

        eax = 4; // get cache info
        ecx = c_id; // cache id

        /* CPUID instruction */
        __asm__ (
            "cpuid" 
            : "+a" (eax) , "=b" (ebx) , "+c" (ecx) , "=d" (edx)
        );

        int cache_type = eax & 0x1F; // bits 04-00: cache type field
        if (cache_type == 0)        // no more cache
            break;
        if (cache_type == 2)        // pass instruction cache
            continue;

        int cache_level = (eax >>= 5) & 0x7; // bits 07-50: cache level, starts at 1

        uint32_t cache_sets = ecx + 1;                                          // bits 31-00
        uint32_t cache_coherency_line_size = (ebx & 0xFFF) + 1;                 // bits 11-00
        uint32_t cache_physical_line_partitions = ((ebx >>= 12) & 0x3FF) + 1;   // bits 21-12
        uint32_t cache_ways_of_associativity = ((ebx >>= 10) & 0x3FF) + 1;      // bits 31-22

        size_t total_cache_size = cache_ways_of_associativity * cache_physical_line_partitions 
                            * cache_coherency_line_size * cache_sets;

        cache_size[cache_level] = total_cache_size; // store only d-cache size
    }
}

void set_block_size(size_t* cache_size, const int NTHREADS, 
                    const int MR, const int NR,
                    int* MC, int* KC, int* NC) {
    float MC_f = (*MC), NC_f = (*NC);

    if(cache_size[1] != 0) {
        (*KC) = cache_size[1] / (NR * sizeof(float));      // L1 = KC * NR
    }
    if(cache_size[2] != 0) {
        MC_f = cache_size[2] / ((*KC) * sizeof(float));   // L2 = MC * KC
        MC_f /= (MR * NTHREADS);
        (*MC) = round(MC_f) * MR * NTHREADS;
    }
    if(cache_size[3] != 0) {
        NC_f = cache_size[3] / ((*MC) * sizeof(float));   // L3 = NC * KC
        NC_f /= (NR * NTHREADS);
        (*NC) = round(NC_f) * NR * NTHREADS;
    }

#ifdef DEBUG
    printf("MC : %ld\n", (* MC));
    printf("KC : %ld\n", (* KC));
    printf("NC : %ld\n", (* NC));
#endif
}

void show_cache(size_t* cache_size) {
    printf("L1 size: %ld bytes\n", cache_size[1]);
    printf("L2 size: %ld bytes\n", cache_size[2]);
    printf("L3 size: %ld bytes\n", cache_size[3]);
}

void cache_opt(const int NTHREADS, const int MR, const int NR, 
               int* MC, int* KC, int* NC) {
    size_t cache_size[32];
    get_cache_size(cache_size);
    set_block_size(cache_size, NTHREADS, MR, NR, &(*MC), &(*KC), &(*NC));
#ifdef DEBUG
    show_cache(cache_size);
#endif
}
#endif