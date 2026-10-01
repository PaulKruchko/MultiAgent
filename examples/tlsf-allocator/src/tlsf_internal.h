/*
 * tlsf_internal.h - block/control layout shared by tlsf.c and the test-only
 * consistency inspector.  Not part of the public API.
 */
#ifndef TLSF_INTERNAL_H
#define TLSF_INTERNAL_H

#include <stdint.h>
#include "tlsf.h"

#define TLSF_SL_LOG2 4 /* 16 second-level classes */
#define TLSF_SL_COUNT (1u << TLSF_SL_LOG2)
#define TLSF_FL_SHIFT (TLSF_SL_LOG2 + TLSF_ALIGN_LOG2)
#define TLSF_SMALL ((size_t)1 << TLSF_FL_SHIFT) /* below: exact classes */
#define TLSF_FL_COUNT (TLSF_MAX_REGION_LOG2 - TLSF_FL_SHIFT + 1)

#define TLSF_FREE ((size_t)1)      /* this block is free */
#define TLSF_PREV_FREE ((size_t)2) /* physical predecessor is free */
#define TLSF_FLAGS (TLSF_FREE | TLSF_PREV_FREE)

/*
 * Every block starts with a TLSF_HDR-byte header.  'size' is the total block
 * size (header included, multiple of TLSF_ALIGN) plus the two flag bits.
 * 'prev_phys' is only valid while TLSF_PREV_FREE is set; the free-list links
 * overlay the payload and are only valid while TLSF_FREE is set.  The region
 * ends with a size-0 "used" sentinel header, so the last real block always
 * has a physical successor and coalescing never runs off the region.
 */
typedef struct tlsf_block {
    struct tlsf_block *prev_phys;
    size_t size;
    struct tlsf_block *next_free;
    struct tlsf_block *prev_free;
} tlsf_block;

#define TLSF_ALIGN_UP(x) (((x) + (TLSF_ALIGN - 1)) & ~(TLSF_ALIGN - 1))
#define TLSF_HDR TLSF_ALIGN_UP(offsetof(tlsf_block, next_free))
/* Smallest block: must hold the free-list links and at least TLSF_ALIGN
 * payload bytes (so every block can serve a request once allocated). */
#define TLSF_MIN_BLOCK                                                     \
    (TLSF_ALIGN_UP(sizeof(tlsf_block)) > TLSF_HDR                          \
         ? TLSF_ALIGN_UP(sizeof(tlsf_block))                               \
         : TLSF_HDR + TLSF_ALIGN)

#ifdef TLSF_STATS
/* Primitive-operation counters (test builds only).  Callers zero them before
 * an operation and read them afterwards. */
typedef struct tlsf_stats {
    unsigned bitmap_checks;   /* bitmap words examined by the class search */
    unsigned list_inserts;    /* free-list insertions */
    unsigned list_removes;    /* free-list removals */
    unsigned neighbor_checks; /* physical-neighbour header inspections */
    unsigned splits;
    unsigned coalesces;
} tlsf_stats;
#endif

struct tlsf {
    uint32_t fl_bitmap;                 /* bit f: some list in row f non-empty */
    uint32_t sl_bitmap[TLSF_FL_COUNT];  /* bit s: heads[f][s] non-empty */
    tlsf_block *heads[TLSF_FL_COUNT][TLSF_SL_COUNT];
    tlsf_lock_fn lock;
    tlsf_lock_fn unlock;
    void *lock_ctx;
#ifdef TLSF_STATS
    tlsf_stats stats;
#endif
};

#define TLSF_CTRL_SIZE TLSF_ALIGN_UP(sizeof(struct tlsf))

#endif
