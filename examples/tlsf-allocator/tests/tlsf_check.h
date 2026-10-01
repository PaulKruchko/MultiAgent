/*
 * tlsf_check.h - test-only heap inspector (freestanding, no libc).
 * Walks the physical block chain and every free list and cross-checks them
 * against the bitmaps.  Uses its own re-implementation of the class mapping
 * so that mapping bugs in tlsf.c are detected rather than mirrored.
 */
#ifndef TLSF_CHECK_H
#define TLSF_CHECK_H

#include <stddef.h>
#include "tlsf.h"

typedef struct tlsf_report {
    size_t free_blocks;
    size_t used_blocks;
    size_t free_payload;     /* F: sum over free blocks of (size - header) */
    size_t used_footprint;   /* sum of used block sizes incl. headers */
    size_t largest_free;     /* largest free block payload */
    size_t largest_alloc;    /* L: largest request tlsf_malloc would satisfy */
} tlsf_report;

/* Returns 0 if consistent, otherwise a positive error code (see .c). */
int tlsf_check(tlsf_t *h, const void *mem, size_t bytes, tlsf_report *r);

/* Largest request the good-fit lookup can satisfy given the free blocks
 * currently present (0 if none); pure computation, heap not modified. */
size_t tlsf_check_largest_alloc(tlsf_t *h);

/* Independent class-mapping helpers used by tests. */
void tlsf_check_mapping(size_t block_size, unsigned *fl, unsigned *sl);
size_t tlsf_check_block_need(size_t request); /* block size for a request */

#endif
