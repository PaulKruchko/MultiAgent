/*
 * tlsf.h - compact Two-Level Segregated Fit allocator (C99, no libc).
 *
 * Contract
 * --------
 * - tlsf_init() formats a caller-owned region and returns a heap handle that
 *   lives *inside* that region (the control structure is placed at the first
 *   TLSF_ALIGN-aligned address).  The caller keeps ownership of the region;
 *   it must stay valid and must not be touched by other code for as long as
 *   the heap is used.  There is no deinit: simply stop using the handle.
 * - Regions may start at any address; the start is rounded up and the end
 *   rounded down to TLSF_ALIGN.  Regions larger than TLSF_MAX_REGION bytes
 *   (default 1 MiB) or too small to hold the control structure plus one
 *   minimum block are rejected (tlsf_init returns NULL).  Nothing is
 *   silently truncated.
 * - Any number of independent heaps may exist; all state is per heap.
 * - tlsf_malloc() returns a TLSF_ALIGN-aligned pointer, or NULL when
 *   size == 0, size > TLSF_MAX_ALLOC, or no suitable free block exists
 *   (heap state is unchanged on failure).
 * - tlsf_free(h, NULL) is a no-op.  Otherwise ptr must have been returned by
 *   tlsf_malloc() on the *same* heap and not freed since (not checked).
 * - Lookup is "good fit": the request is rounded up to the next size-class
 *   boundary so that any block in the selected class fits without searching.
 *   A request may therefore fail although a free block of sufficient size
 *   exists in the request's own class (bounded by 1/16 of the size).
 * - Locking: optional per-heap hooks.  Install them with tlsf_set_lock()
 *   before the heap is shared.  lock(ctx) is called exactly once before and
 *   unlock(ctx) exactly once after every tlsf_malloc()/tlsf_free() that
 *   reaches heap state; calls rejected up front (size 0, oversize, NULL ptr)
 *   return without calling the hooks.  Hooks must not call back into the
 *   same heap.  Waiting time inside lock() is not part of the O(1) bound.
 *
 * Configuration (define before including, identically for every unit):
 *   TLSF_ALIGN_LOG2       log2 of payload alignment, default 3 (8 bytes).
 *                         Limits: 2 <= TLSF_ALIGN_LOG2 <= 5, and
 *                         TLSF_ALIGN >= sizeof(void *) (so 2 is only valid
 *                         on 32-bit targets).
 *   TLSF_MAX_REGION_LOG2  log2 of the largest accepted region, default 20.
 *                         Limits: FL_SHIFT + 1 <= TLSF_MAX_REGION_LOG2 <= 31,
 *                         where FL_SHIFT = 4 + TLSF_ALIGN_LOG2, and it must be
 *                         below the bit width of size_t.
 *   The allocator also requires a 32-bit 'unsigned' (bit scans compute
 *   31 - clz(x)).  All limits are enforced at compile time in tlsf.c.
 */
#ifndef TLSF_H
#define TLSF_H

#include <stddef.h>

#ifndef TLSF_ALIGN_LOG2
#define TLSF_ALIGN_LOG2 3
#endif
#define TLSF_ALIGN ((size_t)1 << TLSF_ALIGN_LOG2)

#ifndef TLSF_MAX_REGION_LOG2
#define TLSF_MAX_REGION_LOG2 20
#endif
#define TLSF_MAX_REGION ((size_t)1 << TLSF_MAX_REGION_LOG2)

/* Largest request tlsf_malloc() will consider (keeps rounded lookups inside
 * the class table).  Equals TLSF_MAX_REGION - TLSF_MAX_REGION/32 - 32. */
#define TLSF_MAX_ALLOC \
    (TLSF_MAX_REGION - (TLSF_MAX_REGION >> 5) - (size_t)32)

typedef struct tlsf tlsf_t;
typedef void (*tlsf_lock_fn)(void *ctx);

tlsf_t *tlsf_init(void *mem, size_t bytes);
void tlsf_set_lock(tlsf_t *h, tlsf_lock_fn lock, tlsf_lock_fn unlock,
                   void *ctx);
void *tlsf_malloc(tlsf_t *h, size_t size);
void tlsf_free(tlsf_t *h, void *ptr);

#endif
