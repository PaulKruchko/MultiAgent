/*
 * tlsf_check.c - test-only heap inspector.  Freestanding: no libc calls, so
 * the same file runs on the host, under FreeRTOS and on bare-metal Cortex-M3.
 *
 * Error codes returned by tlsf_check():
 *   1 handle/region mismatch          8 fl bitmap inconsistent with rows
 *   2 block misaligned / bad size     9 sl bitmap inconsistent with heads,
 *                                        or a stray bit >= TLSF_SL_COUNT
 *   3 block smaller than minimum     10 list node not free / wrong class
 *   4 block overruns region          11 list back-link broken
 *   5 PREV_FREE flag wrong           12 list/physical free count mismatch
 *   6 prev_phys pointer wrong        13 free block links inconsistent
 *   7 two adjacent free blocks       14 sentinel missing
 */
#include "tlsf_check.h"
#include "tlsf_internal.h"

static unsigned floor_log2(size_t x)
{
    unsigned t = 0;
    while (x >>= 1)
        t++;
    return t;
}

void tlsf_check_mapping(size_t size, unsigned *fl, unsigned *sl)
{
    if (size < TLSF_SMALL) {
        *fl = 0;
        *sl = (unsigned)(size / TLSF_ALIGN);
    } else {
        unsigned t = floor_log2(size);
        *fl = t - TLSF_FL_SHIFT + 1;
        *sl = (unsigned)((size - ((size_t)1 << t)) / ((size_t)1 << (t - TLSF_SL_LOG2)));
    }
}

size_t tlsf_check_block_need(size_t request)
{
    size_t n = (request + TLSF_HDR + TLSF_ALIGN - 1) / TLSF_ALIGN * TLSF_ALIGN;
    return n < TLSF_MIN_BLOCK ? TLSF_MIN_BLOCK : n;
}

/* Class index (fl * 16 + sl) searched for a request, or a huge value if the
 * request is rejected outright. */
static unsigned search_index(size_t request)
{
    size_t n;
    unsigned fl, sl;
    if (request == 0 || request > TLSF_MAX_ALLOC)
        return 0xFFFFFFFFu;
    n = tlsf_check_block_need(request);
    if (n >= TLSF_SMALL)
        n += ((size_t)1 << (floor_log2(n) - TLSF_SL_LOG2)) - 1;
    tlsf_check_mapping(n, &fl, &sl);
    return fl * TLSF_SL_COUNT + sl;
}

size_t tlsf_check_largest_alloc(tlsf_t *h)
{
    unsigned fl, sl, top = 0, found = 0;
    size_t lo = 0, hi = TLSF_MAX_ALLOC;
    for (fl = 0; fl < TLSF_FL_COUNT; fl++)
        for (sl = 0; sl < TLSF_SL_COUNT; sl++)
            if (h->heads[fl][sl]) {
                top = fl * TLSF_SL_COUNT + sl;
                found = 1;
            }
    if (!found)
        return 0;
    /* search_index() is monotone in the request: binary search the largest
     * request whose class is <= the highest non-empty class. */
    while (lo < hi) {
        size_t mid = lo + (hi - lo + 1) / 2;
        if (search_index(mid) <= top)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

static size_t bsize(const tlsf_block *b) { return b->size & ~TLSF_FLAGS; }

int tlsf_check(tlsf_t *h, const void *mem, size_t bytes, tlsf_report *r)
{
    const char *lo = (const char *)mem, *hi = lo + bytes;
    tlsf_block *b = (tlsf_block *)((char *)h + TLSF_CTRL_SIZE);
    size_t max_iter = bytes / TLSF_MIN_BLOCK + 2, it, listed = 0;
    int prev_free = 0;
    tlsf_block *prev = 0;
    unsigned fl, sl;
    tlsf_report rep = {0, 0, 0, 0, 0, 0};

    if ((const char *)h < lo || (const char *)h - lo >= (ptrdiff_t)TLSF_ALIGN ||
        ((size_t)(uintptr_t)h & (TLSF_ALIGN - 1)))
        return 1;

    /* Physical walk. */
    for (it = 0;; it++) {
        size_t s;
        if (it > max_iter)
            return 14;
        if ((const char *)b + TLSF_HDR > hi)
            return 4;
        if ((size_t)(uintptr_t)b & (TLSF_ALIGN - 1))
            return 2;
        if (((b->size & TLSF_PREV_FREE) != 0) != prev_free)
            return 5;
        if (prev_free && b->prev_phys != prev)
            return 6;
        s = bsize(b);
        if (s == 0) { /* sentinel */
            if (b->size & TLSF_FREE)
                return 14;
            break;
        }
        if (s & (TLSF_ALIGN - 1))
            return 2;
        if (s < TLSF_MIN_BLOCK)
            return 3;
        if ((const char *)b + s + TLSF_HDR > hi)
            return 4;
        if (b->size & TLSF_FREE) {
            if (prev_free)
                return 7;
            rep.free_blocks++;
            rep.free_payload += s - TLSF_HDR;
            if (s - TLSF_HDR > rep.largest_free)
                rep.largest_free = s - TLSF_HDR;
            tlsf_check_mapping(s, &fl, &sl);
            if (b->prev_free ? b->prev_free->next_free != b
                             : h->heads[fl][sl] != b)
                return 13;
            if (b->next_free && b->next_free->prev_free != b)
                return 13;
            prev_free = 1;
        } else {
            rep.used_blocks++;
            rep.used_footprint += s;
            prev_free = 0;
        }
        prev = b;
        b = (tlsf_block *)((char *)b + s);
    }

    /* Bitmaps and free lists. */
    for (fl = 0; fl < 32; fl++) {
        int row_nonempty = 0;
        if (fl >= TLSF_FL_COUNT) {
            if (h->fl_bitmap & (1u << fl))
                return 8;
            continue;
        }
        /* tlsf_malloc's CTZ could select a stray high bit and index past
         * the row, so bits >= TLSF_SL_COUNT must all be clear. */
        if (h->sl_bitmap[fl] >> TLSF_SL_COUNT)
            return 9;
        for (sl = 0; sl < TLSF_SL_COUNT; sl++) {
            tlsf_block *n = h->heads[fl][sl], *pv = 0;
            int bit = (h->sl_bitmap[fl] >> sl) & 1;
            if (bit != (n != 0))
                return 9;
            if (n)
                row_nonempty = 1;
            for (; n; pv = n, n = n->next_free) {
                unsigned f2, s2;
                if (++listed > rep.free_blocks)
                    return 12; /* too many nodes, or a cycle */
                if ((const char *)n < lo || (const char *)n + TLSF_HDR > hi ||
                    !(n->size & TLSF_FREE))
                    return 10;
                tlsf_check_mapping(bsize(n), &f2, &s2);
                if (f2 != fl || s2 != sl)
                    return 10;
                if (n->prev_free != pv)
                    return 11;
            }
        }
        if (((h->fl_bitmap >> fl) & 1) != (unsigned)row_nonempty)
            return 8;
    }
    if (listed != rep.free_blocks)
        return 12;
    rep.largest_alloc = tlsf_check_largest_alloc(h);
    if (r)
        *r = rep;
    return 0;
}
