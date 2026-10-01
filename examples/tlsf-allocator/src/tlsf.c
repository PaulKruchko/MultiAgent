/*
 * tlsf.c - compact Two-Level Segregated Fit allocator (C99, no libc).
 *
 * Design after Masmano, Ripoll, Crespo & Real, "TLSF: a new dynamic memory
 * allocator for real-time systems" (ECRTS 2004).  tlsf_malloc() and
 * tlsf_free() contain no loops and no recursion: class selection uses at most
 * three bitmap words and two bit scans, and freeing inspects only the two
 * physical neighbours.  See README.md for the operation-count bounds.
 */
#include "tlsf_internal.h"

#ifdef TLSF_STATS
#define STAT(h, f) ((h)->stats.f++)
#else
#define STAT(h, f) ((void)0)
#endif

#ifndef TLSF_CLZ
#define TLSF_CLZ(x) __builtin_clz(x) /* x != 0; CLZ on ARMv7-M */
#endif
#ifndef TLSF_CTZ
#define TLSF_CTZ(x) __builtin_ctz(x) /* x != 0; RBIT+CLZ on ARMv7-M */
#endif

/* Compile-time configuration checks (C99: negative array size on failure). */
typedef char tlsf_check_align[(TLSF_ALIGN_LOG2 >= 2 && TLSF_ALIGN_LOG2 <= 5 &&
                               TLSF_ALIGN >= sizeof(void *)) ? 1 : -1];
typedef char tlsf_check_region[(TLSF_MAX_REGION_LOG2 >= TLSF_FL_SHIFT + 1 &&
                                TLSF_MAX_REGION_LOG2 <= 31 &&
                                TLSF_MAX_REGION_LOG2 < sizeof(size_t) * 8)
                                   ? 1 : -1];
/* mapping() and tlsf_malloc() compute floor(log2) as 31 - clz((unsigned)x). */
typedef char tlsf_check_u32[(sizeof(unsigned) == 4) ? 1 : -1];

static size_t block_size(const tlsf_block *b)
{
    return b->size & ~TLSF_FLAGS;
}

static tlsf_block *next_phys(const tlsf_block *b)
{
    return (tlsf_block *)((char *)b + block_size(b));
}

/* Class of a block size: fl = floor(log2), sl = next 4 bits below the MSB.
 * Sizes below TLSF_SMALL map linearly (one class per TLSF_ALIGN bytes). */
static void mapping(size_t size, unsigned *fl, unsigned *sl)
{
    if (size < TLSF_SMALL) {
        *fl = 0;
        *sl = (unsigned)(size >> TLSF_ALIGN_LOG2);
    } else {
        unsigned t = 31u - (unsigned)TLSF_CLZ((unsigned)size);
        *sl = (unsigned)(size >> (t - TLSF_SL_LOG2)) ^ TLSF_SL_COUNT;
        *fl = t - (TLSF_FL_SHIFT - 1);
    }
}

static void insert_free(tlsf_t *h, tlsf_block *b)
{
    unsigned fl, sl;
    tlsf_block *head;

    STAT(h, list_inserts);
    mapping(block_size(b), &fl, &sl);
    head = h->heads[fl][sl];
    b->next_free = head;
    b->prev_free = 0;
    if (head)
        head->prev_free = b;
    h->heads[fl][sl] = b;
    h->fl_bitmap |= 1u << fl;
    h->sl_bitmap[fl] |= 1u << sl;
}

static void remove_free(tlsf_t *h, tlsf_block *b)
{
    unsigned fl, sl;
    tlsf_block *n = b->next_free, *p = b->prev_free;

    STAT(h, list_removes);
    mapping(block_size(b), &fl, &sl);
    if (n)
        n->prev_free = p;
    if (p) {
        p->next_free = n;
    } else {
        h->heads[fl][sl] = n;
        if (!n) {
            h->sl_bitmap[fl] &= ~(1u << sl);
            if (!h->sl_bitmap[fl])
                h->fl_bitmap &= ~(1u << fl);
        }
    }
}

tlsf_t *tlsf_init(void *mem, size_t bytes)
{
    uintptr_t a = (uintptr_t)mem;
    size_t pad = (size_t)(0u - a) & (TLSF_ALIGN - 1);
    tlsf_t *h;
    tlsf_block *b, *end;
    uint32_t volatile *bm; /* volatile: keep the compiler from emitting memset */
    tlsf_block *volatile *hp;
    unsigned i;

    if (!mem || bytes > TLSF_MAX_REGION || a + bytes < a ||
        bytes < pad + TLSF_CTRL_SIZE + TLSF_MIN_BLOCK + TLSF_HDR)
        return 0;
    h = (tlsf_t *)(a + pad);
    bm = h->sl_bitmap;
    for (i = 0; i < TLSF_FL_COUNT; i++)
        bm[i] = 0;
    hp = &h->heads[0][0];
    for (i = 0; i < TLSF_FL_COUNT * TLSF_SL_COUNT; i++)
        hp[i] = 0;
    h->fl_bitmap = 0;
    h->lock = h->unlock = 0;

    b = (tlsf_block *)((char *)h + TLSF_CTRL_SIZE);
    b->size = ((bytes - pad - TLSF_CTRL_SIZE - TLSF_HDR) & ~(TLSF_ALIGN - 1)) |
              TLSF_FREE;
    end = next_phys(b);
    end->size = TLSF_PREV_FREE; /* size 0, used: terminates the region */
    end->prev_phys = b;
    insert_free(h, b);
    return h;
}

void tlsf_set_lock(tlsf_t *h, tlsf_lock_fn lock, tlsf_lock_fn unlock,
                   void *ctx)
{
    h->lock = lock;
    h->unlock = unlock;
    h->lock_ctx = ctx;
}

static void call_hook(tlsf_t *h, tlsf_lock_fn f)
{
    if (f)
        f(h->lock_ctx);
}

void *tlsf_malloc(tlsf_t *h, size_t size)
{
    size_t need, search;
    unsigned fl, sl;
    uint32_t map;
    tlsf_block *b = 0, *nb;

    if (size - 1 >= TLSF_MAX_ALLOC) /* size == 0 wraps and is rejected */
        return 0;
    need = TLSF_ALIGN_UP(size + TLSF_HDR);
    if (need < TLSF_MIN_BLOCK)
        need = TLSF_MIN_BLOCK;
    /* Round up to the next class boundary: every block in the selected
     * class (or any higher one) is then >= need.  TLSF_MAX_ALLOC keeps
     * 'search' below TLSF_MAX_REGION, so fl < TLSF_FL_COUNT. */
    search = need;
    if (search >= TLSF_SMALL)
        search += ((size_t)1 << (31u - (unsigned)TLSF_CLZ((unsigned)search) -
                                 TLSF_SL_LOG2)) - 1;
    mapping(search, &fl, &sl);

    call_hook(h, h->lock);
    STAT(h, bitmap_checks);
    map = h->sl_bitmap[fl] & (~0u << sl);
    if (!map) {
        STAT(h, bitmap_checks);
        map = h->fl_bitmap & (~0u << (fl + 1)); /* fl + 1 <= 31 */
        if (map) {
            fl = (unsigned)TLSF_CTZ(map);
            STAT(h, bitmap_checks);
            map = h->sl_bitmap[fl]; /* non-zero because fl bit is set */
        }
    }
    if (map) {
        sl = (unsigned)TLSF_CTZ(map);
        b = h->heads[fl][sl];
        remove_free(h, b);
        STAT(h, neighbor_checks);
        if (block_size(b) - need >= TLSF_MIN_BLOCK) {
            /* Split: the tail becomes a free block.  Its successor already
             * has PREV_FREE set because b was free. */
            STAT(h, splits);
            nb = (tlsf_block *)((char *)b + need);
            nb->size = (block_size(b) - need) | TLSF_FREE;
            next_phys(nb)->prev_phys = nb;
            b->size = need; /* a free block never has a free predecessor */
            insert_free(h, nb);
        } else {
            b->size &= ~TLSF_FREE;
            next_phys(b)->size &= ~TLSF_PREV_FREE;
        }
    }
    call_hook(h, h->unlock);
    return b ? (char *)b + TLSF_HDR : 0;
}

void tlsf_free(tlsf_t *h, void *ptr)
{
    tlsf_block *b, *n;

    if (!ptr)
        return;
    b = (tlsf_block *)((char *)ptr - TLSF_HDR);
    call_hook(h, h->lock);
    b->size |= TLSF_FREE;
    STAT(h, neighbor_checks);
    if (b->size & TLSF_PREV_FREE) {
        tlsf_block *p = b->prev_phys;
        STAT(h, coalesces);
        remove_free(h, p);
        p->size += block_size(b); /* flags of p (FREE) are preserved */
        b = p;
    }
    n = next_phys(b);
    STAT(h, neighbor_checks);
    if (n->size & TLSF_FREE) {
        STAT(h, coalesces);
        remove_free(h, n);
        b->size += block_size(n);
        n = next_phys(b);
    }
    n->size |= TLSF_PREV_FREE;
    n->prev_phys = b;
    insert_free(h, b);
    call_hook(h, h->unlock);
}
