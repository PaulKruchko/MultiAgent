/*
 * opcount.h - wrappers that record per-operation primitive counts (requires
 * a TLSF_STATS build).  Freestanding apart from the caller's own printing.
 */
#ifndef OPCOUNT_H
#define OPCOUNT_H

#include "tlsf_internal.h"

#ifndef TLSF_STATS
#error "opcount.h needs -DTLSF_STATS"
#endif

enum { OP_MALLOC_OK, OP_MALLOC_FAIL, OP_FREE, OP_KINDS };

typedef struct opmax {
    unsigned long calls;
    tlsf_stats max;
} opmax;

static opmax g_opmax[OP_KINDS];

static void opmax_zero(tlsf_t *h)
{
    tlsf_stats z = {0, 0, 0, 0, 0, 0};
    h->stats = z;
}

#define OPMAX_UPD(f) if (s->f > m->max.f) m->max.f = s->f
static void opmax_record(int kind, const tlsf_stats *s)
{
    opmax *m = &g_opmax[kind];
    m->calls++;
    OPMAX_UPD(bitmap_checks);
    OPMAX_UPD(list_inserts);
    OPMAX_UPD(list_removes);
    OPMAX_UPD(neighbor_checks);
    OPMAX_UPD(splits);
    OPMAX_UPD(coalesces);
}

static void *oc_malloc(tlsf_t *h, size_t n)
{
    void *p;
    opmax_zero(h);
    p = tlsf_malloc(h, n);
    /* size 0 / oversize are rejected before touching the heap: not counted */
    if (n != 0 && n <= TLSF_MAX_ALLOC)
        opmax_record(p ? OP_MALLOC_OK : OP_MALLOC_FAIL, &h->stats);
    return p;
}

static void oc_free(tlsf_t *h, void *p)
{
    opmax_zero(h);
    tlsf_free(h, p);
    if (p)
        opmax_record(OP_FREE, &h->stats);
}

static const char *const opmax_names[OP_KINDS] = {"malloc_ok", "malloc_fail",
                                                  "free"};

#endif
