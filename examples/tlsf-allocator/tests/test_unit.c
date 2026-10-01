/*
 * test_unit.c - POSIX unit tests for the TLSF allocator.
 * Build: see Makefile target 'unit' (compiled with -DTLSF_STATS).
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "tlsf.h"
#include "tlsf_check.h"
#include "tlsf_internal.h"
#include "opcount.h"

static int g_fail, g_checks;

#define CHECK(c)                                                           \
    do {                                                                   \
        g_checks++;                                                        \
        if (!(c)) {                                                        \
            g_fail++;                                                      \
            printf("  CHECK FAILED %s:%d: %s\n", __FILE__, __LINE__, #c);  \
        }                                                                  \
    } while (0)

#define BIG (1u << 20)
static uint64_t g_big[BIG / 8 + 8];   /* 1 MiB + slack */
static uint64_t g_buf2[65536 / 8];
static uint64_t g_buf3[65536 / 8];

static int consistent(tlsf_t *h, const void *mem, size_t n, tlsf_report *r)
{
    int e = tlsf_check(h, mem, n, r);
    if (e)
        printf("  tlsf_check error %d\n", e);
    return e == 0;
}

static void fill(void *p, size_t n, unsigned seed)
{
    unsigned char *c = p;
    size_t i;
    for (i = 0; i < n; i++)
        c[i] = (unsigned char)(seed * 131u + i * 7u);
}

static int verify(const void *p, size_t n, unsigned seed)
{
    const unsigned char *c = p;
    size_t i;
    for (i = 0; i < n; i++)
        if (c[i] != (unsigned char)(seed * 131u + i * 7u))
            return 0;
    return 1;
}

static size_t blksz_of(void *p)
{
    tlsf_block *b = (tlsf_block *)((char *)p - TLSF_HDR);
    return b->size & ~TLSF_FLAGS;
}

/* Fill every free block so the heap has no free memory left. */
static void fill_heap(tlsf_t *h)
{
    size_t L;
    while ((L = tlsf_check_largest_alloc(h)) != 0)
        if (!oc_malloc(h, L))
            break;
}

static void test_init_invalid(void)
{
    unsigned char *m = (unsigned char *)g_buf2;
    CHECK(tlsf_init(NULL, 4096) == NULL);
    CHECK(tlsf_init(m, 0) == NULL);
    CHECK(tlsf_init(m, TLSF_CTRL_SIZE + TLSF_MIN_BLOCK + TLSF_HDR - 1) == NULL);
    CHECK(tlsf_init(m, TLSF_CTRL_SIZE + TLSF_MIN_BLOCK + TLSF_HDR) != NULL);
    /* misaligned start: one extra byte is consumed by alignment */
    CHECK(tlsf_init(m + 1, TLSF_CTRL_SIZE + TLSF_MIN_BLOCK + TLSF_HDR) == NULL);
    CHECK(tlsf_init(m + 1, TLSF_CTRL_SIZE + TLSF_MIN_BLOCK + TLSF_HDR +
                               TLSF_ALIGN - 1) != NULL);
    CHECK(tlsf_init(g_big, TLSF_MAX_REGION + 1) == NULL);
    CHECK(tlsf_init(g_big, (size_t)-1) == NULL);
    {
        tlsf_t *h = tlsf_init(g_big, TLSF_MAX_REGION);
        tlsf_report r;
        CHECK(h != NULL);
        CHECK(consistent(h, g_big, TLSF_MAX_REGION, &r));
        CHECK(r.free_blocks == 1 && r.used_blocks == 0);
        CHECK(r.free_payload == TLSF_MAX_REGION - TLSF_CTRL_SIZE - 2 * TLSF_HDR);
    }
    /* minimal heap: exactly one minimum block */
    {
        size_t n = TLSF_CTRL_SIZE + TLSF_MIN_BLOCK + TLSF_HDR;
        tlsf_t *h = tlsf_init(m, n);
        void *p;
        CHECK(h && consistent(h, m, n, NULL));
        p = oc_malloc(h, TLSF_MIN_BLOCK - TLSF_HDR);
        CHECK(p != NULL);
        CHECK(oc_malloc(h, 1) == NULL);
        oc_free(h, p);
        CHECK(consistent(h, m, n, NULL));
    }
}

static void test_trivial(void)
{
    tlsf_t *h = tlsf_init(g_buf2, sizeof g_buf2);
    tlsf_report r0, r1;
    CHECK(h != NULL);
    CHECK(consistent(h, g_buf2, sizeof g_buf2, &r0));
    CHECK(tlsf_malloc(h, 0) == NULL);
    CHECK(tlsf_malloc(h, (size_t)-1) == NULL);
    CHECK(tlsf_malloc(h, TLSF_MAX_ALLOC + 1) == NULL);
    CHECK(oc_malloc(h, TLSF_MAX_ALLOC) == NULL); /* larger than this heap */
    CHECK(oc_malloc(h, sizeof g_buf2) == NULL);
    tlsf_free(h, NULL);
    CHECK(consistent(h, g_buf2, sizeof g_buf2, &r1));
    CHECK(r0.free_payload == r1.free_payload && r1.free_blocks == 1);
    /* a 1 MiB heap can serve TLSF_MAX_ALLOC-sized requests up to its size */
    {
        tlsf_t *hb = tlsf_init(g_big, TLSF_MAX_REGION);
        size_t L = tlsf_check_largest_alloc(hb);
        void *p;
        CHECK(L > 0 && L <= TLSF_MAX_ALLOC);
        p = oc_malloc(hb, L);
        CHECK(p != NULL);
        oc_free(hb, p);
        CHECK(oc_malloc(hb, L + 1) == NULL);
        CHECK(consistent(hb, g_big, TLSF_MAX_REGION, NULL));
    }
}

static void test_alignment(void)
{
    unsigned off;
    for (off = 0; off < 2 * TLSF_ALIGN; off++) {
        unsigned char *m = (unsigned char *)g_buf2 + off;
        size_t n = sizeof g_buf2 - 2 * TLSF_ALIGN;
        tlsf_t *h = tlsf_init(m, n);
        void *ptrs[300];
        size_t i;
        CHECK(h != NULL);
        CHECK(((uintptr_t)h & (TLSF_ALIGN - 1)) == 0);
        for (i = 1; i < 300; i++) {
            ptrs[i] = oc_malloc(h, i);
            CHECK(ptrs[i] != NULL);
            CHECK(((uintptr_t)ptrs[i] & (TLSF_ALIGN - 1)) == 0);
            CHECK((unsigned char *)ptrs[i] >= m &&
                  (unsigned char *)ptrs[i] + i <= m + n);
            fill(ptrs[i], i, (unsigned)i);
        }
        CHECK(consistent(h, m, n, NULL));
        for (i = 1; i < 300; i += 2) {
            CHECK(verify(ptrs[i], i, (unsigned)i));
            oc_free(h, ptrs[i]);
        }
        CHECK(consistent(h, m, n, NULL));
        for (i = 2; i < 300; i += 2) {
            CHECK(verify(ptrs[i], i, (unsigned)i));
            oc_free(h, ptrs[i]);
        }
        {
            tlsf_report r;
            CHECK(consistent(h, m, n, &r));
            CHECK(r.free_blocks == 1 && r.used_blocks == 0);
        }
    }
}

static void test_exact_fit_and_split(void)
{
    tlsf_t *h = tlsf_init(g_buf2, sizeof g_buf2);
    char *a, *b, *c, *b2, *d;
    tlsf_report r;
    size_t need = tlsf_check_block_need(100);

    a = oc_malloc(h, 100);
    b = oc_malloc(h, 100);
    c = oc_malloc(h, 100);
    CHECK(a && b && c);
    /* consecutive splits of the initial block are physically adjacent */
    CHECK(b == a + need && c == b + need);
    CHECK(blksz_of(a) == need && blksz_of(b) == need);
    oc_free(h, b);
    CHECK(consistent(h, g_buf2, sizeof g_buf2, &r));
    CHECK(r.free_blocks == 2);
    /* exact fit: same size request reuses the hole without splitting */
    opmax_zero(h);
    b2 = tlsf_malloc(h, 100);
    CHECK(b2 == b);
    CHECK(h->stats.splits == 0 && h->stats.list_inserts == 0);
    /* a slightly smaller request whose remainder < min block: no split */
    oc_free(h, b2);
    d = oc_malloc(h, 100 - TLSF_ALIGN);
    CHECK(d == b && blksz_of(d) == need);
    /* a request needing half of the hole splits it (remainder >= min) */
    oc_free(h, d);
    opmax_zero(h);
    d = tlsf_malloc(h, 40);
    CHECK(d == b);
    CHECK(h->stats.splits == 1);
    CHECK(blksz_of(d) == tlsf_check_block_need(40));
    CHECK(consistent(h, g_buf2, sizeof g_buf2, &r));
    CHECK(r.free_blocks == 2 && r.used_blocks == 3);
}

static void test_coalesce(void)
{
    tlsf_t *h = tlsf_init(g_buf2, sizeof g_buf2);
    char *p[6];
    tlsf_report r;
    int i;
    for (i = 0; i < 6; i++)
        p[i] = oc_malloc(h, 64);
    /* layout: p0 p1 p2 p3 p4 p5 [tail free] */
    oc_free(h, p[1]); /* no free neighbour */
    CHECK(h->stats.coalesces == 0);
    oc_free(h, p[3]); /* no free neighbour */
    CHECK(h->stats.coalesces == 0);
    CHECK(consistent(h, g_buf2, sizeof g_buf2, &r) && r.free_blocks == 3);
    oc_free(h, p[2]); /* both neighbours free */
    CHECK(h->stats.coalesces == 2);
    CHECK(consistent(h, g_buf2, sizeof g_buf2, &r) && r.free_blocks == 2);
    oc_free(h, p[0]); /* next neighbour free only */
    CHECK(h->stats.coalesces == 1);
    CHECK(consistent(h, g_buf2, sizeof g_buf2, &r) && r.free_blocks == 2);
    oc_free(h, p[5]); /* next (tail) free only */
    CHECK(h->stats.coalesces == 1);
    CHECK(consistent(h, g_buf2, sizeof g_buf2, &r) && r.free_blocks == 2);
    oc_free(h, p[4]); /* both neighbours free -> whole heap */
    CHECK(h->stats.coalesces == 2);
    CHECK(consistent(h, g_buf2, sizeof g_buf2, &r) && r.free_blocks == 1);
    /* previous neighbour free only */
    p[0] = oc_malloc(h, 64);
    p[1] = oc_malloc(h, 64);
    p[2] = oc_malloc(h, 64);
    oc_free(h, p[0]);
    oc_free(h, p[1]);
    CHECK(h->stats.coalesces == 1); /* merged with p0 (prev) only */
    CHECK(consistent(h, g_buf2, sizeof g_buf2, &r) && r.free_blocks == 2);
    oc_free(h, p[2]);
    CHECK(consistent(h, g_buf2, sizeof g_buf2, &r) && r.free_blocks == 1);
}

static void test_exhaustion_recovery(void)
{
    static void *ptrs[8192];
    size_t sizes[] = {1, 24, 100, 1000};
    unsigned s, round;
    for (s = 0; s < 4; s++) {
        for (round = 0; round < 2; round++) {
            tlsf_t *h = tlsf_init(g_buf2, sizeof g_buf2);
            tlsf_report r0, r;
            size_t n = 0, i, need = tlsf_check_block_need(sizes[s]);
            unsigned x = 12345;
            CHECK(consistent(h, g_buf2, sizeof g_buf2, &r0));
            while (n < 8192 && (ptrs[n] = oc_malloc(h, sizes[s])) != NULL) {
                fill(ptrs[n], sizes[s], (unsigned)n);
                n++;
            }
            CHECK(n < 8192);
            /* sequential splits: at most one free tail remains, and the
             * used blocks plus that tail account for the whole region */
            CHECK(consistent(h, g_buf2, sizeof g_buf2, &r));
            CHECK(r.free_blocks <= 1 && r.used_blocks == n);
            CHECK(r.used_footprint >= n * need);
            CHECK(r.used_footprint +
                      (r.free_blocks ? r.free_payload + TLSF_HDR : 0) ==
                  r0.free_payload + TLSF_HDR);
            if (round == 0) { /* free in pseudo-random order */
                for (i = n; i > 1; i--) {
                    size_t j;
                    void *t;
                    x = x * 1103515245u + 12345u;
                    j = (x >> 8) % i;
                    t = ptrs[i - 1];
                    ptrs[i - 1] = ptrs[j];
                    ptrs[j] = t;
                }
                for (i = 0; i < n; i++)
                    oc_free(h, ptrs[i]);
            } else { /* reverse order */
                for (i = n; i-- > 0;) {
                    CHECK(verify(ptrs[i], sizes[s], (unsigned)i));
                    oc_free(h, ptrs[i]);
                }
            }
            CHECK(consistent(h, g_buf2, sizeof g_buf2, &r));
            CHECK(r.free_blocks == 1 && r.free_payload == r0.free_payload);
            /* recovery: a large allocation succeeds again */
            CHECK((ptrs[0] = oc_malloc(h, r0.largest_alloc)) != NULL);
            oc_free(h, ptrs[0]);
        }
    }
}

static void test_independent_heaps(void)
{
    tlsf_t *a = tlsf_init(g_buf2, sizeof g_buf2);
    tlsf_t *b = tlsf_init(g_buf3, sizeof g_buf3);
    void *pa[64], *pb[64];
    tlsf_report ra, rb;
    int i;
    CHECK(a && b && a != b);
    for (i = 0; i < 64; i++) {
        pa[i] = oc_malloc(a, 100 + i);
        pb[i] = oc_malloc(b, 300 - i);
        CHECK((char *)pa[i] >= (char *)g_buf2 &&
              (char *)pa[i] < (char *)g_buf2 + sizeof g_buf2);
        CHECK((char *)pb[i] >= (char *)g_buf3 &&
              (char *)pb[i] < (char *)g_buf3 + sizeof g_buf3);
        fill(pa[i], 100 + i, i);
        fill(pb[i], 300 - i, 1000 + i);
    }
    /* exhaust heap a; heap b unaffected */
    while (oc_malloc(a, 256))
        ;
    CHECK(oc_malloc(b, 256) != NULL);
    CHECK(consistent(a, g_buf2, sizeof g_buf2, &ra));
    CHECK(consistent(b, g_buf3, sizeof g_buf3, &rb));
    for (i = 0; i < 64; i++) {
        CHECK(verify(pa[i], 100 + i, i));
        CHECK(verify(pb[i], 300 - i, 1000 + i));
        oc_free(b, pb[i]);
    }
    CHECK(consistent(a, g_buf2, sizeof g_buf2, &ra));
    CHECK(consistent(b, g_buf3, sizeof g_buf3, &rb));
    CHECK(rb.used_blocks == 1);
}

/* Class lower bounds for every (fl, sl). */
static size_t class_lower(unsigned fl, unsigned sl)
{
    if (fl == 0)
        return (size_t)sl * TLSF_ALIGN;
    return ((size_t)1 << (fl + TLSF_FL_SHIFT - 1)) +
           (size_t)sl * ((size_t)1 << (fl + TLSF_FL_SHIFT - 1 - TLSF_SL_LOG2));
}

/* Requests around every class boundary on a 1 MiB heap. */
static void test_mapping_transitions(void)
{
    tlsf_t *h = tlsf_init(g_big, TLSF_MAX_REGION);
    tlsf_report r0;
    unsigned fl, sl, cnt = 0;
    long d;
    CHECK(consistent(h, g_big, TLSF_MAX_REGION, &r0));
    /* independent mapping agrees with class_lower at every boundary */
    for (fl = 0; fl < TLSF_FL_COUNT; fl++)
        for (sl = 0; sl < TLSF_SL_COUNT; sl++) {
            size_t lb = class_lower(fl, sl);
            unsigned f2, s2;
            if (lb < TLSF_MIN_BLOCK)
                continue;
            tlsf_check_mapping(lb, &f2, &s2);
            CHECK(f2 == fl && s2 == sl);
            if (lb > TLSF_ALIGN) {
                tlsf_check_mapping(lb - TLSF_ALIGN, &f2, &s2);
                CHECK(f2 * TLSF_SL_COUNT + s2 == fl * TLSF_SL_COUNT + sl - 1);
            }
            for (d = -2 * (long)TLSF_ALIGN; d <= 2 * (long)TLSF_ALIGN; d++) {
                long req = (long)lb - (long)TLSF_HDR + d;
                char *p;
                if (req <= 0 || (size_t)req > r0.largest_alloc)
                    continue;
                p = oc_malloc(h, (size_t)req);
                CHECK(p != NULL);
                if (!p)
                    continue;
                cnt++;
                CHECK(blksz_of(p) >= tlsf_check_block_need((size_t)req));
                fill(p, (size_t)req, (unsigned)req);
                CHECK(consistent(h, g_big, TLSF_MAX_REGION, NULL));
                CHECK(verify(p, (size_t)req, (unsigned)req));
                oc_free(h, p);
            }
        }
    CHECK(consistent(h, g_big, TLSF_MAX_REGION, NULL));
    printf("  boundary requests exercised: %u\n", cnt);
    /* every request size 1..8192 */
    {
        size_t n;
        for (n = 1; n <= 8192; n++) {
            char *p = oc_malloc(h, n);
            CHECK(p != NULL);
            fill(p, n, (unsigned)n);
            CHECK(verify(p, n, (unsigned)n));
            oc_free(h, p);
        }
        CHECK(consistent(h, g_big, TLSF_MAX_REGION, NULL));
    }
}

/*
 * The core invariant: with exactly one free block of size S, a request
 * succeeds iff its rounded class <= class(S), and then it gets that block.
 * Requests that need more than S must fail.
 */
static void test_fit_invariant(void)
{
    static size_t S_list[1200];
    size_t nS = 0, s, k;
    unsigned long ok = 0, rejected_fit = 0;
    for (s = TLSF_MIN_BLOCK; s <= 4096; s += TLSF_ALIGN)
        S_list[nS++] = s;
    for (k = 12; k < 19; k++) {
        long d;
        for (d = -3; d <= 3; d++)
            S_list[nS++] = ((size_t)1 << k) + (size_t)(d * (long)TLSF_ALIGN);
    }
    for (s = 0; s < nS; s++) {
        size_t S = S_list[s], lo, hi, req;
        unsigned fS, sS;
        tlsf_t *h = tlsf_init(g_big, TLSF_MAX_REGION);
        char *a = oc_malloc(h, S - TLSF_HDR);
        CHECK(a && blksz_of(a) == S);
        fill_heap(h);
        oc_free(h, a);
        {
            tlsf_report r;
            CHECK(consistent(h, g_big, TLSF_MAX_REGION, &r));
            CHECK(r.free_blocks == 1 && r.largest_free == S - TLSF_HDR);
        }
        tlsf_check_mapping(S, &fS, &sS);
        /* sample requests: small ones plus everything near S */
        lo = S > 4 * TLSF_ALIGN + TLSF_HDR ? S - TLSF_HDR - 4 * TLSF_ALIGN : 1;
        hi = S + 2 * TLSF_ALIGN;
        if (S > 600)
            lo = S - S / 8;
        for (req = lo; req <= hi; req++) {
            size_t need = tlsf_check_block_need(req), srch = need;
            unsigned f, sl2, t = 0;
            int expect;
            char *p;
            if (srch >= TLSF_SMALL) {
                size_t x = srch;
                while (x >>= 1)
                    t++;
                srch += ((size_t)1 << (t - TLSF_SL_LOG2)) - 1;
            }
            tlsf_check_mapping(srch, &f, &sl2);
            expect = need <= S && f * 16 + sl2 <= fS * 16 + sS;
            p = oc_malloc(h, req);
            CHECK((p != NULL) == expect);
            CHECK(!p || p == a);
            if (!p && need <= S)
                rejected_fit++;
            if (p) {
                ok++;
                fill(p, req, 7);
                CHECK(consistent(h, g_big, TLSF_MAX_REGION, NULL));
                oc_free(h, p);
            }
        }
    }
    printf("  hole sizes: %lu, successful fits: %lu, good-fit rejections "
           "(block fits but class rounded above): %lu\n",
           (unsigned long)nS, ok, rejected_fit);
}

typedef struct lockstat {
    int depth, max_depth;
    unsigned long locks, unlocks;
} lockstat;

static void t_lock(void *ctx)
{
    lockstat *l = ctx;
    l->locks++;
    if (++l->depth > l->max_depth)
        l->max_depth = l->depth;
}

static void t_unlock(void *ctx)
{
    lockstat *l = ctx;
    l->unlocks++;
    l->depth--;
}

static void test_lock_hooks(void)
{
    tlsf_t *h = tlsf_init(g_buf2, sizeof g_buf2);
    lockstat ls = {0, 0, 0, 0};
    void *p[100];
    unsigned long ops = 0;
    int i;
    tlsf_set_lock(h, t_lock, t_unlock, &ls);
    for (i = 0; i < 100; i++, ops++) {
        p[i] = tlsf_malloc(h, 1 + (size_t)i * 7);
        CHECK(p[i] != NULL);
    }
    CHECK(tlsf_malloc(h, 60000) == NULL); /* failing allocation still locks */
    ops++;
    for (i = 0; i < 100; i++, ops++)
        tlsf_free(h, p[i]);
    tlsf_malloc(h, 0);   /* rejected before locking */
    tlsf_free(h, NULL);  /* no-op, no locking */
    CHECK(ls.locks == ops && ls.unlocks == ops);
    CHECK(ls.depth == 0 && ls.max_depth == 1);
    tlsf_set_lock(h, NULL, NULL, NULL);
    p[0] = tlsf_malloc(h, 10);
    tlsf_free(h, p[0]);
    CHECK(ls.locks == ops);
    CHECK(consistent(h, g_buf2, sizeof g_buf2, NULL));
}

/* The inspector must reject stray sl_bitmap bits >= TLSF_SL_COUNT, which
 * tlsf_malloc's CTZ could otherwise pick as a column past the row. */
static void test_inspector_stray_bits(void)
{
    tlsf_t *h = tlsf_init(g_buf3, sizeof g_buf3);
    unsigned fl;
    CHECK(h != NULL && tlsf_check(h, g_buf3, sizeof g_buf3, NULL) == 0);
    for (fl = 0; fl < TLSF_FL_COUNT; fl++) {
        uint32_t saved = h->sl_bitmap[fl];
        h->sl_bitmap[fl] |= (uint32_t)1 << TLSF_SL_COUNT;       /* bit 16 */
        CHECK(tlsf_check(h, g_buf3, sizeof g_buf3, NULL) == 9);
        h->sl_bitmap[fl] = saved | (uint32_t)1 << 31;          /* top bit */
        CHECK(tlsf_check(h, g_buf3, sizeof g_buf3, NULL) == 9);
        h->sl_bitmap[fl] = saved;
    }
    CHECK(consistent(h, g_buf3, sizeof g_buf3, NULL));
}

static void report_opmax(void)
{
    int k;
    printf("\nOperation-count maxima over all unit tests (TLSF_STATS):\n");
    printf("%-12s %9s %7s %7s %7s %9s %6s %9s\n", "op", "calls", "bitmap",
           "insert", "remove", "neighbor", "split", "coalesce");
    for (k = 0; k < OP_KINDS; k++) {
        tlsf_stats *m = &g_opmax[k].max;
        printf("%-12s %9lu %7u %7u %7u %9u %6u %9u\n", opmax_names[k],
               g_opmax[k].calls, m->bitmap_checks, m->list_inserts,
               m->list_removes, m->neighbor_checks, m->splits, m->coalesces);
    }
}

static void check_bounds(void)
{
    tlsf_stats *m = &g_opmax[OP_MALLOC_OK].max;
    CHECK(m->bitmap_checks <= 3 && m->list_inserts <= 1 &&
          m->list_removes <= 1 && m->neighbor_checks <= 1 && m->splits <= 1 &&
          m->coalesces == 0);
    m = &g_opmax[OP_MALLOC_FAIL].max;
    CHECK(m->bitmap_checks <= 2 && m->list_inserts == 0 &&
          m->list_removes == 0 && m->neighbor_checks == 0 && m->splits == 0);
    m = &g_opmax[OP_FREE].max;
    CHECK(m->bitmap_checks == 0 && m->list_inserts == 1 &&
          m->list_removes <= 2 && m->neighbor_checks == 2 && m->splits == 0 &&
          m->coalesces <= 2);
}

#define RUN(t)                                                             \
    do {                                                                   \
        int before = g_fail;                                               \
        printf("[ RUN  ] %s\n", #t);                                       \
        t();                                                               \
        printf("[ %s ] %s\n", g_fail == before ? " OK " : "FAIL", #t);    \
    } while (0)

int main(void)
{
    printf("TLSF unit tests: TLSF_ALIGN=%u HDR=%u MIN_BLOCK=%u CTRL=%u "
           "FL_COUNT=%u SL_COUNT=%u MAX_REGION=%lu MAX_ALLOC=%lu "
           "sizeof(void*)=%u\n",
           (unsigned)TLSF_ALIGN, (unsigned)TLSF_HDR, (unsigned)TLSF_MIN_BLOCK,
           (unsigned)TLSF_CTRL_SIZE, (unsigned)TLSF_FL_COUNT,
           (unsigned)TLSF_SL_COUNT, (unsigned long)TLSF_MAX_REGION,
           (unsigned long)TLSF_MAX_ALLOC, (unsigned)sizeof(void *));
    RUN(test_init_invalid);
    RUN(test_trivial);
    RUN(test_alignment);
    RUN(test_exact_fit_and_split);
    RUN(test_coalesce);
    RUN(test_exhaustion_recovery);
    RUN(test_independent_heaps);
    RUN(test_mapping_transitions);
    RUN(test_fit_invariant);
    RUN(test_lock_hooks);
    RUN(test_inspector_stray_bits);
    report_opmax();
    RUN(check_bounds);
    printf("\n%d checks, %d failures\n", g_checks, g_fail);
    printf("%s\n", g_fail ? "UNIT TESTS FAILED" : "UNIT TESTS PASSED");
    return g_fail ? 1 : 0;
}
