/*
 * Bare-metal Cortex-M3 tests for the TLSF allocator (production build, no
 * TLSF_STATS), run under QEMU lm3s6965evb with semihosting output.
 * Prints "BAREMETAL PASS" and exits with ADP_Stopped_ApplicationExit only if
 * every check passed.
 */
#include <stddef.h>
#include <stdint.h>

#include "semihost.h"
#include "tlsf.h"
#include "tlsf_check.h"
#include "tlsf_internal.h"

static unsigned long g_checks, g_fail;

#define CHECK(c)                                                           \
    do {                                                                   \
        g_checks++;                                                        \
        if (!(c)) {                                                        \
            g_fail++;                                                      \
            sh_puts("  CHECK FAILED line ");                              \
            sh_putu(__LINE__);                                             \
            sh_puts(": " #c "\n");                                         \
        }                                                                  \
    } while (0)

static uint32_t g_mem[24 * 1024 / 4];
static uint32_t g_mem2[6 * 1024 / 4];
static uint32_t g_mem3[6 * 1024 / 4];

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

static int ok_heap(tlsf_t *h, const void *m, size_t n, tlsf_report *r)
{
    int e = tlsf_check(h, m, n, r);
    if (e) {
        sh_puts("  inspector error ");
        sh_putu((unsigned long)e);
        sh_puts("\n");
    }
    return e == 0;
}

static void t_init(void)
{
    unsigned char *m = (unsigned char *)g_mem;
    size_t minb = TLSF_CTRL_SIZE + TLSF_MIN_BLOCK + TLSF_HDR;
    CHECK(tlsf_init(0, 4096) == 0);
    CHECK(tlsf_init(m, minb - 1) == 0);
    CHECK(tlsf_init(m, minb) != 0);
    CHECK(tlsf_init(m + 1, minb) == 0);
    CHECK(tlsf_init(m + 1, minb + TLSF_ALIGN - 1) != 0);
    CHECK(tlsf_init(m, TLSF_MAX_REGION + 1) == 0); /* rejected, untouched */
    {
        tlsf_t *h = tlsf_init(m, sizeof g_mem);
        tlsf_report r;
        CHECK(h && ok_heap(h, m, sizeof g_mem, &r));
        CHECK(r.free_blocks == 1 &&
              r.free_payload == sizeof g_mem - TLSF_CTRL_SIZE - 2 * TLSF_HDR);
        CHECK(tlsf_malloc(h, 0) == 0);
        CHECK(tlsf_malloc(h, sizeof g_mem) == 0);
        CHECK(tlsf_malloc(h, (size_t)-1) == 0);
        tlsf_free(h, 0);
        CHECK(ok_heap(h, m, sizeof g_mem, &r) && r.free_blocks == 1);
    }
}

static void t_alignment(void)
{
    unsigned off;
    for (off = 0; off < 16; off++) {
        unsigned char *m = (unsigned char *)g_mem + off;
        size_t n = sizeof g_mem - 16, i;
        tlsf_t *h = tlsf_init(m, n);
        void *p[100];
        tlsf_report r;
        CHECK(h && ((uintptr_t)h & (TLSF_ALIGN - 1)) == 0);
        for (i = 1; i < 100; i++) {
            p[i] = tlsf_malloc(h, i);
            CHECK(p[i] && ((uintptr_t)p[i] & (TLSF_ALIGN - 1)) == 0);
            fill(p[i], i, (unsigned)i);
        }
        CHECK(ok_heap(h, m, n, 0));
        for (i = 1; i < 100; i += 2) {
            CHECK(verify(p[i], i, (unsigned)i));
            tlsf_free(h, p[i]);
        }
        for (i = 2; i < 100; i += 2) {
            CHECK(verify(p[i], i, (unsigned)i));
            tlsf_free(h, p[i]);
        }
        CHECK(ok_heap(h, m, n, &r) && r.free_blocks == 1 && r.used_blocks == 0);
    }
}

static void t_split_coalesce(void)
{
    tlsf_t *h = tlsf_init(g_mem, sizeof g_mem);
    char *p[6];
    tlsf_report r;
    size_t need = tlsf_check_block_need(64);
    int i;
    for (i = 0; i < 6; i++)
        p[i] = tlsf_malloc(h, 64);
    for (i = 1; i < 6; i++)
        CHECK(p[i] == p[i - 1] + need); /* split from the same block */
    tlsf_free(h, p[1]);
    tlsf_free(h, p[3]);
    CHECK(ok_heap(h, g_mem, sizeof g_mem, &r) && r.free_blocks == 3);
    tlsf_free(h, p[2]); /* both neighbours */
    CHECK(ok_heap(h, g_mem, sizeof g_mem, &r) && r.free_blocks == 2);
    /* the merged hole p1..p3 is served whole by a request of its size */
    CHECK(tlsf_malloc(h, 3 * need - TLSF_HDR) == p[1]);
    tlsf_free(h, p[1]);
    tlsf_free(h, p[0]); /* next only */
    CHECK(ok_heap(h, g_mem, sizeof g_mem, &r) && r.free_blocks == 2);
    tlsf_free(h, p[5]); /* next (tail) only */
    CHECK(ok_heap(h, g_mem, sizeof g_mem, &r) && r.free_blocks == 2);
    tlsf_free(h, p[4]); /* both */
    CHECK(ok_heap(h, g_mem, sizeof g_mem, &r) && r.free_blocks == 1);
    /* exact-fit reuse of a hole */
    p[0] = tlsf_malloc(h, 100);
    p[1] = tlsf_malloc(h, 100);
    p[2] = tlsf_malloc(h, 100);
    tlsf_free(h, p[1]);
    CHECK(tlsf_malloc(h, 100) == p[1]);
    tlsf_free(h, p[0]);
    tlsf_free(h, p[1]); /* previous neighbour only */
    CHECK(ok_heap(h, g_mem, sizeof g_mem, &r) && r.free_blocks == 2);
    tlsf_free(h, p[2]);
    CHECK(ok_heap(h, g_mem, sizeof g_mem, &r) && r.free_blocks == 1);
}

static void t_exhaustion(void)
{
    static void *ptrs[2048];
    static const size_t sizes[] = {1, 200};
    unsigned s;
    for (s = 0; s < 2; s++) {
        tlsf_t *h = tlsf_init(g_mem, sizeof g_mem);
        tlsf_report r0, r;
        size_t n = 0, i;
        ok_heap(h, g_mem, sizeof g_mem, &r0);
        while (n < 2048 && (ptrs[n] = tlsf_malloc(h, sizes[s])) != 0) {
            fill(ptrs[n], sizes[s], (unsigned)n);
            n++;
        }
        CHECK(n > 0 && n < 2048);
        CHECK(ok_heap(h, g_mem, sizeof g_mem, &r) && r.used_blocks == n);
        CHECK(tlsf_malloc(h, sizes[s]) == 0); /* still exhausted */
        for (i = 0; i < n; i += 2) {
            CHECK(verify(ptrs[i], sizes[s], (unsigned)i));
            tlsf_free(h, ptrs[i]);
        }
        for (i = 1; i < n; i += 2) {
            CHECK(verify(ptrs[i], sizes[s], (unsigned)i));
            tlsf_free(h, ptrs[i]);
        }
        CHECK(ok_heap(h, g_mem, sizeof g_mem, &r) && r.free_blocks == 1 &&
              r.free_payload == r0.free_payload);
        ptrs[0] = tlsf_malloc(h, r0.largest_alloc); /* recovery */
        CHECK(ptrs[0] != 0);
        tlsf_free(h, ptrs[0]);
        sh_puts("  size ");
        sh_putu((unsigned long)sizes[s]);
        sh_puts(": ");
        sh_putu((unsigned long)n);
        sh_puts(" allocations until exhaustion, fully recovered\n");
    }
}

static void t_independent(void)
{
    tlsf_t *a = tlsf_init(g_mem2, sizeof g_mem2);
    tlsf_t *b = tlsf_init(g_mem3, sizeof g_mem3);
    void *pa[16], *pb[16];
    tlsf_report r;
    int i;
    CHECK(a && b && a != b);
    for (i = 0; i < 16; i++) {
        pa[i] = tlsf_malloc(a, 50 + i);
        pb[i] = tlsf_malloc(b, 90 - i);
        CHECK((char *)pa[i] > (char *)g_mem2 &&
              (char *)pa[i] < (char *)g_mem2 + sizeof g_mem2);
        CHECK((char *)pb[i] > (char *)g_mem3 &&
              (char *)pb[i] < (char *)g_mem3 + sizeof g_mem3);
        fill(pa[i], 50 + i, i);
        fill(pb[i], 90 - i, 100 + i);
    }
    while (tlsf_malloc(a, 64))
        ;
    CHECK(tlsf_malloc(b, 64) != 0);
    for (i = 0; i < 16; i++) {
        CHECK(verify(pa[i], 50 + i, i) && verify(pb[i], 90 - i, 100 + i));
        tlsf_free(b, pb[i]);
    }
    CHECK(ok_heap(a, g_mem2, sizeof g_mem2, 0));
    CHECK(ok_heap(b, g_mem3, sizeof g_mem3, &r) && r.used_blocks == 1);
}

/* One free hole of size S: request succeeds iff rounded class <= class(S). */
static void t_fit_invariant(void)
{
    size_t S;
    unsigned long fits = 0, rej = 0;
    for (S = TLSF_MIN_BLOCK; S <= 1024; S += TLSF_ALIGN) {
        tlsf_t *h = tlsf_init(g_mem, sizeof g_mem);
        char *a = tlsf_malloc(h, S - TLSF_HDR), *p;
        size_t L, req;
        unsigned fS, sS;
        while ((L = tlsf_check_largest_alloc(h)) != 0)
            if (!tlsf_malloc(h, L))
                break;
        tlsf_free(h, a);
        tlsf_check_mapping(S, &fS, &sS);
        for (req = S > 64 ? S - 64 : 1; req <= S + 16; req++) {
            size_t need = tlsf_check_block_need(req), srch = need;
            unsigned f, sl, t = 0;
            int expect;
            if (srch >= TLSF_SMALL) {
                size_t x = srch;
                while (x >>= 1)
                    t++;
                srch += ((size_t)1 << (t - TLSF_SL_LOG2)) - 1;
            }
            tlsf_check_mapping(srch, &f, &sl);
            expect = need <= S && f * TLSF_SL_COUNT + sl <= fS * TLSF_SL_COUNT + sS;
            p = tlsf_malloc(h, req);
            CHECK((p != 0) == expect && (!p || p == a));
            if (p) {
                fits++;
                tlsf_free(h, p);
            } else if (need <= S) {
                rej++;
            }
        }
        CHECK(ok_heap(h, g_mem, sizeof g_mem, 0));
    }
    sh_puts("  successful fits ");
    sh_putu(fits);
    sh_puts(", good-fit rejections ");
    sh_putu(rej);
    sh_puts("\n");
}

static uint32_t lcg(uint32_t *s)
{
    *s = *s * 1664525u + 1013904223u;
    return *s >> 8;
}

#define NS 96
static struct { unsigned char *p; uint32_t n, tag; } g_sl[NS];

static void t_stress(void)
{
    const size_t hb = 16 * 1024;
    unsigned char *m = (unsigned char *)g_mem + 5;
    tlsf_t *h = tlsf_init(m, hb);
    uint32_t seed = 0xC0FFEEu, tag = 1;
    unsigned long i, allocs = 0, fails = 0, live = 0;
    tlsf_report r0, r;
    int k;
    ok_heap(h, m, hb, &r0);
    for (i = 0; i < 30000 && g_fail < 5; i++) {
        unsigned s = lcg(&seed) % NS;
        if (g_sl[s].p) {
            CHECK(verify(g_sl[s].p, g_sl[s].n, g_sl[s].tag));
            tlsf_free(h, g_sl[s].p);
            g_sl[s].p = 0;
            live--;
        } else {
            uint32_t c = lcg(&seed) % 100;
            uint32_t n = c < 70 ? 1 + lcg(&seed) % 64
                       : c < 97 ? 65 + lcg(&seed) % 512 : 577 + lcg(&seed) % 3000;
            unsigned char *p = tlsf_malloc(h, n);
            if (p) {
                CHECK(((uintptr_t)p & (TLSF_ALIGN - 1)) == 0 && p >= m &&
                      p + n <= m + hb);
                g_sl[s].p = p;
                g_sl[s].n = n;
                g_sl[s].tag = tag++;
                fill(p, n, g_sl[s].tag);
                allocs++;
                live++;
            } else {
                fails++;
            }
        }
        CHECK(ok_heap(h, m, hb, &r) && r.used_blocks == live);
    }
    for (k = 0; k < NS; k++)
        if (g_sl[k].p) {
            CHECK(verify(g_sl[k].p, g_sl[k].n, g_sl[k].tag));
            tlsf_free(h, g_sl[k].p);
        }
    CHECK(ok_heap(h, m, hb, &r) && r.free_blocks == 1 &&
          r.free_payload == r0.free_payload);
    sh_puts("  seed 0xC0FFEE, 16 KiB heap: ops=");
    sh_putu(i);
    sh_puts(" allocs=");
    sh_putu(allocs);
    sh_puts(" failed_allocs=");
    sh_putu(fails);
    sh_puts("\n");
}

#ifdef FORCE_FAIL
/* Failure-path check (make baremetal-negative): one deliberately false CHECK
 * must turn the run into BAREMETAL FAIL with QEMU exit code 1. */
static void t_force_fail(void)
{
    CHECK(TLSF_ALIGN == 0); /* deliberately false */
}
#endif

#define RUN(t)                                                             \
    do {                                                                   \
        unsigned long before = g_fail;                                     \
        sh_puts("[ RUN  ] " #t "\n");                                      \
        t();                                                               \
        sh_puts(g_fail == before ? "[  OK  ] " #t "\n"                     \
                                 : "[ FAIL ] " #t "\n");                   \
    } while (0)

int main(void)
{
    sh_puts("TLSF bare-metal test, Cortex-M3 (lm3s6965evb), TLSF_ALIGN=");
    sh_putu(TLSF_ALIGN);
    sh_puts(" HDR=");
    sh_putu(TLSF_HDR);
    sh_puts(" MIN_BLOCK=");
    sh_putu(TLSF_MIN_BLOCK);
    sh_puts(" CTRL=");
    sh_putu(TLSF_CTRL_SIZE);
    sh_puts(" sizeof(void*)=");
    sh_putu(sizeof(void *));
    sh_puts("\n");
    RUN(t_init);
    RUN(t_alignment);
    RUN(t_split_coalesce);
    RUN(t_exhaustion);
    RUN(t_independent);
    RUN(t_fit_invariant);
    RUN(t_stress);
#ifdef FORCE_FAIL
    RUN(t_force_fail);
#endif
    sh_putu(g_checks);
    sh_puts(" checks, ");
    sh_putu(g_fail);
    sh_puts(" failures\n");
    sh_puts(g_fail ? "BAREMETAL FAIL\n" : "BAREMETAL PASS\n");
    return g_fail ? 1 : 0;
}
