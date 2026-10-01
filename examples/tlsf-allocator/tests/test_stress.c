/*
 * test_stress.c - deterministic seeded randomized stress test and
 * fragmentation workloads for the TLSF allocator (POSIX host).
 *
 * Usage: test_stress [ops] [outdir]
 *   ops     operations per integrity-stress seed (default 200000)
 *   outdir  directory for CSV snapshots (default ".")
 *
 * Part A: for each seed, random alloc/free over 512 slots on a 64 KiB heap.
 *   Live allocations are tracked independently; every payload carries a
 *   byte pattern verified before free; the heap inspector runs after every
 *   operation; the analytic largest-allocatable size L is cross-checked
 *   every 997 operations: malloc(L) must succeed, and after freeing it (so
 *   the heap is back in its original state) malloc(L+1) must fail.
 * Part B: fragmentation workloads, snapshot every 500 ops to CSV.  Two
 *   metrics: ext_frag = 1 - L/F (includes good-fit lookup rounding) and
 *   phys_frag = 1 - largest_free/F (physical external fragmentation).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tlsf.h"
#include "tlsf_check.h"
#include "tlsf_internal.h"
#include "opcount.h"

static uint64_t g_heap[(1u << 18) / 8]; /* 256 KiB backing store */

typedef struct rng { uint64_t s; } rng;
static uint32_t rnd(rng *r) /* xorshift64* */
{
    r->s ^= r->s >> 12;
    r->s ^= r->s << 25;
    r->s ^= r->s >> 27;
    return (uint32_t)((r->s * 2685821657736338717ull) >> 32);
}
static uint32_t rnd_range(rng *r, uint32_t lo, uint32_t hi)
{
    return lo + rnd(r) % (hi - lo + 1);
}

typedef struct slot {
    unsigned char *p;
    size_t n;
    uint32_t tag;
} slot;

static void pat_fill(unsigned char *p, size_t n, uint32_t tag)
{
    size_t i;
    for (i = 0; i < n; i++)
        p[i] = (unsigned char)(tag + i * 0x9Du + (i >> 8));
}
static int pat_ok(const unsigned char *p, size_t n, uint32_t tag)
{
    size_t i;
    for (i = 0; i < n; i++)
        if (p[i] != (unsigned char)(tag + i * 0x9Du + (i >> 8)))
            return 0;
    return 1;
}

static int g_errors;
#define FAILF(...)                                                         \
    do {                                                                   \
        g_errors++;                                                        \
        printf("ERROR: " __VA_ARGS__);                                     \
        printf("\n");                                                      \
    } while (0)

/* ---------------- Part A: integrity stress ---------------- */
#define NSLOT 512
static slot g_slots[NSLOT];

static uint32_t size_mixed(rng *r)
{
    uint32_t c = rnd(r) % 100;
    if (c < 60) return rnd_range(r, 1, 64);
    if (c < 90) return rnd_range(r, 65, 1024);
    if (c < 99) return rnd_range(r, 1025, 8192);
    return rnd_range(r, 8193, 32768);
}

static void stress_seed(uint64_t seed, unsigned long ops)
{
    const size_t hb = 65536;
    unsigned char *mem = (unsigned char *)g_heap + 3; /* misaligned start */
    tlsf_t *h = tlsf_init(mem, hb);
    rng r;
    unsigned long i, allocs = 0, frees = 0, fails = 0, lchecks = 0;
    size_t live = 0, max_live = 0;
    tlsf_report rep, rep0;
    uint32_t tag = 1;
    int e;

    r.s = seed;
    memset(g_slots, 0, sizeof g_slots);
    if (!h || tlsf_check(h, mem, hb, &rep0)) {
        FAILF("seed %llu: init failed", (unsigned long long)seed);
        return;
    }
    for (i = 0; i < ops && g_errors < 10; i++) {
        slot *s = &g_slots[rnd(&r) % NSLOT];
        if (!s->p) {
            size_t n = size_mixed(&r);
            unsigned char *p = oc_malloc(h, n);
            if (p) {
                if (((uintptr_t)p & (TLSF_ALIGN - 1)) || p < mem ||
                    p + n > mem + hb)
                    FAILF("seed %llu op %lu: bad pointer",
                          (unsigned long long)seed, i);
                s->p = p;
                s->n = n;
                s->tag = tag++;
                pat_fill(p, n, s->tag);
                allocs++;
                live++;
                if (live > max_live)
                    max_live = live;
            } else {
                fails++;
            }
        } else {
            if (!pat_ok(s->p, s->n, s->tag))
                FAILF("seed %llu op %lu: payload corrupted",
                      (unsigned long long)seed, i);
            oc_free(h, s->p);
            s->p = NULL;
            frees++;
            live--;
        }
        if ((e = tlsf_check(h, mem, hb, &rep)) != 0)
            FAILF("seed %llu op %lu: inspector error %d",
                  (unsigned long long)seed, i, e);
        if (rep.used_blocks != live)
            FAILF("seed %llu op %lu: used blocks %lu != live %lu",
                  (unsigned long long)seed, i, (unsigned long)rep.used_blocks,
                  (unsigned long)live);
        if (i % 997 == 0 && rep.largest_alloc) {
            void *p = oc_malloc(h, rep.largest_alloc);
            void *q;
            lchecks++;
            if (!p)
                FAILF("seed %llu op %lu: malloc(L=%lu) failed",
                      (unsigned long long)seed, i,
                      (unsigned long)rep.largest_alloc);
            oc_free(h, p); /* restore the original heap before trying L+1 */
            q = oc_malloc(h, rep.largest_alloc + 1);
            if (q)
                FAILF("seed %llu op %lu: malloc(L+1) succeeded",
                      (unsigned long long)seed, i);
            oc_free(h, q);
        }
        if (i % 10000 == 9999) { /* verify every live payload */
            int k;
            for (k = 0; k < NSLOT; k++)
                if (g_slots[k].p &&
                    !pat_ok(g_slots[k].p, g_slots[k].n, g_slots[k].tag))
                    FAILF("seed %llu op %lu: live payload %d corrupted",
                          (unsigned long long)seed, i, k);
        }
    }
    /* drain */
    {
        int k;
        for (k = 0; k < NSLOT; k++)
            if (g_slots[k].p) {
                if (!pat_ok(g_slots[k].p, g_slots[k].n, g_slots[k].tag))
                    FAILF("seed %llu drain: payload %d corrupted",
                          (unsigned long long)seed, k);
                oc_free(h, g_slots[k].p);
                g_slots[k].p = NULL;
            }
    }
    if ((e = tlsf_check(h, mem, hb, &rep)) != 0 || rep.free_blocks != 1 ||
        rep.free_payload != rep0.free_payload)
        FAILF("seed %llu: heap not restored after drain (err %d)",
              (unsigned long long)seed, e);
    printf("stress seed=0x%llx heap=%lu ops=%lu allocs=%lu frees=%lu "
           "failed_allocs=%lu max_live=%lu L_crosschecks=%lu -> %s\n",
           (unsigned long long)seed, (unsigned long)hb, ops, allocs, frees,
           fails, (unsigned long)max_live, lchecks,
           g_errors ? "ERRORS" : "ok");
}

/* ---------------- Part B: fragmentation workloads ---------------- */
typedef struct workload {
    const char *name;
    const char *desc;
    uint64_t seed;
    size_t heap;
    unsigned nslots;
    unsigned long ops;
    int kind;
} workload;

static uint32_t wl_size(rng *r, int kind)
{
    switch (kind) {
    case 0: return rnd_range(r, 8, 256);
    case 1: {
        uint32_t c = rnd(r) % 100;
        if (c < 70) return rnd_range(r, 16, 128);
        if (c < 95) return rnd_range(r, 129, 2048);
        return rnd_range(r, 2049, 16384);
    }
    case 2: return 1u << rnd_range(r, 3, 12);
    default: return rnd_range(r, 16, 512);
    }
}

static slot g_wslots[4096];

static void run_workload(const workload *w, const char *outdir)
{
    unsigned char *mem = (unsigned char *)g_heap;
    tlsf_t *h = tlsf_init(mem, w->heap);
    rng r;
    unsigned long i, fails = 0, allocs = 0;
    size_t live_req = 0;
    double frag_sum = 0, frag_max = 0, phys_sum = 0, phys_max = 0;
    unsigned long frag_n = 0;
    char path[512];
    FILE *f;
    tlsf_report rep;
    uint32_t tag = 1;

    r.s = w->seed;
    memset(g_wslots, 0, sizeof g_wslots);
    snprintf(path, sizeof path, "%s/frag_%s.csv", outdir, w->name);
    f = fopen(path, "w");
    if (!f) {
        FAILF("cannot write %s", path);
        return;
    }
    fprintf(f, "op,live_requested,used_footprint,free_payload_F,"
               "largest_alloc_L,largest_free,ext_frag,phys_frag,"
               "failed_allocs\n");
    for (i = 1; i <= w->ops; i++) {
        slot *s = &g_wslots[rnd(&r) % w->nslots];
        if (!s->p) {
            size_t n = wl_size(&r, w->kind);
            unsigned char *p = oc_malloc(h, n);
            if (p) {
                s->p = p;
                s->n = n;
                s->tag = tag++;
                pat_fill(p, n < 16 ? n : 16, s->tag);
                live_req += n;
                allocs++;
            } else {
                fails++;
            }
        } else {
            if (!pat_ok(s->p, s->n < 16 ? s->n : 16, s->tag))
                FAILF("%s op %lu: payload corrupted", w->name, i);
            oc_free(h, s->p);
            live_req -= s->n;
            s->p = NULL;
        }
        if (i % 500 == 0) {
            int e = tlsf_check(h, mem, w->heap, &rep);
            double fr = rep.free_payload
                            ? 1.0 - (double)rep.largest_alloc /
                                        (double)rep.free_payload
                            : 0.0;
            double ph = rep.free_payload
                            ? 1.0 - (double)rep.largest_free /
                                        (double)rep.free_payload
                            : 0.0;
            if (e)
                FAILF("%s op %lu: inspector error %d", w->name, i, e);
            fprintf(f, "%lu,%lu,%lu,%lu,%lu,%lu,%.5f,%.5f,%lu\n", i,
                    (unsigned long)live_req, (unsigned long)rep.used_footprint,
                    (unsigned long)rep.free_payload,
                    (unsigned long)rep.largest_alloc,
                    (unsigned long)rep.largest_free, fr, ph, fails);
            if (i > w->ops / 10) { /* skip warm-up */
                frag_sum += fr;
                phys_sum += ph;
                frag_n++;
                if (fr > frag_max)
                    frag_max = fr;
                if (ph > phys_max)
                    phys_max = ph;
            }
        }
    }
    fclose(f);
    tlsf_check(h, mem, w->heap, &rep);
    printf("| %s | 0x%llx | %lu | %u | %lu | %lu | %lu | %lu | %lu | %lu | "
           "%lu | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %lu/%lu |\n",
           w->name, (unsigned long long)w->seed, (unsigned long)w->heap,
           w->nslots, w->ops, (unsigned long)live_req,
           (unsigned long)rep.used_footprint, (unsigned long)rep.free_payload,
           (unsigned long)rep.largest_alloc, (unsigned long)rep.largest_free,
           (unsigned long)(rep.used_footprint - live_req),
           rep.free_payload ? 1.0 - (double)rep.largest_alloc /
                                        (double)rep.free_payload
                            : 0.0,
           frag_n ? frag_sum / frag_n : 0.0, frag_max,
           rep.free_payload ? 1.0 - (double)rep.largest_free /
                                        (double)rep.free_payload
                            : 0.0,
           frag_n ? phys_sum / frag_n : 0.0, phys_max, fails,
           fails + allocs);
    {
        int k;
        for (k = 0; k < (int)w->nslots; k++)
            if (g_wslots[k].p)
                oc_free(h, g_wslots[k].p);
        if (tlsf_check(h, mem, w->heap, &rep) || rep.free_blocks != 1)
            FAILF("%s: heap not restored after drain", w->name);
    }
}

int main(int argc, char **argv)
{
    unsigned long ops = argc > 1 ? strtoul(argv[1], NULL, 10) : 200000;
    const char *outdir = argc > 2 ? argv[2] : ".";
    static const uint64_t seeds[] = {0xC0FFEEull, 0x5EED0001ull, 0xDEADBEEFull};
    static const workload wls[] = {
        {"small_uniform", "sizes U[8,256]", 0x1001, 262144, 1024, 100000, 0},
        {"mixed", "70% U[16,128], 25% U[129,2048], 5% U[2049,16384]",
         0x1002, 262144, 256, 100000, 1},
        {"pow2", "2^k, k U[3,12]", 0x1003, 262144, 256, 100000, 2},
        {"pressure", "U[16,512], 2048 slots on 128 KiB (exhaustion)",
         0x1004, 131072, 2048, 100000, 3},
    };
    unsigned k;
    int b;

    printf("TLSF stress: TLSF_ALIGN=%u HDR=%u MIN_BLOCK=%u sizeof(void*)=%u\n",
           (unsigned)TLSF_ALIGN, (unsigned)TLSF_HDR, (unsigned)TLSF_MIN_BLOCK,
           (unsigned)sizeof(void *));
    printf("\n== Part A: seeded randomized integrity stress ==\n");
    for (k = 0; k < sizeof seeds / sizeof seeds[0]; k++)
        stress_seed(seeds[k], ops);

    printf("\n== Part B: fragmentation workloads (ext_frag = 1 - L/F, "
           "phys_frag = 1 - largest_free/F) ==\n");
    for (k = 0; k < sizeof wls / sizeof wls[0]; k++)
        printf("  %s: %s\n", wls[k].name, wls[k].desc);
    printf("| workload | seed | heap | slots | ops | live requested | used "
           "footprint | free payload F | largest alloc L | largest free "
           "block | header+rounding overhead | final ext_frag | mean ext_frag "
           "| max ext_frag | final phys_frag | mean phys_frag | max phys_frag "
           "| failed/attempted allocs |\n");
    printf("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
           "---|---|\n");
    for (k = 0; k < sizeof wls / sizeof wls[0]; k++)
        run_workload(&wls[k], outdir);

    printf("\nOperation-count maxima over the whole stress run:\n");
    printf("%-12s %9s %7s %7s %7s %9s %6s %9s\n", "op", "calls", "bitmap",
           "insert", "remove", "neighbor", "split", "coalesce");
    for (b = 0; b < OP_KINDS; b++) {
        tlsf_stats *m = &g_opmax[b].max;
        printf("%-12s %9lu %7u %7u %7u %9u %6u %9u\n", opmax_names[b],
               g_opmax[b].calls, m->bitmap_checks, m->list_inserts,
               m->list_removes, m->neighbor_checks, m->splits, m->coalesces);
    }
    {
        tlsf_stats *a = &g_opmax[OP_MALLOC_OK].max,
                   *fl = &g_opmax[OP_MALLOC_FAIL].max,
                   *fr = &g_opmax[OP_FREE].max;
        if (!(a->bitmap_checks <= 3 && a->list_inserts <= 1 &&
              a->list_removes <= 1 && a->neighbor_checks <= 1 &&
              a->splits <= 1 && a->coalesces == 0 && fl->bitmap_checks <= 2 &&
              fl->list_removes == 0 && fl->list_inserts == 0 &&
              fr->list_inserts <= 1 && fr->list_removes <= 2 &&
              fr->neighbor_checks <= 2 && fr->coalesces <= 2))
            FAILF("operation counts exceed analytical bounds");
    }
    printf("\n%s (%d errors)\n", g_errors ? "STRESS FAILED" : "STRESS PASSED",
           g_errors);
    return g_errors ? 1 : 0;
}
