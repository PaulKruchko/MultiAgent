/*
 * FreeRTOS POSIX-simulator test: NUM_WORKERS tasks share ONE TLSF heap whose
 * lock hooks take/give a FreeRTOS mutex.  Each worker allocates random sizes,
 * writes a task-specific pattern, re-verifies it before freeing and finally
 * drains its allocations.  A higher-priority controller task periodically
 * inspects the heap under the same mutex, waits for all workers, reports
 * the result and terminates the process itself (no vTaskEndScheduler).
 */
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"
#include "event_groups.h"

#include "tlsf.h"
#include "tlsf_check.h"

#define NUM_WORKERS 4
#define ITERATIONS 50000
#define SLOTS 48
#define HEAP_BYTES (64 * 1024)
#define TIMEOUT_MS 90000

static uint64_t heap_mem[HEAP_BYTES / 8];
static tlsf_t *g_heap;
static SemaphoreHandle_t g_mutex;
static EventGroupHandle_t g_done;

/* Instrumentation of the lock hooks. */
static volatile int g_inside;           /* tasks currently inside the heap */
static volatile unsigned long g_overlaps; /* entries while another was inside */
static volatile unsigned long g_lock_calls, g_unlock_calls, g_contended;

/* NEGATIVE_CONTROL builds keep the instrumentation but drop the mutex, to
 * demonstrate that the test detects unsynchronised heap access. */
static void heap_lock(void *ctx)
{
    SemaphoreHandle_t m = ctx;
#ifdef NEGATIVE_CONTROL
    (void)m;
#else
    if (xSemaphoreTake(m, 0) != pdTRUE) {
        g_contended++; /* only modified by the task that then owns the lock */
        xSemaphoreTake(m, portMAX_DELAY);
    }
#endif
    if (__atomic_add_fetch(&g_inside, 1, __ATOMIC_SEQ_CST) != 1)
        g_overlaps++;
    g_lock_calls++;
}

#ifdef NEGATIVE_CONTROL
/* Crash attribution for the negative control.  Corruption requires two tasks
 * inside the allocator at once, and the second entrant increments g_overlaps
 * in heap_lock() before touching heap state, so by the time a race-induced
 * crash happens the counter is already non-zero.  The handler reports the
 * counters with async-signal-safe write() only, then _exit()s. */
static size_t fmt_ulong(char *buf, unsigned long v)
{
    char tmp[24];
    size_t n = 0, i;
    do {
        tmp[n++] = (char)('0' + v % 10);
        v /= 10;
    } while (v);
    for (i = 0; i < n; i++)
        buf[i] = tmp[n - 1 - i];
    return n;
}

static size_t put_str(char *buf, const char *s)
{
    size_t n = 0;
    while (s[n]) {
        buf[n] = s[n];
        n++;
    }
    return n;
}

static void crash_handler(int sig)
{
    char buf[160];
    size_t n = 0;
    n += put_str(buf + n, "NEGATIVE CONTROL: caught signal ");
    n += fmt_ulong(buf + n, (unsigned long)sig);
    n += put_str(buf + n, ", lock hooks: lock=");
    n += fmt_ulong(buf + n, g_lock_calls);
    n += put_str(buf + n, " unlock=");
    n += fmt_ulong(buf + n, g_unlock_calls);
    n += put_str(buf + n, " overlaps=");
    n += fmt_ulong(buf + n, g_overlaps);
    n += put_str(buf + n, "\n");
    if (write(STDOUT_FILENO, buf, n) < 0) {
        /* nothing else is safe to do here */
    }
    _exit(128 + sig);
}
#endif

static void heap_unlock(void *ctx)
{
    g_unlock_calls++;
    __atomic_sub_fetch(&g_inside, 1, __ATOMIC_SEQ_CST);
#ifdef NEGATIVE_CONTROL
    (void)ctx;
#else
    xSemaphoreGive((SemaphoreHandle_t)ctx);
#endif
}

typedef struct worker {
    unsigned id;
    uint32_t seed;
    unsigned long allocs, frees, failed, errors, max_live_bytes;
    struct { uint8_t *p; uint32_t n, tag; } slot[SLOTS];
} worker;

static worker g_workers[NUM_WORKERS];

static uint32_t lcg(uint32_t *s)
{
    *s = *s * 1664525u + 1013904223u;
    return *s >> 8;
}

static void pat_fill(uint8_t *p, uint32_t n, uint32_t tag)
{
    uint32_t i;
    for (i = 0; i < n; i++)
        p[i] = (uint8_t)(tag ^ (i * 29u) ^ (i >> 7));
}

static int pat_ok(const uint8_t *p, uint32_t n, uint32_t tag)
{
    uint32_t i;
    for (i = 0; i < n; i++)
        if (p[i] != (uint8_t)(tag ^ (i * 29u) ^ (i >> 7)))
            return 0;
    return 1;
}

static void worker_task(void *arg)
{
    worker *w = arg;
    unsigned long it;
    unsigned long live = 0;
    uint32_t tag = w->id << 24;
    int k;

    for (it = 0; it < ITERATIONS; it++) {
        unsigned s = lcg(&w->seed) % SLOTS;
        if (w->slot[s].p) {
            if (!pat_ok(w->slot[s].p, w->slot[s].n, w->slot[s].tag))
                w->errors++;
            tlsf_free(g_heap, w->slot[s].p);
            live -= w->slot[s].n;
            w->slot[s].p = NULL;
            w->frees++;
        } else {
            uint32_t r = lcg(&w->seed) % 100;
            uint32_t n = r < 70 ? 1 + lcg(&w->seed) % 128
                       : r < 97 ? 129 + lcg(&w->seed) % 1024
                                : 1153 + lcg(&w->seed) % 4096;
            uint8_t *p = tlsf_malloc(g_heap, n);
            if (p) {
                if (((uintptr_t)p & (TLSF_ALIGN - 1)) ||
                    p < (uint8_t *)heap_mem ||
                    p + n > (uint8_t *)heap_mem + sizeof heap_mem)
                    w->errors++;
                w->slot[s].p = p;
                w->slot[s].n = n;
                w->slot[s].tag = ++tag;
                pat_fill(p, n, tag);
                live += n;
                if (live > w->max_live_bytes)
                    w->max_live_bytes = live;
                w->allocs++;
            } else {
                w->failed++;
            }
        }
        if ((it & 255) == 0)
            taskYIELD();
    }
    for (k = 0; k < SLOTS; k++)
        if (w->slot[k].p) {
            if (!pat_ok(w->slot[k].p, w->slot[k].n, w->slot[k].tag))
                w->errors++;
            tlsf_free(g_heap, w->slot[k].p);
            w->slot[k].p = NULL;
            w->frees++;
        }
    xEventGroupSetBits(g_done, 1u << w->id);
    vTaskSuspend(NULL);
}

static void finish(int ok)
{
    printf("%s\n", ok ? "FREERTOS TEST PASS" : "FREERTOS TEST FAIL");
    fflush(stdout);
    _exit(ok ? 0 : 1);
}

static void controller_task(void *arg)
{
    const EventBits_t all = (1u << NUM_WORKERS) - 1;
    TickType_t start = xTaskGetTickCount();
    unsigned long checks = 0, check_errors = 0;
    tlsf_report r0, r;
    int ok = 1, e;
    unsigned k;
    (void)arg;

    unsigned long reported_overlaps = 0;
    tlsf_check(g_heap, heap_mem, sizeof heap_mem, &r0);
    for (;;) {
        EventBits_t b = xEventGroupWaitBits(g_done, all, pdFALSE, pdTRUE,
                                            pdMS_TO_TICKS(20));
        unsigned long ov = g_overlaps;
        if (ov != reported_overlaps) { /* report concurrent entry at once */
            printf("controller: overlapping allocator entries overlaps=%lu\n",
                   ov);
            fflush(stdout);
            reported_overlaps = ov;
        }
        if ((b & all) == all)
            break;
        /* concurrent consistency snapshot, serialized by the heap mutex */
        xSemaphoreTake(g_mutex, portMAX_DELAY);
        e = tlsf_check(g_heap, heap_mem, sizeof heap_mem, &r);
        xSemaphoreGive(g_mutex);
        checks++;
        if (e)
            check_errors++;
        if (xTaskGetTickCount() - start > pdMS_TO_TICKS(TIMEOUT_MS)) {
            printf("TIMEOUT waiting for workers (bits=0x%lx)\n",
                   (unsigned long)b);
            finish(0);
        }
    }
    e = tlsf_check(g_heap, heap_mem, sizeof heap_mem, &r);
    printf("kernel %s, port %s, %d workers x %d iterations, heap %d bytes, "
           "elapsed %lu ticks\n",
           tskKERNEL_VERSION_NUMBER, "ThirdParty/GCC/Posix", NUM_WORKERS,
           ITERATIONS, HEAP_BYTES,
           (unsigned long)(xTaskGetTickCount() - start));
    for (k = 0; k < NUM_WORKERS; k++) {
        worker *w = &g_workers[k];
        printf("worker %u: allocs=%lu frees=%lu failed_allocs=%lu "
               "max_live_bytes=%lu errors=%lu\n",
               k, w->allocs, w->frees, w->failed, w->max_live_bytes,
               w->errors);
        if (w->errors || w->allocs != w->frees || w->allocs == 0)
            ok = 0;
    }
    printf("lock hooks: lock=%lu unlock=%lu contended=%lu overlaps=%lu\n",
           g_lock_calls, g_unlock_calls, g_contended, g_overlaps);
    printf("concurrent heap inspections: %lu, errors: %lu\n", checks,
           check_errors);
    printf("final heap: inspector=%d free_blocks=%lu used_blocks=%lu "
           "free_payload=%lu (initial %lu)\n",
           e, (unsigned long)r.free_blocks, (unsigned long)r.used_blocks,
           (unsigned long)r.free_payload, (unsigned long)r0.free_payload);
    if (g_overlaps || g_lock_calls != g_unlock_calls || check_errors || e ||
        r.free_blocks != 1 || r.used_blocks != 0 ||
        r.free_payload != r0.free_payload)
        ok = 0;
    finish(ok);
}

int main(void)
{
    unsigned k;
    g_heap = tlsf_init(heap_mem, sizeof heap_mem);
    g_mutex = xSemaphoreCreateMutex();
    g_done = xEventGroupCreate();
    if (!g_heap || !g_mutex || !g_done) {
        printf("setup failed\n");
        return 1;
    }
    tlsf_set_lock(g_heap, heap_lock, heap_unlock, g_mutex);
#ifdef NEGATIVE_CONTROL
    {
        struct sigaction sa;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sa.sa_handler = crash_handler;
        sigaction(SIGSEGV, &sa, NULL);
        sigaction(SIGABRT, &sa, NULL);
        sigaction(SIGBUS, &sa, NULL);
    }
#endif
    for (k = 0; k < NUM_WORKERS; k++) {
        char name[8] = {'w', 'o', 'r', 'k', (char)('0' + k), 0};
        g_workers[k].id = k;
        g_workers[k].seed = 0x1234u + 7919u * k;
        xTaskCreate(worker_task, name, configMINIMAL_STACK_SIZE * 4,
                    &g_workers[k], tskIDLE_PRIORITY + 1, NULL);
    }
    xTaskCreate(controller_task, "ctrl", configMINIMAL_STACK_SIZE * 4, NULL,
                tskIDLE_PRIORITY + 2, NULL);
    vTaskStartScheduler();
    printf("scheduler returned unexpectedly\n");
    return 1;
}
