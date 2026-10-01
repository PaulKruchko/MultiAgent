/* startup.c - Cortex-M3 vector table and reset handler for lm3s6965evb. */
#include <stddef.h>
#include <stdint.h>
#include "semihost.h"

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
int main(void);

/* The compiler may emit calls to these for struct copies/initialisers even
 * with -ffreestanding; keep it from turning their loops back into calls. */
__attribute__((optimize("no-tree-loop-distribute-patterns")))
void *memset(void *d, int c, size_t n)
{
    unsigned char *p = d;
    while (n--)
        *p++ = (unsigned char)c;
    return d;
}

__attribute__((optimize("no-tree-loop-distribute-patterns")))
void *memcpy(void *d, const void *s, size_t n)
{
    unsigned char *p = d;
    const unsigned char *q = s;
    while (n--)
        *p++ = *q++;
    return d;
}

__attribute__((optimize("no-tree-loop-distribute-patterns")))
void Reset_Handler(void)
{
    uint32_t *src = &_sidata, *dst = &_sdata;
    while (dst < &_edata)
        *dst++ = *src++;
    for (dst = &_sbss; dst < &_ebss;)
        *dst++ = 0;
    sh_exit(main() == 0);
}

static void fault(const char *what)
{
    sh_puts(what);
    sh_puts("\nBAREMETAL FAIL (fault)\n");
    sh_exit(0);
}

void NMI_Handler(void) { fault("NMI"); }
void HardFault_Handler(void) { fault("HardFault"); }
void MemManage_Handler(void) { fault("MemManage"); }
void BusFault_Handler(void) { fault("BusFault"); }
void UsageFault_Handler(void) { fault("UsageFault"); }
void Default_Handler(void) { fault("unexpected interrupt"); }

__attribute__((section(".isr_vector"), used))
static void (*const vectors[])(void) = {
    __extension__(void (*)(void))(&_estack), /* initial MSP */
    Reset_Handler,
    NMI_Handler,
    HardFault_Handler,
    MemManage_Handler,
    BusFault_Handler,
    UsageFault_Handler,
    0, 0, 0, 0,
    Default_Handler, /* SVCall */
    Default_Handler, /* DebugMon */
    0,
    Default_Handler, /* PendSV */
    Default_Handler, /* SysTick */
};
