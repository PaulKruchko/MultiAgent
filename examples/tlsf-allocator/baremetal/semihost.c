/* semihost.c - ARM semihosting via BKPT 0xAB (Thumb, ARMv7-M). */
#include "semihost.h"

#define SYS_WRITE0 0x04
#define SYS_EXIT 0x18
#define ADP_Stopped_ApplicationExit 0x20026 /* QEMU exits with status 0 */
#define ADP_Stopped_RunTimeErrorUnknown 0x20023 /* QEMU exits with status 1 */

static int semihost_call(int op, void *arg)
{
    register int r0 __asm__("r0") = op;
    register void *r1 __asm__("r1") = arg;
    __asm__ volatile("bkpt 0xab" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}

void sh_puts(const char *s)
{
    semihost_call(SYS_WRITE0, (void *)s);
}

void sh_putu(unsigned long v)
{
    char buf[12];
    int i = 11;
    buf[i] = 0;
    do {
        buf[--i] = (char)('0' + v % 10);
        v /= 10;
    } while (v);
    sh_puts(&buf[i]);
}

void sh_puthex(unsigned long v)
{
    char buf[11];
    int i;
    buf[0] = '0';
    buf[1] = 'x';
    for (i = 0; i < 8; i++)
        buf[2 + i] = "0123456789abcdef"[(v >> (28 - 4 * i)) & 15];
    buf[10] = 0;
    sh_puts(buf);
}

void sh_exit(int ok)
{
    /* On AArch32 the SYS_EXIT parameter is the reason code itself. */
    semihost_call(SYS_EXIT, (void *)(ok ? ADP_Stopped_ApplicationExit
                                        : ADP_Stopped_RunTimeErrorUnknown));
    for (;;)
        ;
}
