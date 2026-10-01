/*
 * host_sizes.c - prints the layout sizes of the production build (no
 * TLSF_STATS) for the configured TLSF_ALIGN_LOG2 (Makefile target
 * 'host-sizes').  The TLSF_STATS test builds add a counter block to the
 * control structure, so their printed CTRL is larger.
 */
#include <stdio.h>

#include "tlsf_internal.h"

int main(void)
{
    printf("host production build: TLSF_ALIGN=%u HDR=%u MIN_BLOCK=%u CTRL=%u "
           "sizeof(void*)=%u\n",
           (unsigned)TLSF_ALIGN, (unsigned)TLSF_HDR, (unsigned)TLSF_MIN_BLOCK,
           (unsigned)TLSF_CTRL_SIZE, (unsigned)sizeof(void *));
    return 0;
}
