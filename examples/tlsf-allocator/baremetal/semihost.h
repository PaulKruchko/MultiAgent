/* semihost.h - minimal ARM semihosting console and exit (no libc). */
#ifndef SEMIHOST_H
#define SEMIHOST_H

void sh_puts(const char *s);          /* SYS_WRITE0 */
void sh_putu(unsigned long v);        /* decimal */
void sh_puthex(unsigned long v);      /* 0x... */
void sh_exit(int ok) __attribute__((noreturn)); /* SYS_EXIT */

#endif
