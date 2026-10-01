## Summary

Implement TLSF with a 32-bit first level and 4-bit second level.

## Options Considered

1. TLSF. 2. Buddy allocator. 3. Fixed pools.

## Chosen Strategy

TLSF: O(1) worst case, and it fits in 2 KB.

## Execution Brief

Write `alloc.c`/`alloc.h`, a POSIX test harness, a FreeRTOS POSIX-port demo and a QEMU semihosting test.

## Acceptance Criteria

- AC-1 [hard]: All tests pass on POSIX, FreeRTOS and QEMU.
- AC-2 [hard]: `.text` < 2048 bytes on Cortex-M3 at `-Os`.

## Risks

- `clz` availability on Cortex-M0 (out of scope).
