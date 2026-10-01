# Compact TLSF allocator for Cortex-M3, FreeRTOS and POSIX

A C99 Two-Level Segregated Fit (TLSF) allocator with 16 second-level
classes, no libc dependency and bounded work in every `malloc`/`free` path.
It runs over caller-supplied regions, supports any number of independent
heaps, has optional per-heap lock hooks, and its payload alignment is set at
compile time (default 8 bytes).

Every number below was produced by a command in the **Reproduction** section
and can be found in the log named next to it (all logs are in `logs/`).

| Requirement | Result | Evidence |
|---|---|---|
| Allocator code size, Cortex-M3 `-Os`, allocator object alone | **692 bytes `.text`** (< 2048); 0 data, 0 bss; no undefined symbols | `make size` → `logs/size.log` |
| No loops or recursion in `malloc`/`free` | Instruction-level CFG of the ARM object has no cycles in anything reachable from `tlsf_malloc`/`tlsf_free` | `make audit` → `logs/audit.log` |
| POSIX unit tests | 1,774,559 checks, 0 failures (align 8); also align 16 and 32 pass | `make unit`, `make unit-align` → `logs/unit.log`, `logs/unit_align.log` |
| Seeded random stress | 3 seeds × 200,000 ops, inspector after every op, 0 errors | `make stress` → `logs/stress.log` |
| ASan + UBSan | unit + 3 × 100,000-op stress, 0 errors (covers the harness, out-of-region accesses and UB; *not* overruns inside a heap, see Tests) | `make asan` → `logs/asan.log` |
| FreeRTOS POSIX simulator | 4 tasks × 50,000 iterations on one mutex-protected heap: PASS | `make freertos` → `logs/freertos.log` |
| FreeRTOS negative control (hooks without mutex) | fails with SIGSEGV (exit 139); the signal handler reports `overlaps=768` concurrent allocator entries, so the failure is attributed to missing locking | `make freertos-negative` → `logs/freertos_negative.log` |
| Bare-metal Cortex-M3, QEMU `lm3s6965evb` | 75,096 checks, `BAREMETAL PASS`, QEMU exit code 0 | `make baremetal` → `logs/baremetal.log` |
| Bare metal, `TLSF_ALIGN_LOG2=2` (4-byte alignment) | 85,257 checks, `BAREMETAL PASS`, QEMU exit code 0 | `make baremetal-a4` → `logs/baremetal_a4.log` |
| Bare-metal failure path (`-DFORCE_FAIL`) | one deliberately false check → `BAREMETAL FAIL`, QEMU exit code 1 | `make baremetal-negative` → `logs/baremetal_negative.log` |

## Layout

```
src/tlsf.h            public API and configuration macros
src/tlsf_internal.h   block/control layout (shared with the test inspector)
src/tlsf.c            the allocator (the only file that ships)
tests/tlsf_check.[ch] test-only heap inspector (freestanding; host/RTOS/bare metal)
tests/opcount.h       per-operation primitive counters (TLSF_STATS builds)
tests/test_unit.c     POSIX unit tests
tests/test_stress.c   seeded randomized stress + fragmentation workloads
tests/host_sizes.c    prints production (no TLSF_STATS) layout sizes on the host
freertos/             FreeRTOSConfig.h + multi-task shared-heap test (POSIX port)
baremetal/            vector table/startup, linker script, semihosting, QEMU test
tools/check_branches.py  loop/recursion audit of the ARM disassembly
tools/plot.py         plots from the stress logs
docs/*.png            plots
logs/                 output of every test run used here
```

## API

```c
#include "tlsf.h"
tlsf_t *tlsf_init(void *mem, size_t bytes);
void    tlsf_set_lock(tlsf_t *h, tlsf_lock_fn lock, tlsf_lock_fn unlock, void *ctx);
void   *tlsf_malloc(tlsf_t *h, size_t size);
void    tlsf_free(tlsf_t *h, void *ptr);
```

- **Initialisation and ownership.** `tlsf_init` formats a region owned by
  the caller and returns a handle to the control structure, which it places
  at the first aligned address *inside* that region. The region must stay
  valid and untouched by anything else while the heap is in use. There is no
  deinit: the caller stops using the handle and takes the memory back.
- **Region rules.** The start can be at any address; it is rounded up to
  `TLSF_ALIGN` and the end is rounded down. `tlsf_init` returns `NULL` for a
  `NULL` region, a region larger than `TLSF_MAX_REGION` (default 1 MiB), a
  region whose address range wraps, or one too small for the control
  structure plus one minimum block plus the end sentinel. Oversized regions
  are rejected, never truncated.
- **Multiple heaps.** All state lives in the heap instance and there are no
  globals. Heaps are independent (tested in `test_independent_heaps` and in
  bare-metal `t_independent`).
- **`tlsf_malloc`** returns a `TLSF_ALIGN`-aligned pointer. It returns
  `NULL`, leaving the heap unchanged, when `size == 0`,
  `size > TLSF_MAX_ALLOC`, or no free block exists in a class that
  guarantees a fit.
- **`tlsf_free(h, NULL)`** does nothing. Any other pointer must have come
  from `tlsf_malloc` on the **same** heap and must not already be freed. This
  is not checked; breaking it is undefined behaviour.
- **Locking.** Hooks are optional and per heap, and must be installed before
  the heap is shared. `lock(ctx)` runs exactly once before and `unlock(ctx)`
  exactly once after every `tlsf_malloc`/`tlsf_free` that touches heap state,
  including failed allocations. Calls rejected before any state is touched
  (`size == 0`, oversized requests, `free(NULL)`) do not call the hooks.
  Internal helpers never call the hooks. Hooks must not re-enter the same
  heap. Time spent waiting in `lock()` is outside the allocator's bound.
- **Configuration** (compile-time, the same in every translation unit;
  the limits are also documented in `src/tlsf.h`):
  - `TLSF_ALIGN_LOG2`: default 3. Limits are `2 ≤ TLSF_ALIGN_LOG2 ≤ 5` and
    `TLSF_ALIGN ≥ sizeof(void*)`, so 2 is only valid on 32-bit targets.
    Executed test coverage: 3 on the host and on Cortex-M3; 4 and 5 on the
    host; 2 on Cortex-M3 (`make baremetal-a4`).
  - `TLSF_MAX_REGION_LOG2`: default 20. Limits are
    `FL_SHIFT + 1 ≤ TLSF_MAX_REGION_LOG2 ≤ 31`, where
    `FL_SHIFT = 4 + TLSF_ALIGN_LOG2`, and it must be below the bit width
    of `size_t`.
  - `unsigned` must be 32 bits wide, because the bit scans compute
    `31 - clz((unsigned)x)`.
  - Invalid configurations fail to compile (C99 negative-array-size checks
    in `src/tlsf.c`).
  - Bit scans use `__builtin_clz`/`__builtin_ctz` (GCC/Clang). Other
    compilers can supply `TLSF_CLZ`/`TLSF_CTZ`.

Memory cost per heap and per block, for the production build (no
`TLSF_STATS`):

| Target | `TLSF_ALIGN` | block header | min block | control struct | Source |
|---|---|---|---|---|---|
| Cortex-M3 (32-bit) | 8 | 8 B | 16 B | 968 B | `logs/baremetal.log` |
| Cortex-M3 (32-bit) | 4 | 8 B | 16 B | 1036 B | `logs/baremetal_a4.log` |
| x86-64 host | 8 | 16 B | 32 B | 1880 B | `make host-sizes` → `logs/host_sizes.log` |
| x86-64 host | 16 / 32 | 16 / 32 B | 32 / 64 B | 1744 / 1632 B | `logs/host_sizes.log` |

The host unit and stress programs are `TLSF_STATS` builds. They add a
24-byte counter block before alignment, so the `CTRL` values they print are
larger (1904 / 1776 / 1664 B). The bare-metal images are production builds.

## Design

### Design comparison (why TLSF)

| Design | malloc worst case | free worst case | Why not chosen |
|---|---|---|---|
| Fixed-block pools | O(1) | O(1) | Arbitrary sizes need several pools, and capacity gets stranded in the wrong size class |
| Binary buddy | O(log(H/min)) split levels | O(log(H/min)) merge levels [3][4] | Cost grows with heap size, and power-of-two rounding can waste up to about 50% |
| Plain segregated free lists | depends on list length (a bin must be searched for a big-enough block) | O(1) | Search is not bounded |
| FreeRTOS `heap_4` (local copy) | O(free blocks): first-fit walk, `FreeRTOS-Kernel/portable/MemMang/heap_4.c:248` | O(free blocks): address-ordered insert, `heap_4.c:511` | Linear in the number of free blocks |
| **TLSF** [1][2] | **O(1)**: bitmap lookup + ≤1 split | **O(1)**: ≤2 neighbour merges | Chosen. Costs more metadata; code size was measured, see below |

Sources: [1] M. Masmano, I. Ripoll, A. Crespo, J. Real, *TLSF: a new dynamic
memory allocator for real-time systems*, ECRTS 2004. [2] M. Masmano,
I. Ripoll, P. Balbastre, A. Crespo, *A constant-time dynamic storage
allocator for real-time systems*, Real-Time Systems 40(2), 2008 (the
mapping and boundary-tag scheme). [3] K. C. Knowlton, *A fast storage
allocator*, Communications of the ACM 8(10), 1965 (the binary buddy
system). [4] D. E. Knuth, *The Art of Computer Programming*, Vol. 1,
§2.5 "Dynamic Storage Allocation" (buddy system and boundary tags). The
Arm Cortex-M3 TRM (DDI 0337) documents the `CLZ` instruction. The QEMU
semihosting documentation covers `SYS_WRITE0` (0x04) and `SYS_EXIT` (0x18)
with `ADP_Stopped_ApplicationExit`. Web access was disabled while this was
built, so these references were not re-checked here. The FreeRTOS `heap_4`
line references were checked against the local kernel copy; [1]–[4]
were cited from memory and not re-checked online. Every
performance and size claim in this README is a local measurement, not a
figure from these sources.

### Block layout

```
 block:  [prev_phys][size|PREV_FREE|FREE][ payload ........................ ]
          valid only   total block size    when FREE: next_free, prev_free
          if PREV_FREE (multiple of ALIGN)   overlay the payload
```

- Header `TLSF_HDR = ALIGN_UP(2 words)`: 8 B on Cortex-M3. Block sizes
  include the header and are multiples of `TLSF_ALIGN`, so the two low bits
  are free for flags.
- `prev_phys` is written into the successor's header only when a block
  becomes free, and read only when `PREV_FREE` is set.
- The region ends with a size-0 "used" sentinel header. The first block
  never has `PREV_FREE`, and the sentinel is never free, so coalescing cannot
  run past either end of the region.
- Invariant: no two physically adjacent blocks are both free, because frees
  coalesce immediately. So a free block's predecessor is always in use, and
  after a split the allocated head needs no flag bookkeeping.

### Class mapping

`SL = 16` classes per power of two (`TLSF_SL_LOG2 = 4`),
`FL_SHIFT = 4 + TLSF_ALIGN_LOG2`, `SMALL = 2^FL_SHIFT` (128 B at align 8).

- For `size < SMALL`: `fl = 0` and `sl = size / ALIGN`. These classes are
  exact.
- Otherwise, with `t = 31 - clz(size)`: `fl = t - FL_SHIFT + 1` and
  `sl = (size >> (t-4)) - 16`.
- With the default configuration there are 14 first-level rows, one
  `uint32_t` first-level bitmap, 14 second-level bitmaps and 14×16
  free-list heads.

**Fit guarantee.** `tlsf_malloc` works out the block size it needs,
`need = max(ALIGN_UP(size + HDR), MIN_BLOCK)`. For `need >= SMALL` it rounds
up to the next class boundary, `search = need + 2^(t-4) - 1`, and looks up
`search`. Every block in the selected class, or in any higher class, is then
at least `need` bytes, so no list is ever searched. `TLSF_MAX_ALLOC =
MAX_REGION - MAX_REGION/32 - 32` guarantees `search < MAX_REGION`, so `fl`
always indexes the table. Each block records its real size. A block is split
only when the remainder is at least `MIN_BLOCK`; otherwise the whole block
is handed out.

The test `test_fit_invariant` (unit) and `t_fit_invariant` (bare metal)
check this exhaustively. They build heaps whose *only* free block has size S
(558 values of S on the host, 127 on ARM) and try every request near S. A
request must succeed exactly when its rounded class ≤ class(S), and when it
succeeds it must return that block.

### Algorithms and bounded work

`src/tlsf.c` contains no loops or recursion in `tlsf_malloc`, `tlsf_free`,
`insert_free`, `remove_free`, `mapping` or `call_hook`. The only loops are
the two table-clearing loops in `tlsf_init`, which have compile-time bounds
of `FL_COUNT` and `FL_COUNT·16` iterations and run once per heap. These
loops use `volatile` stores so the compiler cannot turn them into a
`memset` call. Bit scans are single `CLZ` (or `RBIT`+`CLZ`) instructions on
ARMv7-M. The generated `tlsf_m3.s` contains 5 of them (`logs/size.log`).
The CFG audit in `logs/audit.log` confirms that the compiled ARM code has
no cycles.

Primitive-operation counts per call, derived from the source and counted by
the `STAT()` hooks in `TLSF_STATS` builds:

| Primitive | malloc (success) | malloc (failure) | free | Source lines (`src/tlsf.c`) |
|---|---|---|---|---|
| bitmap words examined by the search | ≤ 3 (SL row, FL map, SL row of the found FL) | ≤ 2 | 0 | 166–174 |
| free-list removals | 1 | 0 | ≤ 2 (prev + next neighbour) | 180, 213, 221 |
| free-list insertions | ≤ 1 (split remainder) | 0 | 1 | 190, 227 |
| physical-neighbour header inspections/updates | 1 (successor flag / `prev_phys`) | 0 | 2 (predecessor flag, successor header) | 181, 209, 218 |
| splits | ≤ 1 | 0 | 0 | 185 |
| coalesces | 0 | 0 | ≤ 2 | 212, 220 |
| lock/unlock hook calls | 1 + 1 | 1 + 1 | 1 + 1 | 165, 196, 207, 228 |

Each insertion or removal is a fixed sequence of pointer and bitmap updates
(`insert_free`: one head read, ≤ 3 link writes, 2 bitmap ORs; `remove_free`:
≤ 2 link writes, ≤ 2 bitmap clears). None of these counts depends on the
heap size, the number of blocks or the request size.

Observed maxima (both tables are identical and equal the bounds above):

| op | calls (unit) | calls (stress) | bitmap | insert | remove | neighbour | split | coalesce |
|---|---|---|---|---|---|---|---|---|
| malloc ok | 541,206 | 467,082 | 3 | 1 | 1 | 1 | 1 | 0 |
| malloc fail | 86,476 | 69,863 | 2 | 0 | 0 | 0 | 0 | 0 |
| free | 539,284 | 467,082 | 0 | 1 | 2 | 2 | 0 | 2 |

Sources: `logs/unit.log` (boundary tests) and `logs/stress.log` (random
workloads, including heaps close to exhaustion); plot in
`docs/opcounts.png`. These sampled maxima only show the counters reach
their bounds. The bound itself comes from the source structure above and
the loop audit, not from sampling.

## Code size (measured)

Command (from `make size`, output in `logs/size.log`):

```
arm-none-eabi-gcc -std=c99 -mcpu=cortex-m3 -mthumb -Os -ffreestanding \
    -Wall -Wextra -Wpedantic -Werror -c src/tlsf.c -o build/tlsf_m3.o
arm-none-eabi-size build/tlsf_m3.o
   text    data     bss     dec     hex filename
    692       0       0     692     2b4 build/tlsf_m3.o
```

Toolchain: arm-none-eabi-gcc 13.2.1 20231009, GNU size 2.42.

- `arm-none-eabi-size -A`: `.text 692`, `.data 0`, `.bss 0`.
- `arm-none-eabi-nm -u` prints nothing, so the object has no libc or libgcc
  helper references.
- Per-function sizes: `tlsf_malloc` 230, `tlsf_free` 126, `tlsf_init` 114,
  `remove_free` 86, `insert_free` 74, `mapping` 40, `call_hook` 10,
  `tlsf_set_lock` 10 bytes.
- Other alignments: `TLSF_ALIGN_LOG2=2` gives 704, `=4` gives 692, `=5`
  gives 696 bytes.
- These figures are for the production object only (no `TLSF_STATS`),
  excluding the test firmware. For comparison, the whole bare-metal test
  image is 9236 bytes of text (`logs/baremetal.log`).

## Tests

### POSIX unit tests (`tests/test_unit.c`, `make unit`, `make unit-align`)
- init validation: `NULL` region, too small, exactly minimal, misaligned
  start, over 1 MiB, `SIZE_MAX`, and a full 1 MiB region
- `malloc(0)`, `free(NULL)`, oversized requests, and `TLSF_MAX_ALLOC` on a
  1 MiB heap
- alignment for region offsets 0..2·ALIGN-1 and request sizes 1..299
- exact fit (a hole reused with no split), split thresholds, and
  physically adjacent splits
- coalescing with no free neighbour, previous only, next only, and both
  (checked through the inspector and the `coalesces` counter)
- exhaustion and recovery for sizes 1, 24, 100 and 1000, freeing in random
  and in reverse order; the heap returns to a single free block and the
  largest allocation works again
- independent heaps, including exhausting one heap while another keeps
  working
- requests at ±2·ALIGN around every class boundary of all 14×16 classes on
  a 1 MiB heap (7,227 requests), plus every size from 1 to 8192
- the fit invariant (above), and lock-hook call counts and nesting depth
- the inspector itself: a stray `sl_bitmap` bit 16 or bit 31 in any row
  must be reported as error 9 (`test_inspector_stray_bits`)
- after every test, observed operation counts are checked against the
  analytical bounds (`check_bounds`)

The **inspector** (`tests/tlsf_check.c`) walks the physical chain and
checks:
- alignment, size, minimum size and region bounds of every block
- the `PREV_FREE` flag and `prev_phys` pointer
- that no two adjacent blocks are free
- the free-list links of every free block

It also walks every free list and checks:
- class membership, using an *independent* mapping implementation
- back links
- that the first-level and second-level bitmaps agree with the list heads,
  and that no `sl_bitmap` bit at or above `TLSF_SL_COUNT` (16) is set,
  since `tlsf_malloc`'s bit scan could otherwise select a column past the
  row
- that the number of listed blocks equals the number of physical free
  blocks

It computes `L`, the largest request the good-fit lookup can satisfy, and
the stress test compares this against real allocations: `malloc(L)` must
succeed, it is freed so the heap is back in its original state, and then
`malloc(L+1)` must fail.

### Seeded stress (`tests/test_stress.c`, `make stress`, 200,000 ops per seed)
Each seed runs random alloc/free on 512 slots of a 64 KiB heap that starts
at a misaligned address. Request sizes are 60% 1–64 B, 30% 65–1024 B,
9% 1–8 KiB and 1% 8–32 KiB. Live allocations are tracked independently of
the allocator. Every payload is pattern-checked before it is freed, and all
live payloads are checked every 10,000 ops. The inspector runs after every
operation, and `L` is cross-checked every 997 ops (as above, `L+1` is
tried only after the `malloc(L)` block is freed). The heap is drained at
the end and must be a single free block again.

| seed | allocs | failed allocs | max live | L cross-checks | result |
|---|---|---|---|---|---|
| 0xC0FFEE | 94,026 | 12,190 | 281 | 201 | ok |
| 0x5EED0001 | 93,756 | 12,739 | 279 | 201 | ok |
| 0xDEADBEEF | 93,840 | 12,557 | 276 | 201 | ok |

The same unit and stress programs also passed under
`-fsanitize=address,undefined -fno-sanitize-recover=all`, with 100,000 ops
per seed (`logs/asan.log`). This run is weak evidence about the
allocator's *internal* memory safety. The heaps live in static arrays that
ASan does not poison, so it cannot see an overrun from one block into the
next or into a block header inside a heap. What the ASan/UBSan run does
cover is the test harness, accesses outside the heap arrays, and undefined
behaviour such as bad shifts, overflow and misaligned accesses. Evidence
for the allocator's internal consistency comes from the inspector, which
runs after every operation, and from the payload pattern checks.

### FreeRTOS (`freertos/`, `make freertos`)
The test builds against the supplied kernel in place:
- **Kernel:** `tskKERNEL_VERSION_NUMBER "V11.1.0+"`. `History.txt` lists
  changes up to V11.3.1.
- **Kernel sources:** `tasks.c queue.c list.c timers.c event_groups.c`
- **Port:** `portable/ThirdParty/GCC/Posix/port.c` with `utils/wait_for_event.c`
- **Kernel heap:** `heap_3.c` (for kernel objects only). The allocator under
  test uses its own 64 KiB static region.
- **Scheduling:** preemptive, time slicing on, 1 kHz tick.

Setup:
- Four worker tasks share one TLSF heap.
- The hooks take and give one FreeRTOS mutex. They also count "tasks inside
  the allocator", and any second entry is recorded as an overlap.
- Each worker runs 50,000 iterations over 48 slots: random sizes, a
  task-specific pattern, and a verify before every free.
- A higher-priority controller inspects the heap under the mutex every
  20 ms. It waits on an event group, with a 90 s timeout, until all workers
  have drained.
- The controller prints the result and ends the process with
  `fflush` + `_exit`, so the test does not depend on `vTaskEndScheduler`.

Observed (`logs/freertos.log`):
- 4 × ~25,006 allocations and matching frees, 0 errors.
- 200,087 lock calls and 200,087 unlock calls.
- 231 acquisitions found the mutex already held, meaning a task was
  preempted inside the allocator. There were 0 overlaps.
- 8 concurrent inspections, 0 errors.
- Final heap: 1 free block with the initial free payload (63,624 B), and
  `FREERTOS TEST PASS`.

**Negative control** (`make freertos-negative`). With `-DNEGATIVE_CONTROL`
the hooks keep their instrumentation but drop the mutex. The binary
installs a `SIGSEGV`/`SIGABRT`/`SIGBUS` handler that prints the `lock`,
`unlock` and `overlaps` counters using only `write()`, then calls `_exit`.
The controller also prints the overlap count whenever it changes. Heap
corruption needs two tasks inside the allocator at once, and the second
one increments `overlaps` in the lock hook before it touches heap state,
so the counter is already set by the time a race causes a crash. The make
target passes only if the exit code is non-zero **and** an `overlaps>0`
line was printed. Observed (`logs/freertos_negative.log`):
`NEGATIVE CONTROL: caught signal 11, lock hooks: lock=19468 unlock=19467
overlaps=768`, exit code 139. The crash is therefore attributed to
concurrent allocator entry. The crash point and counts vary from run to
run.

### Bare metal (`baremetal/`, `make baremetal`)
The image contains its own vector table and reset handler (copies `.data`,
zeroes `.bss`), a linker script for 256 KiB flash at `0x0` and 64 KiB SRAM
at `0x20000000`, and semihosting (`bkpt 0xab`, `SYS_WRITE0`, and `SYS_EXIT`
with reason `0x20026` for pass or `0x20023` for fail). It is linked with
`-nostdlib`, so no newlib is used. It tests the production allocator build
(no `TLSF_STATS`) with:
- init validation, alignment over 16 start offsets, and split/coalesce in
  all four neighbour cases
- exact-fit reuse, and exhaustion/recovery (1,475 × 1-byte and
  113 × 200-byte allocations in 24 KiB)
- two independent heaps, and the fit invariant for S = 16..1024
- a 30,000-op seeded stress run with the inspector after every op

QEMU command:
`qemu-system-arm -M lm3s6965evb -nographic -monitor none -serial none
-semihosting-config enable=on,target=native -kernel build/baremetal.elf`
(run under `timeout 60`). Observed: `75096 checks, 0 failures`,
`BAREMETAL PASS`, QEMU exit code 0. The make target requires both exit
code 0 and the `BAREMETAL PASS` line. QEMU prints a harmless
`Timer with period zero, disabling` line from the emulated board.

- **Failure path** (`make baremetal-negative`). The image is built with
  `-DFORCE_FAIL`, which adds one deliberately false `CHECK`. The target
  passes only if QEMU exits with code 1 and the log contains
  `BAREMETAL FAIL`. Observed (`logs/baremetal_negative.log`):
  `CHECK FAILED line 330: TLSF_ALIGN == 0`, `75097 checks, 1 failures`,
  `BAREMETAL FAIL`, QEMU exit code 1.
- **4-byte alignment** (`make baremetal-a4`). The same suite is built with
  `-DTLSF_ALIGN_LOG2=2`, which is valid on Cortex-M3 because
  `sizeof(void*) = 4`. It gives an 8 B header, a 16 B minimum block and a
  1036 B control structure. Observed (`logs/baremetal_a4.log`):
  `85257 checks, 0 failures`, `BAREMETAL PASS`, QEMU exit code 0.

## Fragmentation measurements (`make stress`, Part B)

Definitions (for `F > 0`):
- `F` is the total free payload (sum of `size - HDR` over free blocks).
- `largest_free` is the payload of the largest free block.
- `L` is the largest request `tlsf_malloc` would satisfy right now. It is
  computed analytically from the highest non-empty class under good-fit
  rounding, and cross-checked against real allocations in Part A.
- **Physical external fragmentation** `phys_frag = 1 - largest_free/F`
  describes how the free memory is split into pieces.
- **Good-fit request fragmentation** `ext_frag = 1 - L/F` describes what a
  caller can actually allocate. Because the lookup rounds a request up to
  the next class boundary, `L` can be up to about 1/16 of the block size
  (plus the header) smaller than `largest_free`. This metric is therefore
  non-zero even when the heap holds a single free block.
- The gap `ext_frag - phys_frag` is lookup rounding, not physical
  fragmentation.

Each workload makes random slot toggles (alloc if empty, free if live) on a
host build (64-bit: 16-byte header). Snapshots are taken every 500 ops,
the mean and max skip the first 10%, and the raw data is in
`logs/frag_<name>.csv`.

| workload | sizes | seed | heap | slots | live requested | used footprint | F | largest_free | L | final phys_frag | mean phys_frag | max phys_frag | final ext_frag | mean ext_frag | max ext_frag | failed/attempted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| small_uniform | U[8,256] | 0x1001 | 256 KiB | 1024 | 66,462 | 78,464 | 180,768 | 169,240 | 163,824 | 0.064 | 0.056 | 0.110 | 0.094 | 0.088 | 0.126 | 0/50,253 |
| mixed | 70% U[16,128], 25% U[129,2048], 5% U[2049,16384] | 0x1002 | 256 KiB | 256 | 86,999 | 89,888 | 169,840 | 117,664 | 114,672 | 0.307 | 0.367 | 0.771 | 0.325 | 0.381 | 0.779 | 0/50,063 |
| pow2 | 2^k, k∈U[3,12] | 0x1003 | 256 KiB | 256 | 92,944 | 95,376 | 164,352 | 94,560 | 94,192 | 0.425 | 0.425 | 0.663 | 0.427 | 0.438 | 0.676 | 0/50,068 |
| pressure | U[16,512], heap oversubscribed | 0x1004 | 128 KiB | 2048 | 107,749 | 124,448 | 3,584 | 224 | 224 | 0.938 | 0.930 | 0.972 | 0.938 | 0.931 | 0.972 | 31,774/66,247 |

`docs/fragmentation.png` plots both metrics, their difference (the
lookup-rounding gap), and `F`, `largest_free` and `L` over time. The CSV
columns are `op, live_requested, used_footprint, free_payload_F,
largest_alloc_L, largest_free, ext_frag, phys_frag, failed_allocs`.

How to read the table:
- These are statistics for these workloads only, not guarantees.
- In the *pressure* workload the heap is essentially full (3.5 KiB free out
  of 128 KiB), so its ratio describes a few scattered small holes.
- `used footprint - live requested` is the header and rounding overhead.
- The 1/16 class width bounds only the rounding of the class *lookup*. It
  does not bound total fragmentation.
- In *small_uniform* about a third of the final `ext_frag` of 0.094 is
  lookup rounding: the physical figure is 0.064, and the plotted gap is
  mostly 0.02–0.045. For *mixed*, *pow2* and *pressure* the gap is small
  (≤ 0.02 at the end) and the fragmentation is mostly physical.

## Reproduction

Prerequisites:
- `gcc`, GNU Make and `bash`
- `arm-none-eabi-gcc`/binutils and `qemu-system-arm`
- Python 3 with matplotlib, used by `audit` and `plots`. The Makefile runs
  `python3` from `PATH` by default. For a virtualenv, pass the interpreter
  explicitly, for example `make PY=/path/to/venv/bin/python all`.
- A FreeRTOS kernel checkout at `./FreeRTOS-Kernel` (this workspace uses
  `V11.1.0+`). Override the location with `make FRTOS=/path/to/FreeRTOS-Kernel`.

From the workspace root:

```
make all            # size host-sizes audit unit unit-align stress asan freertos
                    # freertos-negative baremetal baremetal-a4 baremetal-negative plots
make size           # -> logs/size.log     (text must be < 2048 or the target fails)
make host-sizes     # -> logs/host_sizes.log (production layout sizes, align 8/16/32)
make audit          # -> logs/audit.log    (CFG loop/recursion audit of the ARM object)
make unit           # -> logs/unit.log
make unit-align     # -> logs/unit_align.log (TLSF_ALIGN_LOG2 = 4 and 5)
make stress         # -> logs/stress.log, logs/frag_*.csv   (STRESS_OPS=200000 default)
make asan           # -> logs/asan.log
make freertos       # -> logs/freertos.log
make freertos-negative  # -> logs/freertos_negative.log (expects failure with overlaps>0)
make baremetal      # -> logs/baremetal.log
make baremetal-a4   # -> logs/baremetal_a4.log (TLSF_ALIGN_LOG2 = 2 on Cortex-M3)
make baremetal-negative # -> logs/baremetal_negative.log (expects BAREMETAL FAIL, exit 1)
make plots          # runs stress first; -> docs/fragmentation.png, docs/opcounts.png, logs/plots.log
make toolchain      # -> logs/toolchain.log (tool paths/versions, FreeRTOS kernel version)
make record         # make all, full output -> logs/make_all.log (fails if make all fails)
```

`plots` depends on `stress`, so a standalone `make plots` and parallel
`make -j all` both produce the CSVs and `logs/stress.log` first. A
`make -j8 all` into separate `B=`/`L=` build and log directories also
passed (not retained).

The last clean run was

```
rm -rf build logs docs && make toolchain record
```

It took 88 s and exited with 0. Its complete output is in
`logs/make_all.log`, and the tool versions are in `logs/toolchain.log`:
- gcc 13.3.0
- arm-none-eabi-gcc 13.2.1
- QEMU 8.2.2
- GNU Make 4.3
- Python 3.12.3 with matplotlib 3.11.2 (`python3` from `PATH`, a venv here)

## Limitations

- **Good fit, not best fit.** A request can fail even though a free block
  big enough for it exists in the request's own class, because the lookup
  rounds up to the next class. The waste is at most 1/16 of the size. In
  `test_fit_invariant`, which deliberately samples requests just below
  hole sizes, 68,608 of the 578,705 sampled requests that would have fit
  in the hole (11.9%) were refused this way.
- **1 MiB maximum region per heap** by default (`TLSF_MAX_REGION_LOG2`).
  Heaps cannot be grown or given extra pools after init.
- **No validation of `free` arguments.** Double frees, foreign pointers and
  writes past the end of a block are undefined behaviour and are not
  detected at run time. The inspector is for tests only.
- **Lock hooks.** Their time is not included in the O(1) bound. Calling the
  allocator from an ISR needs hooks that are ISR-safe (for example, masking
  interrupts), and the FreeRTOS mutex used in the test is not ISR-safe.
- **32-bit coverage.** No 32-bit host multilib was available (`gcc -m32`
  failed to link), so 32-bit behaviour was tested only on Cortex-M3 under
  QEMU. QEMU is not cycle-accurate, so no cycle counts are reported. The
  bound is stated in primitive operations and instructions without loops.
- **Compiler.** GCC or Clang builtins are required unless the user defines
  `TLSF_CLZ`/`TLSF_CTZ`.
- **Citations.** The papers and books in the design comparison ([1]–[4])
  were not re-checked during this build because web access was disabled.
- **ASan scope.** The sanitizer run cannot detect overruns between blocks
  inside a heap (see Tests); the allocator's internal consistency rests on
  the inspector and pattern checks.
