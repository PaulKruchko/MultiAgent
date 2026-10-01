# Top-level build for the TLSF allocator, its tests and measurements.
# Every test target tees its output into logs/.
#
# Prerequisites: gcc, arm-none-eabi-gcc, qemu-system-arm, Python 3 with
# matplotlib (override the interpreter with `make PY=/path/to/python`), and a
# FreeRTOS kernel checkout in ./FreeRTOS-Kernel (override with FRTOS=...).

SHELL   := /bin/bash
CC      ?= gcc
ARMCC   := arm-none-eabi-gcc
ARMSIZE := arm-none-eabi-size
PY      ?= python3

B := build
L := logs

HOST_CFLAGS := -std=c99 -O2 -g -Wall -Wextra -Wpedantic -Werror -Isrc -Itests
M3_FLAGS    := -std=c99 -mcpu=cortex-m3 -mthumb -Os -ffreestanding \
               -Wall -Wextra -Wpedantic -Werror

TLSF_SRC  := src/tlsf.c src/tlsf.h src/tlsf_internal.h
CHECK_SRC := tests/tlsf_check.c tests/tlsf_check.h

STRESS_OPS ?= 200000

.PHONY: all size host-sizes audit unit unit-align stress asan freertos freertos-negative \
        baremetal baremetal-a4 baremetal-negative plots toolchain record clean

all: size host-sizes audit unit unit-align stress asan freertos freertos-negative \
     baremetal baremetal-a4 baremetal-negative plots

$(B) $(L):
	mkdir -p $@

# ---- Code size: allocator object alone, Cortex-M3, -Os -------------------
$(B)/tlsf_m3.o: $(TLSF_SRC) | $(B)
	$(ARMCC) $(M3_FLAGS) -c src/tlsf.c -o $@

size: $(B)/tlsf_m3.o | $(L)
	{ echo '$$ $(ARMCC) $(M3_FLAGS) -c src/tlsf.c -o $(B)/tlsf_m3.o'; \
	  $(ARMCC) --version | head -1; \
	  echo '$$ $(ARMSIZE) $(B)/tlsf_m3.o'; $(ARMSIZE) $(B)/tlsf_m3.o; \
	  echo '$$ $(ARMSIZE) -A $(B)/tlsf_m3.o'; $(ARMSIZE) -A $(B)/tlsf_m3.o; \
	  echo '$$ arm-none-eabi-nm -u $(B)/tlsf_m3.o  (undefined symbols)'; \
	  arm-none-eabi-nm -u $(B)/tlsf_m3.o; echo '(end of undefined symbols)'; \
	  echo '$$ arm-none-eabi-nm -S --size-sort $(B)/tlsf_m3.o'; \
	  arm-none-eabi-nm -S --size-sort $(B)/tlsf_m3.o; \
	  for a in 2 4 5; do \
	    $(ARMCC) $(M3_FLAGS) -DTLSF_ALIGN_LOG2=$$a -c src/tlsf.c -o $(B)/tlsf_m3_a$$a.o && \
	    echo "TLSF_ALIGN_LOG2=$$a:" && $(ARMSIZE) $(B)/tlsf_m3_a$$a.o | tail -1; done; \
	  $(ARMCC) $(M3_FLAGS) -c src/tlsf.c -S -o $(B)/tlsf_m3.s; \
	  echo 'CLZ/RBIT instructions in generated code:'; grep -cE '\b(clz|rbit)\b' $(B)/tlsf_m3.s; \
	  text=$$($(ARMSIZE) $(B)/tlsf_m3.o | awk 'NR==2{print $$1}'); \
	  if [ $$text -lt 2048 ]; then echo "SIZE CHECK PASS: text=$$text < 2048"; \
	  else echo "SIZE CHECK FAIL: text=$$text >= 2048"; exit 1; fi; \
	} 2>&1 | tee $(L)/size.log; test $${PIPESTATUS[0]} -eq 0

# ---- Production (no TLSF_STATS) layout sizes on the host ---------------------
host-sizes: tests/host_sizes.c $(TLSF_SRC) | $(B) $(L)
	{ for a in 3 4 5; do \
	    $(CC) $(HOST_CFLAGS) -DTLSF_ALIGN_LOG2=$$a tests/host_sizes.c -o $(B)/host_sizes_a$$a && \
	    $(B)/host_sizes_a$$a || exit 1; done; } 2>&1 | tee $(L)/host_sizes.log; \
	test $${PIPESTATUS[0]} -eq 0

# ---- Machine-code audit: no loops / recursion in malloc/free paths ----------
audit: $(B)/tlsf_m3.o | $(L)
	{ echo '$$ arm-none-eabi-objdump -d $(B)/tlsf_m3.o > $(B)/tlsf_m3.dis'; \
	  arm-none-eabi-objdump -d $(B)/tlsf_m3.o > $(B)/tlsf_m3.dis && \
	  echo '$$ $(PY) tools/check_branches.py $(B)/tlsf_m3.dis' && \
	  $(PY) tools/check_branches.py $(B)/tlsf_m3.dis; } 2>&1 | tee $(L)/audit.log; \
	test $${PIPESTATUS[0]} -eq 0

# ---- POSIX unit tests --------------------------------------------------------
$(B)/test_unit: tests/test_unit.c tests/opcount.h $(TLSF_SRC) $(CHECK_SRC) | $(B)
	$(CC) $(HOST_CFLAGS) -DTLSF_STATS tests/test_unit.c src/tlsf.c tests/tlsf_check.c -o $@

unit: $(B)/test_unit | $(L)
	$(B)/test_unit 2>&1 | tee $(L)/unit.log; test $${PIPESTATUS[0]} -eq 0

# Same unit tests with non-default alignments (16 and 32 bytes).
unit-align: $(TLSF_SRC) $(CHECK_SRC) tests/test_unit.c | $(B) $(L)
	{ for a in 4 5; do \
	    $(CC) $(HOST_CFLAGS) -DTLSF_STATS -DTLSF_ALIGN_LOG2=$$a tests/test_unit.c \
	      src/tlsf.c tests/tlsf_check.c -o $(B)/test_unit_a$$a || exit 1; \
	    $(B)/test_unit_a$$a || exit 1; done; } 2>&1 | tee $(L)/unit_align.log; \
	test $${PIPESTATUS[0]} -eq 0

# ---- Seeded randomized stress + fragmentation workloads ----------------------
$(B)/test_stress: tests/test_stress.c tests/opcount.h $(TLSF_SRC) $(CHECK_SRC) | $(B)
	$(CC) $(HOST_CFLAGS) -DTLSF_STATS tests/test_stress.c src/tlsf.c tests/tlsf_check.c -o $@

stress: $(B)/test_stress | $(L)
	$(B)/test_stress $(STRESS_OPS) $(L) 2>&1 | tee $(L)/stress.log; test $${PIPESTATUS[0]} -eq 0

# ---- Sanitizer build (ASan + UBSan) of unit + stress ------------------------
asan: $(TLSF_SRC) $(CHECK_SRC) tests/test_unit.c tests/test_stress.c | $(B) $(L)
	{ $(CC) $(HOST_CFLAGS) -O1 -fsanitize=address,undefined -fno-sanitize-recover=all \
	    -fno-omit-frame-pointer -DTLSF_STATS tests/test_unit.c src/tlsf.c tests/tlsf_check.c \
	    -o $(B)/test_unit_asan && $(B)/test_unit_asan && \
	  $(CC) $(HOST_CFLAGS) -O1 -fsanitize=address,undefined -fno-sanitize-recover=all \
	    -fno-omit-frame-pointer -DTLSF_STATS tests/test_stress.c src/tlsf.c tests/tlsf_check.c \
	    -o $(B)/test_stress_asan && $(B)/test_stress_asan 100000 $(B); \
	} 2>&1 | tee $(L)/asan.log; test $${PIPESTATUS[0]} -eq 0

# ---- FreeRTOS POSIX simulator, shared heap, mutex hooks ----------------------
FRTOS ?= FreeRTOS-Kernel
FRTOS_SRC := $(FRTOS)/tasks.c $(FRTOS)/queue.c $(FRTOS)/list.c $(FRTOS)/timers.c \
             $(FRTOS)/event_groups.c $(FRTOS)/portable/MemMang/heap_3.c \
             $(FRTOS)/portable/ThirdParty/GCC/Posix/port.c \
             $(FRTOS)/portable/ThirdParty/GCC/Posix/utils/wait_for_event.c
FRTOS_INC := -Ifreertos -I$(FRTOS)/include -I$(FRTOS)/portable/ThirdParty/GCC/Posix \
             -I$(FRTOS)/portable/ThirdParty/GCC/Posix/utils

$(B)/freertos_test: freertos/main.c freertos/FreeRTOSConfig.h $(TLSF_SRC) $(CHECK_SRC) | $(B)
	$(CC) -std=gnu99 -O2 -g -Wall -Wextra -Isrc -Itests $(FRTOS_INC) freertos/main.c \
	  src/tlsf.c tests/tlsf_check.c $(FRTOS_SRC) -pthread -o $@

freertos: $(B)/freertos_test | $(L)
	timeout 120 $(B)/freertos_test 2>&1 | tee $(L)/freertos.log; test $${PIPESTATUS[0]} -eq 0

# Negative control: same test with the mutex removed from the hooks.  The
# target succeeds only if the program fails (non-zero exit, including a crash
# caught by its SIGSEGV/SIGABRT handler) AND reports overlapping allocator
# entries (overlaps>0), so the failure is attributed to missing locking.
freertos-negative: freertos/main.c freertos/FreeRTOSConfig.h $(TLSF_SRC) $(CHECK_SRC) | $(B) $(L)
	$(CC) -std=gnu99 -O2 -g -Wall -Wextra -DNEGATIVE_CONTROL -Isrc -Itests $(FRTOS_INC) \
	  freertos/main.c src/tlsf.c tests/tlsf_check.c $(FRTOS_SRC) -pthread -o $(B)/freertos_negative
	{ timeout 120 $(B)/freertos_negative > $(B)/freertos_negative.out 2>&1; rc=$$?; \
	  cat $(B)/freertos_negative.out; echo "exit code: $$rc"; \
	  if [ $$rc -ne 0 ] && grep -Eq 'overlaps=[1-9]' $(B)/freertos_negative.out; then \
	    echo "NEGATIVE CONTROL OK: failing run with overlapping allocator entries reported"; \
	  else echo "NEGATIVE CONTROL NOT TRIGGERED (need exit!=0 and overlaps>0)"; exit 1; fi; \
	} 2>&1 | tee $(L)/freertos_negative.log; test $${PIPESTATUS[0]} -eq 0

# ---- Bare-metal Cortex-M3 under QEMU (lm3s6965evb) + semihosting ------------
BM_SRC := baremetal/startup.c baremetal/semihost.c baremetal/main.c src/tlsf.c tests/tlsf_check.c
BM_DEP := $(BM_SRC) baremetal/semihost.h baremetal/lm3s6965.ld $(TLSF_SRC) $(CHECK_SRC) | $(B)
BM_LINK = $(ARMCC) $(M3_FLAGS) -g -Isrc -Itests -Ibaremetal -nostdlib -Wl,-T,baremetal/lm3s6965.ld \
	  -Wl,--gc-sections -ffunction-sections -Wl,-Map,$(@:.elf=.map) $(BM_SRC) -lgcc -o $@
QEMU_CMD := qemu-system-arm -M lm3s6965evb -nographic -monitor none -serial none \
	    -semihosting-config enable=on,target=native -kernel

# $(call qemu_run,elf): print the command, versions, image size, run under a
# 60 s timeout and print QEMU's exit code; the subshell exits with that code.
define qemu_run
( echo '$$ $(QEMU_CMD) $(1)'; qemu-system-arm --version | head -1; $(ARMSIZE) $(1); \
	  timeout 60 $(QEMU_CMD) $(1); rc=$$?; echo "qemu exit code: $$rc"; exit $$rc )
endef

$(B)/baremetal.elf: $(BM_DEP)
	$(BM_LINK)

# Same suite with 4-byte alignment (valid on Cortex-M3: sizeof(void*) == 4).
$(B)/baremetal_a4.elf: $(BM_DEP)
	$(BM_LINK) -DTLSF_ALIGN_LOG2=2

# Failure-path image: one deliberately false CHECK (-DFORCE_FAIL).
$(B)/baremetal_neg.elf: $(BM_DEP)
	$(BM_LINK) -DFORCE_FAIL

baremetal: $(B)/baremetal.elf | $(L)
	$(call qemu_run,$<) 2>&1 | tee $(L)/baremetal.log; \
	test $${PIPESTATUS[0]} -eq 0 && grep -q '^BAREMETAL PASS' $(L)/baremetal.log

baremetal-a4: $(B)/baremetal_a4.elf | $(L)
	$(call qemu_run,$<) 2>&1 | tee $(L)/baremetal_a4.log; \
	test $${PIPESTATUS[0]} -eq 0 && grep -q '^BAREMETAL PASS' $(L)/baremetal_a4.log

# Passes only if the forced failure is reported: QEMU exit code 1 and
# a 'BAREMETAL FAIL' line.
baremetal-negative: $(B)/baremetal_neg.elf | $(L)
	{ $(call qemu_run,$<); rc=$$?; \
	  if [ $$rc -eq 1 ]; then echo "BAREMETAL NEGATIVE: QEMU exit code 1 as expected"; \
	  else echo "BAREMETAL NEGATIVE: unexpected QEMU exit code $$rc"; exit 1; fi; \
	} 2>&1 | tee $(L)/baremetal_negative.log; \
	test $${PIPESTATUS[0]} -eq 0 && grep -q '^BAREMETAL FAIL' $(L)/baremetal_negative.log

# ---- Plots --------------------------------------------------------------------
plots: stress | $(L)
	$(PY) tools/plot.py $(L) docs 2>&1 | tee $(L)/plots.log; test $${PIPESTATUS[0]} -eq 0

# ---- Provenance logs -----------------------------------------------------------
toolchain: | $(L)
	{ for t in gcc arm-none-eabi-gcc arm-none-eabi-size qemu-system-arm make; do \
	    echo "\$$ which $$t: $$(which $$t)"; $$t --version | head -1; done; \
	  echo "\$$ which $(PY): $$(which $(PY))"; $(PY) --version; \
	  $(PY) -c 'import matplotlib; print("matplotlib", matplotlib.__version__)'; \
	  uname -srm; \
	  grep -m1 'define tskKERNEL_VERSION_NUMBER' $(FRTOS)/include/task.h; \
	  grep -m1 '^Changes between' $(FRTOS)/History.txt; \
	} 2>&1 | tee $(L)/toolchain.log; test $${PIPESTATUS[0]} -eq 0

# Full run with its complete output captured in logs/make_all.log.
record:
	mkdir -p $(L)
	$(MAKE) --no-print-directory all 2>&1 | tee $(L)/make_all.log; test $${PIPESTATUS[0]} -eq 0

clean:
	rm -rf $(B)
