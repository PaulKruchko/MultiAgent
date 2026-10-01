# Examples

This folder holds the real material from maf's development runs (2026-09-28 to 2026-09-30, Ubuntu 24.04, Claude Code 2.1.284, default tier). The briefs and the allocator were not edited after the runs. The [top-level README](../README.md) has the full tutorials.

## `briefs/`: the exact briefs used in development

Each file is identical to the `brief:` recorded in its run's `run.md`, plus a trailing newline, which `$(cat FILE)` strips. Pass one to `maf run` with `"$(cat FILE)"`:

```bash
maf run "$(cat examples/briefs/smoke-tlsf-vs-buddy.txt)" --budget 3
```

Use a $3 cap for the smoke brief, not the $2 of the development run. That run predates the source audit, which now adds Gemini calls to the cross-check, and it had under $0.01 of headroom at its final call.

| Brief | Mode | Budget | Spent | Rounds | Outcome |
|---|---|---|---|---|---|
| [`smoke-tlsf-vs-buddy.txt`](briefs/smoke-tlsf-vs-buddy.txt): one-page TLSF vs buddy briefing | prose | $2 | $0.75 | 1 | `completed` in about 5 min. 11 notes; the cross-check raised 7 issues (0 critical), verdict PASS. |
| [`portable-allocator.txt`](briefs/portable-allocator.txt): research, design, implement and verify a portable C99 allocator for bare-metal, FreeRTOS and POSIX | code | $25 | $8.29 | 1 | `completed` in about 43 min, verdict PASS (14 issues, 0 critical). The deliverable is `tlsf-allocator/` below. |
| [`fusion-burn-control-thesis.txt`](briefs/fusion-burn-control-thesis.txt): Master's-thesis-quality study of 0-D burn control of a D-T plasma, with simulations | mixed | raised to $55.76 | $45.65 | 3 | `completed_with_issues`: 4 criteria partial, 2 relaxed, 0 critical. It was resumed after a timeout and a budget stop. The thesis is not included here. |

Runs cost real money and are not deterministic, so a rerun produces different work at a different cost. The allocator brief also needs the ARM toolchain, QEMU and a FreeRTOS kernel clone on the host, because Claude Code works without network access. See [Installation](../README.md#installation).

## `tlsf-allocator/`: the verified allocator deliverable

This is a byte-for-byte copy of the `deliverables/` folder of run `2026-09-28-research-design-implement-and-verify-a-portable-2`, as re-exported with `maf export`: 41 files, 888.9 kB. It is a C99 TLSF allocator:

- 692 B of `.text` on Cortex-M3 `-Os`;
- no libc;
- O(1) `malloc`/`free`;
- caller-supplied heaps with optional lock hooks.

| Suite | Result |
|---|---|
| POSIX unit tests | 1,774,559 checks |
| Stress, plus ASan and UBSan | 3 × 200k ops |
| FreeRTOS POSIX simulator | 4 tasks × 50k iterations |
| Bare-metal Cortex-M3 under QEMU `lm3s6965evb` | 75,096 checks |

A separate post-run check added about 85M adversarial fuzz operations and found no bugs; its harness is not included here. Its own [README](tlsf-allocator/README.md) documents the API, the design and every measurement. `logs/` holds the output of the recorded test run.

### Running it outside a maf workspace

That README was written inside the run's workspace. It assumes a kernel at `./FreeRTOS-Kernel` and says "from the workspace root". From the root of your checkout:

```bash
repo=$PWD
rm -rf /tmp/tlsf && cp -r examples/tlsf-allocator /tmp/tlsf && cd /tmp/tlsf     # make rewrites logs/ and docs/*.png
make -j8 all FRTOS="$HOME/.local/share/maf/FreeRTOS-Kernel" PY="$repo/.venv/bin/python"
```

- **`FRTOS=`** points at a FreeRTOS-Kernel clone, which is not shipped. Development used commit `8be86d4a24fd` (V11.1.0+).
- **`PY=`** must be a Python with matplotlib, for the `plots` target.
- **Expected run time:** about 40 s with `-j8` on the development machine.
- **Host-only subset:** `make host-sizes unit unit-align stress asan`.
- **Flaky target:** `freertos-negative`, the FreeRTOS test with the mutex removed, relies on a race and fails to trigger in up to about 8% of runs. If it reports `NEGATIVE CONTROL NOT TRIGGERED`, re-run it with the same kernel path (without `FRTOS=` it fails to compile): `make freertos-negative FRTOS="$HOME/.local/share/maf/FreeRTOS-Kernel"`.

Caveats:

- **No maf gate verified it.** The run predates maf's acceptance and clean-room gates. Its reproducibility was checked by hand, with `make all` on a clean copy.
- **Developer path in a log.** `logs/toolchain.log` records the developer's own interpreter path, as captured during the run.
