## Summary

TLSF implemented, and all three test targets pass.

## Artifacts

- `src/alloc.c` - allocator implementation
- `test/posix_test.log` - POSIX unit and stress test output

## Implementation Notes

Uses `__builtin_clz`; lock hooks are function pointers.

## Verification

`make test` passes: 42 tests. `arm-none-eabi-size` gives text=1804.

## Known Limitations

None.
