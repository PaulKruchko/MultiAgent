## Summary

Two issues raised and both fixed.

## Issues

- [critical] GPT-1: free() does not validate that the block lies inside the region
- [minor] GPT-2: README lacks a build example

## Rulings

- GPT-1 [fix]: a debug-build range check is cheap and the requirement list implies robustness

## Applied Fixes

- GPT-1: added `ALLOC_DEBUG` range check
- GPT-2: README example added

## Unresolved Critical

None.

## Verdict

PASS
